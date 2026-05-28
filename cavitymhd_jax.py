"""
cavity_mhd_jax.py  -  2D lid-driven MHD cavity, fully functional JAX implementation.

Poisson solver: conjugate gradient (matrix-free, JIT-compatible) or SOR (numba).
Time stepping:  explicit RK2 Heun with pressure projection.
Solver design:  pure functions, NamedTuple state.

Solver roadmap (swap via params.solver, callers unchanged):
  'sor'      Gauss-Seidel SOR via numba (sequential, not JIT-able in JAX)
  'cg'       Conjugate gradient via jax.scipy (matrix-free, JIT-compatible) <- default
  'bicgstab' coming in v3
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg, bicgstab
from functools import partial
from numba import njit
import matplotlib.pyplot as plt
from typing import NamedTuple

# ────────────────────────────────────────────────────────────────────
# Numba SOR kernels
# ────────────────────────────────────────────────────────────────────

@njit(cache=True, fastmath=True)
def _sor_sweep(p, Q, alpha, h, N):
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            p[i, j] = p[i, j] + (alpha / 4.0) * (
                p[i-1, j] + p[i+1, j]
                + p[i, j-1] + p[i, j+1]
                - h * h * Q[i-1, j-1] - 4.0 * p[i, j]
            )

@njit(cache=True, fastmath=True)
def _poisson_residual_max(p, Q, h, N):
    res_max = 0.0
    for i in range(N):
        for j in range(N):
            ii, jj = i + 1, j + 1
            lap = (p[ii-1, jj] + p[ii+1, jj]
                   + p[ii, jj-1] + p[ii, jj+1]
                   - 4.0 * p[ii, jj]) / (h * h)
            r = abs(Q[i, j] - lap)
            if r > res_max:
                res_max = r
    return res_max

# ────────────────────────────────────────────────────────────────────
# Params and State
# ────────────────────────────────────────────────────────────────────

class Params(NamedTuple):
    Re:     float
    N_int:  float
    N:      int
    H:      float
    dx:     float
    dy:     float
    dt:     float
    u_lid:  jnp.ndarray   # (N+1,)
    Bx:     jnp.ndarray   # (N+2, N+1)
    By:     jnp.ndarray   # (N+1, N+2)
    Bz:     jnp.ndarray   # (N+1, N+1)
    solver: str           # 'cg' or 'sor'

class State(NamedTuple):
    u:          jnp.ndarray   # (N+2, N+1)
    v:          jnp.ndarray   # (N+1, N+2)
    p:          jnp.ndarray   # (N+2, N+2)
    phi:        jnp.ndarray   # (N+2, N+2)
    t:          float
    step_count: int

# ────────────────────────────────────────────────────────────────────
# Initialization
# ────────────────────────────────────────────────────────────────────

def _lid_profile(N, dx, H, lid_taper=0.01, coeff=1.0):
    x  = np.arange(N + 1) * dx
    xn = x / H
    left  = 0.5 * (1 + np.tanh((xn - lid_taper)       / (lid_taper / 3)))
    right = 0.5 * (1 - np.tanh((xn - (1 - lid_taper)) / (lid_taper / 3)))
    return coeff * left * right

def init(Re, N, B=[1, 0, 0], H=1.0, dt=None, lid_taper=0.01, N_int=0.0, solver='cg'):
    dx    = H / N
    dt    = dt if dt is not None else 0.2 * dx
    u_lid = _lid_profile(N, dx, H, lid_taper)
    params = Params(
        Re=Re, N_int=N_int, N=N, H=H, dx=dx, dy=dx, dt=dt,
        u_lid=jnp.array(u_lid),
        Bx=jnp.full((N+2, N+1), float(B[0])),
        By=jnp.full((N+1, N+2), float(B[1])),
        Bz=jnp.full((N+1, N+1), float(B[2])),
        solver=solver,
    )
    state = State(
        u=jnp.zeros((N+2, N+1)),
        v=jnp.zeros((N+1, N+2)),
        p=jnp.zeros((N+2, N+2)),
        phi=jnp.zeros((N+2, N+2)),
        t=0.0, step_count=0,
    )
    
    return params, state

# ────────────────────────────────────────────────────────────────────
# Poisson solvers
# ────────────────────────────────────────────────────────────────────

def _apply_neumann(p, N):
    p = p.at[:,  0].set(p[:,  1])
    p = p.at[:, -1].set(p[:, -2])
    p = p.at[ 0, :].set(p[ 1, :])
    p = p.at[-1, :].set(p[-2, :])
    return p

def _laplacian_op(p_flat, N, dx):
    p_int  = p_flat.reshape(N, N)
    p_full = jnp.zeros((N+2, N+2)).at[1:-1, 1:-1].set(p_int)
    p_full = _apply_neumann(p_full, N)
    lap = (p_full[:-2,  1:-1] + p_full[2:,   1:-1] +
           p_full[1:-1, :-2]  + p_full[1:-1, 2:] -
           4.0 * p_full[1:-1, 1:-1]) / dx**2
    return lap.ravel()

def _poisson_solve_bicgstab(field, source_interior, N, dx, tol=1e-7):
    src = source_interior - jnp.mean(source_interior)
    A   = partial(_laplacian_op, N=N, dx=dx)
    x0  = field[1:-1, 1:-1].ravel()
    x0  = x0 - jnp.mean(x0)
    sol, _ = bicgstab(A, -src.ravel(), x0=x0, tol=tol, maxiter=10*N*N)
    sol    = sol - jnp.mean(sol)
    p_full = jnp.zeros((N+2, N+2)).at[1:-1, 1:-1].set(sol.reshape(N, N))
    return _apply_neumann(p_full, N)

def _poisson_solve_sor(field, source_interior, N, dx,
                       alpha=1.7, tol=1e-7, max_iter=100_000, check_every=10):
    p      = np.array(field)
    source = np.array(source_interior)
    for it in range(max_iter):
        _sor_sweep(p, source, alpha, dx, N)
        p[:,  0] = p[:,  1];  p[:, -1] = p[:, -2]
        p[ 0, :] = p[ 1, :];  p[-1, :] = p[-2, :]
        if it % check_every == 0:
            if _poisson_residual_max(p, source, dx, N) < tol:
                break
    return jnp.array(p)

def _poisson_solve_jacobi(field, source_interior, N, dx, tol=1e-6, max_iter=50_000):
    # embed source into full grid
    rhs = jnp.zeros((N+2, N+2)).at[1:-1, 1:-1].set(source_interior)

    def body(carry):
        p, _ = carry
        p = _apply_neumann(p, N)
        p_new = (p[:-2, 1:-1] + p[2:, 1:-1] +
                 p[1:-1, :-2] + p[1:-1, 2:] -
                 dx**2 * rhs[1:-1, 1:-1]) / 4.0
        p_new = p_new - jnp.mean(p_new)
        p_full = jnp.zeros((N+2, N+2)).at[1:-1, 1:-1].set(p_new)
        res = jnp.max(jnp.abs(p_full[1:-1, 1:-1] - p[1:-1, 1:-1]))
        return p_full, res

    def cond(carry):
        _, res = carry
        return res > tol

    p_init = field - jnp.mean(field)
    p_out, _ = jax.lax.while_loop(cond, body, (p_init, jnp.inf))
    return _apply_neumann(p_out, N)

def _poisson_solve(field, source_interior, params):
    """Single dispatch point. Swap solver via params.solver."""
    if params.solver == 'bicgstab':
        return _poisson_solve_bicgstab(field, source_interior, params.N, params.dx)
    elif params.solver == 'sor':
        return _poisson_solve_sor(field, source_interior, params.N, params.dx)
    elif params.solver == 'jacobi':
        return _poisson_solve_jacobi(field, source_interior, params.N, params.dx)
    else:
        raise ValueError(f"Unknown solver '{params.solver}'. Choose 'bicgstab' or 'sor' or 'jacobi'.")

# ────────────────────────────────────────────────────────────────────
# Boundary conditions
# ────────────────────────────────────────────────────────────────────

def apply_bcs(state: State, params: Params) -> State:
    u, v, p, phi = state.u, state.v, state.p, state.phi
    u_lid = params.u_lid

    u = u.at[1:-1,  0].set(0.0)
    u = u.at[1:-1, -1].set(0.0)
    v = v.at[ 0, 1:-1].set(0.0)
    v = v.at[-1, 1:-1].set(0.0)
    u = u.at[ 0, :].set(2.0 * u_lid - u[1, :])
    u = u.at[-1, :].set(-u[-2, :])
    v = v.at[:,  0].set(-v[:,  1])
    v = v.at[:, -1].set(-v[:, -2])
    p   = _apply_neumann(p,   params.N)
    phi = _apply_neumann(phi, params.N)

    return state._replace(u=u, v=v, p=p, phi=phi)

# ────────────────────────────────────────────────────────────────────
# Interpolators
# ────────────────────────────────────────────────────────────────────

def _u_to_c(u, N): return 0.5 * (u[1:N+1, 0:N]   + u[1:N+1, 1:N+1])
def _v_to_c(v, N): return 0.5 * (v[0:N,   1:N+1] + v[1:N+1, 1:N+1])
def _u_to_n(u, N): return 0.5 * (u[0:N+1, :]     + u[1:N+2, :])
def _v_to_n(v, N): return 0.5 * (v[:,  0:N+1]    + v[:,  1:N+2])

def _v_to_u(v, N):
    return 0.25 * (v[0:N,   1:N] + v[0:N,   2:N+1]
                 + v[1:N+1, 1:N] + v[1:N+1, 2:N+1])

def _u_to_v(u, N):
    return 0.25 * (u[1:N,   0:N] + u[1:N,   1:N+1]
                 + u[2:N+1, 0:N] + u[2:N+1, 1:N+1])

def _n_to_u(c, N): return 0.5 * (c[0:N,   1:N] + c[1:N+1, 1:N])
def _n_to_v(c, N): return 0.5 * (c[1:N,   0:N] + c[1:N,   1:N+1])

# ────────────────────────────────────────────────────────────────────
# Operators
# ────────────────────────────────────────────────────────────────────

def grad(p, params):
    N, dx, dy = params.N, params.dx, params.dy
    return ((p[:, 1:N+2] - p[:, :N+1])  / dx,
            (p[:N+1, :]  - p[1:N+2, :]) / dy)

def div(u, v, params):
    N, dx, dy = params.N, params.dx, params.dy
    return ((u[1:N+1, 1:N+1] - u[1:N+1, :N])    / dx +
            (v[:N,   1:N+1]  - v[1:N+1, 1:N+1]) / dy)

def laplacian(u, v, params):
    N, dx, dy = params.N, params.dx, params.dy
    lap_x = ((u[1:N+1, 2:N+1] - 2*u[1:N+1, 1:N] + u[1:N+1, 0:N-1]) / dx**2
           + (u[0:N,   1:N]   - 2*u[1:N+1, 1:N] + u[2:N+2, 1:N])   / dy**2)
    lap_y = ((v[1:N,   2:N+2] - 2*v[1:N, 1:N+1] + v[1:N,   0:N])   / dx**2
           + (v[0:N-1, 1:N+1] - 2*v[1:N, 1:N+1] + v[2:N+1, 1:N+1]) / dy**2)
    return lap_x, lap_y

def cross(ax, ay, az, bx, by, bz, N):
    out_x = _v_to_u(ay, N)*_n_to_u(bz, N) - _n_to_u(az, N)*_v_to_u(by, N)
    out_y = _n_to_v(az, N)*_u_to_v(bx, N) - _u_to_v(ax, N)*_n_to_v(bz, N)
    out_z = _u_to_n(ax, N)*_v_to_n(by, N) - _v_to_n(ay, N)*_u_to_n(bx, N)
    return out_x, out_y, out_z

# ────────────────────────────────────────────────────────────────────
# Pressure projection
# ────────────────────────────────────────────────────────────────────

def project(state: State, params: Params) -> State:
    N, dt = params.N, params.dt
    source = div(state.u, state.v, params) / dt
    p_new  = _poisson_solve(state.p, source, params)
    state  = state._replace(p=p_new)
    gp_x, gp_y = grad(state.p, params)
    u = state.u.at[1:N+1, 1:N  ].add(-dt * gp_x[1:N+1, 1:N])
    v = state.v.at[1:N,   1:N+1].add(-dt * gp_y[1:N,   1:N+1])
    return state._replace(u=u, v=v)

# ────────────────────────────────────────────────────────────────────
# Momentum RHS
# ────────────────────────────────────────────────────────────────────

def compute_H(state: State, params: Params):
    u, v = state.u, state.v
    N, dx, dy  = params.N, params.dx, params.dy
    Re, N_int  = params.Re, params.N_int
    Bx, By, Bz = params.Bx, params.By, params.Bz
    uz         = jnp.zeros((N+1, N+1))

    uu_c = _u_to_c(u, N) ** 2
    vv_c = _v_to_c(v, N) ** 2
    uv_n = _u_to_n(u, N) * _v_to_n(v, N)

    H_x = -((uu_c[:, 1:] - uu_c[:, :-1]) / dx +
             (uv_n[0:N, 1:N] - uv_n[1:N+1, 1:N]) / dy)
    H_y = -((uv_n[1:N, 1:N+1] - uv_n[1:N, 0:N]) / dx +
             (vv_c[0:N-1, :] - vv_c[1:N, :]) / dy)

    lap_x, lap_y = laplacian(u, v, params)
    H_x = H_x + lap_x / Re
    H_y = H_y + lap_y / Re

    uxB_x, uxB_y, uxB_z = cross(u, v, uz, Bx, By, Bz, N)
    uxB_x_full = jnp.zeros((N+2, N+1)).at[1:N+1, 1:N  ].set(uxB_x)
    uxB_y_full = jnp.zeros((N+1, N+2)).at[1:N,   1:N+1].set(uxB_y)

    lorentz_src = div(uxB_x_full, uxB_y_full, params)
    phi_new     = _poisson_solve(state.phi, lorentz_src, params)
    state       = state._replace(phi=phi_new)

    gp_x, gp_y = grad(state.phi, params)
    j_x = jnp.zeros((N+2, N+1)).at[1:N+1, 1:N  ].set(-gp_x[1:N+1, 1:N]   + uxB_x)
    j_y = jnp.zeros((N+1, N+2)).at[1:N,   1:N+1].set(-gp_y[1:N,   1:N+1] + uxB_y)

    F_x, F_y, _ = cross(j_x, j_y, uxB_z, Bx, By, Bz, N)
    H_x = H_x + N_int * F_x
    H_y = H_y + N_int * F_y

    return H_x, H_y, state

# ────────────────────────────────────────────────────────────────────
# Time stepping: RK2 Heun
# ────────────────────────────────────────────────────────────────────

def step(state: State, params: Params) -> State:
    N, dt = params.N, params.dt
    u0, v0 = state.u, state.v

    H_x, H_y, state = compute_H(state, params)
    state = state._replace(
        u=u0.at[1:N+1, 1:N  ].set(u0[1:N+1, 1:N]   + dt * H_x),
        v=v0.at[1:N,   1:N+1].set(v0[1:N,   1:N+1] + dt * H_y),
    )
    state = apply_bcs(state, params)
    state = project(state, params)
    state = apply_bcs(state, params)

    H_x, H_y, state = compute_H(state, params)
    state = state._replace(
        u=u0.at[1:N+1, 1:N  ].set(0.5*(u0[1:N+1, 1:N]   + state.u[1:N+1, 1:N])   + 0.5*dt*H_x),
        v=v0.at[1:N,   1:N+1].set(0.5*(v0[1:N,   1:N+1] + state.v[1:N,   1:N+1]) + 0.5*dt*H_y),
    )
    state = apply_bcs(state, params)
    state = project(state, params)
    state = apply_bcs(state, params)

    return state._replace(t=state.t + dt, step_count=state.step_count + 1)

# ────────────────────────────────────────────────────────────────────
# CFL utility
# ────────────────────────────────────────────────────────────────────

def cfl_dt(state: State, params: Params, safety=0.8) -> float:
    u_max   = float(jnp.max(jnp.abs(state.u))) + 1e-12
    v_max   = float(jnp.max(jnp.abs(state.v))) + 1e-12
    dt_conv = min(params.dx / u_max, params.dy / v_max)
    dt_diff = 0.5 * min(params.dx, params.dy)**2 * params.Re
    return safety * min(dt_conv, dt_diff)

# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────

def run(params: Params, state: State,
        n_steps=None, t_end=None, steady_tol=None, log_every=200):
    import time
    state  = apply_bcs(state, params)
    u_prev = state.u
    v_prev = state.v
    t0     = time.perf_counter()

    while True:
        if n_steps is not None and state.step_count >= n_steps: break
        if t_end   is not None and state.t >= t_end:            break

        state = step(state, params)

        if state.step_count % log_every == 0:
            jax.block_until_ready(state.u)
            du      = state.u - u_prev
            dv      = state.v - v_prev
            rms     = float(jnp.sqrt((jnp.sum(du**2) + jnp.sum(dv**2))
                            / (du.size + dv.size))) / params.dt
            max_div = float(jnp.max(jnp.abs(div(state.u, state.v, params))))
            elapsed = time.perf_counter() - t0
            print(f"step {state.step_count:6d}  t={state.t:.3f}  "
                  f"rms_dudt={rms:.3e}  max|div|={max_div:.2e}  "
                  f"wall={elapsed:.1f}s  [{params.solver}]")
            if steady_tol is not None and rms < steady_tol:
                print("Steady state reached.")
                break
            u_prev = state.u
            v_prev = state.v

    return state

# ────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────

def plot_streamline(params: Params, state: State, folder=None, save=False):
    u = np.array(state.u)
    v = np.array(state.v)
    H, N, Re = params.H, params.N, params.Re

    u_cc  = 0.5 * (u[1:-1, :-1] + u[1:-1, 1:])
    v_cc  = 0.5 * (v[:-1, 1:-1] + v[1:,   1:-1])
    x     = np.linspace(0, H, N)
    y     = np.linspace(0, H, N)
    speed = np.sqrt(np.flipud(u_cc)**2 + np.flipud(v_cc)**2)

    fig, ax = plt.subplots(figsize=(6, 6))
    strm = ax.streamplot(x, y, np.flipud(u_cc), np.flipud(v_cc),
                         color=speed, cmap='plasma',
                         linewidth=0.8, density=3, arrowsize=0.6)
    plt.colorbar(strm.lines, ax=ax, label='|u|', shrink=0.8)
    ax.set_xlim(0, H); ax.set_ylim(0, H)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Streamlines  Re={Re}  N={N}  [{params.solver}]')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f'{folder}/streamlines_Re{Re}_N{N}_{params.solver}.png',
                    dpi=150, bbox_inches='tight')
    plt.show()

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dt = 0.05
    params, state = init(Re=5000.0, N=64, B=[1, 0, 0], dt=dt, N_int=0.4, solver='jacobi')

    safe = cfl_dt(state, params)
    print(f"Suggested CFL dt: {safe:.5f}")
    assert dt < safe, "Reduce dt below CFL limit"

    import time

    start = time.time()
    state = run(params, state, t_end=5, steady_tol=1e-3, log_every=100)
    plot_streamline(params, state)
    end = time.time()

    print("total run time:", end - start)