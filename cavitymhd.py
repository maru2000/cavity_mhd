import numpy as np
from numba import njit
from explicit_pressure_correction import *

# ────────────────────────────────────────────────────────────────────
# Module-level JIT kernels
# ────────────────────────────────────────────────────────────────────

# Traveling Magnetic Force
def tmf_forcing_static(sim, t, k=2*np.pi, fx=1.0, fy=0.0):
    """Static traveling-magnetic-field body force in the omega=0 limit.
    f = (sin(k*y), 0) — unidirectional, varies only with y.

    Returns (fx_at_u, fy_at_v) at the appropriate sublattices.
    """
    N, dx, dy = sim.N, sim.dx, sim.dy

    # u-faces live at cell-centered y. Row 0 = top of physical domain,
    # so physical y increases as row index decreases. We don't actually
    # need to invert — sin(k*y_anything) just sets the wave orientation.
    # Use array-row coordinate as y; flip later if the figure wants it.
    y_rows = (np.arange(N) + 0.5) * dx                     # (N,)
    fx_col = -np.sin(k * y_rows)[:, None]                   # (N, 1)
    fx_at_u = fx * np.broadcast_to(fx_col, (N, N - 1)).copy()   # (N, N-1)

    # v-faces: fy = 0 in this forcing
    x_cols = (np.arange(N) + 0.5) * dy                     # (N,)
    fy_col = -np.sin(k * x_cols)[None, :]                   # (1, N)
    fy_at_v = fy * np.broadcast_to(fy_col, (N - 1, N)).copy()   # (N-1, N)


    return fx_at_u, fy_at_v

def rmf_forcing_static(sim, t, omega=1.0):
    """Static rotating-magnetic-field body force in the omega->0 limit.
    
    f = omega * (-(y - y_c), (x - x_c)) — solid-body rotation forcing
    around the center (x_c, y_c) = (H/2, H/2).
    
    This is the RMF analog of tmf_forcing_static: a stylized DC body force
    that captures the qualitative feature of rotating-field stirring (a 
    swirling Lorentz force) without the full electromagnetic derivation.
    
    Returns (fx_at_u, fy_at_v) at the appropriate sublattices.
    """
    N, dx, dy = sim.N, sim.dx, sim.dy
    H = sim.H
    x_c, y_c = H / 2.0, H / 2.0

    # u-faces: x is face-centered (interior x-faces, N-1 of them),
    #          y is cell-centered (N of them).
    # fx = -omega * (y - y_c) — depends only on y, broadcast across x.
    y_rows = (np.arange(N) + 0.5) * dy                          # (N,)
    fx_col = -omega * (y_rows - y_c)[:, None]                   # (N, 1)
    fx_at_u = np.broadcast_to(fx_col, (N, N - 1)).copy()        # (N, N-1)

    # v-faces: x is cell-centered (N of them),
    #          y is face-centered (interior y-faces, N-1 of them).
    # fy = +omega * (x - x_c) — depends only on x, broadcast across y.
    x_cols = (np.arange(N) + 0.5) * dx                          # (N,)
    fy_row = omega * (x_cols - x_c)[None, :]                    # (1, N)
    fy_at_v = np.broadcast_to(fy_row, (N - 1, N)).copy()        # (N-1, N)

    return fx_at_u, fy_at_v

@njit(cache=True, fastmath=True)
def _sor_sweep(p, Q, alpha, h, N):
    """One Gauss-Seidel + over-relaxation sweep. Modifies p in place."""
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            p[i, j] = p[i, j] + (alpha / 4.0) * (
                p[i - 1, j] + p[i + 1, j]
                + p[i, j - 1] + p[i, j + 1]
                - h * h * Q[i - 1, j - 1] - 4.0 * p[i, j]
            )


@njit(cache=True, fastmath=True)
def _poisson_residual_max(p, Q, h, N):
    """Max abs residual |laplacian(p) - Q| over interior cells."""
    res_max = 0.0
    for i in range(N):
        for j in range(N):
            ii, jj = i + 1, j + 1
            lap = (p[ii - 1, jj] + p[ii + 1, jj]
                   + p[ii, jj - 1] + p[ii, jj + 1]
                   - 4.0 * p[ii, jj]) / (h * h)
            r = abs(Q[i, j] - lap)
            if r > res_max:
                res_max = r
    return res_max

def quiver_forcing(fx, fy, H, N, stride=4, title="TMF forcing"):
    fx_aug = np.zeros((N, N+1)); fx_aug[:, 1:N] = fx
    fx_cc = 0.5 * (fx_aug[:, :-1] + fx_aug[:, 1:])
    fy_aug = np.zeros((N+1, N)); fy_aug[1:N, :] = fy
    fy_cc = 0.5 * (fy_aug[:-1, :] + fy_aug[1:, :])

    fx_plot = np.flipud(fx_cc)
    fy_plot = np.flipud(fy_cc)

    x = np.linspace(0, H, N); y = np.linspace(0, H, N)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    s = stride
    ax.quiver(X[::s, ::s], Y[::s, ::s], fx_plot[::s, ::s], fy_plot[::s, ::s],
              scale=20)
    ax.set_xlim(0, H); ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout(); plt.show()

## Plot streamline
def plot_streamline(u, v, Re, H, N, N_int, folder=None, save=False):
    # interpolate staggered u, v to cell centers
    u_phys = u[1:-1, :]          # (N, N+1), strip ghost rows
    v_phys = v[:, 1:-1]          # (N+1, N), strip ghost cols

    u_cc = 0.5 * (u_phys[:, :-1] + u_phys[:, 1:])   # (N, N)
    v_cc = 0.5 * (v_phys[:-1, :] + v_phys[1:, :])   # (N, N)

    # coordinate arrays (row 0 = top in numpy, so flip y for physical orientation)
    x = np.linspace(0, H, N)
    y = np.linspace(0, H, N)

    # streamplot expects u,v on a meshgrid with origin at bottom-left
    # so flip the arrays vertically to match
    u_plot = np.flipud(u_cc)
    v_plot = np.flipud(v_cc)

    fig, ax = plt.subplots(figsize=(6, 6))
    speed = np.sqrt(u_plot**2 + v_plot**2)
    strm = ax.streamplot(x, y, u_plot, v_plot,
                        color=speed,
                        cmap='plasma',
                        linewidth=0.8,
                        density=3,
                        arrowsize=0.6)
    plt.colorbar(strm.lines, ax=ax, label='|u|', shrink=0.8)
    ax.set_xlim(0, H);  ax.set_ylim(0, H)
    ax.set_xlabel('x');  ax.set_ylabel('y')
    ax.set_title(f'Streamlines  Re={Re}  N={N} N_int={N_int}')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f'{folder}/streamlines_Re{Re}_N{N}_Nint{N_int}.png', dpi=150, bbox_inches='tight')
    plt.show()
# ────────────────────────────────────────────────────────────────────
# Solver class
# ────────────────────────────────────────────────────────────────────

class CavityFlow:
    """2D incompressible lid-driven cavity on a staggered grid.

    Layout (row 0 = top of physical domain):
        p : (N+2, N+2)   cell centers, ghost on all four sides
        u : (N+2, N+1)   x-faces,      ghost rows top and bottom
        v : (N+1, N+2)   y-faces,      ghost cols left and right

    Time stepping: explicit RK2 (Heun) with pressure projection at each stage.
    """

    # ── construction ────────────────────────────────────────────────

    def __init__(self, Re, N, B = [1, 0, 0], H=1.0, dt=None, lid_taper=0.01, N_int=0.0, forcing=None):
        
        self.Re = Re
        self.N  = N
        self.H  = H
        self.dx = H / N
        self.dy = self.dx
        self.dt = dt if dt is not None else 0.2 * self.dx

        # Magnetic terms
        self.N_int = N_int # Interaction parameter
        self.forcing = forcing # callable: forcing(x, y, t) -> (fx, fy)

        self.Bx = np.full((N+2, N+1), B[0]) # x-component
        self.By = np.full((N+1, N+2), B[1]) # y-component
        self.Bz = np.full((N+1, N+1), B[2]) # z-component
        self.phi = np.zeros((N+2, N+2)) # electric potential at cell centers

        self.u = np.zeros((N + 2, N + 1))
        self.v = np.zeros((N + 1, N + 2))
        self.p = np.zeros((N + 2, N + 2))

        self.u_lid = self._lid_profile(lid_taper)

        self.t = 0.0
        self.step_count = 0

    def _lid_profile(self, w, coeff=0.0):
        """Flat lid with smooth tanh taper near the corners."""
        x  = np.arange(self.N + 1) * self.dx
        xn = x / self.H
        left  = 0.5 * (1 + np.tanh((xn - w)       / (w / 3)))
        right = 0.5 * (1 - np.tanh((xn - (1 - w)) / (w / 3)))
        return coeff * left * right

    def cfl_dt(self, safety=0.8):
        u_max = max(np.max(np.abs(self.u)), 1e-12)
        v_max = max(np.max(np.abs(self.v)), 1e-12)
        dt_conv = min(self.dx / u_max, self.dy / v_max)
        dt_diff = 0.5 * min(self.dx, self.dy)**2 * self.Re

        dt = min(dt_conv, dt_diff)
        if self.forcing is not None and self.N_int != 0.0:
            # Crude bound: assume |f|_max ~ 1 for the analytic forcings.
            # If you ever use a forcing with larger amplitude, scale here.
            dt_force = np.sqrt(min(self.dx, self.dy) / abs(self.N_int))
            dt = min(dt, dt_force)
        return safety * dt

    def force_calc(self):
        """Calculate the body force term."""
        
    # ── boundary conditions ─────────────────────────────────────────

    def apply_bcs(self):
        """Walls + ghost cells. u_lid at top; no-slip elsewhere; Neumann p."""
        u, v, p = self.u, self.v, self.p
        # TODO: transcribe from your homework's apply_bcs.
        # Wall direct conditions (u on left/right walls, v on top/bottom walls).
        # Ghost rows for u (top: 2*u_lid - u[1,:]; bottom: -u[-2,:]).
        # Ghost cols for v (reflect with sign flip).
        # Pressure ghost cells (Neumann: copy first interior).
        ...
        # direct: u on left/right walls
        u[1:-1, 0]  = 0.0
        u[1:-1, -1] = 0.0

        # direct: v on top/bottom walls
        v[0,  1:-1] = 0.0
        v[-1, 1:-1] = 0.0

        # ghost: u top (lid) and bottom (no-slip)
        u[0,  :] = 2.0*self.u_lid - u[1,  :]   # lid top ghost
        u[-1, :] = -u[-2, :]               # no-slip bottom ghost

        # ghost: v left and right
        v[:, 0]  = -v[:, 1]
        v[:, -1] = -v[:, -2]

        # pressure Neumann
        p[:,  0] = p[:,  1];   p[:, -1] = p[:, -2]
        p[0,  :] = p[1,  :];   p[-1, :] = p[-2, :]

    # ── interpolations ──────────────────────────────────────────────

    def _u_to_c(self, u):
        """Average u onto cell centers. (N+2, N+1) -> (N, N)."""
        N = self.N
        return 0.5 * (u[1:N+1, 0:N] + u[1:N+1, 1:N+1])

    def _v_to_c(self, v):
        """Average v onto cell centers. (N+1, N+2) -> (N, N)."""
        N = self.N
        return 0.5 * (v[0:N, 1:N+1] + v[1:N+1, 1:N+1])

    def _u_to_n(self, u):
        """Average u onto corners. (N+2, N+1) -> (N+1, N+1)."""
        N = self.N
        return 0.5 * (u[0:N+1, :] + u[1:N+2, :])

    def _v_to_n(self, v):
        """Average v onto corners. (N+1, N+2) -> (N+1, N+1)."""
        N = self.N
        return 0.5 * (v[:, 0:N+1] + v[:, 1:N+2])

    # ── divergence and pressure gradients ───────────────────────────

    def _div_uv_to_c(self, u, v):
        """Discrete divergence at interior p-cells. -> (N, N).
        Row 0 is TOP, so dvy uses (north - south) = v[i-1] - v[i]."""
        N = self.N
        div_x = (u[1:N+1, 1:N+1] - u[1:N+1, 0:N]) / self.dx
        div_y = (v[0:N, 1:N+1] - v[1:N+1, 1:N+1]) / self.dy
        return div_x + div_y

    def _grad_x_p_to_u(self, p):
        """dp/dx at interior u-nodes. -> (N, N-1)."""
        N = self.N
        return (p[1:N+1, 2:N+1] - p[1:N+1, 1:N]) / self.dx

    def _grad_y_p_to_v(self, p):
        """dp/dy at interior v-nodes. -> (N-1, N).
        Row 0 is TOP, so north - south = p[i-1] - p[i]."""
        N = self.N
        return (p[1:N, 1:N+1] - p[2:N+1, 1:N+1]) / self.dy

    # ── physics: H = -convection + viscous ──────────────────────────

    def compute_H(self):
        """Right-hand sides of momentum (no pressure):
            H_u at interior u-nodes  -> (N, N-1)
            H_v at interior v-nodes  -> (N-1, N)
        """
        N, dx, dy, Re = self.N, self.dx, self.dy, self.Re

        # interpolations to natural sublattices
        uu_c = self._u_to_c(self.u) ** 2                     # (N, N) at c
        vv_c = self._v_to_c(self.v) ** 2                     # (N, N) at c
        uv_n = self._u_to_n(self.u) * self._v_to_n(self.v)   # (N+1, N+1) at n

        # gradients of velocity
        dudx_c = (self.u[1:N+1, 1:N+1] - self.u[1:N+1, 0:N]) / dx   # (N, N)
        dvdy_c = (self.v[0:N, 1:N+1]  - self.v[1:N+1, 1:N+1]) / dy  # (N, N)
        dudy_n = (self.u[0:N+1, :]    - self.u[1:N+2, :])    / dy   # (N+1, N+1)
        dvdx_n = (self.v[:, 1:N+2]    - self.v[:, 0:N+1])    / dx   # (N+1, N+1)

        # stresses
        txx_c = (2.0 / Re) * dudx_c
        tyy_c = (2.0 / Re) * dvdy_c
        txy_n = (1.0 / Re) * (dudy_n + dvdx_n)

        # H_u at u-faces
        duudx  = (uu_c[:, 1:]      - uu_c[:, :-1])      / dx
        duvdy  = (uv_n[0:N, 1:N]   - uv_n[1:N+1, 1:N])  / dy
        dtxxdx = (txx_c[:, 1:]     - txx_c[:, :-1])     / dx
        dtxydy = (txy_n[0:N, 1:N]  - txy_n[1:N+1, 1:N]) / dy
        H_u = -duudx - duvdy + dtxxdx + dtxydy

        # H_v at v-faces
        dvvdy  = (vv_c[0:N-1, :]    - vv_c[1:N, :])     / dy
        duvdx  = (uv_n[1:N, 1:N+1]  - uv_n[1:N, 0:N])   / dx
        dtyydy = (tyy_c[0:N-1, :]   - tyy_c[1:N, :])    / dy
        dtxydx = (txy_n[1:N, 1:N+1] - txy_n[1:N, 0:N])  / dx
        H_v = -duvdx - dvvdy + dtyydy + dtxydx

        if self.forcing is not None and self.N_int != 0.0:
            fx_at_u, fy_at_v = self.forcing(self, self.t)
            H_u += self.N_int * fx_at_u
            H_v += self.N_int * fy_at_v

        return H_u, H_v

    # ── pressure projection ─────────────────────────────────────────

    def solve_pressure(self, source, alpha=1.7, tol=1e-7,
                       max_iter=100_000, check_every=10):
        """Solve laplacian(p) = source on interior with Neumann BCs.
        Modifies self.p in place. Returns (iters, final_residual)."""
        h, N = self.dx, self.N
        res_max = np.inf
        for it in range(max_iter):
            _sor_sweep(self.p, source, alpha, h, N)
            self.p[:, 0] = self.p[:, 1];  self.p[:, -1] = self.p[:, -2]
            self.p[0, :] = self.p[1, :];  self.p[-1, :] = self.p[-2, :]
            if it % check_every == 0:
                res_max = _poisson_residual_max(self.p, source, h, N)
                if res_max < tol:
                    return it, res_max
        return max_iter, res_max

    def project(self, dt):
        """Make (u, v) discretely div-free; correction potential ends up in p."""
        N = self.N
        source = self._div_uv_to_c(self.u, self.v) / dt
        self.solve_pressure(source)
        self.u[1:N+1, 1:N]   -= dt * self._grad_x_p_to_u(self.p)
        self.v[1:N,   1:N+1] -= dt * self._grad_y_p_to_v(self.p)

    # ── time stepping: Heun's RK2 with projection at each stage ─────

    def step(self):
        """Advance one dt. Stage 1 = forward Euler + project.
        Stage 2 = average with stage-1 RHS + project."""
        N, dt = self.N, self.dt
        u_n = self.u.copy()
        v_n = self.v.copy()

        # --- stage 1 ---
        H_u, H_v = self.compute_H()
        self.u[1:N+1, 1:N]   = u_n[1:N+1, 1:N]   + dt * H_u
        self.v[1:N,   1:N+1] = v_n[1:N,   1:N+1] + dt * H_v
        self.apply_bcs()
        self.project(dt)
        self.apply_bcs()

        # --- stage 2 ---
        H_u, H_v = self.compute_H()
        self.u[1:N+1, 1:N]   = 0.5 * (u_n[1:N+1, 1:N]   + self.u[1:N+1, 1:N])   + 0.5 * dt * H_u
        self.v[1:N,   1:N+1] = 0.5 * (v_n[1:N,   1:N+1] + self.v[1:N,   1:N+1]) + 0.5 * dt * H_v
        self.apply_bcs()
        self.project(dt)
        self.apply_bcs()

        self.t          += dt
        self.step_count += 1

    # ── driver ──────────────────────────────────────────────────────

    def run(self, n_steps=None, t_end=None, steady_tol=None, log_every=200):
        self.apply_bcs()
        u_prev = self.u.copy()
        v_prev = self.v.copy()

        while True:
            if n_steps is not None and self.step_count >= n_steps: break
            if t_end   is not None and self.t          >= t_end:   break
            self.step()

            if self.step_count % log_every == 0:
                du, dv = self.u - u_prev, self.v - v_prev
                rms = np.sqrt((np.sum(du**2) + np.sum(dv**2))
                              / (du.size + dv.size)) / self.dt
                div = np.max(np.abs(self._div_uv_to_c(self.u, self.v)))
                print(f"step {self.step_count:6d}  t={self.t:.3f}  "
                      f"rms_dudt={rms:.3e}  max|div|={div:.2e}")
                if steady_tol is not None and rms < steady_tol:
                    print("Steady state reached.")
                    break
                u_prev, v_prev = self.u.copy(), self.v.copy()


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = CavityFlow(Re=400.0, N=64, N_int=1.0, dt=0.001, forcing=tmf_forcing_static)
    safe_dt = sim.cfl_dt()
    print(f"Suggested dt for stability: {safe_dt:.5f}")
    sim.run(t_end=30, steady_tol=1e-3, log_every=10)
    fx, fy = tmf_forcing_static(sim, 0.0)
    quiver_forcing(fx, fy, sim.H, sim.N, stride=4, title="TMF forcing")
    plot_streamline(sim.u, sim.v, sim.Re, sim.H, sim.N, sim.N_int, folder="./results", save=True)