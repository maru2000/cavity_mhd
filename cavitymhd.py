import numpy as np
from numba import njit
from dataclasses import dataclass
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────
# Module-level JIT kernels
# ────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────

def plot_streamline(u, v, Re, H, N, folder=None, save=False):
    u_phys = u[1:-1, :]
    v_phys = v[:, 1:-1]
    u_cc = 0.5 * (u_phys[:, :-1] + u_phys[:, 1:])
    v_cc = 0.5 * (v_phys[:-1, :] + v_phys[1:, :])

    x = np.linspace(0, H, N)
    y = np.linspace(0, H, N)
    u_plot = np.flipud(u_cc)
    v_plot = np.flipud(v_cc)

    fig, ax = plt.subplots(figsize=(6, 6))
    speed = np.sqrt(u_plot**2 + v_plot**2)
    strm = ax.streamplot(x, y, u_plot, v_plot,
                         color=speed, cmap='plasma',
                         linewidth=0.8, density=3, arrowsize=0.6)
    plt.colorbar(strm.lines, ax=ax, label='|u|', shrink=0.8)
    ax.set_xlim(0, H); ax.set_ylim(0, H)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'Streamlines  Re={Re}  N={N}')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f'{folder}/streamlines_Re{Re}_N{N}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ────────────────────────────────────────────────────────────────────
# Field types
# ────────────────────────────────────────────────────────────────────

@dataclass
class CellScalar:
    """Scalar field at cell centers, including ghosts. Shape (N+2, N+2)"""
    data: np.ndarray


@dataclass
class FaceVector:
    """Vector field with components at u-faces, v-faces, and (optionally) corners."""
    x_at_u: np.ndarray              # shape (N+2, N+1)
    y_at_v: np.ndarray              # shape (N+1, N+2)
    z_at_n: np.ndarray | None = None  # shape (N+1, N+1)


# ────────────────────────────────────────────────────────────────────
# Solver class
# ────────────────────────────────────────────────────────────────────

class CavityMHD:
    """2D incompressible lid-driven cavity on a staggered grid.

    Layout (row 0 = top of physical domain):
        p : (N+2, N+2)   cell centers, ghost on all four sides
        u : (N+2, N+1)   x-faces,      ghost rows top and bottom
        v : (N+1, N+2)   y-faces,      ghost cols left and right

    Time stepping: explicit RK2 (Heun) with pressure projection at each stage.
    """

    # ── construction ────────────────────────────────────────────────

    def __init__(self, Re, N, B=[0, 0, 0], H=1.0, dt=None,
                 lid_taper=0.01, N_int=0.0):
        self.Re = Re
        self.N = N
        self.H = H
        self.dx = H / N
        self.dy = self.dx
        self.dt = dt if dt is not None else 0.2 * self.dx

        # MHD-related storage (unused in pure LDC; ready for Case 2)
        self.N_int = N_int
        self.Bx = np.full((N+2, N+1), B[0])
        self.By = np.full((N+1, N+2), B[1])
        self.Bz = np.full((N+1, N+1), B[2])
        self.phi = np.zeros((N+2, N+2))

        # Primary fields
        self.u = np.zeros((N+2, N+1))
        self.v = np.zeros((N+1, N+2))
        self.p = np.zeros((N+2, N+2))

        self.u_lid = self._lid_profile(lid_taper)

        self.t = 0.0
        self.step_count = 0

    def _lid_profile(self, w, coeff=1.0):
        """Flat lid with smooth tanh taper near the corners."""
        x = np.arange(self.N + 1) * self.dx
        xn = x / self.H
        left = 0.5 * (1 + np.tanh((xn - w) / (w / 3)))
        right = 0.5 * (1 - np.tanh((xn - (1 - w)) / (w / 3)))
        return coeff * left * right

    def cfl_dt(self, safety=0.8):
        u_max = max(np.max(np.abs(self.u)), 1e-12)
        v_max = max(np.max(np.abs(self.v)), 1e-12)
        dt_conv = min(self.dx / u_max, self.dy / v_max)
        dt_diff = 0.5 * min(self.dx, self.dy)**2 * self.Re
        return safety * min(dt_conv, dt_diff)
    
    # ── interior interpolations (used by convective term) ───────────

    def _u_to_c(self, u):
        N = self.N
        return 0.5 * (u[1:N+1, 0:N] + u[1:N+1, 1:N+1])

    def _v_to_c(self, v):
        N = self.N
        return 0.5 * (v[0:N, 1:N+1] + v[1:N+1, 1:N+1])

    def _u_to_n(self, u):
        N = self.N
        return 0.5 * (u[0:N+1, :] + u[1:N+2, :])

    def _v_to_n(self, v):
        N = self.N
        return 0.5 * (v[:, 0:N+1] + v[:, 1:N+2])
    
    # ── additional interpolations for cross product ─────────────────

    def _v_to_u(self, v):
        """v from v-faces to interior u-faces. (N+1, N+2) -> (N, N-1).
        4-point average from the four v-faces surrounding each interior u-face."""
        N = self.N
        return 0.25 * (v[0:N,   1:N]   + v[0:N,   2:N+1]
                    + v[1:N+1, 1:N]   + v[1:N+1, 2:N+1])

    def _u_to_v(self, u):
        """u from u-faces to interior v-faces. (N+2, N+1) -> (N-1, N).
        4-point average from the four u-faces surrounding each interior v-face."""
        N = self.N
        return 0.25 * (u[1:N,   0:N]   + u[1:N,   1:N+1]
                    + u[2:N+1, 0:N]   + u[2:N+1, 1:N+1])

    def _n_to_u(self, c):
        """Corner-stored scalar to interior u-faces. (N+1, N+1) -> (N, N-1).
        2-point average from corners directly above and below each u-face."""
        N = self.N
        return 0.5 * (c[0:N,   1:N] + c[1:N+1, 1:N])

    def _n_to_v(self, c):
        """Corner-stored scalar to interior v-faces. (N+1, N+1) -> (N-1, N).
        2-point average from corners directly left and right of each v-face."""
        N = self.N
        return 0.5 * (c[1:N, 0:N] + c[1:N, 1:N+1])
        
    # ── operators ───────────────────────────────────────────────────

    def grad(self, field: CellScalar) -> FaceVector:
        """Gradient of cell-centered scalar -> face-centered vector."""
        N = self.N
        f = field.data
        dfdx_u = (f[:, 1:N+2] - f[:, :N+1]) / self.dx       # (N+2, N+1)
        dfdy_v = (f[:N+1, :] - f[1:N+2, :]) / self.dy       # (N+1, N+2)
        return FaceVector(x_at_u=dfdx_u, y_at_v=dfdy_v)

    def div(self, field: FaceVector) -> CellScalar:
        """Divergence of face-centered vector -> cell-centered scalar (interior, shape (N, N))."""
        N = self.N
        fx, fy = field.x_at_u, field.y_at_v
        dfx_dx = (fx[1:N+1, 1:N+1] - fx[1:N+1, :N]) / self.dx
        dfy_dy = (fy[:N, 1:N+1] - fy[1:N+1, 1:N+1]) / self.dy
        return CellScalar(data=dfx_dx + dfy_dy)

    def laplacian(self, field: FaceVector) -> FaceVector:
        """Componentwise Laplacian. Valid for incompressible Newtonian flow.
        See note about strain-tensor equivalence for variable-viscosity case."""
        N = self.N
        fx, fy = field.x_at_u, field.y_at_v

        lap_x = ((fx[1:N+1, 2:N+1] - 2*fx[1:N+1, 1:N] + fx[1:N+1, 0:N-1]) / (self.dx ** 2)
               + (fx[0:N,   1:N]   - 2*fx[1:N+1, 1:N] + fx[2:N+2, 1:N])   / (self.dy ** 2))

        lap_y = ((fy[1:N,   2:N+2] - 2*fy[1:N, 1:N+1] + fy[1:N,   0:N])   / (self.dx ** 2)
               + (fy[0:N-1, 1:N+1] - 2*fy[1:N, 1:N+1] + fy[2:N+1, 1:N+1]) / (self.dy ** 2))

        return FaceVector(x_at_u=lap_x, y_at_v=lap_y)
    
    def cross(self, field1: FaceVector, field2: FaceVector) -> FaceVector:
        """Cross product of two face-stored vector fields, field1 x field2.

        Output lattices match the input convention:
        x-component at INTERIOR u-faces, shape (N, N-1)
        y-component at INTERIOR v-faces, shape (N-1, N)
        z-component at all corners,      shape (N+1, N+1)

        Note the asymmetry: in-plane outputs are interior-only (since they
        feed compute_H, which writes to interior u/v faces), while the
        z-component covers all corners (since corners have no ghost concept
        and the z-component will feed jz which is used in the Lorentz force
        via 2-point averages to interior faces).
        """
        a, b = field1, field2

        # ── x-component at interior u-faces: a_y*b_z - a_z*b_y ──
        ay_at_u = self._v_to_u(a.y_at_v)        # (N, N-1)
        bz_at_u = self._n_to_u(b.z_at_n)        # (N, N-1)
        az_at_u = self._n_to_u(a.z_at_n)        # (N, N-1)
        by_at_u = self._v_to_u(b.y_at_v)        # (N, N-1)
        out_x = ay_at_u * bz_at_u - az_at_u * by_at_u

        # ── y-component at interior v-faces: a_z*b_x - a_x*b_z ──
        az_at_v = self._n_to_v(a.z_at_n)        # (N-1, N)
        bx_at_v = self._u_to_v(b.x_at_u)        # (N-1, N)
        ax_at_v = self._u_to_v(a.x_at_u)        # (N-1, N)
        bz_at_v = self._n_to_v(b.z_at_n)        # (N-1, N)
        out_y = az_at_v * bx_at_v - ax_at_v * bz_at_v

        # ── z-component at all corners: a_x*b_y - a_y*b_x ──
        ax_at_n = self._u_to_n(a.x_at_u)        # (N+1, N+1)
        by_at_n = self._v_to_n(b.y_at_v)        # (N+1, N+1)
        ay_at_n = self._v_to_n(a.y_at_v)        # (N+1, N+1)
        bx_at_n = self._u_to_n(b.x_at_u)        # (N+1, N+1)
        out_z = ax_at_n * by_at_n - ay_at_n * bx_at_n

        return FaceVector(x_at_u=out_x, y_at_v=out_y, z_at_n=out_z)

    # ── boundary conditions ─────────────────────────────────────────

    def apply_bcs(self):
        """Walls + ghost cells. u_lid at top; no-slip elsewhere; Neumann p."""
        u, v, p = self.u, self.v, self.p

        # direct: u on left/right walls
        u[1:-1, 0]  = 0.0
        u[1:-1, -1] = 0.0

        # direct: v on top/bottom walls
        v[0,  1:-1] = 0.0
        v[-1, 1:-1] = 0.0

        # ghost: u top (lid) and bottom (no-slip)
        u[0,  :] = 2.0 * self.u_lid - u[1, :]
        u[-1, :] = -u[-2, :]

        # ghost: v left and right
        v[:, 0]  = -v[:, 1]
        v[:, -1] = -v[:, -2]

        # pressure Neumann
        p[:, 0]  = p[:, 1];   p[:, -1] = p[:, -2]
        p[0, :]  = p[1, :];   p[-1, :] = p[-2, :]


    # ── physics: RHS of momentum (convective + viscous) ─────────────

    def compute_H(self, velocity: FaceVector) -> FaceVector:
        """RHS of momentum at interior faces.
        Convective term: conservative form -div(u u). Inlined for now.
        Viscous term: (1/Re) * laplacian(u).
        """
        N, dx, dy, Re = self.N, self.dx, self.dy, self.Re
        u, v = velocity.x_at_u, velocity.y_at_v

        # cell-centered products
        uu_c = self._u_to_c(u) ** 2                          # (N, N)
        vv_c = self._v_to_c(v) ** 2                          # (N, N)
        uv_n = self._u_to_n(u) * self._v_to_n(v)             # (N+1, N+1)

        # convective contributions (conservative form: -div(uu))
        duudx = (uu_c[:, 1:]      - uu_c[:, :-1])      / dx     # (N, N-1)
        duvdy = (uv_n[0:N, 1:N]   - uv_n[1:N+1, 1:N])  / dy     # (N, N-1)
        H_x_conv = -(duudx + duvdy)

        duvdx = (uv_n[1:N, 1:N+1] - uv_n[1:N, 0:N])    / dx     # (N-1, N)
        dvvdy = (vv_c[0:N-1, :]   - vv_c[1:N, :])      / dy     # (N-1, N)
        H_y_conv = -(duvdx + dvvdy)

        # viscous contributions: (1/Re) * laplacian(u)
        visc = self.laplacian(velocity)

        H_x = H_x_conv + visc.x_at_u / Re
        H_y = H_y_conv + visc.y_at_v / Re

        return FaceVector(x_at_u=H_x, y_at_v=H_y)

    def velocity(self) -> FaceVector:
        """Bundle u, v into a FaceVector for operator use."""
        return FaceVector(x_at_u=self.u, y_at_v=self.v)

    def pressure(self) -> CellScalar:
        """Bundle p into a CellScalar for operator use."""
        return CellScalar(data=self.p)

    # ── pressure projection ─────────────────────────────────────────

    def solve_pressure(self, source, alpha=1.7, tol=1e-7,
                       max_iter=100_000, check_every=10):
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
        N = self.N
        source = self.div(self.velocity()).data / dt
        self.solve_pressure(source)
        self.u[1:N+1, 1:N]   -= dt * self.grad(self.pressure()).x_at_u[1:N+1, 1:N]
        self.v[1:N,   1:N+1] -= dt * self.grad(self.pressure()).y_at_v[1:N, 1:N+1]

    # ── time stepping ───────────────────────────────────────────────

    def step(self):
        N, dt = self.N, self.dt
        u_n = self.u.copy()
        v_n = self.v.copy()

        # stage 1
        H = self.compute_H(self.velocity())
        self.u[1:N+1, 1:N]   = u_n[1:N+1, 1:N]   + dt * H.x_at_u
        self.v[1:N,   1:N+1] = v_n[1:N,   1:N+1] + dt * H.y_at_v
        self.apply_bcs()
        self.project(dt)
        self.apply_bcs()

        # stage 2
        H = self.compute_H(self.velocity())
        self.u[1:N+1, 1:N]   = 0.5 * (u_n[1:N+1, 1:N]   + self.u[1:N+1, 1:N])   + 0.5 * dt * H.x_at_u
        self.v[1:N,   1:N+1] = 0.5 * (v_n[1:N,   1:N+1] + self.v[1:N,   1:N+1]) + 0.5 * dt * H.y_at_v
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
            if t_end is not None and self.t >= t_end:              break
            self.step()

            if self.step_count % log_every == 0:
                du, dv = self.u - u_prev, self.v - v_prev
                rms = np.sqrt((np.sum(du**2) + np.sum(dv**2))
                              / (du.size + dv.size)) / self.dt
                div = np.max(np.abs(self.div(self.velocity()).data))
                print(f"step {self.step_count:6d}  t={self.t:.3f}  "
                      f"rms_dudt={rms:.3e}  max|div|={div:.2e}")
                if steady_tol is not None and rms < steady_tol:
                    print("Steady state reached.")
                    break
                u_prev, v_prev = self.u.copy(), self.v.copy()

    # ── operator tests ──────────────────────────────────────────────

    def test_operators(self):
        N, dx, dy = self.N, self.dx, self.dy

        x_c = (np.arange(N+2) - 0.5) * dx
        y_c = ((N+1) - np.arange(N+2) - 0.5) * dy
        Xc, Yc = np.meshgrid(x_c, y_c)

        x_u = np.arange(N+1) * dx
        y_u = ((N+1) - np.arange(N+2) - 0.5) * dy
        Xu, Yu = np.meshgrid(x_u, y_u)

        x_v = (np.arange(N+2) - 0.5) * dx
        y_v = (N - np.arange(N+1)) * dy
        Xv, Yv = np.meshgrid(x_v, y_v)

        # grad of linear scalar
        p = CellScalar(data=(1 + 2*Xc + 3*Yc))
        g = self.grad(p)
        assert np.allclose(g.x_at_u[1:-1, :], 2), "grad x-component wrong"
        assert np.allclose(g.y_at_v[:, 1:-1], 3), "grad y-component wrong"

        # div of (2x, 3y) should be 5
        field = FaceVector(x_at_u=2*Xu.copy(), y_at_v=3*Yv.copy())
        d = self.div(field)
        assert np.allclose(d.data, 5), "div wrong"

        # lap of (x^2, y^2) should be (2, 2)
        field = FaceVector(x_at_u=Xu.copy()**2, y_at_v=Yv.copy()**2)
        l = self.laplacian(field)
        assert np.allclose(l.x_at_u, 2), "lap x-component wrong"
        assert np.allclose(l.y_at_v, 2), "lap y-component wrong"

        print("All operator tests pass")

        # cross product: cyclic basis tests
        # e_x x e_y = e_z, e_y x e_z = e_x, e_z x e_x = e_y
        ex = FaceVector(x_at_u=np.ones((N+2, N+1)),
                        y_at_v=np.zeros((N+1, N+2)),
                        z_at_n=np.zeros((N+1, N+1)))
        ey = FaceVector(x_at_u=np.zeros((N+2, N+1)),
                        y_at_v=np.ones((N+1, N+2)),
                        z_at_n=np.zeros((N+1, N+1)))
        ez = FaceVector(x_at_u=np.zeros((N+2, N+1)),
                        y_at_v=np.zeros((N+1, N+2)),
                        z_at_n=np.ones((N+1, N+1)))

        c = self.cross(ex, ey)
        assert np.allclose(c.x_at_u, 0), "ex x ey: x-component should be 0"
        assert np.allclose(c.y_at_v, 0), "ex x ey: y-component should be 0"
        assert np.allclose(c.z_at_n, 1), "ex x ey: z-component should be 1"

        c = self.cross(ey, ez)
        assert np.allclose(c.x_at_u, 1), "ey x ez: x-component should be 1"
        assert np.allclose(c.y_at_v, 0), "ey x ez: y-component should be 0"
        assert np.allclose(c.z_at_n, 0), "ey x ez: z-component should be 0"

        c = self.cross(ez, ex)
        assert np.allclose(c.x_at_u, 0), "ez x ex: x-component should be 0"
        assert np.allclose(c.y_at_v, 1), "ez x ex: y-component should be 1"
        assert np.allclose(c.z_at_n, 0), "ez x ex: z-component should be 0"

        # antisymmetry: a x b = -(b x a)
        c1 = self.cross(ex, ey)
        c2 = self.cross(ey, ex)
        assert np.allclose(c1.x_at_u, -c2.x_at_u)
        assert np.allclose(c1.y_at_v, -c2.y_at_v)
        assert np.allclose(c1.z_at_n, -c2.z_at_n)

        print("Cross product tests pass")

    # -- collecting snapshots ----------------------------------------
    def snapshot(self) -> dict:
        """Cell-centered (N, N) snapshot of current state, suitable for SciML datasets."""
        u_c = self._u_to_c(self.u)
        v_c = self._v_to_c(self.v)
        return {
            'u': u_c,
            'v': v_c,
            'p': self.p[1:-1, 1:-1].copy(),
            # add phi, c, etc. when those exist
        }

# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    

    # Full LDC run for Ghia validation
    sim = CavityMHD(Re=100.0, N=64, dt=0.005)
    safe_dt = sim.cfl_dt()
    print(f"Suggested dt for stability: {safe_dt:.5f}")

    sim.test_operators()