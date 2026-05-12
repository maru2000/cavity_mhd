"""
Explicit pressure correction
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

## Initialize the fields
def initialize_fields(N):
    """
    Initialize velocity and pressure fields with ghost cells.
    """
    # ghost rows for u at top and bottom
    u = np.zeros((N+2, N+1))   # u[0,:] ghost top, u[-1,:] ghost bottom. Without ghost - np.zeros((N, N+1))
    # ghost cols for v at left and right
    v = np.zeros((N+1, N+2))   # v[:,0] ghost left,   v[:,-1] ghost right. Without ghost - np.zeros((N+1, N))
    p = np.zeros((N+2, N+2))   # Ghost top, bottom, left, right

    return u, v, p

## Lid profile
def lid_profile(N, dx, w=0.01):
    """Flat lid with smooth tanh taper at corners. w controls corner width (fraction of H)."""
    x  = np.arange(N + 1) * dx
    H  = N * dx
    xn = x / H    # normalized 0 to 1
    left  = 0.5 * (1 + np.tanh((xn - w)    / (w/3)))
    right = 0.5 * (1 - np.tanh((xn - (1-w))/ (w/3)))
    return left * right

## Apply BCs
def apply_bcs(u, v, p, u_lid):
    # direct: u on left/right walls
    u[1:-1, 0]  = 0.0
    u[1:-1, -1] = 0.0

    # direct: v on top/bottom walls
    v[0,  1:-1] = 0.0
    v[-1, 1:-1] = 0.0

    # ghost: u top (lid) and bottom (no-slip)
    u[0,  :] = 2.0*u_lid - u[1,  :]   # lid top ghost
    u[-1, :] = -u[-2, :]               # no-slip bottom ghost

    # ghost: v left and right
    v[:, 0]  = -v[:, 1]
    v[:, -1] = -v[:, -2]

    # pressure Neumann
    p[:,  0] = p[:,  1];   p[:, -1] = p[:, -2]
    p[0,  :] = p[1,  :];   p[-1, :] = p[-2, :]

## Compute H
def compute_H(u, v, p, dx, dy, Re, N):
    """
    u: (N+2, N+1),  v: (N+1, N+2),  p: (N, N)
    x-momentum interior: u[1:N+1, 1:N]   → shape (N, N-1)
    y-momentum interior: v[1:N,   1:N+1] → shape (N-1, N)

    H_u = -duu_dx - duv_dy + dtxx_dx + dtxy_dy
    H_v = -dvv_dy - duv_dx + dtyy_dy + dtxy_dx

    """

    # Let's define parameters for x-momentum
    u_p = u[1:N+1, 1:N]

    u_w = u[1:N+1, 0:N-1]
    u_e = u[1:N+1, 2:N+1]
    u_n = u[0:N, 1:N]
    u_s = u[2:N+2, 1:N]

    v_nw = v[0:N, 1:N]
    v_ne = v[0:N, 2:N+1]
    v_sw = v[1:N+1, 1:N]
    v_se = v[1:N+1, 2:N+1]

    dtxx_dx = 2*(1/Re)*(u_e - 2*u_p + u_w)/dx**2

    txy_s = (1/Re)*( (v_se - v_sw)/dx + (u_p - u_s)/dy )
    txy_n = (1/Re)*( (v_ne - v_nw)/dx + (u_n - u_p)/dy )
    dtxy_dy = (txy_n - txy_s)/dy

    duu_dx = ( ((u_e + u_p) / 2)**2 - ((u_w + u_p) / 2)**2 )/ (dx)
    duv_dy = ( ((u_p + u_n) / 2)*((v_nw + v_ne) / 2) - ((u_p + u_s) / 2)*((v_sw + v_se) / 2) )/ (dy)

    # Let's define parameters for y-momentum
    v_p = v[1:N, 1:N+1]

    v_w = v[1:N, 0:N]
    v_e = v[1:N, 2:N+2]
    v_n = v[0:N-1, 1:N+1]
    v_s = v[2:N+1, 1:N+1]

    u_nw = u[1:N, 0:N]
    u_ne = u[1:N, 1:N+1]
    u_sw = u[2:N+1, 0:N]
    u_se = u[2:N+1, 1:N+1]

    dtyy_dy = 2*(1/Re)*(v_n - 2*v_p + v_s)/dy**2

    txy_e = (1/Re)*( (v_e - v_p)/dx + (u_ne - u_se)/dy )
    txy_w = (1/Re)*( (v_p - v_w)/dx + (u_nw - u_sw)/dy )
    dtxy_dx = (txy_e - txy_w)/dx

    dvv_dy = ( ((v_n + v_p) / 2)**2 - ((v_s + v_p) / 2)**2 )/ (dy)
    duv_dx = ( ((v_p + v_e) / 2)*((u_ne + u_se) / 2) - ((v_p + v_w) / 2)*((u_sw + u_nw) / 2) )/ (dx)

    H_u = -duu_dx - duv_dy + dtxx_dx + dtxy_dy
    H_v = -dvv_dy - duv_dx + dtyy_dy + dtxy_dx

    #print("1:", u_p.shape, v_p.shape)

    return H_u, H_v

## Solve pressure poisson equation

@njit
def sor_sweep(p, Q, alpha, h, N):
    for i in range(1, N+1):
        for j in range(1, N+1):
            p[i, j] = p[i, j] + (alpha / 4) * (
                p[i-1, j] + p[i+1, j] + 
                p[i, j-1] + p[i, j+1] - 
                h*h * Q[i-1, j-1] - 4*p[i, j])

@njit(cache=True, fastmath=True)
def poisson_residual_max(p, Q, h, N):
    res_max = 0.0
    for i in range(N):
        for j in range(N):
            ii, jj = i+1, j+1
            lap = (p[ii-1, jj] + p[ii+1, jj] 
                 + p[ii, jj-1] + p[ii, jj+1] 
                 - 4*p[ii, jj]) / (h*h)
            r = abs(Q[i, j] - lap)
            if r > res_max:
                res_max = r
    return res_max

def solve_pressure(p, H_u, H_v, dx, dy, N, alpha=1.5, tol=1e-8, max_iter=100_000, check_every=10):
    
    H_u_aug = np.zeros((N, N+1))
    H_u_aug[:, 1:N] = H_u
    H_v_aug = np.zeros((N+1, N))
    H_v_aug[1:N, :] = H_v
    div_H = (H_u_aug[:, 1:] - H_u_aug[:, :-1]) / dx \
        + (H_v_aug[0:N, :] - H_v_aug[1:N+1, :]) / dy
    
    #print("2:", H_u_aug.shape, H_v_aug.shape, div_H.shape)

    assert dx == dy, "This SOR solver assumes square cells for simplicity"
    h = dx
    Q = div_H   # source term for Poisson equation

    for _ in range(max_iter):
        #p_old = p.copy()
        
        #print("3:", np.max(p_old), np.max(p), np.max(Q))

        sor_sweep(p, Q, alpha, h, N)

        p[:, 0] = p[:, 1];   p[:, -1] = p[:, -2]
        p[0, :] = p[1, :];   p[-1, :] = p[-2, :]

        # pin one point to remove the null space
        #p[1, 1] = 0.0
        # ── Residual check (only every K iterations to save cost) ──
        if _ % check_every == 0:
            # Compute discrete Laplacian on interior cells
            res_max = poisson_residual_max(p, Q, h, N)
            
            if res_max < tol:
                print(f"Converged at iteration {_}, residual={res_max:.2e}")
                return p
    
    print(f"Max iter reached, residual={res_max:.2e}")
    return p

## Update velocity fields
def update_velocity(u, v, H_u, H_v, p, dx, dy, dt, N):
    p_phys = p[1:-1, 1:-1]    # (N, N)

    # check divergence 
    div_pre = (u[1:N+1, 1:N+1] - u[1:N+1, 0:N]) / dx \
        + (v[0:N,   1:N+1]  - v[1:N+1, 1:N+1]) / dy
    # dp/dx at u nodes: east minus west p cells, shape (N, N-1)
    u[1:N+1, 1:N]   += dt * (H_u - (p_phys[:, 1:] - p_phys[:, :-1]) / dx)
    # dp/dy at v nodes: numpy row 0 = top, so north - south = p[j-1] - p[j]
    v[1:N,   1:N+1] += dt * (H_v - (p_phys[:-1, :] - p_phys[1:, :]) / dy)

    # check divergence
    div = (u[1:N+1, 1:N+1] - u[1:N+1, 0:N]) / dx \
        + (v[0:N,   1:N+1]  - v[1:N+1, 1:N+1]) / dy

    #print(f"  max div before velocity update: {np.max(np.abs(div_pre)):.2e}")
    #print(f"  max div after velocity update: {np.max(np.abs(div)):.2e}")

    return u, v, div

## Plot fields
def plot_fields(u, v, p, Re, N, folder = None, save=False):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # strip ghost cells to get physical domain only
    u_phys = u[1:-1, :]      # (N, N+1)
    v_phys = v[:, 1:-1]      # (N+1, N)
    p_phys = p[1:-1, 1:-1]   # (N, N)

    fields = [u_phys, v_phys, p_phys]
    titles = [f'u  ({N} x {N+1})', f'v  ({N+1} x {N})', f'p  ({N} x {N})']

    for ax, field, title in zip(axes, fields, titles):
        im = ax.imshow(field, origin='upper', cmap='RdBu_r', aspect='equal')
        ax.set_title(title)
        ax.set_xlabel('i');  ax.set_ylabel('j')
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'Re={Re},  N={N}', y=1.02)
    plt.tight_layout()
    if save:
        # Append folder to the filename
        plt.savefig(f'{folder}/cavity_Re{Re}_N{N}.png', dpi=150, bbox_inches='tight')
    plt.show()

## Plot streamline
def plot_streamline(u, v, Re, H, N, folder=None, save=False):
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
    ax.set_title(f'Streamlines  Re={Re}  N={N}')
    ax.set_aspect('equal')
    plt.tight_layout()
    if save:
        plt.savefig(f'{folder}/streamlines_Re{Re}_N{N}.png', dpi=150, bbox_inches='tight')
    plt.show()

## Steaduy state residual
def steady_state_residual(u, u_prev, v, v_prev, dt):
    """RMS rate of change of velocity field, normalized by dt."""
    du = u - u_prev
    dv = v - v_prev
    rms = np.sqrt((np.sum(du**2) + np.sum(dv**2)) / (du.size + dv.size))
    return rms / dt

## Extract centerlines
def centerline_profiles(u, v, N):
    """
    Extract u along vertical centerline (x=0.5) and v along horizontal 
    centerline (y=0.5) from staggered-grid fields.
    
    Returns:
        y_u, u_centerline : (N+1,), (N+1,) — u(0.5, y) at u-row positions
        x_v, v_centerline : (N+1,), (N+1,) — v(x, 0.5) at v-col positions
    """
    # u lives at vertical faces. Strip ghost rows: u_phys is (N, N+1).
    # Vertical centerline x=0.5 is between u-columns N//2-1 and N//2 (for even N+1).
    # Average those two columns to get u at x=0.5.
    u_phys = u[1:-1, :]                              # (N, N+1)
    u_cl = 0.5 * (u_phys[:, N//2 - 1] + u_phys[:, N//2])   # (N,)
    
    # u_phys row 0 is top of cavity (lid), row N-1 is bottom.
    # Flip so y increases bottom-to-top, matching physical convention.
    u_cl = u_cl[::-1]                                 # (N,)
    y_u = np.linspace(0, 1, N) + 0.5/N               # cell centers in y
    
    # v lives at horizontal faces. Strip ghost cols: v_phys is (N+1, N).
    # Horizontal centerline y=0.5 is between v-rows N//2-1 and N//2.
    v_phys = v[:, 1:-1]                              # (N+1, N)
    v_cl = 0.5 * (v_phys[N//2 - 1, :] + v_phys[N//2, :])  # (N,)
    
    x_v = np.linspace(0, 1, N) + 0.5/N               # cell centers in x
    
    return y_u, u_cl, x_v, v_cl

def plot_centerlines(u, v, u_g, v_g, x_g, y_g, N, Re, save=False):
    """Plot computed centerline profiles vs Ghia 1982."""
    import matplotlib.pyplot as plt
    
    y_u, u_cl, x_v, v_cl = centerline_profiles(u, v, N)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # u along vertical centerline
    ax = axes[0]
    ax.plot(u_cl, y_u, 'b-', linewidth=2, label=f'Computed (N={N})')
    ax.plot(u_g, y_g, 'ro', markersize=6, label='Ghia 1982')
    ax.set_xlabel('u')
    ax.set_ylabel('y')
    ax.set_title(f'u along vertical centerline (x=0.5), Re={Re}')
    ax.grid(alpha=0.3)
    ax.legend()
    
    # v along horizontal centerline
    ax = axes[1]
    ax.plot(x_v, v_cl, 'b-', linewidth=2, label=f'Computed (N={N})')
    ax.plot(x_g, v_g, 'ro', markersize=6, label='Ghia 1982')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(f'v along horizontal centerline (y=0.5), Re={Re}')
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(f'centerlines_Re{Re}_N{N}.png', dpi=150, bbox_inches='tight')
    plt.show()