# cavitymhd

2D lid-driven cavity flow with inductionless MHD, implemented on a staggered MAC grid. Two backends: a NumPy/Numba reference and a JAX port with JIT-compatible solvers. Validated against Shatrov et al. streamline structure at Re = 5000.

---

## Physics

Low magnetic Reynolds number (Rm ≪ 1) inductionless approximation. The applied field **B** is prescribed and frozen; the induced field is neglected. The governing equations are:

```
∂u/∂t + (u·∇)u = −∇p + (1/Re)∇²u + N · F_L

∇²φ = ∇·(u × B)

j = −∇φ + u × B,    F_L = j × B

∇·u = 0
```

The interaction parameter N = σB²L/ρU absorbs the electrical conductivity. Insulating wall boundary conditions apply to φ (Neumann: ∂φ/∂n = 0 on all walls).

---

## Grid

Standard arrangement:

| Field | Location | Shape |
|---|---|---|
| p, φ | cell centers + ghosts | (N+2, N+2) |
| u, Bx, jx | x-faces | (N+2, N+1) |
| v, By, jy | y-faces | (N+1, N+2) |
| Bz, jz, ωz | corners | (N+1, N+1) |

Row 0 is the top of the physical domain. y increases upward, so y-derivative signs differ from row-0-is-bottom conventions. Retained to keep it consistent with numpy convention.

---

## Time stepping

Explicit RK2 Heun with pressure projection at each stage:

```
1. H = compute_H(uⁿ)          # convection + diffusion + Lorentz
2. u* = uⁿ + dt·H             # predictor
3. u** = project(u*)           # enforce ∇·u = 0

4. H = compute_H(u**)
5. uⁿ⁺¹ = 0.5(uⁿ + u**) + 0.5·dt·H
6. uⁿ⁺¹ = project(uⁿ⁺¹)
```

CFL stability limit: `dt < min(dx/u_max, 0.5·dx²·Re)`.

---

## Solvers

The Poisson solve (both pressure and φ) is dispatched through a single interface. Backend is selected at initialization via the `solver` argument:

| Solver | Backend | Notes |
|---|---|---|
| `sor` | Numba JIT | Gauss-Seidel SOR, ω = 1.7, sequential |
| `jacobi` | JAX `lax.while_loop` | JIT-compatible, GPU-portable |
| `cg` | `jax.scipy.sparse.linalg.cg` | Matrix-free conjugate gradient (in-progress)|
| `bicgstab` | `jax.scipy.sparse.linalg.bicgstab` | Faster than CG for asymmetric problems (in-progress)|

Mean-removal (gauge fixing) is applied to the pressure solve only. The φ solve uses Neumann BCs with no mean constraint, which is the correct gauge for the electric potential.

---

## Usage

```python
from cavity_mhd_jax import init, run, plot_streamline

# Hartmann configuration: B = Bo·ex, Re=5000, N=0.4
params, state = init(Re=5000.0, N=64, B=[1, 0, 0], dt=0.005, N_int=0.4, solver='jacobi')

state = run(params, state, t_end=50, steady_tol=1e-3, log_every=200)
plot_streamline(params, state)
```

```python
# Pure hydrodynamic baseline (N_int=0)
params, state = init(Re=5000.0, N=64, B=[1, 0, 0], dt=0.005, N_int=0.0, solver='jacobi')
state = run(params, state, t_end=50, steady_tol=1e-3, log_every=200)
```

### Numba reference backend

```python
from cavity_mhd_numba import CavityMHD, plot_streamline

sim = CavityMHD(Re=5000.0, N=64, B=[1, 0, 0], dt=0.005, N_int=0.4)
sim.test_operators()
sim.run(t_end=50, steady_tol=1e-3, log_every=200)
plot_streamline(sim.u, sim.v, sim.Re, sim.H, sim.N)
```

---

## Parameters

| Parameter | Description |
|---|---|
| `Re` | Reynolds number |
| `N` | Grid resolution (N × N interior cells) |
| `B` | Applied magnetic field vector [Bx, By, Bz] |
| `N_int` | Interaction parameter N = σB²L/ρU |
| `dt` | Time step (must satisfy CFL) |
| `solver` | Poisson backend: `sor`, `jacobi`, `cg`, `bicgstab` |
| `H` | Domain side length (default 1.0) |
| `lid_taper` | Tanh taper width near lid corners (default 0.01) |

---

## Validation

Shatrov et al. (2003) provides streamline structure and eddy thickness data for the MHD lid-driven cavity at Re = 5000 with B = Bo·ex, Bo = 1 for N = 0.4. The notebook `validation.ipynb` reproduces their Figure 4 (streamlines).

---

## Repository structure

```
cavitymhd/
├── cavity_mhd_jax.py       # JAX backend (JIT, GPU-portable)
├── cavity_mhd_numba.py     # NumPy/Numba reference backend
├── validation.ipynb        # Shatrov comparison notebook
├── README.md
└── .gitignore
```

---

## References

**MHD and validation**
- Shatrov, Mutschke, Gerbeth (2003) "Numerical simulation of the magnetohydrodynamic flow in a lid-driven cavity." *Magnetohydrodynamics* 39(3).
- Davidson (2001) *An Introduction to Magnetohydrodynamics*. Cambridge. Chapter 6 covers the inductionless limit.
- Roberts (1967) *An Introduction to Magnetohydrodynamics*. Longmans.

**Projection method**
- Chorin (1968) "Numerical solution of the Navier-Stokes equations." *Math. Comp.* 22(104):745-762.
- Brown, Cortez, Minion (2001) "Accurate projection methods for the incompressible Navier-Stokes equations." *J. Comp. Phys.* 168:464-499.
- Guermond, Minev, Shen (2006) "An overview of projection methods for incompressible flows." *Comput. Methods Appl. Mech. Engrg.* 195:6011-6045.

**Time stepping**
- Karniadakis, Israeli, Orszag (1991) "High-order splitting methods for the incompressible Navier-Stokes equations." *J. Comp. Phys.* 97:414.
- Sanderse and Koren (2012) "Accuracy analysis of explicit Runge-Kutta methods applied to the incompressible Navier-Stokes equations." *J. Comp. Phys.* 231:3041.
- LeVeque (2007) *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.