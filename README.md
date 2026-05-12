CFD class project: MHD integration into cavity flow. 

Qualtiative study due to lack of benchmarking cases

Steps:
1. Take your cavity flow code
2. Modify compute_H to include Lorentz forces 
3. Add a B_vector/Ha number for input
4. Create cases: only B_x, only B_y, combination of [Bx, By]. This should be generalizable for any given magnetic field that is spatically homogenous

Notes:
- Run at low Rm values to maintain scope
- Parameterize over different (Re, Ha) cases and see how they compare
    - Maybe a baseline of Re and increasing Ha to see how it impacts flow


Projection method and CFD sources for incompressible flow:

- Chorin (1968) "Numerical Solution of the Navier-Stokes Equations" — the original paper. Math Comp 22(104):745-762. Short, readable, foundational. Worth reading once.
- Temam (1969) independently introduced the same idea. Together they're called the Chorin-Temam projection method.
- Brown, Cortez, Minion (2001) "Accurate Projection Methods for the Incompressible Navier-Stokes Equations" J. Comp. Phys. 168:464-499. The careful modern treatment of projection methods, error analysis, and pressure-correction variants. This is the reference for understanding why "φ ≈ p but not exactly" and what the splitting errors look like.
- Guermond, Minev, Shen (2006) "An overview of projection methods for incompressible flows" Comput. Methods Appl. Mech. Engrg. 195:6011-6045. Modern review article. Goes through all the variants (pressure-correction, velocity-correction, rotational form). Heavier on math than Brown-Cortez-Minion.
- Ferziger and Perić (2002) Computational Methods for Fluid Dynamics, ch. 7. Textbook treatment, very practical, what most engineers learn from.
- Pope (2000) Turbulent Flows, ch. 6 (briefly) — has the cleanest statement of the Helmholtz decomposition I know of in a fluids textbook.

For Helmholtz/Hodge decomposition theory itself (the math, not the CFD application):

- Chorin and Marsden (1993) A Mathematical Introduction to Fluid Mechanics, ch. 1.
- Foias, Manley, Rosa, Temam (2001) Navier-Stokes Equations and Turbulence, more advanced functional analysis treatment.

Sources for RK methods

- Hairer, Nørsett, Wanner (1993) Solving Ordinary Differential Equations I: Nonstiff Problems. The standard reference. Heavy but authoritative; Sections II.1-II.3 cover everything from FE through high-order RK.
- LeVeque (2007) Finite Difference Methods for Ordinary and Partial Differential Equations. Very readable, has a clear chapter on RK methods (ch. 5) with stability analysis. This is the one I'd recommend if you only want one book.
- Iserles (2008) A First Course in the Numerical Analysis of Differential Equations, ch. 1-3. Excellent textbook treatment.

For projection methods combined with high-order time stepping specifically:

- Karniadakis, Israeli, Orszag (1991) "High-order splitting methods for the incompressible Navier-Stokes equations" J. Comp. Phys. 97:414. Introduces high-order splitting; relevant if you ever go to RK3/RK4 with projection.
- Sanderse and Koren (2012) "Accuracy analysis of explicit Runge-Kutta methods applied to the incompressible Navier-Stokes equations" J. Comp. Phys. 231:3041. Modern analysis of exactly the question "how do RK and projection interact."

Additional

- Hairer, Lubich, Wanner (2006) Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations. The book on this. Chapter 1 alone is illuminating