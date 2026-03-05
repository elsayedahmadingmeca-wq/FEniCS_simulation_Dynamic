# FEniCS_simulation_Dynamic
# Explicit Nonlinear Structural Dynamics of a Hyperelastic Tube
### FEniCSx · Neo-Hookean · Explicit Newmark · Rayleigh Damping · POD

> **Context:** Personal research project — Master M2 Génie Mécanique, Université d'Orléans.  
> Companion to a quasi-static thermo-mechanical FEM study on the same geometry.  
> Built as preparation for a PhD application in nonlinear structural model reduction.

---

## What this project covers

Dynamic nonlinear simulations are technically demanding: choosing the wrong time integrator,
underestimating wave speeds, or mishandling boundary conditions produces results that *look*
physical but are numerically corrupted. This project was built to confront every one of those
difficulties on a single, well-controlled test case — a pressurised hyperelastic tube subjected
to a prescribed two-axis sinusoidal displacement at one end.

The specific challenges addressed:

| Topic | What was done |
|-------|--------------|
| **Explicit vs. implicit** | Chose explicit Newmark (central difference) — justified by the wave-dominated nature of the problem and avoidance of nonlinear solves at each step |
| **CFL stability** | Computed h_min over all tetrahedral edges; wave speed from `λ + 2μ`; safety factor 0.3 for near-incompressible material |
| **Mass matrix lumping** | Row-sum lumping via unit-vector multiplication; inverted element-wise for O(N) explicit update |
| **Near-incompressibility** | ν = 0.49; handled through Neo-Hookean volumetric term `(λ/2)(ln J)²`; CFL accounts for bulk wave speed inflation |
| **Boundary condition ramping** | Linear ramp over 5 ms before sinusoidal shaking — prevents impulsive loading and spurious high-frequency content |
| **Kinematic BC consistency** | Velocity and acceleration at Dirichlet DOFs overwritten analytically (time derivatives of prescribed law) — prevents boundary drift over thousands of steps |
| **Sinusoidal two-axis motion** | 200 Hz oscillation in Z; coupled Y-motion from geometric constraint |
| **Rayleigh damping** | Mass-proportional term α·M; projected to reduced operator for ROM compatibility |
| **FEniCSx syntax** | UFL automatic differentiation for internal force and tangent; ghost DOF scatter patterns for MPI consistency |
| **POD snapshot analysis** | SVD on lift-subtracted snapshots; singular value decay; homogeneous BC enforcement |

---

## Physical problem

```
         ┌──────────────────┐  ← top face: fully fixed (tag 101)
         │                  │
         │   Neo-Hookean    │
         │   tube           │  E = 5 MPa   ν = 0.49   ρ = 1250 kg/m³
         │                  │
         └──────────────────┘  ← bottom face: prescribed Z + Y motion (tag 102)

Prescribed bottom displacement (t in seconds):
  Ramp phase  (t < 5 ms):   z(t) = -15 mm · (t / 5ms)
  Shock phase (t ≥ 5 ms):   z(t) = -15 mm - 5 mm · sin(2π · 200 · (t - 5ms))
                             y(t) = 0.5 · (15 mm + z(t))
```

Material parameters place this in the **soft tissue / elastomer** regime.
The combination of near-incompressibility and large prescribed displacements
makes this problem representative of biomedical device and offshore cable dynamics.

---

## Numerical method

### Explicit Newmark (central difference)

The semi-discrete equation of motion:

$$\mathbf{M}\ddot{\mathbf{u}} + \mathbf{C}\dot{\mathbf{u}} + \mathbf{f}_{int}(\mathbf{u}) = \mathbf{0}$$

is advanced in time using the **central difference** scheme (β = 0, γ = 1/2):

**Predictor:**
$$\mathbf{u}^{n+1,*} = \mathbf{u}^n + \Delta t\, \dot{\mathbf{u}}^n + \tfrac{1}{2}\Delta t^2\, \ddot{\mathbf{u}}^n$$

**Internal force + acceleration:**
$$\ddot{\mathbf{u}}^{n+1} = \mathbf{M}_L^{-1}\left(\mathbf{f}_{int}(\mathbf{u}^{n+1,*}) - \mathbf{C}\dot{\mathbf{u}}^n\right)$$

**Corrector:**
$$\dot{\mathbf{u}}^{n+1} = \dot{\mathbf{u}}^n + \tfrac{1}{2}\Delta t\left(\ddot{\mathbf{u}}^n + \ddot{\mathbf{u}}^{n+1}\right)$$

where $\mathbf{M}_L$ is the **lumped** (diagonal) mass matrix — enabling the O(N) explicit update
without any linear solve.

### CFL condition

The critical timestep is:

$$\Delta t = \alpha \frac{h_{min}}{c_{mech}}, \qquad c_{mech} = \sqrt{\frac{\lambda + 2\mu}{\rho}}, \qquad \alpha = 0.3$$

The safety factor 0.3 (rather than the theoretical 0.5) accounts for:
- Finite strain stiffening (Neo-Hookean tangent stiffness exceeds small-strain estimate)
- Near-incompressibility (ν = 0.49 → bulk modulus K ≈ 83 μ)

### Neo-Hookean strain energy

$$\psi(\mathbf{F}) = \frac{\mu}{2}(I_c - 3) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2$$

where $\mathbf{F} = \mathbf{I} + \nabla\mathbf{u}$, $J = \det\mathbf{F}$, $I_c = \text{tr}(\mathbf{F}^T\mathbf{F})$.

Internal force assembled via UFL automatic differentiation:
```python
f_int_form = fem.form(-ufl.derivative(psi * dx, u, ufl.TestFunction(V)))
```

---

## Key implementation details

### Why explicit and not implicit?

For wave-propagation problems the timestep is dictated by the physics (CFL), not by
nonlinear convergence. Implicit schemes allow larger Δt but require solving a nonlinear
system at every step — for a near-incompressible hyperelastic material that means
assembling and inverting a poorly-conditioned tangent stiffness. The explicit scheme
avoids this entirely, at the cost of ~50,000 small timesteps instead of ~500 large ones.

### Boundary drift correction

A subtle but critical bug in naive explicit implementations: after the velocity corrector,
Dirichlet DOFs accumulate a numerically wrong velocity. Over thousands of steps this
causes the prescribed boundary to drift from its intended trajectory.

**Fix:** Overwrite boundary velocities analytically after every corrector:
```python
v_new[dofs_bot_z] = vz_bot   # = dz/dt  (analytical derivative)
v_new[dofs_bot_y] = vy_bot   # = dy/dt
a_new[dofs_bot_z] = 0.0      # zero acceleration at constrained DOFs
```

### BC ramping

Without the 5 ms linear ramp, the instantaneous application of a 15 mm displacement
generates a sharp stress wave that immediately violates the CFL condition locally and
produces spurious ringing. The ramp ensures smooth wave initiation.

### Ghost DOF scatter pattern (FEniCSx / MPI)

```python
# After ADD+REVERSE (assembly):
f_int_vec.ghostUpdate(addv=PETSc.InsertMode.ADD,  mode=PETSc.ScatterMode.REVERSE)
# After INSERT+FORWARD (consistency):
f_int_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
```
The second scatter is required for consistent reads across MPI ranks.

---

## POD snapshot analysis

After running the FOM (87,811 snapshots, Δt ≈ 5.7×10⁻⁷ s), a POD basis was extracted
from lift-subtracted snapshots (every 50th step → 1,757 snapshots).

**Lift subtraction** enforces homogeneous boundary conditions on the modes:
$$\mathbf{S}_{deform}^{(k)} = \mathbf{u}^{(k)} - \mathbf{u}_{lift}(t_k)$$

This is a prerequisite for any Galerkin ROM — modes must satisfy zero BCs.

| Modes | Cumulative energy |
|-------|------------------|
| 1 | 57.34% |
| 2 | 86.57% |
| 3 | 96.16% |
| 5 | 99.08% |
| 10 | 99.81% |
| 13 | 99.92% |

The decay confirms that 13 modes capture the essential deformation physics,
with mode 1 corresponding to the dominant axial compression mode.

---

## Repository structure

```
.
├── README.md
├── mesh/
│   ├── tube.xdmf
│   └── tube_facets_linear.xdmf
├── fom/
│   └── explicit_solver.py          # Main FOM — explicit Newmark
└── pod/
    └── compute_pod.py              # Lift-subtracted SVD pipeline
```

---

## Dependencies

```
FEniCSx (dolfinx) >= 0.7
mpi4py
petsc4py
numpy
scipy
```

---

## References

1. Belytschko, T., Liu, W.K., Moran, B. (2000). *Nonlinear Finite Elements for Continua and Structures*. Wiley. — Explicit integration, CFL, lumped mass.
2. Holzapfel, G.A. (2000). *Nonlinear Solid Mechanics*. Wiley. — Neo-Hookean formulation, variational principles.
3. Hughes, T.J.R. (1987). *The Finite Element Method*. Prentice Hall. — Newmark family, stability analysis.
4. Logg, A., Mardal, K.A., Wells, G.N. (2012). *Automated Solution of Differential Equations by the Finite Element Method*. Springer. — FEniCS/UFL formulation.
5. Carlberg, K., Tuminaro, R., Boggs, P. (2015). *Preserving Lagrangian structure in nonlinear model reduction*. SIAM J. Sci. Comput. — POD/ROM theory.
6. Cook, R.D. et al. (2002). *Concepts and Applications of Finite Element Analysis*. Wiley. — Rayleigh damping.
