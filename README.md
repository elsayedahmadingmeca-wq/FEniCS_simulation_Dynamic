# Explicit Nonlinear Dynamics — Hyperelastic Tube (FEniCSx)

3-D explicit finite element solver for the transient dynamics of a hyperelastic tube
subjected to large prescribed displacements, built from scratch in FEniCSx.

The same geometry and mesh are used in a companion project on
[quasi-static thermo-mechanical coupling](https://github.com/elsayedahmadingmeca-wq/Portfolio),
which allows direct comparison of static vs dynamic stress fields.

---

## Physics

- Finite-strain hyperelasticity — compressible Neo-Hookean, ν = 0.49 (near-incompressible)
- Explicit Newmark scheme — central differences (β=0, γ=1/2)
- Lumped mass matrix — O(N) update, no linear solve at any step
- Mass-proportional Rayleigh damping — compatible with explicit integration
- Two-axis sinusoidal prescribed displacement: 200 Hz, 15 mm amplitude

Full governing equations and weak form: [`physics.pdf`](physics.pdf)

---

## Why explicit and not implicit?

Implicit Newmark (β=1/4) is unconditionally stable but requires solving
a nonlinear system at every step. For this problem the near-incompressible
material (ν=0.49) and large ramp displacements force Newton convergence at
Δt ~ 1e-6 s — the same order as the CFL timestep.

At that Δt, implicit integration still demands 87,000 sparse tangent
stiffness assemblies and GMRES solves. In practice: memory spike and
runtime an order of magnitude higher than explicit, for zero accuracy gain.

The problem is wave-propagation dominated. Explicit is the natural choice.

---

## Numerical details

**CFL stability**
```
Δt = α · h_min / c_mech
c_mech = √((λ + 2μ) / ρ) ≈ 115 m/s

Safety factor α = 0.3  (below theoretical 0.5)
Reason: near-incompressible bulk modulus K ≈ 83μ inflates wave speed
        at finite strains beyond the small-strain estimate

Result: Δt ≈ 5.7e-7 s  →  87,811 steps over T = 50 ms
```

**Mass lumping — row-sum**
```python
ones = fem.Function(V);  ones.x.array[:] = 1.0
M_mat.mult(ones.x.petsc_vec, m_lumped.x.petsc_vec)
m_lumped_inv.x.array[:] = 1.0 / m_lumped.x.array
```
Diagonal inverse computed element-wise. No matrix factorisation.

**Kinematic BC consistency**

After the velocity corrector, Dirichlet DOFs accumulate a spurious
velocity from the inertial update. Over 87,000 steps this causes the
prescribed boundary to drift from its trajectory. Fix: overwrite
boundary velocities and accelerations analytically after every corrector.

```python
v_new[dofs_bot_z] = vz_bot(t)   # analytical dz/dt
v_new[dofs_bot_y] = vy_bot(t)
a_new[dofs_bot_z] = 0.0          # zero acceleration at constrained DOFs
a_new[dofs_bot_y] = 0.0
```

**BC ramp**

Linear ramp over 5 ms before sinusoidal phase.
Without it, the instantaneous application of 15 mm displacement
generates a stress wave that violates CFL locally at step 1.

---

## What the simulation produces

- 🔹 Displacement field at each saved step — XDMF → ParaView
- 🔹 Von Mises stress field
- 🔹 Time history: bottom face displacement vs prescribed law

---

## Quick sanity checks

**Energy balance** — for α=0 (no damping), total mechanical energy
E_kin + E_pot must be conserved up to O(Δt²) per step.
A monotonically growing total energy signals a CFL violation.

**Wave speed** — apply a small impulse at t=0, measure arrival time
at the top face. Should match c_mech ≈ 115 m/s.

---

## Repo structure

```
src/
  explicit_solver.py     ← main FOM solver
  compute_pod.py         ← POD basis extraction (lift-subtracted SVD)
mesh/
  tube.xdmf
  tube_facets_linear.xdmf
physics.pdf              ← governing equations and weak form
```

---

## Dependencies

```
dolfinx >= 0.7
mpi4py
petsc4py
numpy
```

---

## References

- Carlberg, Tuminaro, Boggs. *Preserving Lagrangian structure in nonlinear model reduction.*
  SIAM J. Sci. Comput., 37(2):B153–B184, 2015. — [arXiv:1401.8044](https://arxiv.org/abs/1401.8044) (open access)
- Bonet, Wood. *Nonlinear Continuum Mechanics for Finite Element Analysis*, 2nd ed. Cambridge, 2008.
- Logg, Mardal, Wells. *Automated Solution of Differential Equations by the FEM (FEniCS Book).*
  Springer, 2012. — [fenicsproject.org/book](https://fenicsproject.org/book) (open access)
- Dokken. *The FEniCSx Tutorial*, 2023. — [jsdokken.com/dolfinx-tutorial](https://jsdokken.com/dolfinx-tutorial)
