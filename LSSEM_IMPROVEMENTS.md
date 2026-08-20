# LSSEM Solver Improvements

This document summarizes the major improvements, architectural changes, and bug fixes made to the Python Least-Squares Spectral Element Method (LSSEM) solver.

## 1. Performance & Vectorization Upgrades

The original Python port implemented the solver logic via manual `for` loops iterating element-by-element, which was incredibly slow in Python. We applied global vectorization to eliminate these loops:

*   **Fully Batched Operators**: Replaced manual `for` loops across elements with vectorized `np.matmul` and `np.einsum` operations across the entire `(nelem, N, N)` tensor space. This allows numpy to dispatch matrix operations to highly optimized BLAS routines under the hood.
*   **Exact Analytical Preconditioner**: Replaced the slow, iterative unit-vector approach to building the Jacobi preconditioner with an exact analytical diagonal extraction (`compute_jacobi`). Preconditioner assembly time dropped to near zero.
*   **Memory Optimization (`SolverState`)**: We introduced the `SolverState` class to centrally manage and preallocate all intermediate work arrays (such as `su`, `c`, `tmp_x`, `u_x`, etc.). This completely eliminates expensive and repetitive memory allocations inside the inner implicit solver loops.

**Result**: The lid-driven cavity simulation (Re=1000), which previously required significant time to converge, now reaches steady-state convergence (600+ time steps) in approximately 60 seconds.

## 2. Physics & Boundary Condition Fixes

The solver was upgraded from a simple square domain to support complex fluid dynamics problems, exposing and fixing several critical physics bugs:

*   **Complex Geometry Support**: Successfully set up the multi-block **Backward-Facing Step (BFS)** geometry, properly handling re-entrant corners, block connectivity, and heterogeneous boundary condition assignments across the L-shaped domain.
*   **Fixed the Drifting Pressure Bug**: Discovered that the outlet boundary condition (`bc=4`) lacked the crucial `p=0` Dirichlet condition in both `apply_bc` and `apply_mask`. Without this, the global pressure field was mathematically unconstrained and allowed to drift, which previously ruined the upstream flow physics and washed out flow features.
*   **Identified Mass Conservation Defect**: By implementing a custom Gauss-Lobatto quadrature script (`check_mass.py`), we calculated the exact mass fluxes across all boundaries. We conclusively proved that the standard unpenalized LSSEM formulation inherently leaks mass at the step singularity, resulting in a ~400% mass defect that washes out the recirculation bubble. (This sets the stage for adding the $W_{cont}$ continuity penalty).

## 3. Architecture & Infrastructure

The project structure was refactored for better usability and separation of concerns:

*   **Configuration Files**: Moved all hardcoded physical (Reynolds number, timestep) and solver (Newton tolerances, PCG limits) parameters into `.toml` files (`cavity.toml` and `bfs.toml`). The core codebase is now entirely decoupled from specific simulation setups.
*   **Advanced Diagnostics**: Built robust plotting scripts (`plot_bfs.py`, `plot_intermediate_bfs.py`) capable of loading checkpoint restart files (`.npz`) on the fly, allowing for post-processing and intermediate monitoring without blocking the simulation.
*   **Plotting Artifact Fixes**: Manually masked the solid step region in our matplotlib configurations to prevent `scipy.interpolate.griddata` from interpolating fake velocities inside the solid wall. This solved the visual illusion of "flow coming out of the wall" seen in early stream plots.

---

# 3D extension and solver correctness — 2026-08

Detail and evidence for everything below live in
[3D_STATUS.md](./3D_STATUS.md); the plan and its gates are in
[3D_DEVELOPMENT_PLAN.md](./3D_DEVELOPMENT_PLAN.md). This section is the summary.

## 3b. Net-net

The 3D solver works and is verified at its design order — **2.00**, measured two
independent ways against exact solutions (Stokes decay for the implicit path,
Taylor–Green for the **convective** path). Production recipe, each element
decided by measurement: **legacy row weights, no operator-AC, CG tolerance
1e−06**. Twelve silent bugs found, **none of which raised an exception**.
M1–M5 done; M6 and M7 remain, with the open risk being whether the recipe
survives walls.

## 4. `lssem3d`: 3D via a Fourier basis in z

A **new module**. `lssem2d` is called, never edited — the 3D code reuses its
mesh, GLL and gather-scatter, and works around the two places the 2D API did not
fit on the 3D side (`deriv.py` re-derives the contractions rank-agnostically;
`solver3d.gs` folds `(var, mode)` into one axis rather than reimplementing the
connectivity).

Eight modules, **144 tests**. Five of seven milestones complete: the operator
reproduces the 2D cavity at `k_z` = 0 (M2), MMS convergence is spectral in `N`
and exponential in `Nz` (M4), and the `a_mass`/CFL feasibility gate passes in 3D
with a window spanning ~66× in `dt` (M5).

## 5. Row weights: the second least-squares scaling, which `lssem3d` lacked

`lssem2d` writes the momentum row as `a_mass·u + a_flux·N(u)` with the
constraints at weight 1 — so **`a_flux` is the weight of momentum against the
constraints**, and its legacy setting (`a_mass` = 1, `a_flux` = `dt`) makes every
row O(1).

`lssem2d` offers **two** scalings: legacy (`a_mass` = 1, `a_flux` = `dt`), which
the Chan validation uses, and `w_mom` = 1 (`a_mass` = `fac1/dt`, `a_flux` = 1),
which the cavity and BFS studies use. **`lssem3d` hard-coded the second and had
no way to express the first** — which is why it could not reproduce a 2D result
the 2D code produces routinely.

| Stokes decay, AC off, `dt` = 5e−3 | σ | rel err | CG |
|---|---|---|---|
| no row weights | 9.31809 | 4.68e−04 | 600000 (cap) |
| **row weights** | **9.31413** | **4.19e−05** | **22047** |

**27× fewer iterations, 11× more accurate — on this benchmark.**

**The weighting is a problem-dependent choice, not a universal fix.** On the
Re = 1000 cavity the legacy scaling is *27× worse* (688 CG/step against 25), and
AC there is worth 25 against 12320 without it. The plausible discriminant is
viscosity: legacy scales the momentum row to `u + β·dt·(p_x + ν∇×ω)`, and at
ν = 1e−3 that vorticity coupling is ~3e−7, effectively absent. So AC's value is
Reynolds-number dependent and AC is **not** dispensable — an earlier claim to the
contrary, drawn from the Stokes case alone, is withdrawn. Choosing the weighting
per problem is an open design question, exactly as in 2D.

## 6. Accuracy: the scheme now hits its design order

Validated against **Stokes decay** (Chan 1996 Fig. 1), which has an analytic
decay rate — so the error is absolute rather than self-referential:

| `dt` | 0.01 | 0.005 | 0.0025 | 0.00125 | order |
|---|---|---|---|---|---|
| row weights, no operator-AC | 1.68e−04 | 4.19e−05 | 1.05e−05 | **2.61e−06** | **2.00, 2.00, 2.00** |
| operator-AC, `κ_p` = `a_mass` | 6.48e−03 | 6.06e−03 | 6.06e−03 | 6.07e−03 | 0.00 |

Exactly the design order — RK3's third order applies to the explicit half alone
(3.025 measured on the coefficient table); Crank–Nicolson caps the mixed scheme
at 2. This is the **first PDE-level confirmation**: the earlier temporal gate ran
on a scalar model with no pressure and no constraint rows.

**For the Stokes benchmark** the configuration is legacy row weights with no
operator-AC. That does **not** transfer to the cavity or to high Reynolds number
(§5) — a general production recipe is not yet established.

## 7. Performance

* **Mode-parallel solve** (`lssem3d/parallel.py`): whole PCG solves distributed
  across `k_z` chunks. **6.7×** at Nz=128 on 12 performance cores.
* Profiling settled three things by measurement: `normal_op` is **99.4%** of a
  step (FFT and gather-scatter are not worth optimising); BLAS threading buys
  nothing (95.51 → 94.84 ms, 1→8 threads); and threads **tie** processes, so the
  ceiling is memory bandwidth, not the GIL. More cores will not help — which
  re-aims the numba work at *fusing* passes over the data rather than compiling
  the existing ones.
* **Jacobi diagonal assembled across elements** — the probe returned
  `diag/multiplicity`, so `1/diag` over-weighted every element-boundary node by
  2–4×. Worth **1.41–1.44×** fewer CG iterations.

## 8. Correctness fixes

| fix | why it mattered |
|---|---|
| **Row weights** (§5) | the functional itself was mis-scaled |
| **Jacobi probe contamination** | probing the *assembled* operator with a discontinuous unit vector folded off-diagonal couplings into the diagonal — 1.4% error at every interface node. Probing unassembled and keeping the gather is exact (0.0) |
| **Nyquist imaginary half unconstrained** | `fourier.py` stated the invariant and tests asserted it, but nothing *enforced* it in the solve; `irfft` discards those components, so CG was filling a physically invisible direction |
| **True-residual safeguard in `pcg`** | the recursive residual drifts over 10⁴ iterations; CG could declare victory on a number that no longer described the iterate. Now verified against `b − A x`, restarts on drift, and reports the true residual |
| **Pressure pin covered one copy of a shared node** | on a periodic seam the pinned node is shared 2–4 ways, so the global dof was never pinned *and* the mask disagreed with itself across copies — making the assembled operator **non-symmetric**, which CG requires. Symmetry error 1.5e−07 at multiplicity 2, 5.9e−05 at 4; exactly zero once every copy is pinned. On Taylor–Green it was a **240× error floor** that made the convection-active order measurement impossible |
| **CG over-solving** | every 3D solve ran at `tol` = 1e−12 while the 2D driver uses an inexact 1% solve (`cgsfac` = 0.01). Measured policy: **1e−06 costs nothing and saves ~40% of the iterations** |
| **`jacobi_inverse`** | `1.0/np.maximum(d, 1e-30)` put **1e30** at every prescribed dof and survived only because the masked residual is exactly zero. Now exactly 0 there, and it *raises* on a negative diagonal rather than clamping a bug into a live multiplier |

Twelve distinct bugs have been found in the 3D work so far and **not one raised
an exception** — every one produced a correctly-shaped, plausible array.

## 8b. The time-splitting is now verified end to end

The last unverified path was the explicit RK3 convective half, which no test
reached: the Stokes capstone runs with convection **off by construction**, and
the order-3.025 explicit result runs on a scalar model that bypasses the
convective assembly entirely. **Taylor–Green decay** — an exact unsteady
Navier–Stokes solution where `u·∇u` is balanced pointwise by the pressure
gradient — closes it:

| `dt` | 0.1 | 0.05 | 0.025 | 0.0125 | order |
|---|---|---|---|---|---|
| L2 velocity error | 2.53e−06 | 6.33e−07 | 1.58e−07 | 3.95e−08 | **2.00, 2.00, 2.00** |

Supporting: `CV.convective` is spectrally exact against the analytic `u·∇u`
(4.9e−13 at N = 16), and `mesh.periodic_y` — unused anywhere in the repo until
this test — is spectrally accurate.

## 9. Gates that were wrong as written

Four acceptance criteria would each have **failed correct code**. Recording them
was as valuable as the fixes:

| stated criterion | measurement |
|---|---|
| "RKW3 temporal slope 3.0 ± 0.15" | unachievable — CN caps the mixed scheme at 2 |
| "iterations should fall with `k_z`; flat means a bad preconditioner" | flat is *correct* at production `a_mass`: `ν·k_z²` = 1.42 against 1200 |
| "M5 feasibility closed" | closed on **2D** evidence, for a plan whose own risk register rates that transfer as high risk |
| "AC is the enabling technology" | true for the 2D outflow case; in 3D it was compensating for §5 |

## 10. 2D solver

* **Dong (2015) outflow BC** as `bc = 6`, tested through the full ladder — it
  beats P+Z decisively on truncated domains.
* **Jacobi diagonal built at the solve linearisation**, and the dead per-step
  build removed: `newton_step` had been building `M_inv` from the linearisation
  at `U/2` while solving against the Jacobian at full `U`, and `step_bdf`
  computed a diagonal every step that `newton_step` immediately shadowed.

The 2D legacy weighting (`a_mass` = 1, `a_flux` = `dt`) is what the Chan (1996)
validation depends on — it is the canary for any future change to the weighting
path.
