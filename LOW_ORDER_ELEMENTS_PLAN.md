# Low-order FOSLS: linear quadrilateral and triangular elements by direct assembly

Branch `low-order-elements`.  Rewritten 2026-09-20 after establishing that
direct assembly, not a retrofit of the spectral code, is the right route.

## 0. The decision, and why it changed

The first version of this plan tried to reach Q1 by driving the existing
spectral code down to `N = 1`.  That runs into a structural problem (Section
A, kept below as an appendix): the code is COLLOCATED, nodes are the quadrature
points, and two-point GLL is exact only to degree 1 while the least-squares
functional needs degree 2.  Fixing it means decoupling basis from quadrature
throughout `operators.py`, `curvi.py`, `solver.py` and `precond.py`.

**Direct assembly avoids the problem instead of solving it.**  For linear
elements the element matrices can be written down in closed form, there is no
quadrature to get wrong, and `L^T L` is assembled once per linearisation and
handed to a sparse solver.  Everything the spectral code does to make HIGH
order affordable -- tensor products, collocation, fast diagonalisation,
p-multigrid, vertex-patch Schwarz -- is machinery that buys nothing at p = 1
and can simply be left out.

This also makes triangles no harder than quadrilaterals, which the retrofit
route could not (tensor-product derivatives have no simplex analogue).

## 0.1 Decided 2026-09-20

| question | answer | consequence |
|---|---|---|
| objective | **complex geometry via automatic meshing** | triangles are the real target, not a by-product; gmsh/meshio becomes a dependency at that stage; the divergence scaling (V6) still comes out and is kept as a secondary result |
| first increment | **time-dependent from the start** | BDF2 and G6 are in increment 1, not deferred.  The debugging surface is larger, which makes G4 -- the exactly-representable test -- more important, because it isolates assembly errors from time-integration errors |
| first element | **Q1 on affine quads** | closed form applies, and running the identical geometry as the spectral code (cavity, channel) gives a matched-dof comparison against ground truth before moving to P1 triangles where none exists |

Build order: Q1 affine with BDF2 on internally generated structured meshes,
validated against the spectral code; then P1 triangles; then gmsh and a
geometry the current code cannot mesh.  `scipy`, `sympy` and `pyamg` are
already available; `meshio`/`gmsh` are added at the triangle stage, not before.

## 1. What "closed form" means here

It means the element matrix entries are **explicit algebraic expressions in the
node coordinates** -- evaluated arithmetic, not a quadrature loop over sample
points.  There is no rule to choose, no order to verify, and no
under-integration to detect.

The reason it is available on a straight-sided triangle is a single identity.
In barycentric coordinates `L1, L2, L3`, every monomial integrates exactly:

```
    integral over T of  L1^a L2^b L3^c  dA  =  2A * a! b! c! / (a+b+c+2)!
```

with `A` the triangle area.  Specialising it gives the three integrals that
appear in a P1 FOSLS element matrix:

| integral | value | what it is |
|---|---|---|
| `int phi_i` | `A/3` | a = 1 |
| `int phi_i phi_j` | `A/12` (i != j), `A/6` (i = j) | the consistent mass matrix |
| `int phi_i phi_j phi_k` | `A/60`, `A/30`, `A/10` | needed only for the convection coefficient |

and the gradients are **constants**:

```
    dphi_i/dx = b_i / (2A),    dphi_i/dy = c_i / (2A)
    b_1 = y2 - y3,  c_1 = x3 - x2     (cyclic in 1,2,3)
```

so for example the continuity row `u_x + v_y` contributes to `L^T L` exactly

```
    (uu block)  int (dphi_i/dx)(dphi_j/dx) dA  =  b_i b_j / (4A)
```

with no approximation whatever.

**Why the degree works out.**  Within a P1 element the FOSLS residual rows are
at most LINEAR:

| row | expression | degree |
|---|---|---|
| continuity | `u_x + v_y` | constant |
| vorticity | `omega - (v_x - u_y)` | linear |
| momentum | `a_mass*u + a_flux*(fu*u_x + u*fu_x + p_x + nu*omega_y)` | linear |

so `(L U)^T (L U)` is at most quadratic, and the convection coefficient `fu`
(itself a P1 field) raises a few terms to cubic.  The identity above integrates
both exactly.  **The degree-2 problem that blocks the spectral retrofit does
not arise.**

**Where closed form is NOT available.**

| element | exact closed form? |
|---|---|
| P1 triangle, straight-sided | **yes** |
| Q1 on a rectangle or parallelogram (affine map, J constant) | **yes** -- same argument, tensor-product monomials |
| Q1 on a general bilinear quadrilateral | **no** |

For a general quad the map is bilinear, so `J` varies over the element and the
inverse Jacobian makes the integrand a RATIONAL function, not a polynomial.  No
finite quadrature rule is exact for it.  Universal FEM practice is 2x2 Gauss
(degree 3), which is accurate but not exact -- and that is a deliberate,
well-understood approximation rather than the silent under-integration of
Section A.

## 2. Architecture: a standalone assembled code

Not a modification of `lssem2d`.  A separate package, say `fosls_fem/`, sharing
nothing but the problem definition.

The current code is matrix-free: `apply_L` evaluates the operator pointwise and
`L^T L` is never formed.  The low-order code should do the opposite:

1. assemble the sparse `L^T L` and right-hand side `L^T f` per linearisation;
2. solve with sparse Cholesky (2D, up to large problems) or CG + AMG;
3. iterate the Newton/Picard linearisation as now.

`L^T L` is **symmetric positive definite by construction** -- it is a normal
equation -- so Cholesky needs no pivoting and CG is the natural iterative
choice.  Per element the dense block is 12x12 (P1, 4 fields) or 16x16 (Q1),
computed by formula.

That this is an INDEPENDENT implementation is a feature, not a cost.  It shares
no code path with the spectral solver, so agreement between them is real
evidence -- the same argument that made the force audit's
gradient-form-versus-omega-form check worth doing.

## 3. What to build

| # | item | notes |
|---|---|---|
| 1 | `mesh.py` | read a triangular or quad mesh (gmsh `.msh` is the pragmatic choice); node coords, connectivity, boundary tags |
| 2 | `element_p1.py` | closed-form `b_i, c_i, A`; the three barycentric integrals; the 12x12 element `L^T L` and `L^T f` |
| 3 | `element_q1.py` | bilinear shape functions; 2x2 Gauss; the 16x16 block.  Assert the affine case reproduces the closed form |
| 4 | `assemble.py` | scatter into `scipy.sparse`; Dirichlet elimination; the Dong outflow as an edge term |
| 5 | `solve.py` | `scipy.sparse.linalg.splu` / `cholmod`, and a CG+AMG path for larger meshes |
| 6 | `timestep.py` | BDF2 with the SAME `a_mass`, `a_flux`, `kappa_p` definitions as `lssem2d.lssem.ls_coeffs` -- import them, do not re-derive |

Point 6 matters: the weighting is the paper's thesis, and the low-order code
must use the identical coefficients or it tests something else.  They are
scalar row coefficients with no element dependence, so importing them is
correct and also guards against drift.

## 4. Validation ladder

| rung | test | pass criterion |
|---|---|---|
| V0 | element matrix vs symbolic integration (sympy) on one triangle | agreement to machine precision |
| V1 | Q1 on a parallelogram: 2x2 Gauss vs the closed form | identical, confirming the Gauss path |
| V2 | manufactured solution, P1 and Q1, h-refinement | second order in `L2(u)`, first in the gradient |
| V3 | pointwise `max\|div u\|` vs h, and vs the spectral code at matched dof | reported, not judged |
| V4 | lid-driven cavity Re = 1000 vs Ghia | comparable error to the spectral code at matched dof, or a documented reason |
| V5 | cylinder Re = 100 on a triangular mesh | St within the scatter of the spectral N sweep |

V3 is the scientifically interesting one.  V5 is the one that only this route
makes possible, because it needs an automatic mesher.

## 5. What it costs and what it buys

**Cost.** A second code, perhaps 600-1000 lines, and it will be far slower per
degree of freedom than the spectral solver.  For the questions on the table
that does not matter.

**Buys.**

* A genuinely independent implementation to cross-validate against.
* **A measured statement of how the FOSLS divergence advantage scales with p.**
  The paper's divergence results rest on N = 8-10; nothing currently bounds
  them from below.  This is the result the exercise is really for.
* Complex geometry, via automatic triangular meshing.  The present quad mesh is
  a bespoke nine-block construction with an O-ring, built by hand.

**Expected outcome, recorded in advance.**  P1 and Q1 will converge at second
order and will enforce `div u` far more weakly than N = 8 -- plausibly two to
three orders worse in pointwise `max|div u|` at matched dof.  If so, the
product is the scaling statement above, not a production capability.  Writing
this down so that it reads as a finding rather than a disappointment.

## 6. Note on 3D

Trilinear (Q1 hex) and linear tetrahedra follow the same argument: the
barycentric identity generalises to `6V * a!b!c!d!/(a+b+c+d+3)!` on a
tetrahedron, so P1 tets are closed form too.  The existing 3D code is NOT a 3D
element code -- it is a per-Fourier-mode operator with `ddz = i*k_z` -- so
there is nothing there to retrofit either way, and direct assembly is again the
shorter path.  Do 2D first; it answers whether the formulation is usable at low
order at all, for a fraction of the work.

---

# Appendix A -- the retrofit route, and why it was set aside

Kept because the measurements are real and would otherwise be repeated.

**`N = 1` already is the bilinear element.**  The tensor-product GLL basis at
p = 1 has both nodes at the element endpoints.  `lgl_nodes(1)` returns
`[-1, 1]` with weights `[1, 1]`.  On the cylinder mesh at N = 1 the mesh builds
(240 elements, 1104 dof), `SolverState` constructs with correct coefficients,
and `VertexSchwarzCondensed2D` constructs without error.  No plumbing is
missing.

**The blocker is the quadrature.**  GLL with `N+1` points is exact to degree
`2N-1`, so at N = 1 that is degree ONE:

| N | integral of x^2 over [-1,1] | exact | error |
|---|---|---|---|
| 1 | 2.000000 | 0.666667 | **1.3e+00** |
| 2 | 0.666667 | 0.666667 | 0.0e+00 |
| 8 | 0.666667 | 0.666667 | 1.1e-16 |

and the functional needs degree 2.  Under-integrating a sum of squares can
introduce a null space -- the least-squares form of hourglassing.

**It is invisible in geometric checks.**  Measured, 5 BDF steps at N = 1, 2, 4:

| N | 5 steps | quadrature area | exact | error | dof | `\|u\|max` |
|---|---|---|---|---|---|---|
| 1 | OK | 699.235 | 699.215 | 0.003 % | 1,104 | **1.0000** |
| 2 | OK | 699.215 | 699.215 | 0.000 % | 4,128 | 1.4619 |
| 4 | OK | 699.215 | 699.215 | 0.000 % | 15,936 | 1.6090 |

The area is essentially exact because the mesh is mostly straight-sided blocks
on which `J` is CONSTANT, and any rule with `sum(w) = 2` integrates a constant
exactly.  The degree-2 problem lives in the OPERATOR, where the integrand is a
product of two first derivatives -- degree 2 even where `J` is constant.  Any
check of areas, volumes or metric identities will pass and prove nothing.

`N = 1` also runs but is grossly under-resolved on that mesh: `|u|max` is still
exactly the free-stream 1.0000 after five steps while N = 2 reaches 1.4619.
With 16 azimuthal by 4 radial Q1 elements there is nothing to resolve a
Re = 100 boundary layer with.

**Why it was set aside.**  Fixing this means decoupling basis from quadrature
across `operators.py`, `curvi.py`, `solver.py`, `precond.py`, `bc.py` and
`obc.py`, and it still leaves triangles unreachable -- tensor-product
derivatives have no simplex analogue, and the code carries 12 such patterns in
`operators.py`, 11 in `curvi.py`, 8 in `precond.py`.  Direct assembly of
closed-form element matrices does less work and reaches further.

**What transfers to the new route regardless.**  The FOSLS formulation itself
is element-independent: the VVP system and its rows, both least-squares weights
(`a_mass` and `a_flux` are scalar ROW coefficients -- the derivation of
`w = sqrt(dt)` from `dt_eff = dt*w_mom/w_mass` and `a_mass*a_flux = fac1`
contains no element), the AC term, BDF2, and the ADN result that the Riesz map
is block-triangular in the original variables.  `compute_global_indices` hashes
physical coordinates rather than assuming structure, so even the merge layer
would work for any conforming mesh.

---

# Tollgates

One gate per build step, then one per physics claim.  **Each criterion is a
number and each gate can fail.**  A gate that cannot fail is not a gate; where
a quantity is expected to be poor (the divergence), the gate requires it to be
REPORTED and to converge, not to be small.

Nothing proceeds past a red gate.  The ordering matters: G2 and G4 between them
can detect almost every error the later gates would see, and they cost minutes
rather than hours.

## G1 -- `mesh.py`: the mesh is what the file says it is

| test | criterion |
|---|---|
| node and element counts vs the gmsh header | exact |
| `sum` of element areas vs the analytic domain area (unit square; annulus `pi(R^2-r^2)`) | relative error `< 1e-12` (straight-sided elements make this exact) |
| signed area / Jacobian of every element | **all strictly positive** -- no inverted or degenerate element |
| edge incidence | every interior edge shared by exactly 2 elements, every boundary edge by exactly 1 |
| Euler characteristic `V - E + F` | 1 for a simply connected domain, 0 for the annulus -- exact integers |
| each boundary-tagged edge lies on its geometric boundary | distance `< 1e-12` |

The Euler check is cheap and catches the dangling-node and duplicate-node
failures that an area check passes -- the same class of error that produced a
hydrodynamically SLIT cylinder mesh earlier in this project and was caught only
by counting unmerged nodes.

## G2 -- `element_p1.py`: the closed form IS the integral

| test | criterion |
|---|---|
| barycentric identity `2A a!b!c!/(a+b+c+2)!` vs `sympy` symbolic integration, random triangle, all `a+b+c <= 3` | agreement `< 1e-14` relative |
| partition of unity: `sum phi_i = 1`, `sum grad phi_i = 0` | `< 1e-15` |
| gradients vs finite difference of the interpolant | `< 1e-9` |
| **the 12x12 element `L^T L` vs degree-10 Dunavant quadrature of the same integrand** | `< 1e-13` relative, worst entry |
| symmetry of `L^T L` | `< 1e-15` |
| eigenvalues of `L^T L` on one element | **exactly one zero (the pressure constant), next smallest `> 1e-8 * lambda_max`** |

The last row is the hourglassing test, and it is the reason this route was
chosen.  With exact integration it should pass by construction; if a second
near-zero mode appears, something in the row definitions is wrong, and it is
far cheaper to find here than in a diverging simulation.

## G3 -- `element_q1.py`: the Gauss path is calibrated

| test | criterion |
|---|---|
| **parallelogram: 2x2 Gauss vs the closed form** | identical, `< 1e-14` |
| general quad: 2x2 vs 4x4 vs 6x6 Gauss, worst entry | 2x2 reported as the quadrature error; `|2x2 - 6x6| / |6x6| < 1e-2` |
| FEM patch test: a linear field reproduced exactly on a distorted patch | `< 1e-12` |
| symmetry, and the eigenvalue test as G2 | as G2 |

The parallelogram row is the important one: it is the only case where both
paths are exact, so it validates the Gauss machinery against ground truth
before it is used where no ground truth exists.

## G4 -- `assemble.py`: the strongest single test in the plan

| test | criterion |
|---|---|
| global `L^T L` symmetric | `< 1e-14` |
| null space before boundary conditions | exactly one vector, the pressure constant |
| after Dirichlet elimination | Cholesky succeeds; no zero pivot |
| **exactly-representable solution: build `U_exact` from a LINEAR velocity and pressure field, assemble `L^T L` and `L^T f` from the matching forcing, and evaluate the residual** | `\|\|L^T L U_exact - L^T f\|\|_inf < 1e-12` |
| one-element mesh: global matrix equals the element matrix | `< 1e-15` |

The exactly-representable test is worth more than the rest of the gate
combined.  A linear field is in the P1 space, so the discrete solution must
equal it to round-off -- and the test exercises the element matrices, the
scatter, the boundary conditions and the forcing simultaneously.  Any indexing
error, sign error or missing term fails it.  This is the analogue of the
Poiseuille check the spectral code relies on.

## G5 -- `solve.py`

| test | criterion |
|---|---|
| sparse Cholesky vs dense `numpy.linalg.solve`, small mesh | `< 1e-12` relative |
| CG + AMG vs direct, medium mesh | agreement to the requested CG tolerance |
| AMG iteration count under uniform refinement, 4 levels of `h` | **growth slower than `O(1/h)`** -- ideally flat |
| symmetry exploited: Cholesky without pivoting completes | no failure |

## G6 -- `timestep.py`: the same scheme, not a similar one

| test | criterion |
|---|---|
| **`a_mass`, `a_flux`, `kappa_p` imported from `lssem2d.lssem.ls_coeffs`** vs locally computed | **bit-identical** over `dt` in {0.2, 0.1, 0.05, 0.0125} and all three weightings |
| temporal order: manufactured unsteady solution, `dt` halved four times | observed order `2.0 +- 0.1` for BDF2 |
| steady limit: march to steady state vs a direct steady solve | `< 1e-10` |
| `dt_eff = dt` under balanced weighting | verified symbolically, not assumed |

Row 1 is not a formality.  The weighting is the paper's thesis; if the
low-order code re-derives the coefficients it is testing a different scheme,
and the failure would be silent.

## V1--V6 -- the physics gates

| gate | test | criterion |
|---|---|---|
| **V1** | manufactured solution, h-refinement, P1 and Q1 | `L2(u)` second order, `2.0 +- 0.15`; gradient first order |
| **V2** | global eigenvalue check on a real mesh, not one element | one zero mode; no spurious mode below `1e-6 * lambda_max` |
| **V3** | pointwise `max\|div u\|` vs `h`, and vs the spectral code at MATCHED dof | must CONVERGE under refinement at the expected rate; the value is **reported, not judged** |
| **V4a** | **REGULARISED** cavity, lid `u = 16 x^2 (1-x)^2`, J against h | J must DECREASE monotonically; the rate is reported |
| **V4b** | standard discontinuous-lid cavity vs Ghia | centreline profiles and `umin`/`vmin` as a physics sanity check -- explicitly NOT a convergence test, for the reason below |
| **V5** | cylinder Re = 100, triangular mesh, automatic mesher | `St` within `+-0.005` of the spectral value at comparable resolution; `C_D/C_L` harmonic ratio `2.000 +- 0.02` |
| **V1s** | **STOKES** MMS, h-refinement, both L2 and H1, compared against the Q1 INTERPOLANT of the exact solution | H1 must match the interpolant rate (optimal); L2 is reported -- it is NOT expected to reach h^2 under no-slip |
| **V1t** | **BDF2 temporal order**, spatial error identically zero by using a linear-in-space exact solution, started exactly from t0 - dt | observed order `2.00 +- 0.15` for every weighting |
| **V5os** | **ORR-SOMMERFELD** growth rate, plane channel, against the analytic eigenvalue (paper sec 6.2 does this at Re = 7500 spectrally) | growth rate within 5 % at the finest affordable mesh, and CONVERGING under h- and dt-refinement at second order |
| **V6** | the divergence scaling claim: `max\|div u\|` at matched dof for P1, Q1, N = 2, 4, 8 | a monotone trend with a fitted rate, reported with its uncertainty |

### Why V4 had to be split (measured 2026-09-20)

The standard lid-driven cavity is not a valid test of FOSLS convergence, and
the functional says so out loud.  At Re = 100 with Q1, Newton converged to
|dU| < 1e-9 on every mesh:

| lid | n = 8 | n = 16 | n = 32 | rate |
|---|---|---|---|---|
| discontinuous | 3.132e-01 | 3.472e-01 | 3.864e-01 | **-0.15** |
| regularised `16x^2(1-x)^2` | 1.594e-01 | 6.405e-02 | 2.736e-02 | **+1.3** |

The lid velocity is discontinuous at the two top corners, so the exact solution
is not in H^1 there -- and the FOSLS functional measures exactly that norm, so
refining resolves more of an unbounded quantity and J GROWS.  With a lid that
vanishes at both corners the solution is in H^1, the theory applies, and J
falls.

The criterion originally written for V4 therefore asked for convergence on a
problem whose functional does not converge.  This is the same phenomenon as
paper section 6.3, "Where the claim stops: boundary singularities", measured
from the other side -- there through the solution, here through the functional.

AND IT IS THE A POSTERIORI PROPERTY EARNING ITS KEEP.  Nothing about the
iteration looked wrong: Newton converged to |dU| = 8e-9 at every mesh and
Reynolds number, and `umin` moved TOWARD Ghia under refinement (-0.194 ->
-0.205 at Re = 100, -0.218 -> -0.331 at Re = 1000).  Only J revealed that the
iteration was converging to a minimiser which does not approach the continuous
solution.  That is what the Cai-Manteuffel-McCormick equivalence buys in
practice, and it cost one extra line of code.

### Two notes on verifying the orders (measured 2026-09-20)

**Spatial and temporal order must be separated, or neither is measured.**  The
temporal gate V1t uses a solution that is LINEAR IN SPACE -- u = (1+2x-3y)g(t),
v = (4-5x-2y)g(t), omega = -2g(t), p = (0.7+1.3x-0.4y)h(t) -- so every field and
every forcing term lies exactly in the Q1 space, the spatial error is
identically zero, and the mesh size is irrelevant.  Without that, a dt-study
measures the SUM of two errors: second order at coarse dt, then a stall at the
spatial floor, which reads as an order that degrades.  Measured with it, BDF2
gives 2.02, 2.01, 2.00 and a fitted 2.010.

**The exactly-representable test cannot distinguish the weightings, and should
not be expected to.**  Balanced and legacy give 1.6850e-03 and 1.6838e-03 at
the same dt -- identical to three digits.  That is correct: both drive the
residual to zero at the same minimiser, and the weighting only changes the
answer when the residual CANNOT be driven to zero.  V1t verifies BDF2's order;
it says nothing about the paper's thesis, and a test of that needs a case with
an irreducible residual.

### A near-miss worth recording

The first run of V1t showed legacy weighting DIVERGING as dt was refined --
errors of 1.5e2, 6.8e2, 4.8e4, 6.2e5 for dt from 0.08 to 0.01.  That is exactly
what a dramatic confirmation of the small-dt thesis would look like, and it was
entirely my bug: `ls_coeffs`'s table gives the history coefficient as **1** for
legacy, not `w_mass/dt`, because legacy does not pre-multiply the row by 1/dt
(which is why a_flux = dt there).  Using w_mass/dt made the history term dt
times too large.  Corrected, legacy gives 2.009 -- second order, like balanced.

The lesson is narrow and practical: a result that confirms the hypothesis
spectacularly deserves the same scrutiny as one that contradicts it, and the
first place to look is the code path that only that case exercises.

### A note on Newton's convergence rate

Newton converges LINEARLY here -- |dU| falling by a steady factor ~0.5 over 25
iterations, not quadratically.  That is correct, not a defect: `L^T L` is the
GAUSS-Newton Hessian, which drops the term involving second derivatives of L,
and Gauss-Newton degrades to linear convergence precisely when the residual at
the solution is large.  Here J = 0.37 at the minimiser, so it is.
`lssem2d.solver.newton_step` is the same Gauss-Newton and behaves the same way,
which is worth knowing before anyone reports it as a bug.

**V6 is the deliverable.**  V1--V5 establish that the code is correct; V6 is the
result the exercise exists to produce, and the one the paper cannot currently
state.  It should be planned as a figure from the outset rather than extracted
afterwards.

## Measurement discipline, learned the hard way

Applies to V4--V6 and carried over from the cylinder study:

* **Two independent estimators** for any headline quantity, sharing no code.
* **A physics-based validity check** where one exists -- the `C_D`/`C_L`
  harmonic ratio must be 2.000; it belongs to the flow, not to the estimator.
* **Noise floors measured, not assumed**: resample sub-windows of a completed
  run to get the per-sample scatter, and check the lag-1 autocorrelation before
  scaling it by `1/sqrt(n)`.
* **Report differences split across halves of the record.**  A difference that
  does not reproduce between the first and second half is not a measurement.
  This is what exposed a `C_L rms` significance that had been overstated
  roughly fourfold.
