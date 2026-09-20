# Adding bilinear and trilinear elements to the time-dependent FOSLS code

Branch `low-order-elements`.  Written 2026-09-20.

## 0. What this is for, and the question to settle first

A spectral element code does not want low-order elements for their own sake.
The reasons one might still want them here are specific, and which one applies
changes the plan:

| motive | what is actually needed |
|---|---|
| benchmark against low-order methods on equal footing | Q1 that is *correct*, even if slow |
| complex geometry where high-order meshing is hard | Q1 plus a mesh generator; the solver work is the same |
| probe how the FOSLS formulation behaves as p -> 1 | Q1 with the quadrature done right, or the answer is about the quadrature |

**Decide this before starting**, because the third motive is the one with a
predictable and uncomfortable answer: the least-squares functional controls
div u only as well as the space can represent it, and this project's own
divergence results (DIVERGENCE_CONSEQUENCES.md) rest on N = 8-10.  At Q1 the
pointwise divergence will be orders worse, and that is a property of the
formulation, not a bug to be fixed.

## 1. What already works

**N = 1 IS the bilinear element.**  The tensor-product GLL basis at N = 1 has
its two nodes at the element endpoints, so Q1 is already the p = 1 member of
the existing family.  `lgl_nodes(1)` was fixed for a p = 1 multigrid coarse
level and returns `[-1, 1]` with weights `[1, 1]`.

Measured on the cylinder mesh at N = 1: the mesh builds (240 elements, 1104
dof), `SolverState` constructs with the correct coefficients, and
`VertexSchwarzCondensed2D` constructs without error.  So there is no missing
plumbing.  The problem is elsewhere.

## 2. The blocker: the quadrature, not the basis

The code is **collocated** -- nodes and quadrature points are the same set, and
`mesh.wq = J * w_i * w_j` is the GLL weight at each node.  That is what makes a
spectral element method cheap, and it is exactly what fails at p = 1.

GLL with `N+1` points is exact to degree `2N-1`.  At N = 1 that is degree **1**:
the two-point rule is the trapezoidal rule.  Measured:

| N | integral of x^2 over [-1,1] | exact | error |
|---|---|---|---|
| 1 | 2.000000 | 0.666667 | **1.3e+00** |
| 2 | 0.666667 | 0.666667 | 0.0e+00 |
| 8 | 0.666667 | 0.666667 | 1.1e-16 |

The FOSLS functional is a sum of squares of first-order residuals.  For a
bilinear basis, `du/dx` is degree 1 in the transverse direction, so
`(du/dx)^2` is degree 2 -- and degree 2 is precisely what the two-point rule
cannot integrate.  **Every entry of L^T L at p = 1 is under-integrated.**

This is not a small quantitative error.  Under-integrating a sum of squares can
introduce a null space: modes the quadrature cannot see cost nothing in the
functional.  That is the least-squares form of hourglassing, and it is the
principal risk of the whole exercise.

### 2.1 The error is invisible in geometric checks -- do not look for it there

Measured, 5 BDF steps on the cylinder mesh at N = 1, 2, 4:

| N | 5 steps | quadrature area | exact | error | dof | \|u\|max after 5 steps |
|---|---|---|---|---|---|---|
| 1 | OK | 699.235 | 699.215 | **0.003 %** | 1,104 | **1.0000** |
| 2 | OK | 699.215 | 699.215 | 0.000 % | 4,128 | 1.4619 |
| 4 | OK | 699.215 | 699.215 | 0.000 % | 15,936 | 1.6090 |

Two things to take from this, and the first corrects a natural assumption.

**The area is essentially exact at N = 1**, because the mesh is mostly
straight-sided blocks on which the Jacobian is CONSTANT -- and any rule with
`sum(w) = 2` integrates a constant exactly.  The 0.003 % residual comes only
from the curved O-ring, where J varies.  So the degree-2 problem of Section 2
**does not show up in any geometric diagnostic**: it lives in the OPERATOR,
where the integrand is a product of two first derivatives, and that is degree 2
even on a straight-sided element with constant J.  A0 must therefore be an
operator-level or solution-level test.  Checking areas, volumes or metric
identities will pass and prove nothing.

**N = 1 also runs, and is grossly under-resolved on this mesh.**  Five steps
complete without error, but `|u|max` is still exactly 1.0000 -- the free-stream
initial value -- while N = 2 reaches 1.4619 and N = 4 reaches 1.6090.  The flow
has not begun to accelerate around the body.  With 16 azimuthal by 4 radial Q1
elements there is nothing to resolve a Re = 100 boundary layer with, so this
says nothing about the quadrature; it says the cylinder at 240 elements is the
wrong vehicle for A0.  Use a case whose exact solution is representable on a
coarse mesh instead.

## 3. Part A -- bilinear (2D Q1)

### A0. Measure the damage before building anything (half a day)

Run N = 1 against cases with exact solutions and compare with N = 2, 4, 8:

* Kovasznay or Poiseuille, where the exact solution is representable
* the manufactured solution used for the curvilinear gate G4
  (`scratch/curvi_g4.py`), which already sweeps N and reports `eps_u`, `eps_p`
  and pointwise `max|div u|`

Report the convergence rate and, critically, **look for a null space**: assemble
`L^T L` on a small mesh at N = 1 and take its smallest eigenvalues against
N = 2.  If there are spurious zero modes, reduced integration is dead and A1 is
mandatory.  If there are not, A2 becomes a cheap fallback worth measuring.

This step can kill or greatly simplify everything below it, so it comes first.

### A1. Decouple basis from quadrature (the principled route)

The structural change.  Introduce a quadrature rule independent of the nodal
set: `Q` Gauss points with `Q >= N+1`, exact to degree `2Q-1`.  For Q1 with
2x2 Gauss this is degree 3, comfortably above the degree 2 required.

New module, say `lssem2d/quad.py`:

* `gauss_nodes(Q), gauss_weights(Q)`
* `interp_matrix(nodes, qpts)`   -- (Q, N+1), basis evaluated at quadrature pts
* `deriv_matrix(nodes, qpts)`    -- (Q, N+1), basis derivatives at quadrature pts

Then, in order of how much they assume collocation:

1. **`curvi.attach` / `collocation_metrics`** -- evaluate `rx, ry, sx, sy, J` at
   the QUADRATURE points, not the nodes.  `wq` becomes `J_q * wg_i * wg_j`.
   The name `collocation_metrics` stops being accurate and should change.
2. **`operators.py` / `lssem.apply_L`** -- currently multiplies nodal values
   pointwise by `wq`.  Must become: interpolate nodal -> quadrature, form the
   residual there, multiply by `wq`, and for `apply_LT` project back with the
   transpose.  This is the bulk of the work.
3. **`solver.compute_jacobi`** -- the diagonal of A is no longer a pointwise
   product; it becomes a sum over quadrature points of squared basis
   derivatives.  The existing generalisation to curvilinear meshes
   (`Px = rx*phix + sx*phiy`) is the right shape to extend.
4. **`precond.py` / `scratch/vertex_schwarz2d.py`** -- `element_blocks` builds
   the local operator; it inherits whatever (2) does.  NOTE the static
   condensation splits interior from edge dofs via `loc_int[1:N, 1:N]`, which
   at N = 1 is **empty** -- every dof is an edge dof.  The condensation then
   degenerates to a dense patch solve.  That is correct but pointless; add an
   early return so Q1 does not pay for machinery that does nothing.
5. **`bc.py`, `obc.py`** -- the Dong outflow integrates along an edge with its
   own arc weights (`_obc_ws`).  Edge quadrature needs the same treatment or
   the boundary term is inconsistent with the interior.
6. **`assembly.py`, `io.py`, `mesh.py`** -- `compute_global_indices` hashes
   node coordinates and is unaffected.  Field I/O is unaffected.

Everything above N = 1 must be **bit-identical** when `Q = N+1` and the Gauss
points are replaced by the GLL set -- add that as a regression test before
touching anything, the way the curvilinear port did.

### A2. Reduced integration as a measured fallback

If A0 finds no null space, keep collocation and simply document Q1 as
reduced-integration.  Cheap, and standard practice in FEM -- but it must be
*measured*, not assumed, and the convergence rate reported honestly.

## 4. Part B -- trilinear (3D Q1)

**This is a much larger project than Part A, and the reason is structural.**

The existing 3D code is **not** a 3D element code.  `lssem3d/operator.py` is a
per-Fourier-mode VVP operator: z is spectral (`ddz(uh, kz) = 1j*kz*uh`), the
modes decouple, and each mode is a 2D problem in (x, y) with 7 unknowns and 8
residual rows.  Convection is explicit (RKW3), which is what keeps the modes
decoupled and makes each stage Stokes-like.

So "trilinear" cannot be reached by setting N = 1 there.  It requires replacing
the Fourier direction with elements, which means:

1. **A 3D mesh**: hex elements, connectivity, and nine metric terms
   (`rx, ry, rz, sx, ..., tz`) instead of four.
2. **The VVP system without Fourier**: the same 8 rows, but `i k` becomes a
   real `d/dz` operator, so the modes couple and the problem is no longer a
   stack of 2D solves.  The per-mode structure that `fastdiag.py` and the
   preconditioners exploit disappears.
3. **Implicit convection, or not**: if convection stays explicit the stage is
   still Stokes and the balanced weighting is not needed (see paper 2.1 and
   Section 10); if it goes implicit, everything in the 2D weighting analysis
   applies and must be re-derived in 3D.
4. **Preconditioning**: a 3D vertex patch is 8 elements, and static
   condensation splits interior / face / edge / vertex.  At Q1 there are no
   interior or face dofs at all, so the patch solve is dense over its 27 nodes
   per field.  `fastdiag.py`'s fast-diagonalisation depends on the tensor
   structure and per-mode decoupling; it will not carry over unchanged.
5. **Cost**: Q1 in 3D needs many more elements for the same accuracy, and the
   FOSLS system has 7 unknowns per node.

**Recommendation: do not start Part B until Part A is finished and measured.**
Part A answers, in 2D and cheaply, the question that decides whether Part B is
worth attempting at all -- namely whether this formulation is usable at p = 1.
If Q1 in 2D turns out to enforce the divergence constraint too weakly to be
useful, trilinear in 3D will be worse, and the eight-fold cost will buy nothing.

## 5. Validation ladder

Each rung must pass before the next is attempted.

| rung | test | pass criterion |
|---|---|---|
| V0 | regression: N >= 2 with Q = N+1 Gauss-replaced-by-GLL | **bit-identical** to the current code |
| V1 | exact-solution convergence, N = 1, 2, 4 | Q1 shows the expected algebraic rate; no stagnation |
| V2 | spectrum of `L^T L` at N = 1 on a small mesh | no spurious near-zero modes beyond the known pressure constant |
| V3 | pointwise `max\|div u\|` vs N | reported, not judged -- this is expected to be poor at Q1 |
| V4 | lid-driven cavity Re = 1000 vs Ghia at matched dof | comparable error to N = 8 at the same dof count, or a documented reason why not |
| V5 | cylinder Re = 100 at matched dof | St within the scatter of the N sweep |

V4 and V5 at **matched degrees of freedom** rather than matched element count
is the only fair comparison, and it is the one that answers whether low order
is ever the right choice in this formulation.

## 6. Expected outcome, stated in advance

Q1 will converge at the algebraic rate and will enforce the divergence
constraint far more weakly than N = 8 -- probably by two to three orders in
pointwise `max|div u|` at matched dof.  If so, the useful product of this work
is not a Q1 production capability but a **measured statement of how the FOSLS
divergence advantage scales with p**, which is a result the paper does not
currently have and which bounds the claims in DIVERGENCE_CONSEQUENCES.md.

Recording that here so that if it happens it is a finding rather than a
disappointment.
