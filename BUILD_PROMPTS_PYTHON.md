# Build Prompts — 2D LSSEM Incompressible Solver in Python / NumPy

A staged prompt set for driving a coding agent to build a velocity–vorticity–pressure
least-squares spectral element solver, vectorised with NumPy.

**Reference specification:** `DOCUMENTATION.md` (equations, algorithm, transpose
construction). Every prompt below assumes the agent has read it.

Use sem_2d_oo.py as the reference implementation.

---

## How to use this set

Run the prompts **in order**. Each has an **acceptance gate** that must pass before moving
on. Do not let the agent proceed past a failing gate — in this method the failure modes are
silent (a wrong adjoint still converges, to the wrong answer), so the gates are the only
protection.

Paste the **Standing Context** block at the top of each session, then the individual
prompt.

---

## Standing Context (prepend to every prompt)

```
You are building a 2D velocity–vorticity–pressure (VVP) least-squares spectral element
method (LSSEM) solver for incompressible Navier–Stokes, in Python with NumPy.

The specification is in `pmg_dtdiv/DOCUMENTATION.md`. Read it before writing code.

Non-negotiable design rules:
1. NO Python loops over elements or over quadrature nodes in any operator applied inside
   the solver. All element-local work is vectorised across a leading element axis.
2. Arrays are float64 throughout. Shapes are documented in every docstring.
3. Every function that applies an operator must have a matching test that verifies it
   against an independent computation (finite differences, an explicit dense matrix built
   for a tiny case, or an adjoint dot-product identity).
4. No feature is "done" until its acceptance gate passes as an automated test.
5. Prefer clarity over cleverness in the first implementation; optimise only in Stage 9,
   and only against a benchmark you have already recorded.

Array layout convention (use consistently):
    field arrays:   U[e, i, j, k]   e=element, i=x-node, j=y-node, k=field (0..3)
    fields:         k = 0:u, 1:v, 2:p, 3:omega
    1-D operators:  D[i, n]  (differentiation), w[i] (quadrature weights)
```

---

## Stage 1 — LGL basis

```
Implement the Legendre–Gauss–Lobatto basis in `lgl.py`.

Provide:
    lgl_nodes(N)  -> xi[N+1]       LGL nodes on [-1,1], ascending
    lgl_weights(N) -> w[N+1]       LGL quadrature weights
    diff_matrix(N) -> D[N+1,N+1]   D[i,n] = l_n'(xi_i)

Use the standard Newton iteration on the Legendre polynomial derivative for the nodes,
and the closed forms
    w_i = 2 / (N(N+1) [P_N(xi_i)]^2)
    D[i,n] = P_N(xi_i) / (P_N(xi_n) (xi_i - xi_n))     i != n
    D[i,i] = 0 for interior;  D[0,0] = -N(N+1)/4;  D[N,N] = +N(N+1)/4

ACCEPTANCE GATE (write as tests):
  a) Nodes are symmetric about 0 and include ±1 exactly.
  b) Quadrature is exact for polynomials up to degree 2N-1:
     integrate x^m on [-1,1] for m=0..2N-1, compare to the analytic value, tol 1e-12.
  c) D differentiates exactly: for f = x^m (m <= N), D @ f equals m*x^(m-1) to 1e-10.
  d) Row sums of D are zero to 1e-12 (differentiates a constant to zero).
  e) Test at N = 2, 4, 8, 16.
```

---

## Stage 2 — Mesh and DOF layout

```
Implement `mesh.py` for axis-aligned rectangular spectral elements.

class Mesh:
    nelem, N, nterm = N+1
    x0[e], y0[e], hx[e], hy[e]          element origin and size
    xnod[e, i], ynod[e, j]              physical node coordinates
    neighbour[e, 4]                     W,E,S,N neighbour element (-1 = boundary)
    bc[e, 4]                            edge BC codes (see DOCUMENTATION §8.2)

Derived, precomputed once:
    jac[e]  = hx*hy/4
    facx[e] = 2/hx        facy[e] = 2/hy
    wq[e, i, j] = jac[e] * w[i] * w[j]        quadrature weight, shape (nelem, n, n)

Also provide:
    build_channel(...)      simple rectangular grid, for tests
    build_bfs(...)          backward-facing step with an upstream inlet channel
    global_index(e,i,j)     -> unique global node id, for gather/scatter (Stage 5)

ACCEPTANCE GATE:
  a) Node coordinates match a hand-computed 2-element mesh.
  b) sum(wq) over all elements equals the total domain area to 1e-12.
  c) Neighbour/BC arrays are self-consistent: if neighbour[e,E]=f then neighbour[f,W]=e.
  d) A plotting helper renders elements + collocation points (visual check only).
```

---

## Stage 3 — Tensor-product derivatives (the performance core)

```
Implement `operators.py` with element-batched derivative operators.

For U with shape (nelem, n, n):
    d/dx :  dUdx[e,i,j] = facx[e] * sum_m D[i,m] U[e,m,j]
    d/dy :  dUdy[e,i,j] = facy[e] * sum_m D[j,m] U[e,i,m]

and their ADJOINTS, which are the same contractions with D's index pair swapped:
    Dx^T :  out[e,i,j] = facx[e] * sum_m D[m,i] S[e,m,j]
    Dy^T :  out[e,i,j] = facy[e] * sum_m D[m,j] S[e,i,m]

Implement each as a single np.einsum with optimize=True, e.g.
    dUdx = facx[:,None,None] * np.einsum('im,emj->eij', D, U, optimize=True)
    DxT  = facx[:,None,None] * np.einsum('mi,emj->eij', D, S, optimize=True)

Benchmark einsum against the reshape+matmul alternative
    (np.tensordot / D @ U.reshape(...)) and keep the faster one, but ONLY after the
    correctness gate passes. Record timings in a comment.

ACCEPTANCE GATE — this is the most important gate in the project:
  a) Differentiate a polynomial exactly (compare to analytic derivative, 1e-10).
  b) ADJOINT (DOT-PRODUCT) TEST. For random U, S of shape (nelem,n,n):
         <Dx U, S>_w  ==  <U, Dx^T S>_w      to 1e-12 relative
     where <A,B>_w = sum(wq * A * B).
     NOTE: the weight must be handled consistently — decide now whether wq is folded
     into the residual (as the Fortran does) or applied in the inner product, and
     document the choice. The test must pass for BOTH Dx and Dy, at N=4 and N=8, on a
     mesh with NON-UNIFORM element sizes (so facx != facy and they differ per element).
  c) If (b) fails, do not proceed. A wrong adjoint produces a solver that converges
     smoothly to the wrong answer, with no other symptom.
```

---

## Stage 4 — The VVP operator L (pass 1)

```
Implement `lssem.py::apply_L(state, U, fu, fv)` following DOCUMENTATION §7.1.

Given the current linearisation velocities (fu, fv) and an input field U (u,v,p,om),
return the four weighted equation residuals su[e,i,j,k]:

    su_1 = [ fac1/dt * u + fu*u_x + fv*u_y + u*dfu_dx + v*dfu_dy + p_x + nu*om_y ] * wq
    su_2 = [ fac1/dt * v + fu*v_x + fv*v_y + u*dfv_dx + v*dfv_dy + p_y - nu*om_x ] * wq
    su_3 = [ u_x + v_y ] * wq
    su_4 = [ om + u_y - v_x ] * wq

Use the DIVIDED form: dt appears ONLY on the fac1 term. Every spatial term is dt-free.
(This is the pmg_dtdiv convention; see DOCUMENTATION §3.2 for why.)

Precompute dfu_dx, dfu_dy, dfv_dx, dfv_dy once per Newton sub-iteration and cache them —
they do not change while (fu,fv) are frozen.

ACCEPTANCE GATE:
  a) Method of manufactured solutions: choose a smooth (u,v,p,om) satisfying no
     particular equation, compute su analytically by symbolic differentiation
     (sympy) at the LGL nodes, compare to apply_L. Agreement to 1e-10.
  b) With (u,v,p,om) an exact solution of the steady Stokes problem and fac1=0,
     ||su|| is at truncation-error level and decreases spectrally with N.
```

---

## Stage 5 — The transpose L^T (pass 2)

```
Implement `lssem.py::apply_LT(state, su, fu, fv)` following DOCUMENTATION §7.2.

Return c[e,i,j,k] = sum_i (dR_i/dU_k)^T su_i, assembled as:

  c_1 = (fac1/dt)*su1 + dfu_dx*su1 + dfv_dx*su2
        + Dx^T(su3) + Dx^T(fu*su1)
        + Dy^T(su4) + Dy^T(fv*su1)

  c_2 = (fac1/dt)*su2 + dfu_dy*su1 + dfv_dy*su2
        - Dx^T(su4) + Dx^T(fu*su2)
        + Dy^T(su3) + Dy^T(fv*su2)

  c_3 = Dx^T(su1) + Dy^T(su2)                  # NO diagonal term: p enters only via grad

  c_4 = su4 - nu*Dx^T(su2) + nu*Dy^T(su1)

CRITICAL — the three term shapes and their adjoints (DOCUMENTATION §7.2):
  * scalar multiple of identity  ->  adjoint is itself, applied at the TARGET node
  * nodal coefficient (diagonal) ->  adjoint is itself, applied at the TARGET node
  * coefficient x derivative     ->  (diag(a) Dx)^T = Dx^T diag(a):
                                     multiply su by the coefficient FIRST, at the
                                     SOURCE node, THEN apply Dx^T.
  Getting the last one wrong (applying the coefficient after the contraction) is the
  classic bug. It is invisible except through the gate below.

ACCEPTANCE GATE — mandatory, do not proceed without it:
  a) FULL-OPERATOR ADJOINT TEST. For random U, S:
         <apply_L(U), S>  ==  <U, apply_LT(S)>     to 1e-12 relative
     on a mesh with non-uniform elements, with fu,fv random and non-zero, at N=4 and N=8.
  b) SYMMETRY TEST. Define A(U) = apply_LT(apply_L(U)). For random U, V:
         <A(U), V> == <U, A(V)>                    to 1e-12 relative
  c) POSITIVITY. <A(U), U> >= 0 for 100 random U.
  d) DENSE CROSS-CHECK on a tiny case (1 element, N=2): build L explicitly column by
     column by applying apply_L to unit vectors, form L^T numerically, and verify
     apply_LT reproduces it to 1e-12.
```

---

## Stage 6 — Direct stiffness (gather–scatter)

```
Implement `assembly.py` for C0 continuity across element interfaces.

The element-local representation duplicates shared-face nodes. Assembly sums the
contributions and writes the sum back to every copy — the Q Q^T gather-scatter.

Efficient NumPy approach (no loops over faces):
  * Precompute once, in Mesh: gidx[e,i,j] -> global node id  (shape (nelem,n,n))
  * Gather-scatter for a field C[e,i,j,k]:
        flat = C.reshape(-1, 4)
        acc  = np.zeros((n_global, 4))
        np.add.at(acc, gidx.ravel(), flat)        # or np.bincount per component
        C_assembled = acc[gidx]                    # scatter back
  * np.add.at is slow; prefer np.bincount(gidx.ravel(), weights=flat[:,k],
    minlength=n_global) per component, or a precomputed scipy.sparse Q matrix
    (Q^T then Q). Benchmark; document the choice.

ACCEPTANCE GATE:
  a) Assembling a field that is already continuous multiplies interface values by their
     multiplicity (2 on a face, 4 at a cross point) — verify the multiplicity map.
  b) Assembly is self-adjoint: <assemble(A), B> == <A, assemble(B)>.
  c) A constant field remains constant after assemble-then-divide-by-multiplicity.
```

---

## Stage 7 — Boundary conditions

```
Implement `bc.py` with the mask projector described in DOCUMENTATION §8.2.

mask[e,i,j,k] in {0,1};  0 = the DOF is Dirichlet-constrained.

A Dirichlet condition is THREE coordinated actions — implement all three or it is wrong:
  1. write the prescribed value into the solution array each sub-iteration
  2. project the residual:        res *= mask
  3. project the solver update:   U += mask * delta

so the system solved is  (M L^T W L M) dU = M L^T W (f - L U).
This is symmetric elimination; do NOT use the "zero the row, put 1 on the diagonal"
trick, which destroys the symmetry the Krylov solver depends on.

Edge codes: 0 interior, 1 no-slip, 2 lid, 3 inlet profile, 4 outlet (p=0),
5 symmetry (v=0, omega=0), and a free-outflow option that constrains NOTHING.

Pressure pin: because c_3 has no diagonal term, the operator has a constant-pressure
null mode when no Dirichlet pressure exists. Provide pin_node=None to disable it (a
Krylov method from a zero initial guess stays in the range space and tolerates the
singular system) and pin_node=(e,i,j) to fix one DOF.

ACCEPTANCE GATE:
  a) With a Dirichlet BC applied, the constrained DOFs are bit-identical to the
     prescribed values after 50 solver iterations.
  b) The masked operator is still symmetric: <MA M U, V> == <U, M A M V>.
  c) Lid-driven cavity at Re=100 runs without the pin (null mode tolerated) and with it,
     and the two velocity fields agree to 1e-8 after removing the pressure mean.
```

---

## Stage 8 — Time stepping, Newton, and the linear solver

```
Implement `solver.py`.

BDF: fac1,fac2,fac3 = (1,-1,0) for the first step, (1.5,-2,0.5) thereafter.

Time loop (DOCUMENTATION §9):
    for step:
        for sub in range(nsub):
            apply Dirichlet values
            freeze fu,fv = current u,v ; cache their gradients
            r    = apply_LT(apply_L(U) - f_known)     # nonlinear residual
            r   *= mask ; r = assemble(r)
            res0 = norm(r) ; break if res0 <= tol
            cgstol = max(cgsfac*res0, tol)            # INEXACT NEWTON - see below
            dU = bicgstab(A, r, M_inv, tol=cgstol, maxit=nitcgs)
            U += mask * dU
        shift history

Matrix-free BiCGSTAB with a Jacobi preconditioner:
    A(v)     = assemble(mask * apply_LT(apply_L(mask*v)))
    M_inv(v) = v / diag,  diag from the column norms sum_i (dR_i/dU_k)^2  (DOCUMENTATION §7.3)

INEXACT NEWTON (cgsfac): ask the linear solve only for a cgsfac-fold residual reduction,
not for tol. cgsfac=0.1 is a good default. This is not an optimisation — with tight
inner solves a cold start from rest at Re~400 DIVERGES within a handful of steps, because
Newton is being driven hard from a bad initial guess. Loose early solves keep the
correction bounded.

ACCEPTANCE GATE:
  a) BiCGSTAB solves A x = b for random b to 1e-10 on a small mesh; compare against
     scipy.sparse.linalg.bicgstab wrapping the same LinearOperator.
  b) The Jacobi diagonal matches the true diagonal of A extracted column-by-column on a
     tiny case, to 1e-12.
  c) A cold start at Re=389 with nsub=3, cgsfac=0.1 runs 1000 steps without divergence.
```

---

## Stage 9 — Verification suite

```
Implement `tests/verification.py` — these are physics gates, not unit tests.

1. MANUFACTURED SOLUTION (spectral convergence)
   Pick a smooth exact (u,v,p,om), derive the forcing symbolically, solve, and measure
   the L2 error vs N = 4,6,8,10,12 on a fixed mesh.
   GATE: error decreases exponentially in N (straight line on a log-linear plot);
   fit the slope and assert it is steeper than algebraic.

2. KOVASZNAY FLOW (Re=40) — analytic steady Navier-Stokes solution.
   GATE: L2 velocity error < 1e-6 at N=10 on a 4x4 mesh.

3. LID-DRIVEN CAVITY (Re=100, 400, 1000)
   GATE: centreline velocity profiles match Ghia et al. (1982) to within 2%.

4. BACKWARD-FACING STEP (Re=389, Armaly geometry, expansion ratio 2)
   GATE: reattachment length 8.0 +/- 0.3 inlet heights; mass flux Q(x) constant to 0.5%.

5. CONSERVATION
   GATE: divergence-free residual ||div u||_2 at machine-precision-limited level, and
   Q(x) = integral u dy independent of x to 0.1%.
```

---

## Stage 10 — Optimisation

```
Only now, with all gates green and a recorded baseline timing, optimise.

Profile first (cProfile + line_profiler). Expected hot spots, in order:
  1. the einsum contractions in apply_L / apply_LT
  2. gather-scatter in assemble
  3. array temporaries in the residual assembly

Techniques, in the order worth trying:
  a) Precompute einsum paths with np.einsum_path and reuse them.
  b) Replace einsum with reshape + a single BLAS gemm:
         U.transpose(0,2,1).reshape(-1, n) @ D.T     (measure; often 2-4x faster)
  c) Fuse the four field contractions into one gemm by stacking fields on the
     contracted axis.
  d) Preallocate all work arrays in a scratch object; eliminate temporaries with
     out= arguments and in-place ops.
  e) Replace np.add.at with bincount or a precomputed scipy.sparse Q.
  f) If still short: numba @njit(parallel=True) on the residual assembly ONLY, keeping
     the NumPy version as the reference implementation for the gates.

RULE: after every optimisation, re-run Stage 5's adjoint test and Stage 9's Kovasznay
gate. An optimisation that breaks the adjoint is worse than no optimisation, because the
solver will still converge.

GATE: >= 5x faster than the Stage 8 baseline on a 72-element N=10 mesh, with every
earlier gate still passing.
```

---

## Appendix — Failure modes to guard against

These come from debugging the Fortran original. Each cost real time; each is silent.

| trap | symptom | guard |
|---|---|---|
| **Wrong adjoint** (coefficient applied at target instead of source node) | solver converges smoothly to a wrong answer | Stage 5 dot-product test — the only reliable detector |
| **Pass 1 and pass 2 edited out of lockstep** | preconditioner mismatch, or a wrong operator; convergence looks healthy | any operator change re-runs the adjoint test |
| **False convergence** | a bit-identical residual for thousands of steps while the flow still evolves | require BOTH a residual lock AND max\|ΔU\| < 1% of range over several hundred further time units |
| **Judging solutions by max\|Δ\|/range** | 425% "disagreement" between solutions that agree to 2% | report rms, plus a physical scalar (reattachment length); corner singularities dominate pointwise maxima |
| **Over-solving the linear system** | divergence to NaN at high iteration caps on near-singular systems | cap iterations; use the inexact-Newton forcing term |
| **Tight inner solves on a cold start** | Newton diverges within ~4 steps | cgsfac ~ 0.1 through the transient |
| **Testing on a degenerate case** | two correct codes "disagree" by 80% | validate on a well-posed case (channel, Kovasznay, long-domain BFS), never on a truncated outflow |

---

## Suggested file layout

```
lssem2d/
  lgl.py              Stage 1   basis, nodes, weights, D
  mesh.py             Stage 2   geometry, connectivity, quadrature weights
  operators.py        Stage 3   tensor-product derivatives and adjoints
  lssem.py            Stage 4-5 apply_L, apply_LT, diagonal
  assembly.py         Stage 6   gather-scatter
  bc.py               Stage 7   mask projector, BC codes, pressure pin
  solver.py           Stage 8   BDF, Newton, BiCGSTAB, Jacobi
  cases/              meshes and namelists for the verification cases
  tests/
    test_lgl.py       test_operators.py    test_adjoint.py
    test_assembly.py  test_bc.py           verification.py
```
