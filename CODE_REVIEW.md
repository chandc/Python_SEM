# Code Review — `lssem2d` Python/NumPy port

**Reviewed:** `/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/lssem2d`
**Against:** `F90_SEM/pmg_dtdiv` (divided-`dt` VVP-LSSEM) and its `DOCUMENTATION.md`
**Method:** source reading plus executable probes; every quantitative claim below was
measured, not inferred.

---

## Executive summary

The **discretisation is a correct and well-tested port**. The operator, its transpose,
and the inter-element assembly — the three things that are hardest to get right and
whose failure modes are silent — are all correct, and the existing test suite proves it
with the right tests.

The **physics configuration is not a port**. The solver implements a `p = 0` pressure
outlet, not the free outflow that the Fortran study is about, so it currently cannot
reproduce any of the reference results.

Test suite: **30 pass, 2 fail** (both failures are stale/environmental, not real bugs —
see §5, issue 9).

---

## 1. What is correct

### 1.1 The transpose

`DxT` / `DyT` are genuine adjoints:

```python
def DxT(S, D, facx):  out = np.matmul(D.T, S)     # (D^T S)[e,i,j] = sum_m D[m,i] S[e,m,j]
def DyT(S, D, facy):  out = np.matmul(S, D)       # (S D)[e,i,j]   = sum_m S[e,i,m] D[m,j]
```

and every term of `apply_LT` matches the adjoint of `apply_L`. This is verified by the
project's own `tests/test_adjoint.py`, which is exactly the right set of tests:

* dot-product identity `<L U, S> == <U, L^T S>` on a **non-uniform** mesh with non-zero
  `fu, fv`, at N=4 and N=8
* symmetry and positivity of `A = L^T L`
* a dense column-by-column cross-check at N=2

All pass. Given that a wrong adjoint yields a solver that converges smoothly to the
wrong answer, this is the single most important thing in the project and it is done.

### 1.2 Inter-element communication

`Q` / `QT` are named inversely to convention (`Q` gathers local→global, `QT` scatters
back), but the composition in `gather_scatter` is the correct sum-and-scatter, and —
critically — it is **inside the operator**:

```python
def apply_A(state, dU, fu, fv, pin_p=False):
    dU_m = apply_mask(...)          # 1. project
    su   = apply_L(...)             # 2. L
    c    = apply_LT(...)            # 3. L^T
    c_gs = gather_scatter(...)      # 4. Q^T Q   <-- on EVERY mat-vec
    return apply_mask(...)          # 5. project
```

so the incomplete residuals are exchanged with nearest neighbours on every
matrix–vector product, not once per step. The preconditioner diagonal is assembled too.
This is the trap most ports fall into and it was avoided.

### 1.3 Equivalences that are non-obvious but correct

**True nonlinear convection.** The Fortran obtains `u·∇u` from the linearised form using
`fu = U` plus a deferred `−fu·∇fu` term. The Python instead passes `fu = U/2`:

$$
\tfrac{u}{2}u_x + \tfrac{v}{2}u_y + u\,\partial_x\!\left(\tfrac{u}{2}\right) + v\,\partial_y\!\left(\tfrac{u}{2}\right) = u u_x + v u_y
$$

Different route, same result.

**Newton right-hand side.** Both form $J^{T}R$ with $J$ linearised at `fu = U` and $R$
the *true* residual. The Python does this by explicitly switching the cached
linearisation between the two calls — subtle, and correct.

**BDF2.** `fac1 = 1.5`, `alpha = [2.0, -0.5]` reproduces
$(1.5u^{n+1} - 2u^{n} + 0.5u^{n-1})/\Delta t$, matching the Fortran.

**Preconditioner.** Computed by applying `L` to unit vectors rather than analytically as
`dge` does, but it is the same quantity, $\sum_m W (L_{mk})^2$ — the column norms.

---

## 2. Fidelity to the Fortran

### 2.1 Faithful

| item | status |
|---|---|
| VVP first-order system, sign conventions | ✅ |
| **Divided** `dt` form (matches `pmg_dtdiv`, not `pmg_clean`) | ✅ |
| BDF1 / BDF2 coefficients | ✅ |
| Operator `L`, term by term | ✅ |
| Transpose `L^T`, term by term | ✅ |
| Tensor-product derivatives and adjoints | ✅ |
| Direct stiffness on every mat-vec and on the diagonal | ✅ |
| Preconditioner = column norms | ✅ (different method, same quantity) |
| Multiplicity-weighted norm (interface double-counting) | ✅ same as Fortran |
| Mask as three coordinated actions (value / residual / update) | ✅ |

### 2.2 NOT faithful

**(a) There is no free outflow — this is the significant gap.**

Python, `bc.py`:

```python
elif bc_E == 4:
    U_masked[e, -1, :, 2] = 0.0        # p = 0
```

Fortran, `SEM_08_bfs_freeout.f90:334`:

```fortran
else if( ibce(ne).eq.4 ) then
!   FREE OUTFLOW: leave u,v,p,omega all free at the outlet (natural LSSEM
!   condition). No edge-wide pressure mask. Single pin applied after this loop.
endif
```

The Python therefore reproduces `SEM_08_bfs.f90` — the **p = 0 pressure outlet** — not
the free-outflow driver that produced every result in the reference study. It solves a
different boundary-value problem at the outlet.

**(b) The pressure pin is a stub.** `pin_p=True` fixes `U[0,0,0,2]` — element 0, node
(0,0), the *inlet* corner. The Fortran has `npin_e` / `npin_j` with auto-detection,
arbitrary placement, and `-1` to disable entirely. Since pin location was measured to
change the truncated-domain solution by 81 % in `max|Δu|`, a hardcoded pin is not a
substitute.

**(c) Missing solver hardening present in the Fortran:**

* no true-residual safeguard in BiCGSTAB
* no `cgsfac` inexact-Newton forcing term
* no p-multigrid preconditioner (Jacobi only)
* no traction / weak outflow (`trac_mode` 1 / 2 / 3)

**(d) Different Newton exit criterion.** Fortran tests `‖L^T R‖ ≤ tol` *before* solving
and can exit without a linear solve; Python tests `max|dU| < tol` *after* updating, so it
always performs at least one solve per step. The Python criterion is arguably better — a
residual-only test is what produced the false-convergence trap documented in
`F90_SEM/pmg_clean/OUTFLOW_BC_STUDY.md` §10.5 — but it is not the same code.

### 2.3 Practical consequence

On a lid-driven cavity or Kovasznay case the two codes should agree closely. On any BFS
free-outflow case they will not, and the disagreement would say nothing about either
code. **Smallest change to close the gap:** add a BC code that masks nothing at the
outlet, and make the pin location settable. Roughly thirty lines.

---

## 3. Measured findings

### 3.1 The operator is not self-adjoint in the inner product being used

```
plain sum inner product      <Au,v> vs <u,Av> :  relative asymmetry = 4.9e-02
1/multiplicity weighted                        :  relative asymmetry = 2.6e-16
```

After assembly, shared nodes hold identical values, so `np.sum(A*B)` counts them ×2 on
faces and ×4 at corners. Under that weighting $Q^{T}Q\,L^{T}L$ is not self-adjoint;
weight by `1/multiplicity` and it is symmetric to machine precision.

**Consequence — the most valuable available improvement.** The code runs **BiCGSTAB
(2 mat-vecs/iteration) on an operator that is SPD**, where **CG (1 mat-vec/iteration)**
would apply. That is ~2× throughput plus the better robustness of CG, and SPD-ness is
the whole selling point of the least-squares formulation.

*Note:* the Fortran shares this property and also uses BiCGSTAB, so this is not an
infidelity — it is an opportunity in both codes.

### 3.2 No true-residual safeguard — measured drift

Consistent right-hand side, `build_channel(2.0, 1.0, 4, 3, N=6)`, `nu = 1/389`:

| requested `tol` | iterations | **true** relative residual |
|---|---|---|
| 1e-4 | 168 | 7.9e-05 |
| 1e-6 | 361 | 4.7e-07 |
| 1e-8 | 367 | 7.1e-09 |
| **1e-10** | 433 | **1.36e-09**  ← 13× worse than requested |

Convergence is tested on the recurrence residual only. Fine at loose tolerances; it lies
at tight ones. The Fortran recomputes a true residual (`res_true`) as a safeguard.

### 3.3 `compute_jacobi` cost

It builds the diagonal by applying `L` to $4(N+1)^2$ unit vectors, **recomputed on every
`cg_solve` call**.

At N=10, 72 elements (34 848 local DOFs):

```
apply_L          0.413 ms
apply_LT         0.936 ms  (including L)
gather_scatter   0.037 ms
apply_mask       0.054 ms
apply_A total    1.094 ms
compute_jacobi   ~200 ms   =  484 apply_L  ~=  183 matrix-vector equivalents
```

≈30 % overhead on a 300-iteration solve, scaling as $N^{2}$. Correct and elegant, but it
belongs outside `cg_solve` — cached per sub-iteration, or computed analytically as `dge`
does.

### 3.4 Shared preallocated buffers are a live hazard

```
A is B (same buffer)?          True
did the FIRST result mutate?   True
```

`apply_L` returns `state.su` and `apply_LT` returns `state.c`. Two successive calls
return the *same array*, and the earlier result is silently overwritten. Current call
sites happen to be safe because `gather_scatter` and `apply_mask` allocate; the test
suite already works around it with `.copy()` in `test_symmetry_and_positivity`, which
suggests the author hit this once.

**Fix:** return copies, or make the output buffer an explicit argument so aliasing is
visible at the call site.

---

## 4. Mesh

`compute_global_indices` matches nodes by hashing rounded physical coordinates:

```python
key = (round(x, 10), round(y, 10))
```

This works on the current meshes — measured multiplicities are `{1,2,4}` for a channel
and `{1,2,3,4}` for the BFS (multiplicity 3 is legitimate at the re-entrant step corner
where three elements meet).

**But it is unverified.** There is no assertion that the unique-node count matches the
expected value for a given mesh. If two nominally-shared nodes ever hash differently, the
elements silently decouple and the solver converges to a field discontinuous across that
interface. And because multiplicity 3 legitimately occurs, you cannot even eyeball the
multiplicity histogram for anomalies.

**Fix:** each mesh builder should assert its expected unique-node count, and a test
should verify continuity of a solved field across interfaces.

---

## 5. Issue list, ranked

| # | issue | severity | evidence |
|---|---|---|---|
| 1 | **No free outflow** — East code 4 imposes `p=0`; cannot reproduce the reference study | **blocking for the intended purpose** | §2.2(a) |
| 2 | Operator not self-adjoint in the inner product used; CG unavailable, BiCGSTAB costs 2× | **high (perf + robustness)** | 4.9e-2 vs 2.6e-16, §3.1 |
| 3 | No true-residual safeguard in BiCGSTAB | **high** | 13× drift at `tol=1e-10`, §3.2 |
| 4 | `compute_jacobi` = 183 mat-vec equivalents, recomputed every solve | **medium (perf)** | §3.3 |
| 5 | Shared buffers alias between calls | **medium (latent)** | §3.4 |
| 6 | Pressure pin hardcoded to the inlet corner, not selectable | medium | §2.2(b) |
| 7 | No `cgsfac` inexact-Newton forcing | medium | §2.2(c) |
| 8 | Node matching unverified | medium (latent) | §4 |
| 9 | Test suite not green — `test_wq_sum` expects 14, mesh area is 42 (stale test: its comment assumes `L_out=6`, code uses 20); `test_plot_mesh` fails on a matplotlib incompatibility | medium (process) | pytest output |
| 10 | `apply_mask` allocates a full copy per call, twice per mat-vec | low (5 % of mat-vec) | §3.3 |
| 11 | `apply_L` takes `fu,fv` as arguments but reads gradients from `state` — the two can silently disagree | low (latent) | source |
| 12 | `update_linearisation` allocates four arrays per call | low | source |

---

## 6. Recommended order of work

1. **Add a free-outflow BC code** (mask nothing at the outlet) and make the pin location
   settable — without these the port cannot be compared against the Fortran on the cases
   that matter.
2. **Fix the test suite** so it is green; a permanently-failing suite means real
   regressions go unnoticed.
3. **Add the true-residual safeguard** to BiCGSTAB.
4. **Hoist `compute_jacobi`** out of `cg_solve`; cache per sub-iteration.
5. **Switch to CG with the `1/multiplicity` inner product** — verify with the symmetry
   probe in §3.1 first, then measure the iteration-count and wall-clock change.
6. Return copies from `apply_L`/`apply_LT`, or pass output buffers explicitly.
7. Add `cgsfac`; assert expected node counts in the mesh builders; add an
   interface-continuity regression test.

After any change to the operator, **re-run `tests/test_adjoint.py`** — it is the only
reliable detector of the failure mode that matters.

---

## 7. Overall assessment

A competent port. The mathematically dangerous parts are correct *and tested*, which is
more than most ports of this kind achieve. What is missing is operational: solver
hardening, buffer safety, and — decisively — the boundary condition the reference study
is actually about.

Fix items 1–4 and this is a solid, usable solver.
