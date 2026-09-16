# Curvilinear elements for the 2D least-squares SEM — implementation plan

*Branch `curvilinear-2d`, opened 2026-09-16 from `eefae1a`.  Merge only when
every gate below is green and the Cartesian results are bit-for-bit unchanged.*

---

## 0. What is actually being changed

The 2D code is **affine and axis-aligned**.  `mesh.py` stores two scalars per
element,

```python
self.facx = 2.0 / self.hx        # dr/dx, constant per element
self.facy = 2.0 / self.hy        # ds/dy, constant per element
self.wq[e, i, j] = self.jac[e] * w[i] * w[j]
```

and every derivative is a tensor contraction scaled by one of them:

```python
def dUdx(U, D, facx):  return np.matmul(D, U) * facx[:, None, None]
```

Curvilinear geometry replaces each scalar with a **field over the element** and
introduces cross terms:

$$\frac{\partial f}{\partial x} = r_x\,\frac{\partial f}{\partial r}
 + s_x\,\frac{\partial f}{\partial s},\qquad
\frac{\partial f}{\partial y} = r_y\,\frac{\partial f}{\partial r}
 + s_y\,\frac{\partial f}{\partial s},$$

with $J = x_r y_s - x_s y_r$ and $r_x = y_s/J$, $r_y = -x_s/J$, $s_x = -y_r/J$,
$s_y = x_r/J$, all of them $(N+1)\times(N+1)$ arrays per element.  **Every
derivative acquires a second contraction**, which is the whole cost of the
change: one matmul becomes two, plus pointwise metric multiplies.

**Blast radius**, by call sites touching `facx`/`facy`:

| file | sites | what changes |
|---|---|---|
| `lssem2d/lssem.py` | 24 | the residual rows and their transposes — the functional itself |
| `lssem2d/kernels_numba.py` | 12 | the fused kernels, which must match the reference path |
| `lssem2d/operators.py` | 8 | `dUdx`, `dUdy`, `DxT`, `DyT` — the primitives |
| `lssem2d/obc.py` | 6 | outflow terms, which need true boundary normals |
| `lssem2d/mesh.py` | 5 | construction of the metric itself |
| `lssem2d/solver.py` | 2 | plumbing |

**What does not change:** the least-squares formulation, the weighting result of
the paper (it is algebraic and geometry-independent), the vertex-patch
preconditioner's *structure* (it probes the assembled operator, so it inherits
whatever the operator is), and the global `gidx` connectivity.

**What is at risk and must be watched:** any preconditioner that assumes
separability.  The FDM/fast-diagonalisation path is exactly separable only for
affine rectangles; on curvilinear elements it becomes a surrogate.  The patch
preconditioner is safe, the PMG transfers are reference-element operations and
are safe, and the paper's results rest on the patch method.

---

## 1. Design decisions, taken up front

**D1 — Metrics stored, not recomputed.**  Add to `Mesh`: `rx, ry, sx, sy, jac`
each `(nelem, n, n)`, replacing the scalars.  Keep `facx`/`facy` as *derived
properties* that raise on a curvilinear mesh, so every unconverted call site
fails loudly instead of silently using a wrong constant.

**D2 — Collocation metrics, computed by differentiating the mapping with the
same `D`.**  $x_r = D x$ etc. on the GLL grid.  This is what makes the discrete
metric identities hold and free-stream preservation exact (gate G0); computing
metrics analytically from the mapping does *not* guarantee it.

**D3 — Conservative (curl) form is not needed in 2D.**  The metric identity
$\partial_r(J r_x) + \partial_s(J s_x) = 0$ is satisfied by collocation metrics
in two dimensions with the same $D$.  (It is 3D where the cross-product form
becomes necessary; note it for the eventual 3D port and do not build it now.)

**D4 — Isoparametric mapping.**  The geometry is represented in the same
degree-$N$ GLL basis as the solution.  Curved boundaries are specified by a
callback that maps the reference boundary to the physical one.

**D5 — The affine path stays.**  `Mesh.curvilinear` is a flag; when false the
code takes the existing scalar path unchanged.  This keeps every result in the
paper reproducible on the branch and makes the rotation-invariance gate (G1) a
genuine comparison rather than a self-consistency check.

---

## 2. The gates

Each step ends at a gate with a **known answer**.  A step is not done until its
gate is green and recorded in this file with the measured numbers.  Gates are
cumulative: every later step re-runs all earlier ones.

### G0 — Geometry: metric identities and free-stream preservation

*Known answer: exactly zero, to round-off.*

On a deliberately distorted mesh (sinusoidal perturbation of interior nodes):

1. **Constant preserved:** $\nabla f = 0$ for $f \equiv 1$, to $10^{-14}$.
2. **Linear exact:** $\nabla f = (1,0)$ for $f = x$, and $(0,1)$ for $f = y$, to
   $10^{-13}$ — this is the discrete metric identity in its usable form.
3. **Divergence of a constant vector field is zero**, to $10^{-14}$.
4. **Area:** $\sum_e \sum_{ij} w_{ij} J_{ij}$ equals the analytic domain area to
   $10^{-13}$.

*Why first:* every later error is uninterpretable if the metrics are wrong, and
these four catch nearly every implementation mistake in the mapping.

### G1 — Affine rotation invariance

*Known answer: the existing Cartesian results, to round-off.*

Take the lid-driven cavity and the Kovasznay case, rotate the mesh by 30° and
the boundary data with it, and require the solution to match the unrotated run
**rotated back**, to $10^{-12}$.  A constant-Jacobian rotated element exercises
all four metric terms and every cross term while having an exactly known answer.

*This is the strongest gate in the plan.*  It cannot be passed by a
self-consistent but wrong metric, and it separates "curvilinear machinery works"
from "curvilinear machinery is merely stable".

### G2 — Spectral convergence on a curved mesh

*Known answer: a manufactured solution, and the convergence rate.*

Use the existing MMS machinery (`scratch/mms2d_temporal.py` builds the forcing)
on a sinusoidally deformed mesh.  Require:

* error falling **exponentially** in $N$ for $N = 4 \dots 12$ on a fixed mesh
  (the spectral signature; algebraic decay means the metrics are inconsistent);
* the same asymptotic rate as the Cartesian case at equal degrees of freedom,
  within a factor of ~2 in the constant.

### G3 — Taylor–Couette: an exact solution on a genuinely curved domain

*Known answer: analytic.*  Steady flow between concentric rotating cylinders,

$$u_\theta(r)=Ar+\frac{B}{r},\qquad
A=\frac{\Omega_2R_2^2-\Omega_1R_1^2}{R_2^2-R_1^2},\qquad
B=\frac{(\Omega_1-\Omega_2)R_1^2R_2^2}{R_2^2-R_1^2},$$

with $u_r = 0$ and $p$ from radial equilibrium.  An annular mesh is the natural
curvilinear test: no straight element edges anywhere, an exact solution, and a
pressure field that is not constant.

Require: $\lVert u-u_{\rm exact}\rVert_\infty$ at the discretisation level and
falling spectrally with $N$; the velocity field divergence-free pointwise to the
usual $10^{-5}$; and $\omega$ matching the analytic $2A$ in the core.

*This gate is the reason to do the work:* it is unreachable on the current code.

### G4 — Kovasznay on a deformed mesh

*Known answer: the existing `KOVASZNAY_VALIDATION.md` result.*

Re-run Kovasznay with interior nodes perturbed by 10% of the element size.  The
error must match the Cartesian run to within the discretisation error — a curved
mesh should cost accuracy, but not much, and not change the convergence order.

### G5 — The preconditioner survives

*Known answer: the existing iteration counts.*

The vertex-patch preconditioner on a curvilinear mesh must stay flat in $N$:
19–21 iterations on the cavity at $N = 5\dots20$ is the Cartesian benchmark
(`LOW_MEMORY_PATCH_SOLVERS.md` §3.1).  Allow degradation up to ~1.5× on a
strongly deformed mesh; more than that means the patch blocks are no longer
capturing the local operator and the cause must be found before merging.

Also check that element-interior condensation still pays: the Schur complements
are unchanged in structure, only their entries differ.

### G6 — A case that needs the geometry

*Known answer: published data.*  Flow over a circular cylinder at $Re = 40$
(steady, symmetric): recirculation length $L/D \approx 2.2$ and separation angle
$\approx 53.5°$ against the standard compilations.  This is the first result the
Cartesian code could not produce at all, and it is the one to put in a paper.

---

## 3. Steps

| # | work | gate | estimate |
|---|---|---|---|
| 1 | `Mesh` carries `rx, ry, sx, sy, jac`; collocation metrics; `facx`/`facy` become raising properties; a `deform(f)` helper and an annular builder | **G0** | 2 days |
| 2 | `operators.py` primitives take the metric fields; `dUdx`/`dUdy` gain the second contraction; `DxT`/`DyT` become the adjoints in the $J$-weighted inner product | **G0** re-run, plus an adjoint test $\langle Du,v\rangle=\langle u,D^{\mathsf T}v\rangle$ to $10^{-14}$ | 2 days |
| 3 | `lssem.py`: the four residual rows and their transposes on the new primitives; the affine path preserved behind `mesh.curvilinear` | **G1** | 3 days |
| 4 | `kernels_numba.py` updated to match, with the existing backend-parity test extended to curvilinear meshes | parity to $10^{-14}$ against the reference path | 2 days |
| 5 | MMS and Kovasznay on deformed meshes | **G2**, **G4** | 2 days |
| 6 | Annular builder and the Taylor–Couette case | **G3** | 2 days |
| 7 | Patch preconditioner and condensation on curvilinear meshes | **G5** | 2 days |
| 8 | `obc.py`: true boundary normals for the outflow terms | Gartling re-run matches `GARTLING_VALIDATION.md` on a graded curved mesh | 2 days |
| 9 | Cylinder at $Re = 40$ | **G6** | 3 days |

**Total ≈ 20 working days.**  Steps 1–4 are the irreducible core; 5–7 are
validation; 8–9 are what the capability is *for*.

---

## 4. Risks, and what to do about them

| risk | signal | response |
|---|---|---|
| Metrics inconsistent (G0 passes, G1 fails) | linear fields exact but rotation invariance broken at $10^{-6}$ | the adjoint is wrong, not the metric: check `DxT` carries $J$ correctly |
| Cost more than doubles | wall time per apply > 2.5× Cartesian | expected 2×; more means the metric multiplies are not fused — hoist them into the kernels |
| Patch preconditioner degrades badly (G5 red) | iterations grow with deformation | the ring assembly may be picking up badly scaled blocks; equilibrate per element before Cholesky |
| FDM preconditioner silently wrong | Helmholtz solves converge slowly on curved meshes | it is a surrogate now, not exact; either accept more iterations or fall back to the patch method |
| The affine path drifts | a paper result changes on this branch | CI on the branch re-runs the Cartesian validations and diffs against recorded values |

---

## 5. Branch discipline

* Work on `curvilinear-2d`; **rebase on `main`, never merge back until G0–G5 are
  green** and the Cartesian regression suite is bit-identical.
* Every step commits with its gate's measured numbers in the message, as the
  rest of this repository does.
* This file is the running record: each gate gets its table of results appended
  under it as it is passed, so the branch explains itself.
* The 3D code is untouched.  The eventual 3D port needs the conservative metric
  form (D3) and is a separate plan.
