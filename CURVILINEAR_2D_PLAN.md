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

**PASSED 2026-09-16** (`scratch/curvi_g0.py`, `figs/curvi_g0.png`).  All values
are **multiples of the round-off floor**, not absolute errors — see below:

| identity | affine | rotated 30° | deformed 10 % | annulus |
|---|---|---|---|---|
| grad(1) = 0 | 4.2 | 6.5 | 4.0 | 4.8 |
| grad(x) = (1,0) | 0.5 | 1.0 | 1.5 | 1.0 |
| grad(y) = (0,1) | 0.5 | 1.0 | 1.5 | 1.0 |
| div(const) = 0 | 5.3 | 5.9 | 4.2 | 4.8 |
| area | 11.0 | 11.0 | — | 11.9 |
| adjoint x | 0.0 | 0.1 | 0.0 | 0.0 |
| adjoint y | 0.1 | 0.0 | 0.0 | 0.0 |

Worst 11.9 against a gate of 50.  Quarter-annulus area 0.589048622548 against an
exact $\pi(1-0.25)/4 = 0.589048622548$.

**The gate had to be rewritten before it meant anything, twice, and both fixes
were to the test rather than the code.**  An absolute threshold of $10^{-13}$
failed the annulus at $5\times10^{-13}$ — but differentiating a constant cannot
beat round-off *amplified by the operator*, $\epsilon\lVert D\rVert\lVert
\text{metric}\rVert$, since $D$ has entries of order $N^2$ and the metric of
order $1/h$.  Measuring that ratio across meshes settled it: 4–7 everywhere,
**unchanged when the annulus is refined twofold** (which doubles both the metric
and the error) and when $N$ is raised. That is round-off, and an absolute gate
would have failed every fine mesh for no reason.  Separately the adjoint test
read 61 units on the affine mesh, which is the *cancellation* in a dot product
of random fields — normalising by the summation condition
$\sum|t|/|\sum t|$ puts it at 0.1.

A real metric bug lands at $10^6$ units or more, so the rescaled gate is in no
danger of passing one; the first version was simply measuring the wrong thing.

### G1 — Affine rotation invariance

*Known answer: the existing Cartesian results, to round-off.*

Take the lid-driven cavity and the Kovasznay case, rotate the mesh by 30° and
the boundary data with it, and require the solution to match the unrotated run
**rotated back**, to $10^{-12}$.  A constant-Jacobian rotated element exercises
all four metric terms and every cross term while having an exactly known answer.

*This is the strongest gate in the plan.*  It cannot be passed by a
self-consistent but wrong metric, and it separates "curvilinear machinery works"
from "curvilinear machinery is merely stable".

**PASSED 2026-09-16** (`scratch/curvi_g1.py`, `figs/curvi_g1.png`), and the
Cartesian regression is untouched: **90/90 existing tests pass**.

Rather than a solve, which would mix the operator with boundary conditions and a
preconditioner, the gate tests the assembled operator directly in three ways:

| test | what it rules out | result |
|---|---|---|
| **T1** affine path vs curvilinear path, same mesh unmoved | the branch having broken the Cartesian code | 6.6e−15 (30 ε) |
| **T2** rotation covariance of $L$, 0°–90° | a metric that is self-consistent but wrong | ≤ 1.7e−15 (≤ 8 ε) |
| **T3** adjoint pair $\langle LU,S\rangle=\langle U,L^{\mathsf T}(wq\,S)\rangle$ | a transposed index or dropped weight, which T1 and T2 both survive | ≤ 8.9e−16 (≤ 4 ε) |

Worst 6.6e−15 against a gate of $10^{-12}$.

**T2 is the substance.**  Put a field on the unrotated mesh and its rotated image
on the rotated mesh; the residual rows must come back as the rotated image of the
original rows — momentum as a vector, continuity and the vorticity definition as
scalars.  Flat at a few ε across every angle, including 45° where the cross terms
are largest.

**One thing the gate caught was the convention, not the code.**  `apply_L`
multiplies its rows by `wq`, so `apply_LT` expects the weight already inside its
argument.  Testing $\langle LU,S\rangle=\langle U,L^{\mathsf T}S\rangle$ with an
unweighted $S$ reads **89**, which looks like a catastrophic adjoint failure and
is entirely a property of the interface.  With $wq\,S$ it is 1e−16 on every mesh.
Worth recording because the same trap is waiting in the numba kernels at step 4.

### Step 3 as built

`_apply_L_numpy` and `_apply_LT_numpy` dispatch through one pair of closures
rather than a branch per call site: on an affine mesh they are the existing calls
verbatim, so that path stays bit-identical, and on a curvilinear mesh they carry
the metrics and the second contraction.  20 call sites converted (8 forward
derivatives, 6 `DxT`, 6 `DyT`) plus the linearisation's own four derivatives in
`update_linearisation` — those feed the convective Jacobian terms, and an affine
derivative there on a curvilinear mesh corrupts the operator in a way **no
forward identity catches**; only T2 would have seen it.

### G2 — Spectral convergence on a curved mesh

*Known answer: a manufactured solution, and the convergence rate.*

Use the existing MMS machinery (`scratch/mms2d_temporal.py` builds the forcing)
on a sinusoidally deformed mesh.  Require:

* error falling **exponentially** in $N$ for $N = 4 \dots 12$ on a fixed mesh
  (the spectral signature; algebraic decay means the metrics are inconsistent);
* the same asymptotic rate as the Cartesian case at equal degrees of freedom,
  within a factor of ~2 in the constant.

**GREEN** — `scratch/curvi_g2.py`, `figs/curvi_g2.png`.  Steady manufactured
solution (the `mms2d_temporal.py` stream function with $\cos t$ removed, so BDF2
contributes exactly zero and the spatial error is not mixed with a temporal one
of fixed order).  Relative $L^2$ error in $\mathbf u$, $E = 2$:

| $N$ | affine | curvilinear-affine | rotated 30° | deformed 10 % |
|---|---|---|---|---|
| 4 | 1.544e−02 | 1.544e−02 | 2.986e−02 | 2.909e−02 |
| 8 | 7.520e−07 | 7.520e−07 | 9.164e−06 | 2.358e−04 |
| 14 | 4.319e−14 | 4.349e−14 | 1.994e−12 | 4.038e−08 |

Exponential rates 2.72 / 2.71 / 2.49 / 1.43.  The affine and curvilinear code
paths agree to **3.0e−16**, which is the regression protecting every Cartesian
result in the paper.  Insensitive to $dt$ across 1e4–1e8 (6.998132e−06 →
6.998259e−06), so the steady limit is a limit and not a tuning knob.

**THE ROTATED MESH IS WHAT TESTS `bc.py`, NOT THE DEFORMED ONE.**  `curvi.deform`
perturbs by a product of sines that vanishes on the domain boundary, so the
DOMAIN is still the unit square and boundary nodes have not moved — their
tensor-product coordinates are still correct.  A rotated square's edges are not
axis aligned, $x$ and $y$ both vary along every edge, and that column would not
converge at all if `bc.py` were still reading `xnod[e,0]` as a single $x$ for a
whole edge.

### G2a — How the exponential claim was actually established

Worth recording because the first version of this gate used the wrong statistic
and would have licensed a wrong answer.

**An $R^2$ comparison between an exponential and an algebraic fit cannot separate
them over $N$ spanning less than a factor of three.**  The deformed column
returned $R^2 = 0.9955$ exponential against 0.9869 algebraic — a difference with
no discriminating power, since a high-order power law and an exponential are
nearly identical over such a range.

The test that needs no fitting: for $e \sim e^{-bN}$ the **drop per order is
constant** and the implied algebraic order $q$ **rises without bound**; for
$e \sim N^{-q}$, $q$ is constant and the drop per order falls.  Read that way the
deformed column was *ambiguous* — $q$ ran 5.8, 8.6, 15.8, 15.7, 14.9, going FLAT
over three intervals, which is the algebraic signature.  Extending to $N = 16, 18$
settled it (`scratch/curvi_rate_check.py`):

| $N$ | error | drop | local $b$ | implied $q$ |
|---|---|---|---|---|
| 12 | 4.028e−07 | 17.4 | 1.43 | 15.7 |
| 14 | 4.038e−08 | 10.0 | 1.15 | **14.9** |
| 16 | 2.536e−09 | 15.9 | 1.38 | **20.7** |
| 18 | 9.882e−11 | 25.7 | 1.62 | **27.6** |

$q$ resumes climbing while the error falls another 2.6 orders: the flat stretch
was a transient, not a tail.  The alternative explanation is also excluded — the
deformed mesh has min $J$ = 4.29e−02 and max/min = 1.916, **identical at
$N = 8, 12, 16$**, so no element is near-singular and the metric is a property of
the map rather than of the order, as collocation metrics must be.

Lesson for the remaining gates: quote the successive-ratio table, not a fit
quality.

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

**GREEN** — `scratch/curvi_g3.py`, `figs/curvi_g3.png`, fields in
`scratch/curvi_g3_fields.py` → `figs/curvi_g3_fields.png`.  Quarter annulus,
$r \in [0.5, 1]$, 3×4 elements, $\Omega_{\rm in} = 1$, $\nu = 0.05$, Re = 5.
**Zero forcing** — this is a genuine Navier–Stokes solution, not a manufactured
one — and started **from rest**, so a broken operator cannot sit still and report
zero error.

| $N$ | $\lVert\mathbf u\rVert$ rel $L^2$ | $u_r$ (exactly 0) | max\|∇·u\| | max\|ω − 2A\| |
|---|---|---|---|---|
| 4 | 2.065e−06 | 1.214e−06 | 2.687e−04 | 7.156e−05 |
| 6 | 7.662e−09 | 5.596e−09 | 3.514e−06 | 1.537e−06 |
| 8 | 4.598e−11 | 3.478e−11 | 3.066e−08 | 1.899e−08 |
| 10 | 2.529e−13 | 1.922e−13 | 2.916e−10 | 1.745e−10 |
| 11 | 1.879e−14 | — | — | — |

Divergence and vorticity finish five orders under the $10^{-5}$ requirement.  By
the successive-ratio test of G2a this is unambiguous: the drop per order is
**13.4, 13.6, 13.5** over the last three intervals (local $b$ = 2.59, 2.61, 2.60)
while $q$ climbs 22.0 → 24.8 → 27.3.  A relative error of 1.879e−14 at $N = 11$ is
about 85 ε, so the exact solve is **still not the limiting factor** — the floor
was looked for and not found.

The two analytic CONSTANTS are the sharp instruments here.  $u_r \equiv 0$ and
$\omega \equiv 2A$ have no shape for a metric error to hide inside, and $u_r$ is
the component the mapping mixes into the Cartesian pair the solver stores — the
one most exposed to an inconsistent metric.  Both track the discretisation error
rather than sitting above it.

### G4 — Kovasznay on a deformed mesh

*Known answer: the existing `KOVASZNAY_VALIDATION.md` result.*

Re-run Kovasznay with interior nodes perturbed by 10% of the element size.  The
error must match the Cartesian run to within the discretisation error — a curved
mesh should cost accuracy, but not much, and not change the convergence order.

**GREEN** — `scratch/curvi_g4.py`, `figs/curvi_g4.png`.  Re = 40, domain
$[-0.5,1]\times[-0.5,0.5]$, 8 elements, pure steady Newton ($w_{\rm mass} = 0$),
exact linear solves.  $\epsilon_u$ is the rms over **unique global nodes**, as in
`KOVASZNAY_VALIDATION.md`.

| $N$ | Cartesian | deformed 10 % | ratio | Chan (1996) |
|---|---|---|---|---|
| 4 | 3.7245e−03 | 3.8998e−03 | 1.0 | 3.724e−03 |
| 8 | 2.8658e−07 | 6.7114e−07 | 2.3 | — |
| 9 | 6.3851e−09 | 1.1189e−07 | 17.5 | 6.380e−09 |
| 12 | 3.8595e−12 | 1.2256e−10 | 31.8 | — |

The Cartesian column reproduces the published values at **×1.00** on both points
where a reference exists — an end-to-end check of the whole chain against a
number nobody could tune after the fact.  Rates: Cartesian $b = 2.61$, deformed
$b = 2.17$.  **The deformation costs a constant (up to ~32×), not an order.**

*A note on the gate's own threshold.*  The first version required pointwise
$\nabla\!\cdot\mathbf u < 10^{-4}$ **at every $N$** and so reported FAIL on the
$N = 4$ value of 0.28 — where the solution itself is resolved only to 4e−03.  In
a least-squares method div $\mathbf u$ is minimised, not enforced, so a coarse
discretisation has a large one by construction.  Judged at the finest order,
where the requirement belongs, it is 6.94e−08.

### G5 — The preconditioner survives

*Known answer: the existing iteration counts.*

The vertex-patch preconditioner on a curvilinear mesh must stay flat in $N$:
19–21 iterations on the cavity at $N = 5\dots20$ is the Cartesian benchmark
(`LOW_MEMORY_PATCH_SOLVERS.md` §3.1).  Allow degradation up to ~1.5× on a
strongly deformed mesh; more than that means the patch blocks are no longer
capturing the local operator and the cause must be found before merging.

Also check that element-interior condensation still pays: the Schur complements
are unchanged in structure, only their entries differ.

**GREEN** — `scratch/curvi_g5.py`, `figs/curvi_g5.png`.  Ghia cavity Re = 1000,
4×4 elements, condensed vertex patch + $p = 2$ coarse, CG tol 1e−10, the
preconditioner rebuilt every Newton step.  Mean CG iterations per solve:

| $N$ | ndof | affine | curvilinear-affine | deformed 10 % | annulus |
|---|---|---|---|---|---|
| 5 | 1 764 | 31.7 | 31.7 | 32.0 | 26.5 |
| 8 | 4 356 | 32.0 | 32.0 | 32.3 | 28.2 |
| 12 | 9 604 | 32.8 | 32.8 | 33.0 | 29.8 |

Growth over $N = 5 \dots 12$: 1.04× affine, 1.03× deformed, 1.13× annulus —
**flat**.  Worst deformed/affine ratio **1.01** against the plan's 1.5 allowance,
so the curvature is essentially free.  Affine and curvilinear-affine agree to
**0.0 iterations**, exactly.  (The annulus is a different problem — Taylor–Couette
at Re = 5 — so only its flatness is meaningful, not its ratio to the cavity; it is
there because it has no affine counterpart, which is the point of the gate.)

`element_blocks` needed no change: it probes `apply_L`/`apply_LT` with local unit
vectors and so inherited the curvilinear operator for free.  The COARSE level did
not — `PMG2` builds it with a shallow `copy` of the mesh and then lowers $N$,
leaving the fine-order metric fields attached to a coarse-order state.  The coarse
geometry is now the fine geometry *interpolated to the coarse nodes*, so both
levels describe the same domain to the accuracy each can represent.

### G5a — A case that must NOT be run, and was

The first version of this gate included a **rotated cavity** and reported 44 → 48
iterations against the affine 32.  That reads as a 1.5× curvature penalty and is
nothing of the kind.

Boundary code 2 prescribes the **Cartesian** pair $(u,v) = (\text{lid}, 0)$.
Rotate the mesh 30° and the lid rotates with it, but the prescribed velocity does
not: $(1,0)$ against an outward normal $(-\sin 30°, \cos 30°)$ has a **normal
component of $-0.5$**.  Every other wall is no-slip, so the net flux through the
boundary is $-0.5$ and the incompressibility constraint cannot be satisfied
anywhere in the domain.

A least-squares method does not diverge on an inconsistent constraint — it
minimises a residual it can never zero — so the run **completes and returns an
iteration count that looks like a measurement**.  That is the trap: the failure
mode of this formulation is a plausible number, not a crash.

Rotation covariance is already established exactly, at 1e−15, by G1's T2, which
tests the operator and therefore needs no boundary conditions at all.  Nothing is
lost by dropping the case, and the annulus replaces it with a curved domain whose
data is consistent.

*Consequence for `bc.py`:* codes 2 and 3 cannot express a velocity that is not
axis-aligned.  Not needed for any gate here (every curved case uses code 1 with
`exact_solution`, which prescribes both components), but it is the reason step 8
must give `obc.py` true boundary normals before any outflow problem on a curved
mesh, and a vector-valued lid/inlet would need the same treatment.

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
