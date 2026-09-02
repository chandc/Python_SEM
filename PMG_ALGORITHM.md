# The p-multigrid preconditioner (`PMG2`)

Reference for `lssem2d/precond.py`. Covers the operator being preconditioned,
every step of the V-cycle, the multilevel recursion, both coarse-grid solvers,
and the measurements taken so far.

> **`PMG2` is no longer two-level.** The class name is historical. `pc` accepts
> a **sequence of orders**, so `pc=(4,2)` gives `p → 4 → 2` and
> `pc=(15,7,3,2)` gives a five-level hierarchy. §6.6 measures why that matters:
> **only a halving ladder is p-robust** — fixed 2- and 3-level hierarchies
> degrade almost as badly as Jacobi as the order rises.

Companion documents: [FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) (why the near-null
space matters), [OUTFLOW_BC_STUDY.md](./OUTFLOW_BC_STUDY.md) (why the soft
outflow mode exists), [OUTFLOW_DONG_OBC_PLAN.md](./OUTFLOW_DONG_OBC_PLAN.md).

---

## 1. What is being preconditioned

The LSSEM VVP system solves, at each Newton step, `A δU = b` with four unknowns
per node, `U = (u, v, p, ω)`. From `lssem2d/solver.py::apply_A`:

$$
A \;=\; M\,Q^{T} Q\,\Bigl(L^{T} W L \;+\; B^{T} W_b B\Bigr)\,M
$$

| symbol | meaning | code |
|---|---|---|
| `L` | first-order VVP residual operator | `lssem.apply_L` |
| `W` | quadrature weights (forward operator only) | `mesh.wq` |
| `QᵀQ` | direct stiffness summation — imposes C⁰ continuity | `assembly.gather_scatter` |
| `M` | Dirichlet mask, applied on **both** sides to keep `A` symmetric | `bc.get_global_mask` |
| `B` | Dong OBC boundary rows on `bc == 6` edges | `obc.apply_B` |

`A` is symmetric positive definite — this is what makes CG admissible, and it is
gated in [FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) §F0 (asymmetry ~1e-16).

### The Dong outflow rows

For an outflow plane at `x = xmax` with outward normal `n = (1,0)`
(Dong 2015, eq. 4):

$$
R_x = \nu D_0 \frac{\partial u}{\partial t} - p + \nu \frac{\partial u}{\partial x} - E_x(u), \qquad
R_y = \nu D_0 \frac{\partial v}{\partial t} \;\;\;\;\;\;\;+ \nu \frac{\partial v}{\partial x} - E_y(u)
$$

entering the functional as a boundary term, **not** imposed strongly:

$$
J \;\longrightarrow\; J + w_{\text{obc}}^{2}\int_{\Gamma_{\text{out}}} \bigl(R_x^{2}+R_y^{2}\bigr)\,ds
$$

The backflow switch `E(n,u) = ½[|u|²n + (n·u)u]·Θ₀`, with
`Θ₀ = ½(1 − tanh(u/(δU₀)))`, is treated **explicitly** (lagged), so only the
linear part reaches the operator:

$$
R_x^{\text{lin}} = c_b u - p + \nu\,\partial_x u, \qquad
R_y^{\text{lin}} = c_b v + \nu\,\partial_x v, \qquad
\boxed{\,c_b = \frac{\nu D_0\,\mathrm{fac1}}{\Delta t}\,}
$$

Edge quadrature weight `ws_j = (h_y/2)·w_j`. **East edges only** — any other
edge carrying `bc == 6` raises `NotImplementedError`.

### Why a diagonal preconditioner is not enough

The VVP outflow pressure lives in a very soft direction of `A` — measured
**~8×10³ times softer** than a generic direction on the Chan mesh. A diagonal
preconditioner rescales pointwise and cannot touch a near-null mode, so CG stops
with it unresolved and **the converged state becomes path dependent**. That is a
correctness failure, and it is the reason `PMG2` exists.

This is the same structure [FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) §F2 found from
the other direction: the softest mode of the assembled FOSLS operator carries
**97.7% of its energy in ω**, invariant across N = 4…12.

---

## 2. The V-cycle

Shown two-level for legibility. §2.1 gives the recursion that makes it
multilevel; the per-level steps are identical at every level.

```mermaid
flowchart TD
    R["residual r<br/>(fine, order p)"] --> S1
    S1["<b>1. pre-smooth</b><br/>z = S(r)<br/>deg-k Chebyshev, k operator applies"] --> D1
    D1["<b>2. defect</b><br/>res = r − A z"] --> RE
    RE["<b>3. restrict</b><br/>r_c = Q<sup>T</sup>(P<sup>T</sup>⊗P<sup>T</sup>)(res ⊙ w)<br/>order p → p_c"] --> CS
    CS["<b>4. coarse solve</b><br/>e_c = A_c<sup>−1</sup> r_c<br/><i>Chebyshev</i> or <i>Direct</i>"] --> PR
    PR["<b>5. prolong</b><br/>z ← z + (P⊗P) e_c<br/>order p_c → p"] --> D2
    D2["<b>6. defect</b><br/>res = r − A z"] --> S2
    S2["<b>7. post-smooth</b><br/>z ← z + S(res)"] --> OUT
    OUT["preconditioned vector z<br/>= M<sup>−1</sup> r"]
```

Verbatim from `PMG2.__call__`:

```python
z   = self.smooth(r)                     # 1. pre-smooth
res = r - apply_A(state, z, fu, fv)      # 2. defect
ec  = self._coarse_solve(self._restrict(res))   # 3-4. restrict + coarse solve
z   = z + self._prolong(ec)              # 5. prolong
res = r - apply_A(state, z, fu, fv)      # 6. defect
z   = z + self.smooth(res)               # 7. post-smooth
```

Cost per application: **2 fine operator applies** (the two defects) + `2·deg`
from the two smooths + the coarse solve.

### 2.1 The recursion — how depth is built

Step 4 says *"solve `A_c e_c = r_c`"*. It does not say the coarse level must be
solved cheaply, only that whatever solves it must be a **fixed linear operator**
(§2.2). A V-cycle is one. So the hierarchy is built by making the coarse solver
another `PMG2`:

```
pc = 2          p ─────────────────────────► 2          two levels
pc = (4, 2)     p ──────► 4 ──────► 2                   three levels
pc = (15,7,3,2) 30 ─► 15 ─► 7 ─► 3 ─► 2                 five levels  (N=30 ladder)
                └────┴─────┴─────┴────┴─ DirectCoarse at the bottom only
```

This works because `PMG2.__call__(r)` **already has the coarse-solver
signature** — residual in, correction out — so no separate multilevel driver is
needed. Each level Chebyshev-smooths, restricts, recurses, prolongs, smooths
again. Only the *bottom* level is ever assembled.

$$
M^{-1}_{(p)} \;=\; \mathcal{V}\bigl(p,\; M^{-1}_{(p_1)}\bigr), \qquad
M^{-1}_{(p_k)} \;=\; \mathcal{V}\bigl(p_k,\; M^{-1}_{(p_{k+1})}\bigr), \qquad
M^{-1}_{(p_{\text{last}})} \;=\; A_{p_{\text{last}}}^{-1}
$$

**Every level stays a fixed linear operator**, so CG's symmetry requirement holds
all the way down — the argument in §2.2 applies unchanged at each level.

**Why depth is not optional.** A single `p → 2` jump asks one coarse grid to
represent everything the fine grid cannot resolve. Heys *et al.* (2005) identify
this directly: `p/2` coarsening retains only ~25% of points where AMG retains
~50%, which is why aggressive p-coarsening underperforms. §6.6 measures the
consequence — over `N = 5…30`, iterations grow **7.90×** for a fixed 2-level
hierarchy, **5.56×** for 3-level, and **1.41×** for the halving ladder.

### 2.2 Why `M⁻¹` must be a fixed linear operator

`PMG2` is used as a preconditioner inside CG, which requires `M⁻¹` to be a
**fixed, symmetric, positive-definite linear operator**. Three design choices
follow, and none is optional:

1. **`R = Pᵀ` exactly.** An independent fine→coarse interpolation would break
   symmetry.
2. **Multiplicity weighting before restriction.** Fields live in redundant local
   storage where every copy of a shared node already holds the assembled value,
   so a plain `Pᵀ` would count a shared node once per owning element. Dividing
   by the multiplicity first makes the transfer the true adjoint of `_prolong`.
3. **No inner Krylov solve on the coarse grid** — see §4.

---

## 3. The components

### 3.1 Smoother — 4th-kind Chebyshev

Degree-`k` polynomial in `D⁻¹A` started from `z = 0`. Fourth-kind needs only an
**upper** bound on the spectrum, no lower edge, which is what makes it robust
when the spectrum is poorly known. With `ρ = 1.3·λmax` (`λmax` from 20 power
iterations on `M⁻¹A`), for `k = 1…deg`:

$$
d_k = \frac{2k-3}{2k+1}\,d_{k-1} \;+\; \beta_k\frac{8k-4}{(2k+1)\rho}\,M^{-1}r_k,
\qquad z_k = z_{k-1} + d_k, \qquad r_{k+1} = r_k - A\,d_k
$$

`β = 1` gives plain 4th-kind; the optimised weights `_BETA4` are Phillips &
Fischer / Lottes, *Optimal Chebyshev smoothers*, Table 5 — the same table as
`solver_pmg2.f90`. Cost: `deg` operator applications per call.

### 3.2 Inter-level transfer

`P` is the 1-D Lagrange interpolation matrix from `LGL(p_c)` to `LGL(p_f)`
nodes, applied as a tensor product in each direction:

$$
\mathcal{P} = P \otimes P, \qquad
(\mathcal{P}x_c)_{ab} = \sum_{i}\sum_{j} P_{ai}P_{bj}\,(x_c)_{ij}
$$

```
fine  p = 8   ●--●---●----●----●----●---●--●     9 LGL nodes
                   \    \   |   /    /
                    \    \  |  /    /              P  (interpolate)
                     \    \ | /    /               R = Pᵀ (adjoint)
coarse p = 2      ●---------●---------●            3 LGL nodes
```

Note the **non-uniform** LGL spacing — clustering near element ends as
`O(1/p²)`. That anisotropy is what makes plain Jacobi/Gauss–Seidel smoothing
degrade at high order, and it is why a polynomial smoother is used here.

### 3.3 The coarse operator

**Re-discretised**, not Galerkin: the same elements and geometry at order `p_c`,
with `A_c` built by the same `apply_A`. (`solver_pmg2.f90` offers this as
`pmg_galerkin=.false.`; its production path is the Galerkin variant, which is
more accurate and much more code.)

> **Every coefficient must be carried down.** The coarse `SolverState` is built
> fresh, so anything not forwarded silently reverts to a default and the
> V-cycle returns a correction **to the wrong operator**:
>
> | coefficient | if omitted | measured cost |
> |---|---|---|
> | `w_mom`, `w_mass` | `ls_coeffs` takes its LEGACY branch — `(fac1, dt)` instead of `(fac1·w_mass/dt, w_mom)` | CG needed **~2000 iterations instead of tens** |
> | `dtau` | drops off the momentum diagonal | — |
> | `obc_w`, `obc_D0`, `obc_delta`, `obc_U0` | boundary rows lose the `ν D₀ ∂u/∂t` term (`c_b → 0`) | ~0.5% — see §6.2 |
>
> `obc_active()` is decided by the **mesh** (`bc == 6` edges), which `copy(m)`
> preserves, so the coarse operator gets an OBC term whether or not its
> coefficients were forwarded. Probe with `scratch/pmg_coarse_probe.py`.

---

## 4. Coarse-grid solvers

Both options are **fixed linear operators**. An inner CG would not be — its
polynomial depends on the right-hand side it is handed, so `M⁻¹` would become
nonlinear, the outer CG's orthogonality relations would fail, and the three-term
recurrence would stop being valid. Symptoms are stagnation or erratic residuals
rather than an honest error. If an inner Krylov solve is ever wanted, the outer
solver must change to **flexible** CG or FGMRES.

### 4.0 A nested `PMG2` — chosen automatically when `pc` is a sequence

Not really a third option so much as the recursion of §2.1: when `pc` has more
than one entry, the coarse solver is another V-cycle and the *last* entry gets
whichever leaf solver `coarse_solver` names. This is the configuration §6.6
finds p-robust.

### 4.1 `Chebyshev4` (default)

Degree-10 4th-kind polynomial on `A_c`, Jacobi-preconditioned. Cheap, matrix
free, symmetric. **But it smooths the coarse problem, it does not solve it** —
a polynomial damps a spectral *band*, and the soft end is precisely what the
coarse level is supposed to remove.

### 4.2 `DirectCoarse`

Assemble `A_c` once, factorise, reuse.

```
for each free coarse DOF g = (node j, field f):
    U ← 0
    U[all local copies of j, f] ← 1     ← the global basis function is the
    col_g ← apply_A(U)  reduced to one    SUM of its local copies
            representative per node
A_c ← ½(cols + colsᵀ);   splu(A_c)
```

**Assembly is by probing `apply_A`, which is correct by construction**: masking,
the Dong boundary term and the least-squares weights are all in the matrix, with
no second implementation to drift out of sync. Measured asymmetry: **exactly
0.00e+00**.

**A singular coarse operator is handled, not an error.** With `pin_p=False` on a
closed domain the pressure is defined only up to a constant, the coarse operator
inherits that null mode, and `splu` fails with *"Factor is exactly singular"*.
That is real, not a bug — a preconditioner does not need the null direction
resolved. `DirectCoarse` shifts the diagonal by ~1e−12 of its scale and
continues; measured, the shift is **exactly 0.0 with `pin_p=True`** and 8.9e−12
without.

Cost is `O(ndof_coarse)` applications of a *small* operator, paid once per
linearisation. It scales with the **element count, not with `p`** — which is why
§6.6's setup time is flat (0.01 s at N=5, 0.04 s at N=30) across a 31× growth in
fine-grid DOF:

| mesh | `p_c = 2` coarse DOF | dense `A_c` |
|---|---|---|
| Poiseuille 6×2 | 260 (202 free) | 0.5 MB |
| Gartling 11×4 | 828 | 5.5 MB |
| 16×16 | 4356 | 152 MB |

At large element counts this needs element-local probing (as
`scratch/fosls_assemble.py` does, `O(1)` in mesh size) or an AMG V-cycle in
place of the factorisation — which is what Pazner uses for his coarse solver
`R₀`, and where [FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) §F2's near-null-space
result would apply.

---

## 5. Test cases

| # | case | why chosen | script |
|---|---|---|---|
| **T1** | Plane Poiseuille + Dong outlet, `[0,12]×[0,1]`, `bcs=(3,6,1,1)`, Re=100, `dt=0.5`, `D₀=1`, `δ=0.05` | **Gate.** Exact solution known, and it *zeroes the Dong rows* (`p=0`, `∂u/∂x=0`, `v=0` at the exit), so the outflow is exercised while the right answer stays unambiguous | `scratch/pmg_direct_coarse.py` |
| **T2** | Same, driven to steady state from 4 different initial conditions (`zero`, `uniform`, `parabolic`, `noisy`) | Tests the **correctness** claim — path dependence — which iteration counts cannot | `scratch/pmg_path_dependence.py` |
| **T3** | Gartling BFS Re=800 + Dong outlet, Chan's 11×4 grid at N=7, steady (`w_mass=0`) | The soft outflow mode actually bites; published reattachment length to check against | `scratch/dong_gartling.py` |
| **T4** | Short-domain BFS, Re=389 | Recirculation **crosses** the outlet, so the boundary sits in genuine backflow — where `D₀`/`δ` should matter | `scratch/dong_bfs.py` |

---

## 6. Results

### 6.1 T1 — single linear solve, `b = A x_rand`, tol 1e-10

`N=8`, 6×2 elements, 3888 local DOF, `p_c = 2`, fine smoother degree 4:

| preconditioner | CG iterations | solve wall | setup |
|---|---|---|---|
| Jacobi | 1275 | 0.284 s | — |
| PMG2 + Chebyshev(10) coarse | 142 | 0.508 s | 0.008 s |
| **PMG2 + Direct coarse** | **107** | **0.253 s** | 0.03 s |

The direct coarse solve wins on **both** axes: 1.33× fewer iterations, and 2.0×
less solve wall because one triangular solve replaces ten coarse operator
applications per V-cycle.

### 6.2 T1 — order sweep, and isolating the two changes

`cheby(bug)` reproduces the pre-fix coarse operator by resetting the coarse OBC
coefficients to their defaults after construction.

| N | coarse DOF | Jacobi | cheby **(bug)** | cheby **(fixed)** | **Direct** | Jacobi/Direct |
|---|---|---|---|---|---|---|
| 6 | 202 | 846 | 97 | 98 | **70** | 12.1× |
| 8 | 202 | 1275 | 141 | 142 | **107** | 11.9× |
| 10 | 202 | 1741 | 195 | 195 | **152** | 11.5× |
| 12 | 202 | 2196 | 242 | 243 | **194** | 11.3× |
| **growth 6→12** | | **2.60×** | 2.49× | 2.48× | **2.77×** | |

Three things this settles:

1. **The OBC propagation fix is worth ~0.5% here** (97 vs 98, 141 vs 142). It is
   a genuine inconsistency and worth fixing, but it is *not* what makes the
   direct solve faster. T4 is where it should matter.
2. **The direct coarse solve is a real 1.25–1.40×, and it shrinks with order**
   (1.40× at N=6 → 1.25× at N=12). It is not a `p`-robustness fix.
3. **`PMG2` at `p_c = 2` is a constant-factor preconditioner, not a
   `p`-independent one.** Jacobi/PMG2 is flat at 11.3–12.1× while *both* grow
   ~2.5–2.8× over the range. This is the same pattern
   [FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) §F2g found for AMG under
   `p`-refinement.

### 6.3 T2 — path dependence: **the ordering is confirmed**

Twelve configurations (4 initial conditions x 3 preconditioners), driven to
`|dU|inf < 1e-11` or a 400-step cap, run 12-way parallel on the DGX Spark's CPU
cores. Raw: `scratch/pmg_path_dependence_*.npz`,
`scratch/pmg_path_dependence_spark.log`.

| IC | Jacobi | + Chebyshev coarse | + **Direct** coarse |
|---|---|---|---|
| `zero` | conv@173 | conv@159 | **conv@143** |
| `uniform` | conv@159 | conv@190 | **conv@144** |
| `parabolic` | conv@214 | conv@106 | **conv@26** |
| `noisy` | **NaN@67** | **NaN@67** | diverged (`|dU|~1e43`) |

**Spread of the converged states** — `max_{a,b} ||U_a - U_b||_inf` over the three
initial conditions that converged for every preconditioner:

| preconditioner | max pairwise spread | vs Jacobi |
|---|---|---|
| Jacobi | 1.257e-06 | 1x |
| PMG2 + Chebyshev(10) | 1.548e-07 | **8.1x** |
| **PMG2 + Direct** | **1.074e-08** | **117x** |

**Monotone, in the predicted order.** Resolving the coarse problem exactly
reduces path dependence by **117x against Jacobi and 14x against the polynomial
coarse solve**. The spread is five orders of magnitude above the per-step
convergence criterion (1e-11), so these are genuinely different converged states,
not numerical noise on the stopping test.

**The `parabolic` row is the clearest single signal.** That initial condition is
essentially the exact solution, so a solver that resolves the system should
recognise it at once. Direct does — **26 steps against Jacobi's 214, 8.2x
fewer**. The diagonal preconditioner leaves something unresolved that then has to
relax away over hundreds of time steps, which is the mechanism `precond.py`
describes.

**But the magnitudes are small, and that matters for how much to claim.** A
1.3e-06 spread on a field of O(1) is not the dramatic path dependence that would
produce visibly different flow. The ~8e3x soft direction was measured on the
**Chan mesh**, not on Poiseuille, and Poiseuille is a benign case chosen as a
gate precisely because its answer is known. T3 (Gartling) is where the effect
should be larger and where this ordering needs to be reconfirmed before the
default is changed.

**The `noisy` initial condition was a test-design error.** A 0.3-amplitude random
velocity perturbation drives the Dong outlet into a regime where the
**formulation** fails, not the solver: Jacobi and Chebyshev both reach NaN at
**exactly step 67**, and two very different preconditioners failing at the same
step is the signature of a problem-level blow-up. Direct did not overflow but
diverged all the same (`|dU| ~ 1e43` at step 125, ~300 s/step, killed at 10.4 h).
The amplitude was chosen without checking it sat in a stable basin -- the same
mistake as the `minchan` trip amplitude. It is excluded from the spread, so the
comparison above is like-for-like over the same three initial conditions.

### 6.4 T3 — Gartling BFS: the iteration never converges, and that is the result

Gartling Re = 800 with the Dong outlet, 44 elements at N = 7 (9048 global DOF),
steady form, 4 initial conditions x 3 preconditioners, 12-way parallel on the
DGX Spark. Raw: `scratch/pmg_t3_gartling_*.npz`.

**All twelve configurations ran to the step cap. None converged.** Raising the
cap from 300 to 2000 -- 5.5x the wall time -- returned results **identical to
every printed digit**. The extra 1700 steps changed nothing.

**It is not a limit cycle.** `OUTFLOW_BC_STUDY.md` sec 6 documents a period-2
orbit for a related failure, and that was the first diagnosis; it is wrong here.
Measured with `scratch/pmg_t3_drift.py`:

| | `\|U(k)-U(k-1)\|` | `\|U(k)-U(k-2)\|` | ratio |
|---|---|---|---|
| Jacobi | 3.461e-08 | 6.922e-08 | **0.5x** |
| Direct | 2.079e-09 | 4.158e-09 | **0.5x** |

A period-2 orbit gives `\|U(k)-U(k-2)\| ~ 0`. **Exactly 2x is the signature of
constant-rate linear drift** -- the iterate sliding along one fixed direction
forever. That is the soft mode, quantified: the solver cannot pin down the
component along the near-null direction, so the solution creeps along it at a
constant rate and the steady test can never fire.

**The drift rate is the cleanest measure this study has produced:**

| preconditioner | drift per step | vs Jacobi | reattachment spread | wall |
|---|---|---|---|---|
| Jacobi | 3.461e-08 | 1x | **0.186** (3.0%) | ~7150 s |
| PMG2 + Chebyshev | — | — | 0.086 (1.4%) | ~5900 s |
| **PMG2 + Direct** | **2.079e-09** | **17x** | **0.082 (1.3%)** | **~2750 s** |

**And it moves a benchmarked physical quantity.** Lower-wall reattachment against
Gartling's 6.10:

| IC | Jacobi | Chebyshev | Direct |
|---|---|---|---|
| `zero` | 6.2778 | 6.1409 | 6.1417 |
| `uniform` | 6.1879 | 6.1646 | 6.1909 |
| `inlet` | 6.2895 | 6.2094 | 6.2051 |
| `pzseed` | 6.1035 | 6.1234 | 6.1234 |
| **spread** | **0.186** | 0.086 | **0.082** |

**Under Jacobi the reattachment length depends on the initial condition by 3% of
its own value.** That is far more consequential than T2's 1e-06 field spread --
it is the benchmark quantity moving. Both PMG2 variants more than halve it.

**Verdict.** T3 reconfirms the T2 ordering on a case where the soft mode bites,
and adds two things T2 could not show: a **17x reduction in drift rate**, and a
**2.6x wall-time win** (2750 s against 7150 s) that Poiseuille did not exhibit.
The direct coarse solve is better on iterations, on wall time, and on the
physics.

**What it does not show.** No configuration reaches a fixed point, so "converged
state" is not defined for this case and the T2-style spread is not computable
(`common = []`). Reducing drift 17x is not eliminating it. **The underlying
problem is the formulation's soft direction, and a better coarse solve slows the
symptom rather than removing the cause** -- the same relationship
`ROW7_WEIGHT` has to the omega cluster in 3D
([FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) sec F2).

### 6.5 T4 — not yet run; the case is characterised

T4 (the preconditioner comparison under genuine backflow) has **not** been run.
What exists is a characterisation of the case itself, from the saved fields --
`scratch/plot_short_streamlines.py`, figure `scratch/short_bfs_streamlines.png`.

Short-domain Armaly BFS, Re = 389, ER 1.94, 72 elements at N = 10, step at
`x = 0`, outlet at `x = 5`. The validated reference is
[ARMALY_VALIDATION.md](./ARMALY_VALIDATION.md)'s long-domain P+Z result,
**`x_r/S = 8.145`**, within 1.2% of Armaly's experiment — so `x_r ≈ 7.7`, **past
this domain's outlet**. The recirculation crosses the boundary and every outflow
condition is asked to handle reversed flow.

| outflow condition | `x_r/S` | vs 8.145 | outlet backflow | min `u` | max `|u|` |
|---|---|---|---|---|---|
| free | 3.66 | **−55%** | 56.8% | −1.128 | **2.3175** |
| P+Z | 5.17 | −36% | 6.8% | −0.069 | 1.5000 |
| Dong, switch disarmed | >5.32 | closest | 20.5% | −0.153 | 1.5000 |
| **Dong, switch armed** | **>5.32** | **closest** | **18.2%** | **−0.102** | 1.5000 |

**Free outflow is visibly corrupted** — a spurious secondary vortex near
`x ≈ 3.7–5`, the `u = 0` line kinking upward and out of the top of the domain,
and `max|u| = 2.32` against a physical 1.5, a 55% overshoot. It truncates the
bubble to 45% of its true length.

**Dong is the only condition that does not artificially reattach.** Both variants
carry the bubble past the outlet, which is what the reference says should happen,
and the armed switch cuts peak outlet backflow 33% against disarmed — the `Θ₀`
term doing the job it exists for.

Two caveats: *"no reattachment in domain"* only bounds `x_r/S > 5.32`, which is
consistent with 8.145 but does not measure it; and the armed run finished
`WALLCAP` at 547 steps (`|dU| = 1.1e-10`), near-converged rather than converged.

**This is why T4 is the right next test.** `obc_D0 = 2 ≠ 0` here, so unlike T3 the
coarse OBC-propagation fix is live; and the outlet sits in reversed flow, which
is where a preconditioner that resolves the soft mode should matter most.

### 6.6 High order: p = 5 to 30. **The halving ladder is p-independent**

`scratch/pmg_high_p.py`. 2×2 elements so N=30 is affordable, steady, `w_mom=1`,
`pin_p=True`, tol 1e−8. `PMG2` now nests, so `pc` may be a sequence.

| N | gDOF | ladder | Jacobi | 2-lvl | 3-lvl | **ladder** | Jac/lad |
|---|---|---|---|---|---|---|---|
| 5 | 484 | (2) | 475 | 71 | 54 | **71** | 6.7× |
| 6 | 676 | (3,2) | 650 | 94 | 63 | **72** | 9.0× |
| 8 | 1156 | (4,2) | 942 | 130 | 78 | **78** | 12.1× |
| 10 | 1764 | (5,2) | 1253 | 157 | 99 | **89** | 14.1× |
| 12 | 2500 | (6,3,2) | 1546 | 193 | 108 | **80** | 19.3× |
| 16 | 4356 | (8,4,2) | 2132 | 259 | 141 | **82** | 26.0× |
| 20 | 6724 | (10,5,2) | 2742 | 329 | 177 | **89** | 30.8× |
| 24 | 9604 | (12,6,3,2) | 3412 | 429 | 222 | **91** | 37.5× |
| 30 | 14884 | (15,7,3,2) | 4365 | 561 | 300 | **100** | 43.6× |
| **growth 5→30** | | | **9.2×** | **7.90×** | **5.56×** | **1.41×** | |

**71 → 100 iterations across a 6× increase in order and 31× in DOF.**

**Depth is the whole story.** Fixed 2-level (7.90×) and 3-level (5.56×) both
degrade badly — they coarsen too fast, exactly the failure Heys *et al.* (2005)
identify for p/2 schemes, which retain only ~25% of points where AMG retains
~50%. Only the halving ladder is p-robust.

**Setup stays flat, 0.01 s → 0.04 s**, confirming the design prediction:
`DirectCoarse` assembles only `p_c = 2`, whose cost scales with the ELEMENT
count, not with p. The coarse solve at N=30 costs what it costs at N=5.

**Against [FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) §F2g this is the answer to high
order.** AMG degraded 2.16× over N=4…12 and both AmgX schemes stalled outright
from N=6–8. This holds to N=30. **Neither LOR nor AMG is needed here** — because
p-multigrid never assembles the fine operator, so the O(p^{2d}) density that
defeats AMG (§F2h(ii)) never arises.

Consistency checks: at N=5 `ladder=(2)` so ladder ≡ 2-level (71 = 71); at N=8
`ladder=(4,2)` so ladder ≡ 3-level (78 = 78).

**Scope.** One 2×2 mesh, Stokes (zero linearisation), manufactured RHS,
lid-driven-cavity masking. This measures the OPERATOR's conditioning, which is
the right thing for a preconditioner study, but it is not a flow computation.
**h-independence at high p is untested here** (F2 tested it only for AMG at N=4),
and solve wall times were not recorded — only iteration counts, which per §F2f
are the quantity that transfers.

### 6.7 Ghia Re=1000 cavity — p-independence survives real convection

§6.6 is an operator study: Stokes, zero linearisation, manufactured RHS. This is
the real thing. **Ghia, Ghia & Shin (1982) lid-driven cavity, Re = 1000**, 1×1,
moving lid, 4×4 elements, `dt=1`, `pin_p=True`, steady tol 1e−8 — with genuine
nonlinear convection, so `fu`/`fv` change every Newton step and **the
preconditioner is rebuilt every step**. That per-step rebuild is a cost §6.6
never paid. `scratch/pmg_ghia_cavity.py`.

| N | gDOF | Jacobi | 2-lvl | 3-lvl | **ladder** | RMS vs Ghia |
|---|---|---|---|---|---|---|
| 6 | 2500 | 294.2 | 28.3 | 19.5 | **22.8** | 2.4087e−01 |
| 8 | 4356 | 605.8 | 53.9 | 32.4 | **32.4** | 4.5740e−02 |
| 10 | 6724 | 733.2 | 60.5 | 35.7 | **31.1** | 3.0528e−02 |
| 12 | 9604 | 893.6 | 70.4 | 39.2 | **28.8** | 2.0159e−02 |
| 16 | 16900 | 1383.2 | 105.8 | 57.3 | **32.0** | 7.5927e−03 |
| **growth 6→16** | | **4.70×** | **3.74×** | **2.94×** | **1.40×** | |

*(CG iterations per Newton step.)*

**The §6.6 ordering holds under convection.** The ladder is flat — 22.8 → 32.0,
**1.40×**, essentially the 1.41× measured for Stokes — while Jacobi grows 4.70×
and the fixed hierarchies 3.74× / 2.94×. At N=16 the ladder needs **43× fewer
iterations than Jacobi** and 3.3× fewer than 2-level.

**Three correctness checks, all passed.**

1. **RMS is identical across all four preconditioners** at every N (to 4–5
   digits). They converge to the *same* answer — the preconditioner changes cost,
   not physics. This is the check that matters: a preconditioner which moves the
   converged state is a different solver, not a faster one.
2. **RMS matches an independent prior run** — 4.5740e−02 here against the stored
   4.5708e−02 at 4×4/N=8 in `cavity_ghia_res.npz`.
3. **`lad` ≡ `p3` at N=8** (32.4 both), since `ladder(8) = (4,2)`. And
   steps-to-steady barely move across preconditioners (172/172/165/171 at N=6),
   so outer Newton convergence is unaffected.

#### Wall time: the crossover is at N ≈ 10

| N | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|
| Jacobi | 9.5 s | 29.9 s | 39.8 s | 50.1 s | 135.1 s |
| **ladder** | 19.4 s | 35.1 s | 34.0 s | 37.6 s | **67.3 s** |
| **speedup** | **0.49×** | **0.85×** | 1.17× | 1.33× | **2.01×** |

**43× fewer iterations buys only 2.01× wall time**, and the ladder *loses* below
N ≈ 10. The gap widens with N (0.49 → 2.01×), so the crossover is real and the
ladder is right at high order, but **the iteration ratio badly overstates the
practical gain.**

> **Correction (§6.8).** This section originally attributed the modest wall gain
> to the per-step rebuild — *"~200 factorisations per run"*. **That diagnosis was
> wrong.** §6.8 measures the build cost directly at **24% / 14% / 10% / 8.4%** of
> wall for N = 8/16/24/30 — a *falling* share, because `DirectCoarse` scales with
> the element count, not with `p`. The real cost is that each ladder iteration is
> intrinsically expensive: several levels, each with Chebyshev applies.

**This is the strongest evidence in this document**, because unlike T1–T3 and
§6.6 it is a benchmarked flow with published reference data, an independent
prior result to check against, and a correctness check that all four
preconditioners agree.

### 6.8 Freezing the factorisation, and p to 30

§6.7 blamed the per-step rebuild for the modest wall-clock gain, so the obvious
remedy is to stop rebuilding. `refresh` controls it: `1` rebuilds every Newton
step, `k` every `k` steps, and *frozen* builds once. A frozen preconditioner is
built on a **snapshot** `SolverState`, not the live one — `apply_A` takes `fu`/`fv`
explicitly but reads `dfu_dx` from the *state*, so holding one while the other
keeps being re-linearised would give an operator frozen in half and live in the
other. CG only needs `M⁻¹` fixed *within* a solve, so refreshing between solves
is legitimate either way.

Ghia Re=1000, 4×4 elements (16), N = 8…30. CG/step — wall:

| N | gDOF | Jacobi | **ladder `r1`** | `r25` | **frozen** |
|---|---|---|---|---|---|
| 8 | 4356 | 605.8 — 30.3 s | 32.4 — 34.6 s | 37.7 — **30.8 s** | 51.1 — 40.7 s |
| 16 | 16900 | 1383.2 — 146.0 s | **31.9 — 69.4 s** | 40.2 — 75.7 s | 54.5 — 105.0 s |
| 24 | 37636 | 2422.9 — 442.5 s | **34.1 — 159.9 s** | 45.9 — 180.9 s | 62.0 — 234.6 s |
| 30 | 58564 | 3186.1 — 735.6 s | **37.1 — 245.0 s** | 49.7 — 267.4 s | 67.3 — 327.8 s |
| **growth 8→30** | | **5.26×** | **1.14×** | | |

#### Freezing is a consistent loss

**Rebuilding every step wins at every order except N=8.** Freezing raises
iterations **+58% / +71% / +82% / +81%**, which costs far more than the
factorisations save. Frozen is **1.18× to 1.51× slower** than `r1`.

**And it refutes §6.7's diagnosis.** The build cost is:

| N | 8 | 16 | 24 | 30 |
|---|---|---|---|---|
| build | 8.2 s | 9.8 s | 16.3 s | 20.5 s |
| **share of wall** | **24%** | **14%** | **10%** | **8.4%** |

**The share falls with order**, because `DirectCoarse` scales with the *element*
count (fixed at 16), not with `p`, while the solve grows with `p`. So the rebuild
was never dominant, and freezing helps least exactly where the ladder matters
most. The real cost is the intrinsic expense of a multilevel iteration.

#### The ladder is p-independent to N=30 on a real flow

**CG/step 32.4 → 31.9 → 34.1 → 37.1 across N = 8…30 — 1.14× growth** against
Jacobi's 5.26%, on a 58,564-DOF Re=1000 cavity with convection. At N=30 that is
**86× fewer iterations and 3.00× less wall**, and **the wall speedup keeps
growing**: 0.88× → 2.10× → 2.77× → 3.00×.

#### How the counts scale: `its ∝ N^a`

| | exponent `a` | R² | `its/N` |
|---|---|---|---|
| Jacobi — Ghia Re=1000 | **1.26** | 0.9990 | 75.7 → 106.2 |
| Jacobi — Stokes (§6.6) | **1.21** | 0.9974 | 95.0 → 145.5 |
| ladder — Ghia | **0.09** | 0.61 | — |
| ladder — Stokes | **0.17** | 0.82 | — |

**Jacobi is ~N^1.25 — mildly superlinear, not linear.** "Linear in N" is a fair
rule of thumb but under-predicts: over N = 8…30 it gives 3.75× where the measured
growth is 5.26×, a 40% underestimate. The fit is tight (R² = 0.999). The two
datasets agree to within 0.05 in the exponent despite being different physics on
different meshes (Stokes/4 elements vs Re=1000/16), which makes the
characterisation robust. A naive SEM argument would give `√cond ~ N^1.5`; the
measurement sits below that.

**For the ladder the exponent is ~0.1, and the low R² is itself the evidence** —
a power law barely fits because the data is constant with scatter. `its/N` *falls*
3–4× across the range, which is what a constant over a growing N does.

**The clean statement: Jacobi grows like N^1.25; the ladder does not grow.**

#### Cost per iteration — the deepening ladder is nearly free

A flat iteration count is not by itself flat *work*: the ladder gets deeper as
`p` grows (3 levels at N=8, 5 at N=30), so each iteration does more. Measured:

| N | gDOF | levels | CG/step | ms/iter | ms/iter per MDOF |
|---|---|---|---|---|---|
| 8 | 4356 | 3 | 32.4 | 5.2 | 1190 |
| 16 | 16900 | 4 | 31.9 | 12.8 | 757 |
| 24 | 37636 | 5 | 34.1 | 28.2 | 751 |
| 30 | 58564 | 5 | 37.1 | 40.3 | **688** |

Over N = 8 → 30: **13.4× the DOF, 1.67× the levels, 1.15× the iterations, 7.77×
the per-iteration cost, 7.08× the wall.** The extra levels cost essentially
nothing — coarse levels are small — and the per-iteration cost *per DOF* falls.

Flat iterations × sublinear per-iteration cost puts total wall at
**~DOF^0.75** over this range, i.e. close to optimal complexity for a system
whose condition number squares.

> **Read the sublinear exponent with suspicion.** Per-iteration cost measures
> **DOF^0.80**, which is *below* work-proportional — and a matvec-bound method
> cannot genuinely beat O(DOF), since every DOF must be touched. The likely cause
> is that the small cases are **numpy/Python overhead-bound**: at N=8 the arrays
> are small enough that per-`einsum` overhead dominates, and it amortises by
> N=30. A compiled implementation should bring the exponent back to ~1.0, giving
> total wall ~DOF^1.0 — still good, but not 0.75. **Quote "flat iterations"
> confidently; treat the sublinear wall scaling as provisional.**

#### What would break this: `h`, not `p`

Every p-independence measurement here — §6.6, §6.7, §6.8 — is at a **fixed
element count** (4 or 16). Under `h`-refinement two things move against the
method at once:

1. **The coarsest problem grows.** `DirectCoarse` scales with the element count,
   so the direct solve stops being free — §6.8's build share (24% → 8.4%) falls
   with `p` but would *rise* with `h`.
2. **The coarse space shrinks relative to the fine space.** p-multigrid coarsens
   in `p` only; it does nothing about `h`.

Classical theory then wants an **`h`-cycle underneath the p-ladder** — which is
exactly where AMG on the `p_c = 2` operator returns, as Pazner's coarse solver
`R₀` does (§F2h(ii)). That operator is low-order and sparse, so the O(p^{2d})
density that defeats AMG at high order does not arise there.

#### Accuracy: both profiles, and a floor

RMS against Ghia Tables I and II, **identical across all four configurations at
each N** (4–5 digits) — freezing changes the iteration count, not the answer:

| N | 8 | 16 | 24 | 30 |
|---|---|---|---|---|
| `rms_u` (x=0.5) | 4.5740e−02 | 7.5928e−03 | 3.3045e−03 | **3.2096e−03** |
| `rms_v` (y=0.5) | 6.6528e−02 | 7.5242e−03 | 5.0727e−03 | **6.6584e−03** |

**Convergence stops after N=24.** `rms_u` moves only 3.30e−03 → 3.21e−03 and
`rms_v` gets *worse*, 5.07e−03 → 6.66e−03. That is an accuracy **floor**, not
p-convergence, and it is **not the solver** — all four configurations agree to
five digits. Candidates: the 16-element mesh, the `dt=1` steady weighting, or the
`1e-8` steady tolerance. **These should not be quoted as benchmark agreement
until that is run down.**

*(Ghia Table I was verified against the repo's stored `cavity_re1000_data.npz`
— max difference 5e−05, its 4-dp rounding. Table II is transcribed from the same
source and has no independent check here.)*

---

## 8. Conclusions

Two separate findings, and the second is the larger one.

### 8.1 Depth is what makes p-multigrid work at high order

**A halving ladder with a direct coarse solve is p-independent to N = 30**
(§6.6): 71 → 100 iterations across a 6× increase in order and 31× in DOF, i.e.
**1.41×** growth. Fixed hierarchies are not — 2-level grows **7.90×** and
3-level **5.56×**, against Jacobi's 9.2×. **Depth, not the coarse solver, is what
separates a p-robust method from a constant-factor one.**

This resolves the high-order problem [FOSLS_2D_PLAN.md](./FOSLS_2D_PLAN.md) §F2g
opened. AMG degraded 2.16× over N=4…12 and *both* AmgX schemes stalled outright
from N=6–8; this holds to N=30. The reason is structural rather than
parametric: **p-multigrid never assembles the fine operator**, so the O(p^{2d})
density that defeats AMG (§F2h(ii)) never arises. Only `p_c = 2` is assembled,
and its cost scales with elements, not order — setup is flat at 0.01→0.04 s.

**It also makes LOR unnecessary here**, which is stronger than the three earlier
refutations in §F2/F2e/F2g: those established LOR was not *needed for
convergence*; this exhibits a method that is p-robust without it.

**Confirmed on a real flow.** §6.7 repeats this on the **Ghia Re=1000 cavity**
with genuine nonlinear convection and the preconditioner rebuilt every Newton
step: the ladder grows **1.40×** over N=6…16 against Jacobi's 4.70×, essentially
the Stokes result. All four preconditioners converge to the same answer (RMS
identical to 4–5 digits), and the RMS matches an independent prior run.

**The wall gain is smaller than the iteration gain, but it grows.** §6.8 to
N=30: **86× fewer iterations, 3.00× less wall**, with the speedup rising 0.88× →
2.10× → 2.77× → 3.00×. Use a ladder above N ≈ 10; below it Jacobi is cheaper
despite needing 20× the iterations.

**Freezing the coarse factorisation does not help** (§6.8) — it is 1.18–1.51×
*slower*, because it costs +58–82% iterations to save a build that is only
8–24% of wall and *falling* with `p`.

**Still untested: the h × p cross.** §6.6 is one 2×2 mesh and §6.7 one 4×4;
§F2 tested `h` only for AMG at N=4. Whether flat-in-p survives mesh refinement is
the obvious next measurement.

### 8.2 The direct coarse solve

**It is the better choice on every axis measured, and the evidence is
consistent across two problems and three independent measures.**

| | Poiseuille (T1/T2) | Gartling BFS (T3) |
|---|---|---|
| CG iterations vs Chebyshev coarse | 1.25–1.40× fewer | — |
| path dependence | **117×** less than Jacobi | **17×** less drift |
| benchmark physics | — | reattachment spread 3.0% → **1.3%** |
| wall time | ~2× | **2.6×** |

The ordering is monotone and in the predicted direction every time — Jacobi
worst, Chebyshev coarse in between, direct best — and the wall-time advantage
*grows* with problem size (2× at 3,888 DOF, 2.6× at 9,048), because the coarse
problem stays fixed while the fine solve gets harder.

**Three qualifications bound the claim.**

1. **It slows the symptom, not the cause.** T3's drift falls 17× but never
   reaches zero; the iteration still does not converge. The soft direction is a
   property of the **formulation**, and no coarse solver removes it. This is the
   same relationship `ROW7_WEIGHT` has to the ω cluster in 3D — attack the
   symptom pointwise, buy a large factor, leave the cause standing.
2. **It is not a p-robustness fix.** The advantage *shrinks* with order (1.40× at
   N=6 → 1.25× at N=12), and `PMG2` remains a flat ~11–12× constant-factor
   preconditioner rather than a p-independent one (§6.2).
3. **Assembly is `O(elements)`** — one `apply_A` per free coarse DOF, per
   linearisation. Trivial at 202–828 coarse DOF (0.03 s), but it scales with the
   **mesh**, not with `p`. This is the one thing blocking it as a universal
   default.

**Scope.** Every case tested had a Dong outflow. The generic iteration-count gain
should transfer anywhere, but the large wins — 117× spread, 17× drift, the
reattachment result — are all soft-outflow-mode effects. On closed-boundary
problems it is untested, and less should be expected.

**Recommendation.** Make `coarse_solver='direct'` the default **for open-outflow
cases** once T4 confirms it under genuine backflow, and convert the assembly to
element-local probing (as `scratch/fosls_assemble.py` does, `O(1)` in mesh size)
before using it on large meshes. Keep `Chebyshev4` as the fallback where
assembly cost would dominate.

**And use a halving ladder, not a fixed depth, whenever `N > 8`** (§8.1). At
N=8 the ladder and the 3-level hierarchy coincide; above it they diverge sharply
— by N=30 the ladder needs 100 iterations against the 3-level's 300.
