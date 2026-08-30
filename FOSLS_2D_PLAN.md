# Testing McCormick's FOSLS claims on `lssem2d` — review and plan

Review of `Steve McCormick's  VVP Least Squares.md`, and a staged plan to test its
three claims against the 2D solver.

---

## 0. Conclusion

**FOSLS theory applies to this solver — established by measurement, and it made a
correct quantitative prediction before the fact.** F1 measured the ellipticity
constant at $1.55\times10^4$, predicting $\sqrt{c_2/c_1}\approx124$ preconditioned
iterations; F2 then measured AMG at 113–138, flat across a 16× increase in
elements. Two independent routes, one number.

| claim | verdict |
|---|---|
| **C1** functional is $H^1$-norm-equivalent | **CONFIRMED** — $c_2=1.000$, $c_2/c_1$ saturates at 1.55e4 (steps 1.16 → 1.03 → **1.01**) |
| **C3** $h$-independent convergence | **CONFIRMED** — AMG flat at 113→131 while Jacobi grows 3.1×; ratio widens 5.9× → 16.1× |
| **C2** weak BCs in the functional | **UNTESTED** (F3) |

**But three limits bound what it is worth today.**

**Steady only.** At production $\Delta t$ the mass term dominates, the problem is
ill-conditioned regardless of preconditioner, and Jacobi is near-optimal. AMG has
nothing to offer the time-stepper.

**CG is carrying the multigrid.** The standalone V-cycle factor is $\rho\approx0.97$
against a textbook 0.1–0.3. "$h$-independent" is supported; "works well" is not.

**Wall time is not yet a win** — 0.85× at 16×16 even with 16× fewer iterations —
though both sides of that comparison are Python and the iteration counts are what
transfer.

### The finding that outlived the question

The softest mode carries **97.7% of its energy in $\omega$** — the 2D analogue of
§7J's 3D cluster, which carried 100% in $(\omega_x,\omega_y)$. That single
structure explains three separate results: why the redundant $\nabla\cdot\omega$
row was ruinous in 3D, why `ROW7_WEIGHT = 1e-4` buys 14.6×, and why AMG stalls at
$\rho\approx0.97$ with a constants-only coarse space. Supplying those modes takes
CG from 138 to 52.

**`ROW7_WEIGHT` treats the symptom pointwise; a coarse space containing the cluster
treats the cause.** That reframing — not the preconditioner — is the most useful
thing this study produced, and it points at deflation or a formulation without the
cluster rather than at more preconditioner tuning.

### New capability, obtained as a side-effect

The least-squares functional $J$ — already computed, never used — is a **validated
error estimator** (F4′: effectivity constant to 1.40× across six orders of error;
$J$ rises $8.6\times10^9$ on the `minchan_001` defect). It works where there is no
exact solution, which is where this project has no accuracy measure at all. **Free,
and immediately usable.**

### Refuted — including three of my own predictions

* **F1b (rescale $\omega$)** — $c_2/c_1$ is *invariant* under change of variables.
* **"accuracy vs ellipticity conflict"** — a comparison made in a limit production
  never visits; the code already chooses the right weighting in each regime.
* **"AMG must go on a low-order refined operator"** — smoothed aggregation handled
  the dense-block SEM matrix directly.

**Scope, honestly:** 2D, steady, no convection, one geometry, $\nu \ge 10^{-3}$,
and $c_2/c_1 = 1.55\times10^4$ — bounded, but far from the O(1) a textbook FOSLS
achieves, degrading as $\nu^{-2}$.

---

## 1. What the document claims, and what it omits

The summary is a fair statement of the FOSLS framework (Cai–Manteuffel–McCormick).
Three claims are load-bearing:

| # | claim | testable here? |
|---|---|---|
| **C1** | The functional is SPD and **$H^1$-norm-equivalent**, hence uniformly elliptic | yes — measurable |
| **C2** | Boundary conditions belong **in the functional**, weakly, to keep coercivity | yes — we currently do the opposite |
| **C3** | AMG then gives **$h$-independent** convergence | yes — and we already have contrary evidence |

**Four things the document omits that decide whether any of this transfers.**

**Norm equivalence is a statement about *scaling*, not just about form.** The theory
requires constants $0 < c_1 \le c_2$ with

$$
c_1 \,\|\mathbf{Q}\|_{1,\Omega}^2 \;\le\; \mathcal{F}(\mathbf{Q};\mathbf{0}) \;\le\; c_2 \,\|\mathbf{Q}\|_{1,\Omega}^2 ,
$$

and everything downstream — ellipticity, AMG's $h$-independence, the error bound —
follows from $c_2/c_1$ being $O(1)$. **Arbitrary row weights destroy it.** Our
functional is
$\mathcal{F} = \sum_r \rho_r \|R_r\|^2_{0,\Omega}$ with $\rho_r$ chosen for
*time-stepping* reasons, not for norm equivalence (§5.1). That is the first thing
to measure and the document never raises it.

**The 3D VVP system needs the redundant $\nabla\cdot\boldsymbol{\omega} = 0$ row
for ellipticity — and we down-weight it by $10^{-4}$.** §7J measured that
down-weighting as worth **14.6×** in Jacobi iterations at production mesh size.
If FOSLS theory is right, we have bought Jacobi conditioning by *breaking the
very ellipticity AMG would need*. **In 2D this tension does not exist** — $\omega$
is a scalar, $\nabla\cdot\boldsymbol{\omega}$ is not a thing — which is precisely
why 2D is the clean testbed and why testing here first is the right call.

**"Elliptic" is not our operator.** FOSLS $h$-independence is proved for the
*steady* elliptic system. Ours carries a mass term $a_{\text{mass}}\,\mathbf{u}$
from the implicit time step, making it reaction-dominated at production
$\Delta t$ ($a_{\text{mass}} \sim 10^3$). That *helps* conditioning — and it also
means an $h$-independence result obtained at large $a_{\text{mass}}$ proves
nothing about the elliptic limit. Any test must sweep $a_{\text{mass}}$.

**AMG needs a *sparse* matrix, and the SEM operator is not one.** The document
says AMG applies "naturally" to the FOSLS stiffness matrix. For a *low-order* FOSLS
discretisation it does. For a **spectral element** discretisation it does not: the
element block is dense ($324\times324$ at $N=8$), which is exactly what AMG cannot
coarsen. The fix is low-order refined preconditioning (F2) — standard practice for
high-order methods, and the single most important thing the document omits for our
setting.

---

## 2. The central prediction — which our data currently contradicts

C3 says multigrid convergence should be **bounded away from 1 independently of
$h$**. §7K measured p-multigrid on this operator:

| $N$ | Jacobi its | PMG its | ratio | wall |
|---|---|---|---|---|
| 8 | 755 | 102 | 7.4× | 0.28× |
| 12 | 1183 | 159 | 7.4× | 0.43× |
| 16 | 1580 | 217 | **7.3×** | 0.48× |

**The ratio is pinned at 7.3–7.4× while both counts grow.** That is a
constant-factor preconditioner, *not* a mesh-independent one — textbook multigrid
would hold PMG's column flat. And the 3D $h$-scan showed the same: 1118 → 2297
iterations over a 9× element increase.

So either FOSLS theory does not apply to our functional, or our multigrid is the
wrong kind. **Three candidate explanations, and the plan is built to separate
them:**

| | hypothesis | discriminator |
|---|---|---|
| **H1** | the functional is **not norm-equivalent** (row weights break it) | measure $c_2/c_1$ directly |
| **H2** | **p-MG ≠ AMG** — algebraic coarsening succeeds where polynomial coarsening failed | run AMG on the assembled $A$ |
| **H3** | **strong BC masking** destroys the coercivity FOSLS gets from weak BCs | compare masked vs functional BCs |

---

## 3. Plan

### F0 — Assemble $A$ (½ day, low risk, unlocks everything)

`lssem2d` is matrix-free; AMG is not. Build
$A = M Q^{T} Q\, L_0^{T} (\rho W) L_0 M$ as a `scipy.sparse` matrix by
element-wise assembly, **not** by probing (probing is $O(n^2)$ matvecs and was
how `spectrum.py` got expensive).

**Gate:** $\|A x - \texttt{apply\_LT}(\texttt{apply\_L}(x))\|/\|Ax\| < 10^{-12}$
on random continuous $x$ — the assembled matrix must *be* the matrix-free
operator, not a lookalike. Also assert symmetry to $10^{-14}$ and positive
definiteness of the free block.

*This gate matters more than it looks: §2.1 of 3D_STATUS records four missing
factors in $A$ that survived a full suite of symmetry and convergence tests.*

### F1 — Measure norm equivalence (½ day) — tests **H1**

Compute the generalised eigenvalues of

$$
\mathcal{F}(\mathbf{Q}) = \mathbf{Q}^{T} A\, \mathbf{Q}
\quad\text{against}\quad
\|\mathbf{Q}\|_{1}^2 = \mathbf{Q}^{T} H \,\mathbf{Q},
\qquad H = \text{block-diag}(K + M),
$$

with $K$ the SEM stiffness and $M$ the mass matrix. Then
$c_1 = \lambda_{\min}$, $c_2 = \lambda_{\max}$, and **$c_2/c_1$ is the FOSLS
ellipticity constant** — the quantity the whole theory rests on.

Sweep: `a_flux`/`a_mass` weightings (legacy vs `w_mom=1`), $a_{\text{mass}}$ from
the steady limit to production, and $N \in \{4, 6, 8\}$.

**Gate:** if $c_2/c_1$ is $O(1)$ and $h$-independent, the functional *is*
norm-equivalent and H1 is refuted — then C3 should hold and any failure is the
solver's. If $c_2/c_1$ grows with $h$ or $N$, **H1 is confirmed and no
preconditioner can deliver $h$-independence** until the weights are fixed.

*This single measurement decides whether the rest of the plan is about the
preconditioner or about the formulation.* Run it first.

### F2 — AMG on a **low-order refined** operator (1 day) — tests **H2**

**Not on the SEM matrix.** The high-order element block is *dense* — $324\times324$
at $N=8$ — and AMG's whole premise is exploiting sparsity and strong connections
in the matrix graph. In a dense block every DOF is strongly connected to every
other, so there is nothing to coarsen and the setup cost is quadratic in the
block size:

| $N$ | SEM block | LOR nnz/elem | sparser |
|---|---|---|---|
| 4 | $100^2 = 10{,}000$ | 900 | 11× |
| **8** | $\mathbf{324^2 = 104{,}976}$ | **2,916** | **36×** |
| 16 | $1156^2 = 1{,}336{,}336$ | 10,404 | **128×** |

The standard remedy is **low-order refined (LOR) preconditioning**, going back to
Orszag (1980) and used by Nek5000, MFEM and libParanumal: build the $Q_1$ FEM
operator on the *same GLL nodes*, treating each GLL cell as a bilinear quad. That
matrix $A_{\text{LOR}}$ is sparse (9-point stencil), shares the DOF set with
$A_{\text{SEM}}$, and — for an $H^1$-elliptic operator — is **spectrally
equivalent with a constant independent of $N$** ($\approx \pi^2/4$ for the
Laplacian). AMG then coarsens $A_{\text{LOR}}$, and the result preconditions
$A_{\text{SEM}}$.

**So there are three matrices, and only the middle one is ever handed to AMG:**

| | what | assembled? | used for |
|---|---|---|---|
| $A_{\text{SEM}}$ | the actual operator | F0, small cases only | verification, spectra (F1) |
| $A_{\text{LOR}}$ | $Q_1$ on GLL nodes | **yes, sparse** | **AMG input** |
| coarse levels | AMG's own hierarchy | by pyamg | the V-cycle |

Production stays **matrix-free** for $A_{\text{SEM}}$; only the preconditioner is
assembled.

**F1 is a precondition, not merely an input.** The LOR–SEM equivalence is an
$H^1$ result. It extends to a FOSLS system *if and only if* the functional is
$H^1$-norm-equivalent — which is exactly what F1 measures. **If F1 fails, F2's
whole approach loses its basis**, not just its expected performance. The chain is
C1 $\Rightarrow$ LOR-AMG viable $\Rightarrow$ C3.

Then, in order:

1. **Smoothed aggregation, block size 4** (`pyamg.smoothed_aggregation_solver`),
   since $A_{\text{LOR}}$ is a coupled 4-field system and scalar AMG has no notion
   of that.
2. **Supply the near-null space $B$ explicitly** rather than letting pyamg guess
   constants. §7J found the softest 3D modes were $(\omega_x,\omega_y)$-dominated;
   extract the 2D analogue with `eigsh` and hand it over.
3. **Expect anisotropy trouble.** GLL nodes cluster at element edges with spacing
   $O(1/N^2)$, so the LOR mesh is severely stretched there — a known difficulty
   for AMG coarsening, and the most likely reason a first attempt underperforms.

**Gate — stated in advance:** iterations **flat within 20%** across a 4×
$h$-refinement at fixed $N$. A constant-factor reduction is the result §7K already
rejected and must be recorded as such, not reported as success.

### F3 — Weak boundary conditions (1 day) — tests **H3**

Currently `bc.apply_mask` imposes BCs **strongly** by zeroing rows. FOSLS puts
them in the functional:

$$
\mathcal{F} = \|\mathcal{L}_0\mathbf{Q}-\mathbf{F}\|_{0,\Omega}^2
            \;+\; \beta\,\|\mathcal{B}\mathbf{Q}-\mathbf{g}\|_{0,\partial\Omega}^2 .
$$

Add a boundary term with weight $\beta$ and compare against masking on the
**cavity** (essential BCs, singular corners) and the **BFS** (outflow — where
`obc.py` already exists and OUTFLOW_BC_STUDY.md records a real bug).

**Gate:** unchanged accuracy on Kovasznay and Poiseuille — both have exact
solutions — plus the ellipticity constant from F1 *improving*, which is the
mechanism the theory claims. Weak BCs that change the answer are a bug, not a
trade-off.

*Genuine upside here beyond conditioning:* weak outflow BCs are exactly what
`OUTFLOW_BC_STUDY.md` wants, and the FOSLS functional gives a principled way to
impose them without over-specifying characteristics.

### F4 — Decide (½ day)

| F1 result | F2 result | conclusion |
|---|---|---|
| $c_2/c_1$ $O(1)$ | AMG flat | **FOSLS works; port to 3D** — and re-examine `ROW7_WEIGHT` |
| $c_2/c_1$ $O(1)$ | AMG not flat | our AMG setup is wrong (near-null space?), not the theory |
| $c_2/c_1$ grows | — | **the weights break norm equivalence** — fix the functional before any preconditioner |

**Total ≈ 3 days.** F1 alone (1 day including F0) may settle it, and should be run
before committing to F2–F3.

---

## 4. What could sink it, and one thing that would be worth knowing anyway

**The row weights are load-bearing and were measured, not guessed.** §7F, §7J and
WEIGHT_VS_TIMESTEP_STUDY.md establish that the legacy weighting is required for
accuracy — Poiseuille is 1875× worse under the alternative. If F1 shows those
weights break norm equivalence, we have a genuine conflict between *accuracy* and
*ellipticity*, not a bug to fix. That would be the most interesting outcome and
the plan should not be written to avoid it.

**Convection makes the operator non-symmetric in the underlying PDE**, though the
normal equations remain SPD. FOSLS theory for convection-dominated problems is
weaker, and our production cases run at $Re$ up to 1000. Test at low $Re$ first,
then sweep.

**AMG setup cost may exceed the win.** §7K's PMG was rejected on wall time despite
7.4× fewer iterations. AMG's setup is *more* expensive than PMG's. The gate in F2
is deliberately about $h$-independence rather than wall time, because a
constant-factor win is already known not to pay — but if F2 passes, wall time
becomes the next question, not a settled one.

**Regardless of the outcome, F0 and F1 are worth having.** An assembled $A$ enables
direct solves for small cases, exact spectra, and a check on every future
preconditioner claim; the ellipticity constant is the invariant that four wrong
preconditioner conclusions in §7I–§7K were reached without.

---

## RESULTS — F0 and F1 (2026-08-24)

### F0 — assembled, and gated: **PASS**

| case | ndof | matvec vs `apply_A` | asymmetry | κ(A) |
|---|---|---|---|---|
| N=4, 2×2 | 324 | 2.21e−16 | 4.25e−17 | 5.50e+04 |
| N=6, 3×2 | 988 | 3.84e−16 | 9.05e−17 | 1.61e+05 |
| N=8, 3×3 | 2500 | 4.60e−16 | 9.69e−17 | 6.28e+05 |

`scratch/fosls_assemble.py`. Probed **element-locally** (324 probes at N=8, not
`nelem`×324) using the code's own `apply_L`/`apply_LT` — reimplementing $L_0$ is
the L1 trap that let four missing factors survive a full suite (§2.1).

Density falls 6.2% → 2.9% as the mesh grows: **sparse between elements, dense
within them** — the concrete reason AMG must go on the LOR operator.

### F1 — the ellipticity constant: **the functional IS norm-equivalent, but only when correctly scaled**

**The C3 test — $h$-refinement in the elliptic limit ($dt\to\infty$), `w_mom=1`:**

| mesh | $c_1$ | $c_2$ | $c_2/c_1$ | step |
|---|---|---|---|---|
| 1×1 | 8.49e−05 | 1.000 | 1.178e+04 | |
| 2×2 | 7.31e−05 | 1.000 | 1.369e+04 | 1.16× |
| 4×4 | 6.54e−05 | 1.000 | 1.530e+04 | 1.03× |
| **6×6** | 6.44e−05 | 1.000 | **1.552e+04** | **1.01×** |

**It saturates. The constant is bounded independently of $h$**, which is exactly
what FOSLS predicts — so C1 holds and C3 is live. $c_2 = 1.000$ exactly.

**The weighting decides everything, and only in the elliptic limit:**

| dt | $a_{\text{mass}}$ | legacy | `w_mom=1` |
|---|---|---|---|
| 1 | 1 | 8.89e+03 | 8.89e+03 *(identical — see below)* |
| 100 | 0.01 | 2.96e+06 | 1.33e+04 |
| 10⁴ | 10⁻⁴ | **2.00e+10** | **1.55e+04** |

*Both* saturate, so legacy is not non-equivalent in the technical sense — **its
constant is 1.3 million times larger.** Legacy sets $a_{\text{flux}} = \Delta t$,
scaling momentum by $\Delta t$ against unweighted continuity and vorticity, so the
functional stops being a balanced sum of residuals.

*Methodological note: the first version of this sweep ran at $dt = 1$, where the
two weightings **coincide** ($a_{\text{mass}} = a_{\text{flux}} = 1$) — it
reported three identical numbers as though they were a result. $\Delta t$ is what
separates them.*

**And the residual constant is a variable-scaling artefact:**

| ν | $c_2/c_1$ | vs $\nu^{-2}$ |
|---|---|---|
| 0.1 | 4.89e+02 | — |
| 0.01 | 1.37e+04 | 0.28× |
| 0.001 | 1.34e+06 | 0.27× |

$c_2/c_1 \propto \nu^{-2}$ (the ratio to the $\nu^{-2}$ extrapolation is constant
at 0.28). The mechanism is visible in the rows: momentum carries
$p_x + \nu\,\omega_y$ while the vorticity definition carries $\omega + u_y - v_x$,
so **the same variable $\omega$ enters one row $\nu$ times weaker than the
other** and no single $H^1$ norm bounds both tightly. Classical Stokes FOSLS
avoids this by rescaling the variables (Bochev & Gunzburger).

### What this means — and a correction

**RETRACTED: "the accuracy-vs-ellipticity conflict is real."** I compared legacy
against `w_mom=1` in the *elliptic limit* and concluded they were irreconcilable.
Production never visits that limit. Sweeping $\Delta t$ shows a **crossover at
$\Delta t \approx 0.03$**:

| dt | regime | legacy | `w_mom=1` | winner |
|---|---|---|---|---|
| 1e−3 | **3D production** | 7.54e+06 | 2.04e+09 | **legacy, 270×** |
| 3e−2 | — | 7.39e+04 | 8.33e+04 | legacy, 1.1× |
| 1e−1 | — | 6.15e+04 | 2.09e+04 | `w_mom=1`, 2.9× |
| 1e+4 | steady (Kovasznay) | 1.94e+10 | 1.37e+04 | `w_mom=1`, 1.4e6× |

**Each weighting is optimal in its own regime, and the code already chooses
correctly** — Kovasznay runs `dt=1e30, w_mom=1`; unsteady runs legacy. §7F's
accuracy finding and F1's ellipticity finding are about *different $\Delta t$
regimes* and are consistent. There was never a conflict; there was a comparison
made in the wrong limit.

**A consistency check falls out of it.** At production $\Delta t = 10^{-3}$ the
constant is $7.5\times10^6$, so $\sqrt{c_2/c_1} \approx 2740$ predicted CG
iterations — against the **~4000 measured** in the 3D minimal channel. The
ellipticity constant predicts the observed iteration count to within a factor
of 1.5, which is the first independent check that any of this describes the real
solver.

**And it scopes F2 sharply.** At small $\Delta t$ the mass term dominates, the
problem is ill-conditioned *regardless of preconditioner*, and Jacobi is close to
optimal. **AMG has little to offer the time-stepper.** Its payoff is confined to
the STEADY solver — Kovasznay, the steady cavity, BFS, Gartling — which is
already where `w_mom=1` is used and where the ceiling is flat at 1.55e4.

### Revised next steps

| | | |
|---|---|---|
| ~~**F1b**~~ | ~~rescale $\omega$~~ — **REFUTED**: $c_2/c_1$ is *invariant* under variable rescaling, since $q = D\tilde q$ maps $Aq=\lambda Hq$ to $(DAD)\tilde q = \lambda (DHD)\tilde q$. Measured: 1.368939e+04 for scalings of 100, 0.01 and 10 — ratio exactly 1.000. Only a change to the **rows** (which alters the answer) or to the first-order **system** can move it | ½ day saved |
| **F2** | LOR-AMG — now scoped to **steady** problems only, at $\ge 6\times6$ elements (they cross at $4\times4$; below that Jacobi wins and the test would mislead) | ceiling $\sqrt{1.55\times10^4} \approx 124$ iterations flat, vs Jacobi unbounded |
| **F3** | weak BCs | unchanged |
| **F4′** | **the functional as an error monitor** — the consequence of ellipticity that costs nothing and the project has never exploited | see below |

### F4' — the cheapest consequence, and the one this week argued for

Ellipticity means $c_1\|e\|_1^2 \le J(Q) \le c_2\|e\|_1^2$: the functional
**already being minimised** is a computable two-sided bound on the error, needing
no exact solution and no auxiliary problem.

The project has no a-posteriori error estimate; accuracy is measured only where an
analytic answer exists. `minchan_001` ran 14 h looking healthy on every logged
diagnostic while carrying an 11% divergence defect — and $J$ aggregates *every*
row residual, so it would have been visibly elevated from the first step. L16
asked for "a conserved quantity and a validated reference"; a computable error
bound is stronger than both and works on the cavity, the BFS and the turbulent
channel, none of which have exact solutions.

Caveat: the bound is loose by $\sqrt{c_2/c_1}$ — ~124× at $\nu=10^{-2}$, ~1160×
at $10^{-3}$ — so it is an honest *monitor* and a poor *quantitative* estimate.
As a relative instrument across runs of one configuration, it needs no constants
at all.

F2's gate is now *quantitative* rather than qualitative: an $H^1$-optimal
preconditioner should give $O(\sqrt{c_2/c_1})$ iterations, so ~124 at ν=0.01 and
flat under refinement. Anything far above that indicts the AMG setup; anything
that *grows* indicts the LOR equivalence.

---

## F4′ RESULTS — the functional as an error estimator: **PASS**

`scratch/fosls_estimator.py`. Kovasznay, Re=40, 4×4 elements, steady
(`dt=1e30`, `w_mom=1` — the regime F1 measured).

### The success measure, and why "J correlates with error" is not it

Any monotone function correlates. The theory gives the sharp test: the
**effectivity index** $\theta = \sqrt{J(Q)}\,/\,\|e\|_1$ must satisfy
$\sqrt{c_1} \le \theta \le \sqrt{c_2}$. Three gates:

| | gate | why |
|---|---|---|
| **G1** | $\theta \in [\sqrt{c_1},\sqrt{c_2}]$ | theory check — if it escapes, F1's constants or the functional are wrong |
| **G2** | $\theta \to$ const under refinement, **< 2× spread** | **the one that matters**: a drifting effectivity cannot decide anything — a falling $J$ would not distinguish a better solution from a coarser yardstick |
| **G3** | $J$ **rises** with an injected defect | the `minchan_001` test — an estimator blind to the failure we actually suffered is useless however elegant |

### Results

| N | $J$ | $\|e\|_1$ | $\theta$ | eps_rms |
|---|---|---|---|---|
| 4 | 4.520e−05 | 1.819e−01 | **0.0370** | 9.72e−05 |
| 6 | 1.585e−09 | 8.711e−04 | **0.0457** | 1.60e−07 |
| 8 | 1.363e−14 | 2.261e−06 | **0.0516** | 2.49e−10 |
| 10 | 5.652e−19 | 1.201e−07 | 0.0063 | 1.09e−10 |
| 12 | 5.817e−19 | 1.519e−07 | 0.0050 | 1.22e−10 |

**G2 PASS** — $\theta$ = 0.0370, 0.0457, 0.0516, spread **1.40×** over three
resolutions spanning six orders of magnitude in error.
**G1 PASS** — all inside $[0.0080, 1.0]$.
**G3 PASS, decisively** — injecting non-solenoidal noise at amplitude $10^{-4}$
raises $J$ by **8.6 × 10⁹**, and $J$ scales exactly as amp² (a squared norm, as
the theory requires): 1.17e−4, 1.17e−2, 1.17e0, 1.17e2 across four decades.

### Three methodological traps, all of which produced a false FAIL first

**The pressure gauge.** $p$ is determined only up to a constant, so the raw
difference carries an O(1) offset. Measured: $\|e\|_1$ pinned at **0.993 for
every N** while the velocity error fell to 1e−10 — the norm was reporting the
constant, not the solution. `kov.py` already subtracts the mean; this now does too.

**The gate read past the round-off floor.** Beyond N=8 this case is converged to
machine precision: $J \approx 5.7\times10^{-19}$, i.e. zero, and `eps_rms` stops
improving. $\theta$ there is a ratio of two noise levels. Taking "the last three
N" — the obvious choice — selects **exactly** the meaningless points and reported
an 8.25× spread.

**The convergence detector on the wrong quantity.** Using $\|e\|_1$ admitted N=10,
where it still fell 19× while `eps_rms` moved only 2.3×. Detecting on `eps_rms`
— the driver's own metric — gives the clean regime.

### What this is now worth

$J$ is **already computed** by `kov.residual()` and never used. It gives a
computable error bound on problems with **no exact solution** — the cavity, the
BFS, the turbulent channel — which is precisely where this project currently has
no accuracy measure at all.

The effectivity 0.037–0.052 means $\|e\|_1 \approx \sqrt{J}/0.045$, so **a single
free number estimates the error to within ~40%** in this configuration. That is
far sharper than the $\sqrt{c_2/c_1} \approx 124$ worst case, because $\theta$ sits
well inside its bounds rather than at an extreme.

**Caveat, and it is the important one:** $\theta$ is constant *for one
configuration under p-refinement*. It has **not** been shown constant across ν,
across geometries, or under h-refinement, and F1 says the bounds themselves
degrade as $\nu^{-2}$. So $J$ is established as a **relative monitor within a
run** — which needs no constants at all and is what `minchan_001` lacked — and
not yet as a portable quantitative estimator.

---

## F2 RESULTS — AMG gives h-independent iterations: **GATE PASSED**

`scratch/fosls_amg.py`. Steady (`dt=1e4`, `w_mom=1`), ν=1/100, N=4.

### h-refinement — the gate

| mesh | ndof | Jacobi | **AMG** | ratio | wall (incl. setup) |
|---|---|---|---|---|---|
| 4×4 | 1027 | 671 | **113** | 5.9× | 0.62× |
| 8×8 | 4099 | 1245 | **138** | 9.0× | 0.62× |
| 12×12 | 9219 | 1611 | **137** | 11.8× | 0.68× |
| 16×16 | 16387 | 2103 | **131** | 16.1× | 0.85× |

**AMG iterations are flat — 113, 138, 137, 131 over a 16× increase in elements**
(1.16× spread, gate was <1.2×). Jacobi grows 3.1× over the same range and the
ratio widens monotonically, 5.9× → 16.1×.

**Two independent routes agree.** F1 predicted the ceiling at
$\sqrt{c_2/c_1} = \sqrt{1.55\times10^4} \approx 124$ iterations. AMG measures
113–138. The ellipticity constant predicted the preconditioned iteration count
before it was run.

### The near-null space was the whole game

| variant | 4×4 | 6×6 | 8×8 | 12×12 | growth |
|---|---|---|---|---|---|
| Jacobi | 671 | 1033 | 1245 | 1611 | 2.40× |
| scalar `B` (pyamg default) | 153 | 198 | 249 | 349 | 2.28× |
| 4-field `B` | 124 | 169 | 144 | 173 | 1.40× |
| **4-field `B` + energy prolongation** | **113** | **142** | **138** | **137** | **1.21×** |

The default single all-ones vector cannot represent a mode constant in one field
and zero in the others, so it coarsens a 4-field system as though it were scalar
— and grows almost exactly like Jacobi (2.28× vs 2.40×). **Supplying one constant
per field is what converts a constant-factor preconditioner into an
$h$-independent one**, which was item 2 on F2's list and turns out to be the only
item that mattered.

### **LOR was not needed** — a prediction of this plan, refuted

§F2 argued at length that AMG *must* go on a low-order refined operator because
SEM element blocks are dense (324² at N=8), leaving AMG "nothing to coarsen".
**Smoothed aggregation handled the SEM matrix directly.** The dense-block concern
was real but not decisive: aggregation works on the strength-of-connection graph,
and a dense block still has structure once the near-null space identifies the
fields. LOR remains untested and may yet matter at higher order — the density
argument bites hardest at N ≥ 12 — but it is not required for the result.

### Wall time: **not yet a win, and the measurement is not representative**

0.62× → 0.85×, trending up but still below 1 at 16×16 even with 16× fewer
iterations: an AMG iteration costs ~19× a Jacobi one here. Under $p$-refinement
it breaks even (1.01× at N=8, 1.05× at N=10).

This is the §7K pattern — iterations vindicated, wall time not — but with a
caveat §7K also carried: **both sides are Python**. scipy CG with a Python
callback against pyamg's Python V-cycle, while production runs a fused
numba/CUDA matvec. **The iteration counts transfer; the wall times do not.** The
ratio widens 5.9× → 16.1× over the range tested, so the wall-clock crossover lies
beyond 16×16 on this harness and somewhere different in production.

### Verdict

**C3 is confirmed, not merely available.** The remaining question is engineering
— whether a compiled V-cycle beats a compiled Jacobi at production mesh sizes —
not whether the mathematics works. And the scope limit from F1 stands: this is
the **steady** solver only. At production $\Delta t$ the mass term dominates and
Jacobi is near-optimal.

### F2e — every F2 number above was measured at **N=4**; here it is at N=8

A fair question, and the gap was real: the h-refinement gate, $\rho$, the
$\omega$-dominated near-kernel and the 138\u2009\u2192\u200952 result were **all** at N=4,
the lowest order this code supports. **Production is N=8.** It matters for one
specific reason already written into this plan: LOR was rejected on N=4 evidence,
and the dense-block concern it rested on bites hardest at high order.

Fixed 6\u00d76 mesh, p-refined:

| N | element block | ndof | Jacobi | AMG | ρ | ω frac of softest mode | CG w/ computed `B` |
|---|---|---|---|---|---|---|---|
| 4 | 100² | 2307 | 1033 | 142 | 0.9710 | **0.978** | 58 (2.45×) |
| 6 | 196² | 5187 | 1529 | 184 | 0.9704 | **0.977** | 72 (2.56×) |
| **8** | **324²** | 9219 | 2327 | **209** | **0.9682** | **0.977** | **94 (2.22×)** |

**Three of the four are flat and the conclusions carry.** $\rho$ does not degrade
(0.9710 → 0.9682) even though the element block grows 10.5× in entries — so
**the LOR rejection now stands at the order where its own counter-argument was
strongest**, which is exactly where I had flagged it as untested. The
$\omega$ fraction is invariant to three figures (0.978 / 0.977 / 0.977): the
near-kernel's structure is a property of the *first-order system*, not of the
discretisation. And the computed-mode gain holds at 2.2–2.6×.

**The fourth is not flat, and it is a genuine limit.** AMG iterations grow 1.47×
(142 → 209) as the order doubles, against Jacobi's 2.25×. **AMG is
$h$-independent but not $p$-independent** — which is what FOSLS theory actually
promises, so this is a correction to my reporting, not to the theory. The C3
gate was and remains an $h$-refinement gate; nothing here claimed p-independence,
but nothing here had tested it either.

### F2 QUALIFIED — AMG is h-independent, but CG is carrying it

Prompted by the question *"did we try it as a preconditioner?"* — every F2 number
is preconditioned CG, and CG can compensate for a weak preconditioner. The honest
measure is the **standalone V-cycle factor**:

| mesh | ρ | V-cycles to 1e−8 | CG its |
|---|---|---|---|
| 4×4 | 0.9715 | 637 | 113 |
| 8×8 | 0.9673 | 555 | 138 |
| 12×12 | 0.9671 | 550 | 137 |
| 16×16 | 0.9704 | 613 | 131 |

**ρ ≈ 0.97 — the V-cycle removes ~3% of the error per cycle**, against a textbook
0.1–0.3. AMG alone needs ~600 cycles; AMG+CG needs 131. So "h-independent" is
supported and "works well" is not — they are different claims.

It is *not* §7K's stall (ρ = 1.0000 exactly). ρ < 1 genuinely converges, and
**ρ is flat**, which is where the h-independence actually comes from.

**Not the smoother.** Block Gauss–Seidel at blocksize 4 gives results *identical*
to scalar (0.9700/138 both — pyamg falls back silently on a CSR matrix), and
doubling the sweeps moves CG (138→102) while leaving ρ at 0.9701. More smoothing
does not help, so the bottleneck is the **coarse-grid correction**.

**It is the near-null space — and §7J predicted which one.** Computing the true
softest modes:

```
softest-mode energy by field (u, v, p, ω):  [0.018  0.005  0.00004  0.977]
```

**97.7% in ω** — the 2D analogue of §7J's 3D cluster, which carried 100% of its
energy in $(\omega_x,\omega_y)$. Four field constants cannot span that:

| `B` | ρ | CG |
|---|---|---|
| 4 field constants | 0.9673 | 138 |
| constants + 4 computed modes | 0.9583 | **59** |
| 8 computed near-null modes | 0.9476 | **52** |

**138 → 52, a further 2.7×**, and it closes the loop on §7J: the same
ω-dominated cluster that made the redundant row ruinous in 3D is what limits AMG
in 2D. The row-7 down-weighting attacked the *symptom* pointwise; a coarse space
containing those modes attacks the cause.

**Production route.** Shift-invert eigensolves are not usable in production — they
need a factorisation, which is what AMG exists to avoid. The standard answer is
**adaptive / bootstrap AMG** (`pyamg.adaptive_sa_solver`), which discovers the
near-kernel algebraically by relaxing on $Ax=0$. Untested here and the obvious
next step.

### F2f — AmgX on the GB10: the compiled-vs-compiled test F2 could not run

F2's wall times were Python-vs-Python and I said explicitly they don't transfer.
AmgX 2.5.0 (built 2026-08-30, `sm_121`, CUDA 13.1) supplies the missing leg: a
compiled GPU V-cycle against a compiled GPU Jacobi, same matrices, same
`tol=1e-8`, same RHS seed. Driven through the C API by ctypes
(`scratch/amgx_run.py`) because the point-block path needs `block_dimx=4`, which
the MatrixMarket reader cannot express.

**The pipeline validates exactly.** AmgX's PCG+Jacobi reproduced F2's scipy
CG+Jacobi on all four meshes — **671 / 1245 / 1611 / 2103**, not one iteration
out. Every number below sits on that footing.

| mesh | ndof | Jacobi | AmgX agg (blk 4) | AmgX classical | *pyamg SA+energy* |
|---|---|---|---|---|---|
| 4×4 | 1027 | 671 — 0.044 s | 568 — 0.278 s | 421 — 0.165 s | *113* |
| 8×8 | 4099 | 1245 — 0.082 s | 571 — 0.332 s | 503 — 0.249 s | *138* |
| 12×12 | 9219 | 1611 — 0.118 s | 969 — 0.597 s | 653 — 0.352 s | *137* |
| 16×16 | 16387 | 2103 — 0.187 s | 1005 — 0.752 s | 542 — 0.352 s | *131* |
| **growth** | | **3.13×** | **1.77×** | **1.29×** | ***1.16×*** |

**AmgX loses to compiled Jacobi at every mesh** — 0.16× to 0.53×. Setup is
negligible (0.01–0.02 s); the cost is per-iteration, 7–10× Jacobi's.

**And neither AmgX scheme reproduces the h-independence.** Aggregation grows
1.77×, classical 1.29×, against pyamg's 1.16×. **The missing ingredient is
identifiable**: AmgX's AGGREGATION is *plain*, with no energy-minimising
prolongation — and F2 already showed that `smooth='energy'` is what converted a
constant-factor preconditioner into an h-independent one (4-field `B` alone:
1.40× growth; with energy: 1.21×).

**Point-block aggregation does work, and confirms F2's mechanism.** On the same
matrix, `block_dimx=4` versus scalar: **740 → 568, a 1.30× gain** — against
pyamg's scalar-`B` → 4-field-`B` step of 153 → 124 = 1.23×. Two different
libraries recovering the same field-constant structure by different means.
Classical AMG cannot take it at all (*"Unsupported block size for strong
connections"* — it is scalar-only).

**The projection is the actual answer to F2's open question.** At 16×16 AmgX
runs a V-cycle iteration in 0.649 ms. A compiled preconditioner of *pyamg's
quality* — 131 iterations — would take **0.085 s against Jacobi's 0.187 s, i.e.
2.20× faster**, and the margin widens with h since Jacobi grows 3.13×.

> **So a compiled V-cycle does beat compiled Jacobi on this operator — but not
> with anything AmgX ships.** The win requires energy-minimised prolongation on a
> 4-field near-null space, which AmgX has no API for. The engineering question
> F2 deferred now has a number attached and a specific missing feature named.

Raw data: `scratch/amgx_mats/amgx_f2f_results.json`.
