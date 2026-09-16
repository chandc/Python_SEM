# Paper plan: one parameter, two pathologies, two fixes

**Supersedes** `PAPER_PLAN_LS_TIMESTEP.md`, which planned a narrow theory paper on
the weighting alone.  That material is not discarded — it becomes Levels 0–2 of
this one.  The change of shape is deliberate: the audience we want is people who
would *use* a least-squares spectral element method for unsteady flow and
currently do not, and they need both halves of the answer.

**Working title.** *Time-marching least-squares spectral elements for unsteady
incompressible flow: the weighting that makes them accurate and the
preconditioner that makes them affordable.*

---

## The spine

A least-squares discretisation is attractive for the obvious reasons: the
algebraic system is symmetric positive definite whatever the physics, there is no
inf-sup condition to satisfy, and the functional is its own error estimator.  It
is nevertheless absent from production unsteady CFD, for two reputational
reasons, and both turn out to be the same number wearing two hats.

Write the weighted momentum row as $a_{\rm mass}\mathbf u+a_{\rm flux}\mathcal N(U)$
and let $c=a_{\rm mass}/a_{\rm flux}=\mathrm{fac}_1/\Delta t$ be the mass
coefficient the implicit step imposes.  Then

| $c$ appears as | and controls | pathology | fix |
|---|---|---|---|
| the ratio $ma=w_{\rm mass}w_{\rm mom}\mathrm{fac}_1/\Delta t$ | how strongly the momentum equation is represented in the *fixed point* of the step map | refining $\Delta t$ makes the answer worse; the steady state depends on $\Delta t$; a mesh-scale mode appears near singularities | $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$ — **uniquely** |
| the size of the zeroth-order term in the *operator* | whether velocity and vorticity decouple; above $c^\ast\approx\nu p^4/h^2$ they do not, and the divergence-free directions become invisible to pointwise relaxation | Jacobi, block Jacobi and p-multigrid all stall; thousands of iterations per solve | overlapping vertex-patch Schwarz with a coarse level, condensed |

**Thesis:** get the weighting right and precondition for the regime, and the
method becomes a usable DNS tool.  We demonstrate that by running one.

---

## The ladder

Each level is a complete statement that stands on the one below, and each is
either already measured or has a named piece of work outstanding.

### Level 0 — What must be true (proved)

Algebra only; no inequalities, no numerical evidence required.

* At a fixed point the mass and history terms cancel **for every BDF order**,
  because $\mathrm{fac}_1=\sum_m\alpha_m$ is exactly that statement, and the
  momentum residual collapses to $a_{\rm flux}(AU^\ast-f)$.
* Stationarity then gives
  $[\,m\Pi_u^HWA+aA^HWA+\tfrac1aC^HWC\,]U^\ast=m\Pi_u^HWf+aA^HWf$.
  **Every individual step is symmetric positive definite; its fixed point is
  not**, and it is not the minimiser of any steady least-squares functional.
* The relative size of the non-symmetric part is exactly $ma$, so $ma$ *is* the
  weight with which the momentum equation enters.  Under the conventional
  scaling $ma=\Delta t$: the operator becomes symmetric as $\Delta t\to0$, and
  that symmetry is the disease, because what survives is the constraint block.
* **Uniqueness theorem.** Momentum survives the limit iff
  $w_{\rm mass}w_{\rm mom}=O(\Delta t)$, and the ratio it depends on is invariant
  under any row or column scaling, so no reformulation can evade it.  Time
  consistency independently forces $w_{\rm mom}/w_{\rm mass}=1$.  Together:
  $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$ and nothing else.

*Status: complete.*  Sited inside Bochev & Gunzburger's weighted-$L^2$ taxonomy
(SIAM Review 40(4) 1998) as its missing parameter-dependent member — their
estimate (3.34) already separates the momentum and constraint residuals by one
Sobolev order, their §4.2 exists to emulate that on a finite element space, and
their scope statement puts time-dependent problems outside the review.

### Level 1 — The mechanism in closed form (done)

The one-dimensional model: one Fourier mode of velocity–vorticity–pressure
Stokes reduced to a two-point boundary-value problem, discretised exactly as the
production code is.  Linear, so the fixed point is one dense solve.  Gives:
the symmetry defect measured as $ma$ over four decades; the error flat in
$\Delta t$ for the balanced weighting and growing for the conventional one; the
condition number after field-dependent row scaling flat only for the balanced
weighting (a balanced norm); pressure identified as the worst-damaged field.
**Figure 1** (three panels) is built.

*Outstanding:* nothing required.  Optional: a discrete Helmholtz plus inf-sup
proof that the scaled limit is nonsingular, a week, which would let us say
"proved" instead of "verified across $h,p,k,\nu$".  **Not on the critical path**
— the energy argument we tried is reported as non-sharp and its viscosity
dependence is backwards, and that is stated rather than hidden.

### Level 2 — How much accuracy it costs (done)

Transient manufactured solution, full nonlinear Navier–Stokes, four polynomial
orders × six time steps.  Gives the error model
$C_2\Delta t^2+\Phi\lVert R_h\rVert$ with $C_2$ identical for both weightings
(five digits at $N=12$) and $\Phi=0.19$ against $0.05$; the crossover
$\Delta t^\ast=\sqrt{\Phi\lVert R_h\rVert/C_2}$; and that crossover **verified as
a prediction**, not a fit — computed from coarse meshes, then the separation
found where it was placed, with the conventional error rising 2.8× per halving
below it.  Solver error excluded by a tolerance sweep.  **Figure 2** is built.

Carries the paper's most quotable practitioner warning: with the solution
resolved, both weightings agree to five digits, so **a code verified against a
well-resolved manufactured solution passes this defect silently**.

### Level 3 — What it does to engineering answers (done)

* Lid-driven cavity, $Re=1000$: the conventional steady state depends on the time
  step non-monotonically and is twice as far from Ghia as the balanced one, with
  a node-to-node mode under the lid reproduced in an independent implementation
  and in the original Fortran code.
* Orr–Sommerfeld growth rate, $Re=7500$, known answer: conventional weighting
  3–6 % wrong at steps of $10^{-2}$ and below, drifting from −10 % to +4 %
  *within a single run*, and **worse when the linear solver is tightened**;
  balanced within 0.02–0.08 %.
* The singular-corner case, where both weightings converge only sublinearly in
  $\Delta t$ at a level an order below the spatial error — the honest boundary of
  the claim.

*Outstanding:* nothing required.

### Level 4 — What it costs to run (mostly done)

The same $c$, now in the operator.  Diagnosis first: above $c^\ast\approx\nu
p^4/h^2$ the $(\mathbf u,\boldsymbol\omega)$ pair is coupled and the
divergence-free kernel is invisible to pointwise relaxation, which explains in
one statement why point Jacobi, block Jacobi and p-multigrid all stall — four
separate failed experiments in our own history, plus low-order-refined AMG.
Then the remedy: overlapping vertex patches with a $p=2$ coarse space, made
affordable by static condensation that is *exact*.

| | conventional solver | patch + coarse |
|---|---|---|
| 2D, $N=5\to20$ | 435 → 4010 iterations | **19–21, flat** |
| 3D, $N=4\to10$ | 1158 → 4802 | **36–38, flat** |
| minimal channel, production | 4675 per stage | **72**, step identical to every logged digit |

**Third geometry, done 2026-09-12** (`scratch/gartling_precond.py`,
LOW_MEMORY_PATCH_SOLVERS.md §4): Gartling's backward-facing step at $Re=800$ —
inflow/outflow rather than closed or periodic, a re-entrant corner, non-uniform
and graded grids.  Patch iterations **flat in both $p$ and $h$** (48→56 over
orders 5–7; 52.8→53.3 over three meshes including a graded one) while Jacobi
doubles in each; ratios 123–204×, wall speed-up 3–4×.  Independent of which of
the two outflow treatments is used (142× with the free outlet).

And the sharpest statement of the regime argument in the project: sweeping
$\Delta t$ at fixed mesh, the patch preconditioner gets **better** as $c$ grows
(109 → 54 → 44 iterations at $c$ = 15, 150, 1500) while Jacobi degrades
(3883 → 13687, worst solve 25647).  At the largest step it is not worth its cost
(0.9× wall); by $c=1500$ it is 6.7× faster.  *Outstanding: nothing.*

### Level 5 — Making it fast (done, one confirmation outstanding)

The part that is hard to find written down anywhere:

* on a uniform mesh with an operator fixed by explicit convection, the ~2000
  patch factorisations collapse to about a dozen distinct ones — *verified*
  during the build, not assumed, so a graded mesh fails loudly;
* batched triangular solves are the wrong GPU primitive here: storing explicit
  inverses and applying them as batched matrix products is 3–5× faster, because
  the former runs as thousands of small kernels;
* graph capture removes the launch cost that remains.

Apply: 160 ms → 8.3 ms.  Build: 40 min → 30 s.  Channel step: 64 min → 25 s on a
GB10 and a projected ~2 s on an A100.

*Outstanding:* **confirm the A100 step time end to end** at the current commit
(one hour).

### Level 6 — Production (the work that remains)

Two statements, one negative and one positive, and the negative one is what makes
the paper trustworthy.

* **The delimitation.** With convection explicit, each stage is a Stokes
  *projection* whose right-hand side is not solenoidal, so the balanced weighting
  makes the divergence grow (3e−1 → 6e−1 in ten steps against 8e−4) while the
  physics is unchanged to four digits.  **Do not use it there.**  The 2D
  mechanism also does not fire there — run01's fields carry no mesh-scale mode,
  measured against a fractional-step field on the same mesh.  So the practical
  rule is conditional, and we say on what.
* **The DNS.** Turbulent channel at $Re_\tau=180$ in a minimal box, statistics
  against five reference databases.

*Outstanding, and this is the critical path:*
1. **Time-step sensitivity study** (1 day): restart from run01's checkpoint at
   several steps, one turnover each, and ask whether the production settings sit
   in the affected regime.  This decides the settings for (2) and is a result in
   itself either way.
2. **The long run** (1–2 days on an A100, ~7 days on the GB10): 20 turnovers
   against run01's 3.7, which is what second-order statistics need.
3. Statistics and comparison figures.

### Level 7 — What a user does on Monday (writing)

A one-page recipe, and the diagnostic is free: compute
$ma=w_{\rm mass}w_{\rm mom}\mathrm{fac}_1/\Delta t$ from your own configuration.
Order one is safe; much less than one is not, and the conventional choice gives
exactly $\Delta t$.  Then: the halve-the-step confirmation test; the fix and its
cost (≈2× iterations under a diagonal preconditioner, nothing under the patch
one, which is insensitive to $c$); the explicit-convection exception; and the
warning that verification against a resolved manufactured solution will not
detect any of this.  Ship it as a function in the code that prints the verdict at
start-up, not only as a paragraph.

---

## Figures

| # | content | status |
|---|---|---|
| 1 | mechanism on the 1D model: fixed-point error, symmetry defect $=ma$, scaled condition number | **built** |
| 2 | transient MMS: common $\Delta t^2$ line, different floors, one panel | **built** |
| 3 | cavity: centreline profiles vs Ghia and the under-lid mode; steady state vs $\Delta t$ | built, needs merging |
| 4 | Orr–Sommerfeld: $\ln E'$ traces and the growth-rate table | built |
| 5 | iterations vs polynomial order for Jacobi / block Jacobi / p-multigrid / patch, 2D and 3D | to draw from existing data |
| 6 | apply time and step time across three machines | to draw |
| 7 | channel statistics vs the five reference databases | **built** — `colab/section10.py`, four panels (mean, fluctuations, shear stress, total-stress balance) over the $t=5.2$–30 window, with the fractional-step twin and the box-validity line |

## Venue and shape

**Journal of Computational Physics**, as a full-length methods paper.  It takes
work of this shape — a mechanism, a fix, a solver, and a production
demonstration — and its readership is the one that needs it.  Computers & Fluids
is the fallback and would take it largely as is.

The theory is stated as propositions with proofs where they are proofs
(Level 0), and as measured statements where they are measurements (the
well-posedness of the scaled limit).  No claim is dressed above its evidence;
the one place we tried and failed to prove something is reported as such.

## Campaign closed (2026-09-15)

The DNS ran to $t=30$: 37,500 steps, 1.62 s/step, 14.1 h of A100 time for 25
turnovers.  Averaged over $t=5.2$–30 (24.8 turnovers, ~24 independent samples),
$u_\tau=1.0025$ against a prescribed 1, the total-stress identity closes to
0.009, four of six quantities fall inside the mutual spread of the five
databases, and the result is closer to them than the fractional-step twin on all
six (mean deviation 1.2 % against 2.9 %).  Sections 1–10 of the draft are
written; only Appendix A is outlined.

## Drafting status (2026-09-12)

`PAPER_DRAFT.md`, 8,300 words.

| section | state |
|---|---|
| 1 Introduction | written |
| 2 The weighted step, 3 The fixed point | written (Level 0) |
| 4 Closed form, 5 Accuracy cost, 6 Benchmarks | written (Levels 1–3) |
| 7 The same parameter in the operator | written (Level 4 diagnosis), now including the 2D steady $p$-multigrid study as the positive control and the same-code $c$-sweep |
| 8 The patch preconditioner | written, including the Gartling geometry and the $c$-sweep |
| 9 Implementation and cost | written, three machines |
| 10 DNS and the RKW3 exception | **written** — run complete to $t=30$, 24.8-turnover window, closer to the databases than the fractional-step twin on all six quantities |
| Appendix A Energy argument | outlined |
| References | 20 entries, complete for §§2–9 |

Remaining writing is Section 10's statistics and Appendix A. Everything else on
the critical path is now a computation, not a drafting task.

## After the reviewer's cost objection

The per-step decomposition $t_{\rm step}=3\,n_{\rm it}(t_{\rm op}+t_{\rm pc})$
puts 84 % of the A100 step in the preconditioner apply, most of that in reading
the 5.3 GB coarse inverse.  COARSE_AND_ITERATIONS_PLAN.md lays out the two
levers — the coarse read and the iteration count — with the prize quantified
(2.1 s → ≈0.3 s in fp64 if both land) and a three-rig measurement order.  Week 1
of it is existing switches and two-line changes.

## The fractional-step comparison (2026-09-15/16)

Vendored the projection solver (`fractional_step/`), profiled it on the same
A100, and optimised it until the comparison was fair — because publishing the
first measurement would have compared attention, not methods.

| | as vendored | after | |
|---|---|---|---|
| E path, h per eddy turnover | 15.84 | **0.67** | torch port 1.78×, captured V-cycle 5.9×, Δt 2.3× |
| against least squares (0.56) | 28× | **1.19×** | |

So **cost does not separate these methods**, and the paper cannot lean on it.
What remains is the divergence and the vorticity, and
`DIVERGENCE_CONSEQUENCES.md` records exactly how far that evidence reaches: no
measurable effect on low-order statistics (measured twice), a real effect on
vorticity (measured, with a caveat), and a mechanism-level argument for particle
tracking and scalar transport that we have **not** demonstrated.  It proposes the
one-day experiment that would.

## Schedule

| | work | days |
|---|---|---|
| 1 | channel time-step sensitivity study | 1 |
| 2 | A100 end-to-end confirmation | 0.5 |
| 3 | long channel DNS (wall-clock, unattended) | 2–7 |
| 4 | preconditioner $h$-study + second problem class | 5 |
| 5 | statistics, figures 3, 5, 6, 7 | 3 |
| 6 | drafting | 15 |

Critical path is 1 → 3 → 5 → 6; item 4 runs alongside.  Optional and off the
path: the discrete Helmholtz proof (5 days), which would upgrade one sentence.

## Risks

| risk | mitigation |
|---|---|
| the sensitivity study finds run01's settings *were* in the affected regime | that is a finding, not a failure; it strengthens Level 6 and changes the settings for the long run |
| the long run relaminarises (the minimal box is intermittent by design) | run01 sustained to $t=5$; checkpoint often, and report the intermittency rather than hide it |
| a referee asks for the nonsingularity proof | Level 0's uniqueness theorem is the load-bearing result and is complete; well-posedness is supported across $h,p,k,\nu$ and the failed argument is disclosed |
| the preconditioner looks incremental against the Schwarz literature | the contribution is the diagnosis (why standard smoothers fail in this regime), the exactness of the condensation, and the implementation — positioned as such, not as a new algorithm |
