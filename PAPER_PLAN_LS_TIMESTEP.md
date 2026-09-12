# Paper plan: the time-step weighting of least-squares finite element methods

**Working title.** *How the time step is weighted decides what a least-squares
finite element method converges to*, or, more conventionally, *On the scaling of
the temporal residual in first-order system least-squares discretisations of
incompressible flow*.

**One-sentence claim.** In a transient least-squares (FOSLS/LSFEM) discretisation
the fixed point of the time-stepping map is the solution of a *non-symmetric*
problem in which the momentum equation enters weighted by $\Delta t^2$ relative to
the constraints, so that refining the time step degrades the solution and the
scheme converges to a constraint-satisfying field that is not the intended one;
rescaling the momentum row by $\sqrt{\Delta t}$ — the balanced-norm scaling known
for steady singularly perturbed problems — removes the defect, and the transient
error separates as $C_2\Delta t^2 + \Phi(w)\lVert R_h\rVert$ with $R_h$ the
irreducible spatial least-squares residual and $\Phi$ a weighting-dependent
constant, measured at $0.19$ for the conventional weighting and $0.05$ for the
balanced one.  The consequence is not a loss of temporal order — there is none —
but a floor, and below the crossover $\Delta t^\ast=\sqrt{\Phi\lVert R_h\rVert/C_2}$
the legacy error *rises* with refinement: **there is an optimal time step for the
conventional weighting, refining past it is harmful, and at a given mesh there is
an accuracy it cannot reach at any time step.**  Verified as a prediction, not a
fit: the crossover was computed from coarser meshes and the separation then found
where it was placed (§2.3).

---

## 1. Novelty assessment (searched 2026-09-12)

| prior work | what it does | distance from our claim |
|---|---|---|
| Adler, MacLachlan, Madden, *First-order system least squares finite-elements for singularly perturbed reaction-diffusion equations* (arXiv:1909.08598; SINUM) | weighted FOSLS inducing a **balanced norm** for a **steady** singularly perturbed reaction-diffusion problem on Shishkin meshes | the same scaling idea, in a steady setting and for layer resolution; no time step, no constraints, no fixed-point/consistency question |
| **Führer & Karkulik**, *New a priori analysis of first-order system least-squares finite element methods for parabolic problems* (arXiv:1805.04147) | the a priori theory for **time-stepping** LSFEM on parabolic first-order systems: an elliptic projection in a non-symmetric bilinear form, optimal estimates in the natural norm and in $L^2$ | the closest *analysis* of the setting we work in, and it treats the weighting as given: the "natural norm" is assumed, not chosen. It is also **unconstrained** (reaction–convection–diffusion, no divergence constraint), and our mechanism needs constraint rows for the solution to hide in — a delimitation worth stating explicitly in the paper |
| Adler, Lashuk, MacLachlan & Zikatanov, *Discrete Energy Laws for the First-Order System Least-Squares Finite-Element Approach*, LNCS 10665 (2018); arXiv:1709.00385 | **time-stepping** FOSLS (Crank–Nicolson) for heat and Stokes; energy laws converge at $O(h^{2p})$ | **the nearest miss, and it names our gap**: its constants are $C(\tau)=O(1/\tau^2)$ and the authors remark that *"a rescaling of the equations may ameliorate this worst-case scenario"* — and do not pursue it. Our $\Delta t^2$ momentum-to-constraint ratio is exactly that $1/\tau^2$, and the rescaling is exactly what we identify and measure |
| Bochev & Gunzburger, *Least-Squares Finite Element Methods* (Springer 2009); SINUM 1994 analysis of LSFEM for Navier–Stokes | the norm-equivalence framework, ADN ellipticity, the VVP formulation and its non-equivalence under some boundary conditions | steady theory; weighting discussed for mass conservation and for Reynolds number, not for the time step |
| Weighted/nonlinear-weight LSFEM (Chen et al., Deang & Gunzburger, and the viscoelastic and generalized-Newtonian literature) | weights on the continuity row and nonlinear residual weights to improve mass conservation | weights chosen against *mass conservation* at fixed $\Delta t$; the $\Delta t$-dependence of the weighting is not the subject |
| Space–time LSFEM (Pontaza & Reddy, JCP 2004; Gerritsma and co-workers; recent adaptive space–time LSFEM, arXiv:2509.11955, 2309.14300) | minimise the residual over a space–time slab, so time is a coordinate and the question does not arise in this form | a different discretisation; our result is a statement about *time-marching* LSFEM, which is what practical codes use. Worth citing as the alternative that avoids the problem at higher cost |
| Proot & Gerritsma; Pontaza & Reddy (LSSEM foundations) | least-squares spectral elements, $hp$ convergence without $H^1$-coercivity | the discretisation we use; no treatment of the temporal weighting |

**Where the mechanism does and does not apply.** It needs (a) a time-marching
least-squares discretisation, so the space–time formulations are outside it, and
(b) *constraint rows with a nontrivial discrete kernel* — continuity and the
vorticity definition here — for the solution to hide in when the momentum row
loses weight. A pure parabolic first-order system (Führer & Karkulik) has no such
kernel, and we should say so rather than let a referee say it.

**Conclusion: the claim appears to be new**, and the strongest evidence that it is
both new and wanted is the sentence in arXiv:1709.00385 that identifies the
$O(1/\tau^2)$ constant and leaves the rescaling as a suggestion. The paper should
open there rather than pretend the gap was unnoticed.

**Still to check before submission** (library access needed, not web search):
Bochev & Gunzburger's book §12 on time-dependent problems; the Proot–Gerritsma
thesis chapters on transient LSSEM; and a citation search forward from
arXiv:1709.00385.

---

## 2. What we have

### 2.1 Theory — the fixed-point equation (`scratch/model1d_fixedpoint.py`)

With BDF1 the history scaling equals the mass coefficient, so at a fixed point the
momentum residual collapses and stationarity gives

$$
\Big[\,m\,\Pi_u^{H}WA \;+\; a\,A^{H}WA \;+\; \tfrac1a\,C^{H}WC\,\Big]U^\ast
= m\,\Pi_u^{H}Wf + a\,A^{H}Wf ,
\qquad m=a_{\rm mass},\; a=a_{\rm flux}.
$$

Three statements follow, all verified numerically in the 1D model:

1. Every step is symmetric positive definite; **the fixed point is not**, and it is
   not the minimiser of any steady least-squares functional (the cross term
   $\Pi_u^H W A$ has no counterpart there).
2. **The relative size of the non-symmetric part is exactly $ma$**: $\Delta t$ for
   the legacy scaling, $1$ for the balanced one. Measured: symmetry defect
   $2.0\times10^{-3}\to2.0\times10^{-7}$ linearly over four decades of $\Delta t$
   (legacy) against a constant $2.0\times10^{-2}$ (balanced). The legacy
   fixed-point operator becomes *symmetric* as $\Delta t\to0$ — and that symmetry
   is the disease, because what remains is the constraint block alone.
3. The balanced scaling has the $\Delta t$-independent limit
   $[\Pi_u^HWA+C^HWC]U=\Pi_u^HWf$.

### 2.2 The model problem itself

One Fourier mode of 2D velocity–vorticity–pressure Stokes reduced to a two-point
boundary-value problem in $x$ — the per-mode structure of the 3D code — with C⁰
spectral elements, GLL collocation and GLL quadrature. Linear, so the fixed point
is one dense solve and no marching is needed. Two cases:

| case | irreducible residual | legacy error, $\Delta t=10^{-5}$ | balanced error | legacy $\kappa$ |
|---|---|---|---|---|
| solution in the space (control) | $10^{-15}$ | 4.0e−8 (round-off × $\kappa$) | 3.5e−11 | 3.4e12 |
| $\psi=\sin^2\pi x$, 4 elements $N=4$ | 9.8e−4 | 1.8e−2 (6×) | 2.9e−3 | 4.8e12 |
| $\psi=\sin^2\pi x$, 8 elements $N=6$ | 7.7e−8 | 4.1e−6 (25×) | 1.7e−7 | 5.3e13 |

**No irreducible residual, no pathology** — which is the $C_1\Delta t\lVert R_h\rVert$
term made visible, and it explains why the defect is invisible on manufactured
solutions that the space represents exactly (a trap for anyone verifying such a
code).

### 2.4 The weighting is unique, and the scaled problem is uniformly well posed

Two results obtained while trying to prove the obvious thing, which is false.

**The naive limit operator is singular.** $\Pi_u^HWA+C^HWC$ has *no rows* in the
pressure component ($\Pi_u^H$ selects velocity rows, $C$ does not involve $p$), so
its smallest singular value is numerically zero. Pressure is carried only by the
Gauss–Newton term $aA^HWA$, i.e. at $O(a)$. The right statement is therefore not
"the limit is nonsingular" but a **balanced-norm statement**: scaling the velocity
and vorticity rows by $a$ and the pressure rows by $1/a$, the balanced operator's
condition number becomes **independent of $\Delta t$** (2.2e7, flat from
$\Delta t=10^{-4}$ to $10^{-6}$, 8 elements $N=6$), while the legacy operator under
the same scaling still grows like $1/\Delta t$ (1.7e11 → 1.7e13).

**Uniqueness, and it needs no search over scalings.** Within the velocity row
block the constraint and momentum contributions stand in the ratio $1/(ma)$ with
$ma=w_{\rm mass}w_{\rm mom}\mathrm{fac}_1/\Delta t$, and any row or column scaling
multiplies both equally — the ratio is scaling-invariant. So the momentum equation
survives the limit **iff** $w_{\rm mass}w_{\rm mom}=O(\Delta t)$. Time consistency
independently forces $w_{\rm mom}/w_{\rm mass}=1$ (otherwise the scheme integrates
$\Delta t_{\rm eff}=\Delta t\,w_{\rm mom}/w_{\rm mass}$). Together:
$w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$, **uniquely**. The legacy weighting has
$ma=\Delta t$ and loses momentum; the unit weighting has $ma=1/\Delta t$ and loses
the constraints, which is the measured stall of that variant. This is the paper's
main theoretical statement and it is a one-paragraph argument.

**Pressure is the field the legacy weighting damages most** — three orders worse
than velocity at $\Delta t=10^{-5}$, and $10^4$ worse than balanced — which
identifies the near-null pressure direction known at large $c$ as a consequence of
the weighting rather than of the operator.

### 2.5 What is proved, and what is only observed (2026-09-12)

Stated plainly, because the distinction decides the venue:

| statement | status |
|---|---|
| the fixed-point equation, and that every step is SPD while its fixed point is not | **proved** (algebra, §2.1) |
| the relative size of the non-symmetric part is $ma$, hence $\Delta t$ legacy and $1$ balanced | **proved**, and verified to three digits |
| scheme-independence over BDF order via $\mathrm{fac}_1=\sum\alpha_m$ | **proved**, verified at BDF2 |
| **uniqueness of $w=\sqrt{\Delta t}$** from scaling-invariance plus time consistency | **proved** (two lines, §2.4) |
| nonsingularity of the scaled limit | **observed** across $h,p,k,\nu$; the energy argument covers only a sufficient condition that nothing satisfies and has the $\nu$-dependence backwards (§4.14) |
| the two-term error model and the crossover $\Delta t^\ast$ | **measured**, and the crossover verified as a prediction |

So the theoretical core — the mechanism and the uniqueness — is rigorous, and the
one gap is a well-posedness statement that is numerically solid and analytically
incomplete.

### 2.3 Numerical evidence in 2D and 3D (already measured)

| evidence | result | where |
|---|---|---|
| lid-driven cavity, $Re=1000$, $N=15$ | legacy steady state is $\Delta t$-dependent and non-monotone (rms vs Ghia 0.0095 / 0.0256 / 0.0122 at $\Delta t=1/0.1/0.01$), carries a node-to-node mode under the lid (7–8 of 8 sign changes); balanced is flat over three decades (0.0061–0.0062) with no sign changes | ZIGZAG_CURE_RESEARCH.md §4.1, §4.6 |
| the same mode in an independent implementation and in the original Fortran code | reproduced (6 of 8 sign changes in Fortran) — not a bug in one code | CAVITY_N15_MARCH.md |
| Orr–Sommerfeld growth rate, $Re=7500$, known answer $\sigma=0.00223497$ | legacy off by 3–6 % at $\Delta t=0.02$ and $0.01$ with a local rate drifting −10 %→+4 % within one run, and *worse* when the linear solver is tightened; balanced within 0.02–0.08 % | ZIGZAG_CURE_RESEARCH.md §4.5 |
| temporal order, start-up Poiseuille | 2.04 balanced (2.039 legacy) — the cure costs no order, and the test cannot discriminate because its solution is representable | §4.4 |
| **crossover predicted then tested** | $\Delta t^\ast$ computed from the $N=6,8$ constants places the $N=12$ separation at 2.6e−4; continuing that row past it gives ratios 1.00 → 1.08 → 3.34 → 29.1, with the legacy error reaching a minimum at the predicted step and then rising 2.8× per halving.  Excluded as a solver artefact by a tolerance sweep (identical to five digits and identical iteration counts at $10^{-12}$ and $10^{-16}$) | §4.12 |
| **transient manufactured solution, 2D nonlinear Navier–Stokes, four orders × six time steps** | the two error terms separated: identical $\Delta t^2$ behaviour for both weightings (five digits at $N=12$, order 1.98), then floors at $0.19\lVert R_h\rVert$ (legacy) against $0.05\lVert R_h\rVert$ (balanced), the constants stable over three decades of $\lVert R_h\rVert$; the $N=10$ floor predicted from $N=6,8$ to within 1 % | §4.11, `figs_fosls_vs_fs/mms2d_temporal.png` |
| Richardson self-convergence, regularised vs singular cavity lid | smooth: order → 2 for balanced, 0.6–0.7 legacy; singular lid: 0.3 for both, at a level an order below the spatial error, separating the two error terms | §4.7 |
| 3D channel, explicit convection (the delimiting negative result) | balanced weighting makes $\nabla\!\cdot\mathbf u$ grow (3e−1 → 6e−1 in ten steps against legacy's 8e−4), because the RKW3 stage is a Stokes *projection* whose right-hand side is not solenoidal; physics identical to four digits | BALANCED_CONDENSED_PLAN.md §1.5, §5.3 |
| 3D channel, legacy weighting | the 2D mode is *absent* — same node-alternation statistics as a fractional-step field and less top-mode energy | §4.8 |

That last pair is what makes the paper honest rather than promotional: the cure
applies to implicit-convection time stepping and must not be applied to an
explicit-convection projection stage, and we can say why.

---

## 3. What is missing before submission

| item | effort | why it matters |
|---|---|---|
| ~~A transient **manufactured solution** convergence table~~ — **done 2026-09-12**, §4.11 and the figure; it also corrected the error model (the second term is a floor, not a first-order term) | — | done |
| Extend the 1D model to **BDF2** and confirm the same structure with $\mathrm{fac}_1=3/2$ | half a day | the production scheme is BDF2; the derivation above is BDF1 |
| nonsingularity of the limit — **§2.5 / ZIGZAG §4.14: the energy argument is disposed of.** Its constants are computable exactly, and the sufficient condition is violated by up to three orders while $\sigma_{\min}$ stays positive, *and* it wants small $\nu$ whereas the limit actually degenerates as $\nu\to0$. A proof matching the observation needs a discrete Helmholtz decomposition plus an inf-sup argument | **1 week**, not 1 day | this is what separates a JCP paper from a SISC one; **decision needed** |
| ~~extend the 1D model to BDF2~~ — **done**: the analysis is order-independent because fac₁ = Σαₘ is what makes the mass and history terms cancel at a fixed point; measured $ma$ = 1.5 and symmetry defect 3.06e−2 = 1.5 × the BDF1 value | — | done |
| Convert the 1D model to a **figure**: error vs $\Delta t$ for both weightings, plus the symmetry defect | half a day | this is the paper's Figure 1; the MMS figure (merged to one panel, dashed legacy / solid balanced per order) is Figure 2 |
| ~~Forward citation search from arXiv:1709.00385~~ — **done 2026-09-12**: no follow-up pursues the rescaling; the nearest time-stepping analysis (Führer & Karkulik) treats the weighting as given. Remaining: Bochev & Gunzburger §12 (needs the book, not web search) | half a day | priority |
| Decide whether the preconditioning story is a **second paper** or a section | — | it is a second paper; mixing them weakens both |

---

## 4. Where to submit

**First choice: Journal of Computational Physics.** The contribution is a
method-level defect with a mechanism and an inexpensive fix, demonstrated on two
standard benchmarks, three independent implementations and a closed-form model.
JCP's readership is the one that needs the warning, and it accepts derivations at
the level of rigour we have (a fixed-point argument plus numerical verification)
without demanding theorems. The 3D negative result is the kind of delimitation JCP
reviewers reward.

**Second: Computer Methods in Applied Mechanics and Engineering.** Equivalent fit,
strong FEM readership, slightly more tolerant of a long numerical-evidence section.

**Third, only if item 3 of §3 is done properly: SIAM Journal on Scientific
Computing.** The FOSLS community (Adler, MacLachlan, Manteuffel, Ruge and
co-workers) publishes there and in SINUM, and arXiv:1709.00385 — the paper we are
answering — is from that group, so it is the natural venue *if* we can state a
theorem about the limit operator. Without it, SISC referees will ask for one.

**Fallback, faster and lower-profile: International Journal for Numerical Methods
in Fluids** or **Journal of Scientific Computing**, both of which host the LSSEM
literature (Proot & Gerritsma, Gerritsma) and would take the paper largely as it
stands.

**Recommendation, revised 2026-09-12:** target **JCP now**.  The mechanism and the
uniqueness result are rigorous, the evidence is unusually complete, and the one
analytical gap (well-posedness of the scaled limit) is honestly reportable as a
numerical result — JCP will take that; SISC would not.  Going for SISC means a
week of analysis first (discrete Helmholtz plus inf-sup) with no guarantee, and
it delays a result whose practical warning is worth publishing now. The uniqueness argument of
§2.4 is short enough to state as a proposition with a two-line proof, which is
what moves the paper from "we tried a scaling that worked" to "this is the only
scaling that can work".
