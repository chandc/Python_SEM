# Curing the small-dt oscillation of the least-squares time step: literature and measurements

Status 2026-09-09. Companion to CAVITY_N15_MARCH.md (the symptom, its code
review and the Fortran cross-check) and H_MINUS_ONE_PROPOSAL.md (the
negative-norm attempt, closed).

## 1. What has to be cured, stated so that the literature can be matched to it

Per time step the FOSLS solve minimises

$$
J(\mathbf u,\omega,p)\;=\;\big\|\,\mathrm{fac}_1\,\mathbf u-\mathbf h+\Delta t\,N(\mathbf u)\,\big\|^2
\;+\;\|\nabla\!\cdot\mathbf u\|^2\;+\;\|\nabla\!\times\mathbf u-\omega\|^2
\qquad\text{(legacy weighting)},
$$

with $\mathbf h=\sum_m\alpha_m\mathbf u^{\,n-m}$ the BDF history and
$N(\mathbf u)=\mathbf u\!\cdot\!\nabla\mathbf u+\nabla p+\nu\nabla\!\times\omega$,

so the momentum equation enters with weight $\Delta t^{2}$ relative to the two
constraint rows, and at a steady state ($\mathbf h=\mathrm{fac}_1\mathbf u$) the
march converges to the minimiser of
$J_\infty=\Delta t^{2}\|N(\mathbf u)\|^2+\|\nabla\!\cdot\mathbf u\|^2+\|\nabla\!\times\mathbf u-\omega\|^2$. Consequences measured on the Re = 1000 cavity at N = 15:
(i) the steady state depends on dt (rms vs Ghia $0.0095/0.0256/0.0218/0.0122$ at
$\Delta t=1/0.1/0.03/0.01$); (ii) below a momentum weight of
about $10^{-3}$ ($\Delta t\lesssim0.03$) a node-to-node mode appears under the lid — a
discretely divergence-free, curl-consistent velocity oscillation that the
momentum rows are too weak to suppress; (iii) three independent
implementations reproduce it (Python, the assembled direct-solve
prototype, the original Fortran code). A cure must therefore keep the
momentum equation enforced at O(1) at every dt, without destroying the
solver, and without changing the converged answer at dt = 1 (which is
validated).

## 2. What the literature says

### 2.1 Balanced-norm FOSLS for reaction-dominated problems — the direct analogue

Our time step is a reaction–diffusion-type problem with reaction
coefficient $c=\mathrm{fac}_1/\Delta t$ ($c=150$ at $\Delta t=0.01$, $5405$ in the channel). Adler,
MacLachlan & Madden, *FOSLS for singularly perturbed reaction–diffusion
equations* (arXiv:1909.08598, 2019; building on Lin & Stynes' "balanced
norm"), show that the unweighted $L^2$ FOSLS of $-\varepsilon\Delta u+bu=f$ loses the
layer structure as $\varepsilon\to0$ because its energy norm is too weak, and
that the cure is to **re-weight the first-order system** so that the induced
norm is balanced:

$$
\mathcal L\,\mathcal U=
\begin{pmatrix}
\varepsilon^{1/2}\,(\mathbf w-\nabla u)\\[2pt]
-\varepsilon\,b^{-1/2}\,\nabla\!\cdot\mathbf w+b^{1/2}\,u\\[2pt]
\varepsilon^{k/2}\,\nabla\!\times\mathbf w
\end{pmatrix},
$$

i.e. the reaction/mass equation is scaled by $b^{-1/2}$ and the flux
definition by $\varepsilon^{1/2}$, giving the balanced norm

$$
|||\mathcal U|||^2=b\,\|u\|^2+\varepsilon\,\|\nabla u\|^2+\varepsilon\,\|\mathbf w\|^2
+\varepsilon^{2}\,\|\nabla\!\cdot\mathbf w\|^2+\varepsilon^{k}\,\|\nabla\!\times\mathbf w\|^2
$$

with $\varepsilon$-independent coercivity and continuity constants (their
Theorem 1). Translated to the time step ($b\leftrightarrow c$, the flux
definition $\leftrightarrow$ our constraint rows): **the momentum row should be
scaled by $c^{-1/2}$, not by $c^{-1}$ (legacy) nor by $1$
($w_{\rm mom}=w_{\rm mass}=1$)**. In `lssem2d`'s parameters that is

$$
w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}
\quad\Longrightarrow\quad
a_{\rm mass}=\frac{\mathrm{fac}_1}{\sqrt{\Delta t}},\qquad a_{\rm flux}=\sqrt{\Delta t},
$$

so the momentum row becomes $\Delta t^{-1/2}\times$ the legacy row and its
weight relative to the constraints is $\Delta t^{-1}\times$ legacy. The steady
state is then the minimiser of

$$
J_\infty^{\rm bal}=\Delta t\,\|N(\mathbf u)\|^2+\|\nabla\!\cdot\mathbf u\|^2+\|\nabla\!\times\mathbf u-\omega\|^2
$$

(one power of $\Delta t$ instead of two), and at $\Delta t=0.01$ the momentum
weight equals that of the legacy $\Delta t=0.1$ run — which had no zigzag.
This is also the geometric mean of the two weightings already tried, which
is what the parameter-dependent norm $c\|\mathbf u\|^2+\|\nabla\mathbf u\|^2$ of the
ADN analysis (ADN_FOSLS.md §3) asks for.

### 2.2 Space–time FOSLS: the constraint needs its time derivative controlled

Gantner & Stevenson, *A well-posed FOSLS formulation of the instationary
Stokes equations* (arXiv:2201.10843; SISC 2022/23), and Führer & Karkulik
(heat equation, 2021), obtain well-posed space–time least-squares
formulations with all residuals in computable norms. The decisive detail:
**the divergence residual is measured in $H^1(I;L^2(\Omega))$, not in
$L^2(I\times\Omega)$**, and they report that well-posedness is lost when it is
measured in $L^2$
(§1, "numerical experiments … indicate that well-posedness is indeed lost").
In a time-stepping reading this means the constraint must be enforced on
the *increment* as strongly as on the state: a term

$$
\Big\|\frac{\nabla\!\cdot\mathbf u^{\,n+1}-\nabla\!\cdot\mathbf u^{\,n}}{\Delta t}\Big\|^2
\quad\text{alongside}\quad\|\nabla\!\cdot\mathbf u^{\,n+1}\|^2 .
$$ That is a formulation-level explanation of why the
weighted compromise drifts between steps at small dt, and a second
candidate (§3, C). Their momentum residual is in $L^2(I\times\Omega)$ with unit weight.

### 2.3 Pontaza's regularised divergence constraint and consistent splitting

Pontaza (JCP 217, 2006, "A least-squares finite element formulation for
unsteady incompressible flows with improved velocity–pressure coupling")
identified that in least-squares formulations of the *non-stationary*
equations the pressure is not a Lagrange multiplier for the constraint, so
the velocity–pressure coupling is weak and spurious temporal pressure
oscillations appear; his remedy is a **regularised divergence-free
constraint** that re-couples p to ∇·u, giving smooth pressure evolution
and improved mass conservation (paywalled; exact form not retrieved). In
JCP 225 (2007) he went further and abandoned the monolithic LS time step
for a **consistent splitting** (Guermond–Shen type): a least-squares solve
for the velocity subproblem only, with the pressure from a separate
consistent step. Both are answers to the same weakness identified here.
Our artificial-compressibility option (`dtau_p`, continuity row
$\nabla\!\cdot\mathbf u+\kappa_p\,p$) is the closest existing knob to the
regularised constraint.

### 2.4 Filtering (spectral-element practice)

Fischer & Mullen (C. R. Acad. Sci. 2001) and Fischer, Kruse & Loth
(J. Sci. Comput. 2005): a modal filter that damps only the degree-N mode
(interpolate to degree $N-1$ and blend: $F=(1-\alpha)I+\alpha\,P_{N-1\to N}P_{N\to N-1}$) stabilises SEM Navier–Stokes
and removes spurious highest-mode content at negligible cost. It treats the
symptom, not the weighting.

### 2.5 Negative-norm (H⁻¹) momentum residual — closed

Tested in H_MINUS_ONE_PROPOSAL.md: does not remove the dt-dependence, does
not cure the oscillation (it down-weights exactly the rough residual that
the oscillation produces), not better than legacy at dt = 1. Not a cure.

## 3. Candidate cures, ranked, with what has been measured

| # | candidate | principle | changes the dt = 1 answer? | solver impact | status |
|---|---|---|---|---|---|
| A | **balanced weighting $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$** (momentum row $\times\,c^{-1/2}$) | Adler–MacLachlan–Madden balanced norm; ADN parameter-dependent norm | no (equals legacy at $\Delta t=1$) | momentum weight $\Delta t^{-1}\times$ legacy; patch preconditioner works (30 it/step measured at $\Delta t=0.01$) | **running**: dt = 0.01 (w = 0.1), dt = 0.001 (w = 0.0316); §4 |
| B | fixed momentum weight $w=1$ ($w_{\rm mom}=w_{\rm mass}=1$) | keep momentum at weight 1 at every $\Delta t$ | no | momentum weight $\Delta t^{-2}\times$ legacy; $(\mathbf u,p)$ coupling $O(c)$; present preconditioner stalls (>500 s/step) | needs its own preconditioner design |
| C | add the increment constraint $\lVert(\nabla\!\cdot\mathbf u^{n+1}-\nabla\!\cdot\mathbf u^{n})/\Delta t\rVert^2$ (and the curl analogue) | Gantner–Stevenson: divergence in $H^1(I;L^2)$ | no (vanishes at steady state) | adds a row; same operator structure | untested; one-day prototype in `hminus1_2d.py` |
| D | regularised divergence constraint / artificial compressibility on the continuity row (`dtau_p`) | Pontaza 2006 | consistent ($\kappa_p p-\kappa_p p_{\rm prev}$ cancels at steady state) | the patch preconditioner built for the plain operator stalls (172 it then >1000 s/step); needs the AC term inside the preconditioner build | untested for the zigzag |
| E | Fischer–Mullen modal filter on $u,v$ each step | damp the degree-$N$ mode | slightly (filtering error) | none | **tested**: α = 0.1: mean \|Δu\| 0.039 → 0.032, sign pattern intact; α = 0.3: 0.025, sign changes 7 → 2 of 8, rms vs Ghia unchanged. Palliative: the LS step regenerates the mode; the oscillation is not pure degree-N |
| F | consistent splitting: LS for the velocity subproblem, pressure by a separate consistent step | Pontaza 2007 | yes (different discretisation) | different solver (Helmholtz-type LS) | not attempted; large change |
| G | space–time least squares | Gantner–Stevenson, Führer–Karkulik | yes | (d+1)-dimensional problem | out of scope for DNS |
| H | $H^{-1}$ momentum norm | Bramble–Lazarov–Pasciak | yes | new operator | **closed** (H_MINUS_ONE_PROPOSAL.md §7) |

## 4. Measurements

### 4.1 Candidate A — balanced weighting $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$ (`scratch/ghia_n15_w0.1`, `…_wsqrt_dt1e-3`)

Same mesh (4×4, N = 15), same preconditioner (condensed vertex patch,
refreshed every 50 steps), from rest, compared with the legacy run at the
same dt and t. Zigzag = statistics of $\Delta u$ between consecutive GLL nodes on
$x=0.5$, $0.75<y<0.95$. Figure `figs_fosls_vs_fs/zigzag_balanced_weighting.png`.

| weighting | $\Delta t$ | $t$ | zigzag mean $\lvert\Delta u\rvert$ | max | sign changes | rms vs Ghia ($u$, $v$) | CG it/step | s/step (8 thr) |
|---|---|---|---|---|---|---|---|---|
| legacy (weight dt) | 0.01 | 10 | 0.0376 | 0.058 | 7 / 8 | 0.1276, 0.1440 | 15 | 1.1 |
| **balanced (w = 0.1)** | 0.01 | 10 | **0.0093** | 0.028 | **2 / 8** | 0.1274, 0.1414 | 30 | 2.1 |
| legacy (weight dt) | 0.001 | 1 | 0.0410 | 0.081 | 7 / 8 | 0.2142, 0.2893 | 12 | 0.9 |
| **balanced (w = 0.0316)** | 0.001 | 1 | **0.0143** | 0.084 | **3 / 8** | 0.2155, 0.2892 | 38 | 2.5 |

Reading. The balanced weighting removes the alternating sign pattern (7–8
of 8 → 2–3 of 8) and cuts the node-to-node amplitude 3–4×, at both dt, while
leaving the bulk transient solution unchanged (rms vs Ghia identical to
three digits — it is the same flow, without the spurious mode). The
solver stays healthy: 30–38 CG iterations per step against 12–15, twice
the time per step, no sign of the stall that w = 1 produced. The remaining
max $\lvert\Delta u\rvert$ at $\Delta t=0.001$ (0.084) is a single interval at the
lid corner, not an alternating pattern.

### 4.2 Candidate E — Fischer–Mullen filter (`scratch/ghia_n15_filt0.1`, `…_filt0.3`)

Applied to u, v (free dofs) after every step, from the legacy dt = 0.01
steady state, 300 steps:

| $\alpha$ | zigzag mean $\lvert\Delta u\rvert$ | max | sign changes | rms vs Ghia ($u$, $v$) |
|---|---|---|---|---|
| 0 (reference) | 0.0393 | 0.083 | 7 / 8 | 0.0122, 0.0164 |
| 0.1 | 0.0319 | 0.066 | 7 / 8 | 0.0123, 0.0162 |
| 0.3 | 0.0246 | 0.055 | 2 / 8 | 0.0127, 0.0157 |

A palliative: it damps the degree-N content each step, the least-squares
step regenerates the rest. Kept as a cheap option to combine with A.

### 4.3 Candidate D — artificial compressibility on the continuity row

With $\tau_p=1/a_{\rm mass}$ (`dtau_p`) the operator changes and the vertex-patch
preconditioner built for the plain operator no longer fits (172 CG
iterations, then the next step did not complete in 5 minutes); stopped.
Untested for the oscillation; would need the AC term inside the patch and
coarse builds (it is, in principle: the term is element-local).

## 5. Recommendation

**Adopt the balanced weighting, $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$, for
time-accurate runs.** It is the weighting the balanced-norm FOSLS theory prescribes for a
reaction-dominated first-order system, it coincides with the legacy
weighting at $\Delta t=1$ (so every validated steady result stands), it removes
the spurious mode's alternating structure and most of its amplitude at
$\Delta t=0.01$ and $0.001$, it leaves the physical solution unchanged, it keeps
second-order temporal accuracy (§4.4), it holds the Orr–Sommerfeld growth
rate to <0.1 % at $\Delta t=0.02$ and $0.01$ where the legacy weighting is off by
3–6 % (§4.5), and it costs at most 2× per step with the existing
preconditioner. Two follow-ups:

1. **Steady state under the balanced weighting** at $\Delta t=0.01$
   (`scratch/ghia_n15_w0.1_steady`, 9000 steps to $t=90$, 24 CG it/step
   mean, 4.2 h wall): rms vs Ghia ($u$, $v$)
   0.0320/0.0347 at $t=20$, 0.0087/0.0101 at $t=30$, 0.0065/0.0069 at
   $t=40$, 0.0063/0.0063 at $t=50$, 0.0062/0.0061 at $t=60$, **0.0062/0.0060
   from $t=70$ to $90$** (max $\lvert\Delta U\rvert/\Delta t$ halving every 10 time
   units, 2.0e−4 at $t=80$; the step cap ended the run before the 1e−6
   criterion), already
   below the legacy plateau (0.0122/0.0164) and the $\Delta t=1$ solve
   (0.0095/0.0101), heading for the N=30 reference (0.0032/0.0067). The
   $\Delta t$-dependence of the steady state is removed. Profiles and a
   near-lid zoom: `figs_fosls_vs_fs/ghia_balanced_steady.png`
   (`scratch/plot_ghia_balanced.py`); at $t=90$ max $\lvert u-u_{\rm Ghia}\rvert$ =
   0.011 against 0.022 (legacy plateau) and 0.020 ($\Delta t=1$), and the
   extrema $u_{\min}=-0.374$, $v_{\min}=-0.510$, $v_{\max}=0.364$ against
   Ghia's $-0.383$, $-0.516$, $0.371$ (N=30: $-0.386$, $-0.522$, $0.374$).
2. **Channel DNS:** the 3D operator weights the squared momentum residual by
   $c^{-2}$; the balanced prescription is $c^{-1}$, i.e. `MOM_WEIGHT` and the
   history scaling both multiplied by $c=5405$ relative to now (a
   $\sqrt{5405}\approx73\times$ stronger momentum row than legacy, $73\times$
   weaker than weight 1). The preconditioner regime
   changes accordingly (c* is unchanged, the row scale is not); measure on
   the assembled harness before running, as in ADN_FOSLS.md §5.

Candidate C (the increment-divergence term of the space–time theory) is
the next thing to try if A's steady state is still dt-dependent; F
(consistent splitting) is the escape route if the LS time step cannot be
made dt-robust at all.

### 4.4 Temporal order under the balanced weighting (`scratch/pois_temporal_balanced.py`)

The harness of TEMPORAL_ACCURACY_STUDY.md (start-up plane Poiseuille with its
closed-form unsteady solution, BDF2 seeded exactly, $t=0.02\to0.12$, $N=14$,
Newton iterated out, CG to $10^{-14}$), with $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$
set per $\Delta t$:

| $\Delta t$ | $w=\sqrt{\Delta t}$ | $a_{\rm mass}$ | rms $u$ error |
|---|---|---|---|
| 0.01 | 0.100 | 15.0 | 7.31e−5 |
| 0.005 | 0.0707 | 21.2 | 1.72e−5 |
| 0.0025 | 0.0500 | 30.0 | 4.16e−6 |
| 0.00125 | 0.0354 | 42.4 | 1.03e−6 |
| 0.000625 | 0.0250 | 60.0 | 2.54e−7 |

Fitted slope **2.040** over all five (2.067 over the coarse three), against
2.039 (legacy) and 2.042 ($w=1$) in the 2026-08-12 study. **No temporal
accuracy is lost**: the row weight changes which compromise the
least-squares step strikes when the residual cannot vanish, not the BDF2
equation it discretises, and on an unsteady solution the BDF truncation
error dominates that compromise for every weighting tried.

### 4.5 Orr–Sommerfeld growth rate under the balanced weighting (`scratch/os_run_balanced.py`)

The time-accuracy benchmark of CHANNEL_VALIDATION.md §6 (Chan 1996 Fig. 2):
plane Poiseuille at $Re=7500$, least-stable mode at $\alpha=1$, amplitude
$10^{-4}$, mesh $1\times3$ elements with wall elements $0.3/1.4/0.3$, periodic
in $x$, integrated to $t=100$; $\sigma$ from the slope of $\ln(E'/E'_0)$ over
$t\in[25,100]$, reference $\sigma_{\rm ref}=0.00223497$ (Streett). Two
Newton sub-iterations per step, CG to a relative residual reduction
`cgsfac` (0.01 as in the original harness; $10^{-4}$ where marked). The
forcing $f_x=2\nu$ carries the momentum-row weight ($a_{\rm flux}=w_{\rm mom}$)
in both weightings. Figure `figs_fosls_vs_fs/os_balanced_vs_legacy.png`.

**Order sweep at $\Delta t=0.1$** ($w=\sqrt{\Delta t}=0.316$):

| $N$ | legacy $\sigma$ (error) | balanced $\sigma$ (error) |
|---|---|---|
| 8 | 0.00172525 (−22.8 %) | 0.00200301 (−10.4 %) |
| 10 | 0.00223825 (+0.147 %) | 0.00223471 (−0.012 %) |
| 14 | 0.00223443 (−0.024 %) | 0.00222936 (−0.251 %) |

**Time-step sweep** (the test that matters for the cure — the legacy
weighting is $\Delta t$ on the momentum row, the balanced one $\sqrt{\Delta t}$):

| $N$ | $\Delta t$ | `cgsfac` | legacy error | balanced error | balanced $w$ |
|---|---|---|---|---|---|
| 10 | 0.1 | 0.01 | +0.147 % | −0.012 % | 0.316 |
| 10 | 0.02 | 0.01 | −3.32 % | −0.097 % | 0.141 |
| 10 | 0.02 | $10^{-4}$ | **+6.36 %** | **+0.083 %** | 0.141 |
| 10 | 0.01 | 0.01 | +3.88 % | −0.566 % | 0.100 |
| 10 | 0.01 | $10^{-4}$ | +0.399 % (drifting, see below) | **+0.019 %** | 0.100 |
| 14 | 0.02 | 0.01 | −0.626 % | −0.153 % | 0.141 |

Local growth-rate error per 20-unit window, $N=10$, `cgsfac` $=10^{-4}$:

| $t$ window | 0–20 | 20–40 | 40–60 | 60–80 | 80–100 |
|---|---|---|---|---|---|
| legacy $\Delta t=0.02$ | +5.6 % | +8.5 % | +6.3 % | +6.3 % | +6.2 % |
| legacy $\Delta t=0.01$ | −10.1 % | −2.3 % | −1.0 % | +1.6 % | +4.3 % |
| balanced $\Delta t=0.02$ | −0.5 % | +0.5 % | +0.06 % | +0.04 % | +0.09 % |
| balanced $\Delta t=0.01$ | −0.6 % | +0.5 % | −0.01 % | 0.00 % | +0.04 % |

Reading.

1. **At $\Delta t=0.1$ the two weightings are equivalent** (both within
   0.25 % at $N\ge10$; $w=0.316$ is not far from $\Delta t=0.1$), and the balanced
   weighting reproduces the $N=8$ under-prediction that Chan's figure
   shows, so the benchmark's character is unchanged.
2. **Below $\Delta t=0.1$ the legacy weighting loses the growth rate.** At
   $\Delta t=0.02$ the error is 3–6 % and at $\Delta t=0.01$ the local growth rate
   drifts monotonically from −10 % to +4 % within one run — the 0.4 % global
   figure is the fit crossing zero, not accuracy. Tightening CG makes it
   *worse* (3.3 % → 6.4 % at $\Delta t=0.02$): the error is in the discrete
   problem, i.e. the same mechanism as the cavity zigzag — with the momentum
   row weighted $\Delta t^2$ in the functional, the constraint rows dominate and
   the step no longer tracks the momentum equation. Refining $\Delta t$ is
   supposed to reduce the temporal error; under the legacy weighting it
   *raises* the least-squares compromise faster than it lowers the BDF
   truncation, so the benchmark gets worse with a smaller step.
3. **The balanced weighting is $\Delta t$-robust**: 0.02–0.08 % at every $\Delta t$
   once CG is converged, with a flat local growth rate. The 0.57 % at
   $\Delta t=0.01$ with the loose tolerance is solver error accumulated over
   $10^4$ steps (a relative reduction of $10^{-2}$ per step is not enough at
   that step count); it goes to 0.019 % at $10^{-4}$.
4. Cost: the balanced runs took the same or less wall time than legacy at
   every $\Delta t$ (e.g. 512 s vs 569 s at $\Delta t=0.01$, `cgsfac` $10^{-4}$) —
   on this 3-element problem with Jacobi the extra iterations of the
   stronger momentum row are offset by the better-conditioned (u, ω)
   coupling.

Together with §4.4 (temporal order 2.04 preserved) this is the
time-accurate half of the cure's validation: the balanced weighting keeps the
benchmark that the code was validated on and extends it to time steps at
which the legacy weighting fails.

### 4.6 $\Delta t$-dependence of the steady state, legacy vs balanced (`scratch/plot_rms_vs_dt.py`)

N = 15, 4×4, condensed patch preconditioner, relative CG tolerance $10^{-6}$.
Balanced runs at $\Delta t=10^{-3}$ and $10^{-4}$ were restarted from the
$\Delta t=10^{-2}$ plateau field (two identical history levels) and are the
state after 1.3 and 0.13 time units; both drift rates are falling and the
centreline numbers have not moved since their first checkpoint.
Figure `figs_fosls_vs_fs/ghia_rms_vs_dt.png`; profiles
`figs_fosls_vs_fs/ghia_balanced_dt_study.png`.

| $\Delta t$ | legacy rms $u$ / $v$ | balanced rms $u$ / $v$ |
|---|---|---|
| 1 | 0.0095 / 0.0101 | — |
| 0.1 | 0.0256 / 0.0335 | 0.0090 / 0.0099 |
| 0.03 | 0.0218 / 0.0288 | — |
| 0.01 | 0.0122 / 0.0164 | 0.0061 / 0.0060 |
| 0.001 | — | 0.0061 / 0.0057 |
| 0.0001 | — | 0.0061 / 0.0057 (final, $t=90.5$) |
| N = 30 reference | 0.0032 / 0.0067 | |

The legacy steady state is non-monotone in $\Delta t$ (worst at 0.1, 3× the
$\Delta t=1$ value) and never reaches the balanced value; the balanced steady
state is flat over three decades below $\Delta t=0.1$ and at $\Delta t=0.1$ still
better than the best legacy point. The $v$ error at $\Delta t\le10^{-2}$ is below
the N = 30 reference's own value, i.e. at the resolution floor of the
comparison.

### 4.7 Transient time accuracy on a nonlinear problem: Richardson self-convergence of the cavity (`scratch/richardson_where.py`)

N = 15, 4×4, condensed patch preconditioner, CG relative $10^{-6}$, one
Newton step per time step (checked: four Newton steps change the
$\Delta t=0.01$ solution by $1\times10^{-6}$, CG $10^{-9}$ by $2\times10^{-9}$ — both
negligible against the $\Delta t$ differences below).  All runs restart from
the same smooth state at $t=5$ (one-level self-starting BDF step,
`--restart bdf1`) and march to $t=6$; the table gives
$\lVert U(\Delta t)-U(\Delta t/2)\rVert$ (weighted $L^2$ of the velocity) and the
observed order $\log_2$ of successive ratios.

**Ghia lid (singular corners):**

| weighting | $\Delta t$ pair | $L^2$ diff | same, excluding corners | order |
|---|---|---|---|---|
| balanced | 0.02 / 0.01 | 5.8e−4 | 4.5e−4 | |
| balanced | 0.01 / 0.005 | 4.7e−4 | 3.4e−4 | 0.31 |
| balanced | 0.005 / 0.0025 | 3.8e−4 | 2.6e−4 | 0.32 |
| legacy | 0.01 / 0.005 | 1.1e−3 | 9.6e−4 | |
| legacy | 0.005 / 0.0025 | 8.1e−4 | 7.3e−4 | 0.37 |

**Regularised lid $u_{\rm lid}=16x^2(1-x)^2$ (smooth, same Re, mesh, start):**

| weighting | $\Delta t$ pair | $L^2$ diff | order |
|---|---|---|---|
| balanced | 0.02 / 0.01 | 3.1e−6 | |
| balanced | 0.01 / 0.005 | 9.7e−7 | **1.65** (1.76 away from the lid, 1.95 in $v$ on the centreline) |
| balanced | 0.005 / 0.0025 | 5.6e−7 | 0.80 (0.98 away from the lid) |
| legacy | 0.01 / 0.005 | 1.1e−6 | |
| legacy | 0.005 / 0.0025 | 6.9e−7 | 0.68 |

Repeated with four Newton iterations per step and CG $10^{-8}$ (`rich6_*`):
balanced 8.9e−7 / 5.5e−7 (order 0.69, 0.85 away from the lid), legacy
1.0e−6 / 6.9e−7 (0.60).  The single-Newton error itself is second order
(4.6e−7 → 1.1e−7 on halving $\Delta t$) and is not what limits the third
pair.  The maximum difference sits on the right wall at $x=0.996$,
$y\approx0.79$ and is $\Delta t$-independent (6.3e−6 → 6.2e−6).

Reading.

1. **The singular corner, not the time step, sets the $\Delta t$-convergence
   of the Ghia cavity.** With the corners regularised the $\Delta t$
   differences fall by 150× and the balanced weighting converges at
   second order on the first pair (1.65–1.95), then at first order once the
   differences are below $\sim10^{-6}$ (0.7–0.85, with a $\Delta t$-independent
   residue of 6e−6 on the wall).  That tail is the weight-dependent
   least-squares compromise of §1.2: an $O(\Delta t)$ term whose coefficient is
   the size of the *spatial* residual (here the N = 15 wall boundary layer),
   sitting under the $O(\Delta t^2)$ BDF2 error and taking over below
   $\Delta t\approx0.005$ at this resolution.  It vanishes with spatial
   refinement, not with $\Delta t$.  With the singular lid the
   differences are spread over the whole domain (excluding the corner boxes
   changes them by 25 %) and shrink only as $\Delta t^{0.3}$: the least-squares
   residual cannot vanish at the corner, and the compromise it strikes
   depends on the row weights, which depend on $\Delta t$.  A Richardson study
   from rest gave the same order 0.4, so the impulsive start was not the
   cause either.
2. **The legacy weighting is below first order even on the smooth problem**
   (0.68), consistent with §4.5: its momentum row loses weight as $\Delta t^2$,
   so refining $\Delta t$ changes what the step enforces.
3. Practical consequence: the FOSLS transient error separates as
   $C_2\,\Delta t^2 + \Phi(w)\,\lVert R_h\rVert$ with $R_h$ the spatial
   least-squares residual and $\Phi$ a weighting-dependent constant (§4.11
   measures $\Phi\approx0.19$ legacy, $0.05$ balanced; an earlier draft of this
   line wrote the second term as $C_1\Delta t\lVert R_h\rVert$, which the
   manufactured-solution study disproves: the term is a $\Delta t$-independent
   floor, not a first-order term).  Where $R_h$ is negligible (Poiseuille order 2.04,
   Orr–Sommerfeld <0.1 %) the scheme is second order; on the regularised
   cavity at N = 15 the second term appears at the $10^{-6}$ level, three
   orders below the spatial error.  On a problem with a boundary singularity
   the discrete transient converges in $\Delta t$ only sublinearly, at a level
   (∼5e−4 in $L^2$ velocity) an order of magnitude below the spatial error at
   N = 15 (∼3e−3 rms on the centrelines) — the singularity is a spatial
   resolution problem that the time step exposes, and mesh grading at the
   corners is the remedy, not a smaller $\Delta t$.

### 4.8 Does the 3D channel (legacy weighting) carry the 2D zigzag? — No (`scratch/zigzag3d.py`)

By the 2D criterion it should: the channel's momentum rows carry squared
weight $1/c^2\approx3\times10^{-8}$ ($c\approx6000$), far below the $10^{-3}$ at
which the cavity zigzag appeared.  Measured on run01's final field
($t=4.96$, 6×18 N=8, 32 planes, legacy weighting, $\Delta t=8\times10^{-4}$) and
on the fractional-step state on the same mesh and basis
(`results/minchan_re180_E/state_t15.95.npz`), in physical space:

| field | sign changes of consecutive $\Delta u$ along $y$: all / wall elements | energy fraction in the top two Legendre modes along $y$: $u$ / $v$ / $w$ | along $x$: $u$ / $v$ / $w$ |
|---|---|---|---|
| FOSLS run01, $t=4.96$ | 0.076 / 0.054 | 1.2e−6 / 5.3e−6 / 4.7e−5 | 1.5e−5 / 1.0e−2 / 5.6e−3 |
| fractional step, $t=15.95$ | 0.075 / 0.065 | 3.8e−6 / 1.3e−4 / 2.8e−4 | 6.6e−5 / 1.5e−2 / 1.3e−2 |

Over the run the top-mode fractions along $y$ *fall* (u 3.8e−6 → 1.2e−6,
v 3.3e−5 → 5.3e−6, w 3.0e−4 → 4.7e−5 from $t=0.08$ to $4.96$) while the
sign-change fraction stays at 0.07: the FOSLS step smooths the seed's
mesh-scale content rather than building any.

A zigzag would put the sign-change fraction near 1 and pile energy into the
top modes; the FOSLS field has the same alternation statistics as the
fractional-step field and *less* top-mode energy in every component.  (The
$10^{-2}$ fractions along $x$ are the streamwise under-resolution both codes
share at $\Delta x^+=17.7$.)

Why the 2D mechanism does not fire here: the zigzag is a property of the
*fixed point* of a step whose momentum row has lost its weight, and it needs
a source — the O(1) least-squares residual at the singular lid corner —
that the constraint rows cannot see.  The channel has no singular source
and no fixed point; convection is explicit, so the implicit step is a
Stokes-like solve whose right-hand side is the previous state plus a smooth
increment, and the mesh-scale kernel of the constraint operator is never
driven.  The 3D price of the legacy weighting is the one §4.5 measured in
2D — time accuracy below $\Delta t\approx0.1$ in cavity units — not the
spatial zigzag; and the balanced remedy is blocked in 3D by the divergence
of the explicit convective increment (BALANCED_CONDENSED_PLAN.md gate 5.3).

### 4.9 The mechanism in closed form: a 1D model problem (`scratch/model1d_fixedpoint.py`)

Everything above is a measurement.  This section is the derivation, on a model
small enough that the fixed point can be written down and solved in one linear
solve rather than marched to.

**The fixed-point equation.**  With BDF1 ($\mathrm{fac}_1=1$) the history scaling
equals the mass coefficient, $\mathrm{hist}=a_{\rm mass}=:m$, so at a fixed point
$U^\ast=U^n$ the momentum residual collapses to $a_{\rm flux}(AU^\ast-f)$, and
stationarity of the step functional gives, after dividing by $a_{\rm flux}=:a$,

$$
\Big[\;\underbrace{m\,\Pi_u^{H}WA}_{\text{non-symmetric}}\;+\;\underbrace{a\,A^{H}WA}_{\text{Gauss--Newton}}\;+\;\underbrace{\tfrac1a\,C^{H}WC}_{\text{constraints}}\Big]U^\ast
\;=\;m\,\Pi_u^{H}Wf+a\,A^{H}Wf ,
$$

with $A$ the spatial momentum operator, $C$ the constraint rows and $\Pi_u$ the
velocity selection.  **Every individual time step is a symmetric positive-definite
problem; its fixed point is not.**  Nor is it the minimiser of any steady
least-squares functional — that would be $[a^2A^HWA+C^HWC]U=a^2A^HWf$, which has
no cross term.

**What the weighting decides.**  The three terms scale as

| weighting | cross | Gauss–Newton | constraints | momentum : constraints |
|---|---|---|---|---|
| legacy ($w_{\rm mom}=w_{\rm mass}=\Delta t$) | $1$ | $\Delta t$ | $1/\Delta t$ | $\Delta t^{2}$ |
| **balanced** ($w=\sqrt{\Delta t}$) | $1/\sqrt{\Delta t}$ | $\sqrt{\Delta t}$ | $1/\sqrt{\Delta t}$ | $1$ |

Under the legacy scaling the momentum equation enters the fixed-point problem
*only* through the non-symmetric cross term, $\Delta t^{2}$ below the constraints:
the fixed point is any constraint-satisfying field, selected at $O(\Delta t^2)$,
and that selection is what appears as a mesh-scale mode.  Under the balanced
scaling the cross term and the constraints stay at the same order for every
$\Delta t$, and multiplying through by $\sqrt{\Delta t}$ leaves the
$\Delta t$-independent limit $[\Pi_u^HWA+C^HWC]U=\Pi_u^HWf$.

**The model.** One Fourier mode (wavenumber $k$ in $y$) of the 2D
velocity–vorticity–pressure Stokes system reduced to a two-point boundary-value
problem in $x$ — exactly the per-mode structure of the 3D channel code —
discretised with C⁰ spectral elements, residuals collocated at the GLL nodes and
summed with the GLL weights, as `lssem3d` does.  Measured (relative error of the
fixed point against the exact state; "sym defect" is
$\lVert K-K^H\rVert/\lVert K\rVert$; "zz" is the mean node-to-node increment of the
error in $v$):

*Control, solution in the discrete space* (2 elements, $N=6$, polynomial data;
irreducible residual $10^{-15}$):

| $\Delta t$ | legacy err | legacy cond | balanced err | balanced cond |
|---|---|---|---|---|
| $10^{-1}$ | 1.8e−12 | 7.0e6 | 2.4e−13 | 7.3e5 |
| $10^{-3}$ | 7.2e−10 | 9.8e8 | 4.8e−13 | 1.2e6 |
| $10^{-5}$ | 4.0e−08 | 3.4e12 | 3.5e−11 | 8.2e7 |

*Unresolved, $\psi=\sin^2\pi x$* (4 elements, $N=4$; irreducible residual
9.8e−4 momentum / 2.6e−2 constraints):

| $\Delta t$ | legacy err | sym defect | zz | balanced err | sym defect | zz |
|---|---|---|---|---|---|---|
| $10^{-1}$ | 9.3e−3 | 2.0e−3 | 6.9e−3 | 2.3e−3 | 2.0e−2 | 1.7e−3 |
| $10^{-2}$ | 1.7e−2 | 2.0e−4 | 1.2e−2 | 2.8e−3 | 2.0e−2 | 2.1e−3 |
| $10^{-3}$ | 1.8e−2 | 2.0e−5 | 1.2e−2 | 2.9e−3 | 2.0e−2 | 2.2e−3 |
| $10^{-5}$ | 1.8e−2 | 2.0e−7 | 1.2e−2 | 2.9e−3 | 2.0e−2 | 2.2e−3 |

Reading, and these are the paper's Section 2.

1. **No irreducible residual, no pathology.** In the control both weightings
   return the exact state for every $\Delta t$; the only degradation is
   conditioning.  This *is* the $C_1\Delta t\lVert R_h\rVert$ term of §4.7: the
   defect is proportional to what the discrete space cannot satisfy.
2. **The legacy fixed-point error grows as $\Delta t$ falls and then plateaus**
   at 6× the balanced error (and 60× on the finer 8×$N$=6 mesh), with 5× the
   mesh-scale content, while the balanced error is flat in $\Delta t$ at the
   best-approximation level.  Refining the step makes the answer worse and then
   stops improving — the behaviour §4.5 measured on the growth rate.
3. **The symmetry defect *is* the momentum weight.**
   $\lVert K-K^H\rVert/\lVert K\rVert$ is exactly $m a$: $\Delta t$ under the
   legacy scaling (2.0e−3 → 2.0e−7, linear over four decades) and constant under
   the balanced one.  The legacy fixed-point system becomes *symmetric* as
   $\Delta t\to0$, and that symmetry is the disease: what remains is the
   constraints alone.
4. **Conditioning.** The legacy fixed-point operator reaches $\kappa=5\times10^{13}$
   at $\Delta t=10^{-5}$ on the finer mesh — in double precision its fixed point is
   no longer numerically reachable — against $10^{8}$–$10^{9}$ balanced.

### 4.10 The weighting is unique, and the balanced problem is uniformly well posed

Two further results from the same model, and together they are the theoretical
core of the paper.

**(a) The balanced fixed-point problem is uniformly well conditioned in a
field-weighted norm; the legacy one is not.**  The operator's block structure
(fields $u,v\mid\omega\mid p$; note $AU=\nabla p+\nu\nabla\times\omega$ has no
velocity columns and $C$ has no pressure columns) is

$$
\begin{array}{r|ccc}
 & u,v & \omega & p\\\hline
u,v & \tfrac1a C^HWC & \tfrac1a C^HWC + m\,(\cdot) & m\,(\cdot)\\
\omega & \tfrac1a C^HWC & \tfrac1a C^HWC + a A^HWA & a A^HWA\\
p & 0 & a A^HWA & a A^HWA
\end{array}
$$

so the pressure rows are $O(a)$ and the rest $O(1/a)$.  Scaling the velocity and
vorticity rows by $a$ and the pressure rows by $1/a$ gives (`scaling_study`):

| $\Delta t$ | legacy raw | legacy scaled | balanced raw | **balanced scaled** |
|---|---|---|---|---|
| $10^{-1}$ | 5.2e7 | 5.2e7 | 5.2e6 | 5.2e6 |
| $10^{-3}$ | 1.6e10 | 1.6e10 | 2.0e7 | 2.0e7 |
| $10^{-4}$ | 5.3e11 | 1.7e11 | 1.2e8 | **2.2e7** |
| $10^{-5}$ | 5.3e13 | 1.7e12 | 1.2e9 | **2.2e7** |
| $10^{-6}$ | 5.2e15 | 1.7e13 | 1.2e10 | **2.2e7** |

(8 elements, $N=6$.)  The balanced condition number is **flat in $\Delta t$** once
scaled — the defining property of a balanced norm — while the legacy one keeps
growing like $1/\Delta t$ under the same scaling.

**(b) The weighting is unique.**  No search over scalings is needed to know the
legacy case cannot be repaired, because within the velocity row block the
constraint and momentum contributions stand in the ratio

$$
\frac{\lVert a^{-1}C^HWC\rVert}{\lVert m\,\Pi_u^HWA\rVert}\;\sim\;\frac1{ma},
\qquad ma=\frac{w_{\rm mass}\,w_{\rm mom}\,\mathrm{fac}_1}{\Delta t},
$$

and **any row or column scaling multiplies both equally**, so the ratio is
invariant.  The momentum equation survives the limit if and only if
$w_{\rm mass}w_{\rm mom}=O(\Delta t)$.  Time consistency independently requires
$w_{\rm mom}/w_{\rm mass}=1$, since otherwise the scheme integrates
$\Delta t_{\rm eff}=\Delta t\,w_{\rm mom}/w_{\rm mass}$ (this is already stated in
`lssem2d.lssem.ls_coeffs`).  The two conditions together give

$$
\boxed{\;w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}\;}
$$

uniquely.  Legacy ($w=\Delta t$) has $ma=\Delta t\to0$ and loses momentum; the
unit weighting ($w=1$) has $ma=1/\Delta t\to\infty$ and loses the constraints
instead — which is exactly the measured behaviour of that variant, whose
divergence and solver stall were recorded in §4.1 and CAVITY_N15_MARCH.md.  The
balanced choice is not one option among several; it is the only one that is both
time-consistent and non-degenerate.

**(c) Which field the legacy weighting damages most: pressure.**  Per-field
relative error at $\Delta t=10^{-5}$, 8 elements $N=6$:

| | $u$ | $v$ | $p$ | $\omega$ |
|---|---|---|---|---|
| legacy | 1.7e−7 | 3.9e−7 | **1.9e−4** | 5.4e−6 |
| balanced | 2.5e−9 | 5.8e−9 | **1.6e−8** | 1.7e−7 |

Pressure is three orders worse than velocity under the legacy weighting and a
factor $10^4$ worse than under the balanced one — the same near-null pressure
direction that FOSLS_TIME_DEPENDENT.md §2 identifies at large $c$, here seen as a
consequence of the weighting rather than of the operator.

### 4.11 The two error terms separated: a transient manufactured solution (`scratch/mms2d_temporal.py`)

The test the earlier sections could not do.  Start-up Poiseuille gives order 2 for
*both* weightings because its solution is essentially representable, so
$\lVert R_h\rVert\approx0$ and the second term is invisible — a null test, exactly
as §4.9's control case predicts.  Orr–Sommerfeld measures a growth rate, not a
norm error.  Richardson measures differences between step sizes, not errors.  So:
a transcendental transient solution the space cannot represent at any order,
$\psi=\cos t\,\sin^2\pi x\,\sin^2\pi y$ on the unit square (divergence-free by
construction, $u=v=0$ on all four walls for every $t$), full nonlinear
Navier–Stokes with the convective term carried in the manufactured forcing,
$\nu=0.01$, $T=0.2$, 2×2 elements, BDF2 seeded exactly at both levels, Newton to
$10^{-13}$, CG to $10^{-12}$ relative with the condensed patch preconditioner.
Relative weighted-$L^2$ error in $(u,v)$ at $T$.  Figure
`figs_fosls_vs_fs/mms2d_temporal.png`.

| $N$ | $\lVert R_h\rVert$ | regime over $\Delta t\in[7.8\times10^{-4},2.5\times10^{-2}]$ | legacy floor | balanced floor | worst ratio |
|---|---|---|---|---|---|
| 6 | 5.4e−3 | floor only | 9.8e−4 = 0.18 $\lVert R_h\rVert$ | 3.0e−4 = 0.056 | 4.1 |
| 8 | 6.0e−5 | floor only | 1.14e−5 = 0.19 | 2.6e−6 = 0.044 | 5.4 |
| 10 | 4.1e−7 | **crossover** | 7.86e−8 = 0.19 | ≤1.4e−8 = ≤0.035 | 5.5 |
| 12 | 1.9e−9 | $\Delta t^2$ only | not reached | not reached | 1.0 |

and the $N=10$ curve, which is the whole result in six lines:

| $\Delta t$ | legacy | order | balanced | order | ratio |
|---|---|---|---|---|---|
| 2.5e−2 | 3.413e−6 | | 3.412e−6 | | 1.0 |
| 1.25e−2 | 9.387e−7 | 1.86 | 9.339e−7 | 1.87 | 1.0 |
| 6.25e−3 | 2.609e−7 | 1.85 | 2.448e−7 | 1.93 | 1.1 |
| 3.13e−3 | 1.059e−7 | **1.30** | 6.394e−8 | 1.94 | 1.7 |
| 1.56e−3 | 8.252e−8 | **0.36** | 2.072e−8 | 1.63 | 4.0 |
| 7.81e−4 | 7.859e−8 | **0.07** | 1.417e−8 | 0.55 | 5.5 |

Reading.

1. **The weighting is not a temporal-order effect.**  While the $\Delta t^2$ term
   dominates the two weightings give the *same* error — to five digits at $N=12$
   (3.4117e−6 against 3.4118e−6) and to 1 % at $N=10$ — with order 1.98–1.99.
   Any study in that regime, which includes every order study we ran before,
   cannot distinguish them.
2. **The weighting sets the floor.**  Each curve flattens at
   $\Phi\lVert R_h\rVert$ with $\Phi=0.19$ (legacy) and $0.04$–$0.06$ (balanced),
   *constant across three decades of $\lVert R_h\rVert$*.  The $N=10$ legacy floor
   predicted from the $N=6$ and $N=8$ constants is $0.19\times4.10\times10^{-7}
   =7.8\times10^{-8}$; measured $7.86\times10^{-8}$.
3. **So at a given mesh there is an accuracy the legacy weighting cannot reach at
   any time step, and the balanced weighting reaches it.**  That is the practical
   statement, and it is stronger than the transient-accuracy claim we started
   with: the defect is in the *spatial* accuracy of a time-dependent computation.
4. The floor is visible in pressure too, and larger there
   (§4.10(c) measured the same asymmetry in the 1D fixed point).
