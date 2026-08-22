# Gartling Re = 800 backward-facing step: reproducing Chan & Mittal figs 3–6

Study date: 2026-08-14/16. Reproduces all four backward-facing-step figures of

> D. C. Chan & R. Mittal, *Large-eddy simulation of a backward facing step flow
> using a least-squares spectral element method*, Center for Turbulence Research,
> **Proceedings of the Summer Program 1996**, 347–358.
> (`reference/chan_mittal_CTR_summer_program_1996.pdf`, NTRS 19970014673)

against the benchmark of

> D. K. Gartling, *A test problem for outflow boundary conditions — flow over a
> backward-facing step*, Int. J. Numer. Meth. Fluids **11** (1990) 953–967.

Reproduce: `scratch/mesh_gartling.py` (grids), `scratch/gartling_run.py` (solver
driver), `scratch/gartling_plot.py` (figs 3), `scratch/gartling_fig4.py` (fig 4),
`scratch/gartling_fig56.py` (figs 5/6), `scratch/gartling_wmom_plot.py`,
`scratch/gartling_permissible.py`, `reference/gartling_digitize.py` (benchmark
extraction).

**Note on the citation.** This is the *Proceedings of the Summer Program 1996*,
not the Annual Research Briefs. Several files in this repo previously referred to
"Chan's CTR fig. 2"; the BFS figures are **figs 3–6 of the Summer Program
proceedings**, and fig. 2 there is the profile comparison, not the grid.

---

## Executive summary

1. **All four figures reproduce.** At 7th order on the uniform 11×4 grid the
   steady solver gives lower reattachment **6.100** against Chan's 6.1, upper
   separation 4.863 (4.8) and upper reattachment 10.484 (10.5). Figures 5 and 6
   reproduce as a pair: the 11×4 grid sustains an oscillation
   (peak-to-peak `max|v|` = 1.8e-01 at $t = 200$) that the 18×4 grid damps to a
   true fixed point (**1.6e-10** at $t = 400$).

2. **There is a floor on the usable time step, and it is structural.** The
   continuity and vorticity rows of the least-squares functional carry weight
   exactly 1, so `a_mass = w_mass·fac1/dt` measures the time-derivative term
   against incompressibility. Eliminating `dt` and `w_mass` through
   `dt_eff = dt·w_mom/w_mass` gives the identity

       a_mass = fac1 · a_flux / dt_eff          (exact; verified on 20 runs)

   Refining `dt_eff` at fixed `w_mom` therefore drives `a_mass` up. Measured on
   this flow: **every run with `a_mass` ≤ 6.05 stayed bounded and every run with
   `a_mass` ≥ 12.1 diverged — 34 runs, no crossover.**

3. **Escaping by lowering `w_mom` costs accuracy.** Pressure appears only in the
   momentum rows, so the pressure block of `LᵀL` scales as `a_flux²`. Sweeping
   `w_mom` on the converged 18×4 solution: within ±1.2% of Gartling for
   `w_mom ∈ [0.25, 2.0]`, but −4.6% at `w_mom` = 0.1 (−11.6% for the steady
   form). The two constraints close on each other at **dt ≈ 0.06**.

4. **The periodic-channel cases are exempt — but the reason is now in doubt.**
   *(Flagged 2026-08-18: a plane channel with an outflow and a parabolic inlet —
   residual ≈ 0 — diverges at `a_mass` = 60/120/300 identically to the same
   channel with a residual of 8e−02. Zero residual with an outflow gives no
   protection, so the discriminator looks like the **outflow boundary**, not the
   residual. Periodic channels lack both, so the evidence below cannot separate
   them. **Resolved in `AMASS_RESOLVED.md`: the discriminator is the outflow
   boundary.**)* The original argument was:
   `TEMPORAL_ACCURACY_STUDY.md` ran the channel at `a_mass` = 30 with no trouble,
   where the BFS diverges. Poiseuille/Stokes are exactly representable
   (`J = 5.94e-27`), so all four rows vanish together and the weighting is
   irrelevant; Gartling sits at `L2(div u) = 2.3e-02`, two to three orders
   higher, so the weights decide which row is sacrificed.

5. **Steady form and time-marching converge to different answers.** At matched
   `w_mom` on the same grid, both converged to ~1e-10: 6.158 (steady form) vs
   6.044 (unsteady) at `w_mom` = 1, and 5.392 vs 5.819 at `w_mom` = 0.1. The BDF2
   mass term cancels identically at a fixed point, so these ought to coincide.
   **Unexplained.**

---

## 1. Setup, and the viscosity trap

Gartling's problem is a plain rectangle — unlike every other BFS grid in this
repo there is **no upstream inlet channel**:

| quantity | value |
|---|---|
| domain | `[0, 17] × [-0.5, 0.5]` |
| inflow | at `x = 0`, `y ∈ [0, 0.5]`, `u = 24y(0.5-y)` (u_max 1.5, mean 1) |
| step face | `x = 0`, `y ∈ [-0.5, 0]`, no-slip |
| walls | top and bottom no-slip |
| outlet | `x = 17`, P+Z (`p = 0` and `∂ω/∂x = 0`) |
| ν | **1.25e-03 = 1/800** |

**The trap.** Chan writes "the Reynolds number based on the step height and mean
velocity is 800". Taken literally with `S = 0.5` and mean inlet velocity 1 that
is `ν = 0.5/800 = 6.25e-04`. The value that reproduces his own quoted
reattachment of 6.1 is `ν = 1/800`, i.e. Re built on the inlet hydraulic
diameter `2h = 1`. Check: `x_r = 6.1` with `S = 0.5` is `x_r/S = 12.2`, and
Armaly's curve gives `x1/S ≈ 12` at $Re = 800$ in that convention. Taking the
literal reading would put the case at $Re = 1600$. This is the same convention slip
already documented for the Armaly runs in `armaly_run.py`.

### Grids

`scratch/mesh_gartling.py` builds both:

| grid | elements | spacing |
|---|---|---|
| uniform 11×4 | 44 | `dx = 1.5455` throughout |
| **Chan-graded 11×4** | 44 | `x = 0,1,2,3,4,5,7,9,11,13,15,17` — five of width 1, six of width 2 |
| uniform 18×4 | 72 | `dx = 0.9444` |

The graded boundaries were measured off Chan's own grid skeleton (top panel of
his fig. 5) at 600 dpi. Matching his stated "11 elements in the streamwise
direction", it is graded exactly 2:1 with the fine half over the recirculation.

> **Correction.** An earlier reading of that figure gave 13 elements at
> 0.89/1.78. That was wrong: too high a darkness threshold missed the faint
> interior lines in the right half of the scan, and the axis frame — which is
> inset from the grid block — was taken as the domain edge. Runs on that
> mis-measured mesh are void.

---

## 2. Figure 3 — profiles at x = 7 and x = 15

`figs/gartling_steady_profiles.png`. Steady form (`w_mass = 0`, `w_mom = 1`,
loose solve, `line_search=True` — the configuration `STEADY_FORM_STUDY.md` §9
recommends).

Max error vs the benchmark, as % of the benchmark's own range:

| case | u@7 | **v@7** | ω@7 | u@15 | v@15 | ω@15 |
|---|---|---|---|---|---|---|
| uniform N=5 | 22.9 | **378.4** | 46.8 | 1.9 | 10.8 | 1.6 |
| uniform N=6 | 4.1 | 49.9 | 3.1 | 1.9 | 6.4 | 1.8 |
| **uniform N=7** | **4.1** | **14.6** | **1.5** | 2.1 | 7.6 | 1.8 |
| graded N=7 | 4.1 | 49.6 | 2.5 | 1.9 | 5.8 | 1.6 |

Two things worth reading off this:

**`u@7` sits at exactly 4.1% for four independent discretisations**, and `u@15`
at 1.9% for five. When solutions that differ from each other agree on their
distance from the reference, the residual is most likely in the *reference*.
That 4.1% is the digitisation floor, not solver error. Gartling's own tables
would settle it; they are paywalled (see §3).

**`v` at x = 7 is the outlier — and so is Chan's.** He writes: "All except the
vertical velocity profile at the axial location of 7 show an excellent agreement
with the benchmark data of Gartling." We reproduce not only his numbers but his
one failure mode. `v ~ 0.02` against `u ~ 1.1` makes it the most
resolution-sensitive quantity in the problem.

Mass conservation, computed from the fields (`∫u dy` should be 0.5 at every
station): 0.4995–0.5011 for every case at both stations.

---

## 3. The benchmark data, and a digitisation trap worth recording

Gartling's tables are **not freely available** — IJNMF 11:953 is paywalled, and
Notus, the SU2 laminar-step tutorial and the Abaqus verification manual all cite
it while publishing only their own results. `reference/gartling_digitize.py`
therefore extracts the solid line (Gartling's curve) from Chan's fig. 3.

**The staircase was a marker-pitch bias, not quantisation.** A per-row median of
the ink columns carries a *periodic* error: Chan's ○/△/□ markers recur about
every 40 rows and each is centred on **his** data point rather than on Gartling's
line, so wherever a marker straddles the curve the median is pulled aside.
Measured signature: residual wavelength **38–44 rows in all three panels**, at
0.65–0.69% rms of full range.

Two attempts to *reject* markers made it worse, and both failures are
instructive:

- narrowest-run selection — a marker outline's edges are the same ~4 px width as
  the line, so it locks onto circle edges (rms rose to 1.4–4.0%);
- exactly-one-run rows — correct in principle, but the v panels are marker-dense
  and collapsed to 7 and 28 surviving points.

What works is exploiting the bias's structure: because a marker *straddles* the
line its pull is nearly zero-mean over one pitch, so Savitzky–Golay over **two
pitches** (81 rows, cubic) cancels it while preserving features that span 200+
rows. Residual after: **0.18–0.52%** of range, full coverage retained, and the
conservation gates unmoved (+0.7% at x = 7, −1.0% at x = 15).

**Axis calibration.** Every panel puts its last tick label on the frame except
`x15_u`, whose "0.9" stops one minor tick short. Counting ticks along the top
frame gives 15.04 minor intervals of 0.1, so that frame is at **1.0, not 0.9**.
Taking 0.9 compresses u by 1.4/1.5 and breaks conservation: `∫u dy` came out
0.4292 against 0.5 (−14%); with 1.0 it is 0.4952 (−1.0%).

---

## 4. Figure 4 — wall vorticity

`figs/gartling_fig4_wall_vorticity.png`. ω is a *solved* variable in this VVP
formulation, so wall values are read straight off the solution — no
differentiation of `u` is involved, making the zero crossings an independent
confirmation of the reattachment numbers.

| case | lower zeros | upper zeros |
|---|---|---|
| N=5 | 6.868 | 5.627, 10.534 |
| N=6 | 6.181 | 4.947, 10.505 |
| **N=7** | **6.100** | **4.863, 10.484** |
| 18×4 N=6 | 6.158 | 4.916, 10.466 |
| **Chan & Mittal** | **6.1** | **4.8, 10.5** |

Monotone p-convergence in every quantity. On a wall `v ≡ 0` so `dv/dx = 0` there
and `ω = −du/dy` exactly, which is why a zero crossing is a separation or
reattachment point.

### Grading did not explain the N=5 discrepancy

The obvious hypothesis for our N=5 `v@7` being 4.6× the benchmark was that our
mesh was uniform where Chan's is graded 2:1. Tested on his actual measured mesh,
it is **not supported** — graded is *worse* at N=5 and N=6:

| order | uniform | graded | Chan |
|---|---|---|---|
| N=5 | 6.868 | 5.155 | 6.1 |
| N=6 | 6.181 | 5.844 | 6.1 |
| N=7 | 6.100 | 6.187 | 6.1 |

The two meshes converge to the reattachment from **opposite sides**, bracketing
6.1 to within 1.4% at N=7. Why our N=5 differs so much from his 5th-order result
remains unexplained; untested candidates are his free outflow versus our P+Z and
his solver settings, neither recoverable from the paper.

---

## 5. Figures 5 and 6 — the grid artifact

The paper's point is that figs 5 and 6 are a **pair**: the same physics on two
grids, where the coarse one invents a limit cycle ("the transient flow predicted
above is a numerical artifact") and the finer one does not ("evolves
asymptotically towards a steady state"). Both runs start from **rest**, as Chan
does. `figs/gartling_fig5_nx11_evolution.png`,
`figs/gartling_fig6_nx18_evolution.png`, `figs/gartling_fig56_history.png`.

| grid | dt | `a_mass` | ran to | terminal p2p `max|v|` | verdict |
|---|---|---|---|---|---|
| 11×4 | 0.1 | 1.5 | 140 | ~1e-01 | periodic |
| 11×4 | 0.025 | 6.0 | 200 | 1.79e-01 | periodic |
| 11×4 | 0.0124 | 6.05 | 200 | 3.23e-01 | periodic |
| **18×4** | **0.1** | **1.5** | **400** | **1.64e-10** | **steady** |

**Nine orders of magnitude** between the two grids, and the 11×4 periodicity is
invariant to timestep, `a_mass` and weights — only the mesh changes it. The 18×4
case needed $t \approx 400$ to show it: at Chan's $t = 140$ it was still climbing at 94.5%
of its final reattachment, and a geometric extrapolation from that data
(projecting a limit of ~6.03) turned out to be wrong.

**These runs required `w_mom = w_mass = 0.1`.** At `w_mom = w_mass = 1` this flow
cannot be time-integrated at all — see §6.

---

## 6. The `a_mass` stability threshold

At `w_mom = w_mass = 1`, dt = 0.1 (`a_mass` = 15) the Gartling flow diverges from
rest by $t \approx 19$ — and, decisively, **diverges even when started exactly on the
converged steady field**, at $t = 62.1$ on the 11×4 grid and $t = 74.0$ on the 18×4.
So the discrete steady state is *unstable* under that time-stepping operator; it
is not a basin-of-attraction problem, and refining the mesh only delays it.

Sorting every run with a trustworthy time axis by `a_mass`:

| `a_mass` | runs | outcome | peak `max\|u\|` |
|---|---|---|---|
| 0.3 – 6.05 | 18 | **all stable** | 1.55 – 1.59 |
| 12.1 – 60 | 16 | **all diverged** | 11.5 – 20 |

34 runs, **no crossover**, and the split cuts across every other variable — both
grids, both initial conditions, sub-iterations 1/3/5/10, `a_flux` from 0.1 to 1,
dt from 0.01 to 2. None of those predict the outcome; `a_mass` does.

> **Scope.** The threshold is measured on *this flow*, which has an outflow
> boundary. It does **not** transfer to a closed domain: the Ghia $Re = 1000$
> lid-driven cavity converges at `a_mass` = 30 with no remedy at all
> (`|dU|` = 5.4e−10 after 518 steps, `ARTIFICIAL_COMPRESSIBILITY.md` §5.1). Read
> 6.05 / 12.1 as a property of the weighting imbalance *interacting with an
> outflow condition*, not of `a_mass` alone.
>
> **It can be moved.** Adding artificial compressibility to the continuity row
> raises the usable limit on this flow to `a_mass` = 60 at `κ_p` = `a_mass`/2 —
> dt = 0.1, 0.05 and 0.025 all reach t = 140 within 0.9% of Gartling's
> reattachment. `a_mass` = 120 still fails at every `κ_p` tried. See
> `ARTIFICIAL_COMPRESSIBILITY.md` §4.

`peak|u|` is a perfect secondary discriminator: every stable run peaks at
1.55–1.59 against the inlet peak of 1.5, every diverged run reaches 11.5–20,
with nothing in between. That includes the `nsub=1` run that never tripped a
20.0 cutoff but reached 11.5 and is correctly classed as diverged.

**`a_flux` sets the timescale of failure, not the outcome.** At `a_mass` = 15,
`a_flux` = 1 survives to $t \approx 9$–$19$ while `a_flux` = 0.1 dies at 7.90; same at
`a_mass` = 30 (18.45 vs 5.26). So lowering `w_mom` *alone* makes failure sooner.

> **Retraction.** This study initially reported that "`w_mom` = 0.1 fixes the
> instability". It does not. Lowering `w_mom` helped only because `w_mass` was
> lowered with it, cutting `a_mass` tenfold. `a_mass` is the control variable.

### Two schemes are identical when `a_mass`, `a_flux` and `hist` match

`(w_mass, w_mom, dt) = (1.0, 0.1, 0.1)` and `(0.1, 0.1, 0.01)` both give
`a_mass = 15, a_flux = 0.1, hist = 10`. They produce **bit-identical** fields
(`max|dU| = 0.000000e+00`, same 789 history rows, time axes agreeing to 1.8e-15)
and fail at the identical instant. `dt_eff` is the real timestep; nominal `dt` is
a label.

A corollary: **`w_mass = dt` pins `a_mass = fac1 = 1.5` for any dt**, so that
family is unconditionally stable — but then `dt_eff = w_mom`, so nominal dt
becomes decorative and no time refinement actually occurs. Verified: dt = 0.05,
0.025 and 0.0125 at `w_mass = dt` give identical numbers at every checkpoint.

---

## 7. The permissible region, and the dt floor

`figs/gartling_permissible_region.png`. Two constraints bound the usable region
and they close on each other.

**Stability** — `w_mom ≤ (a_crit/fac1)·dt_eff`, slope between 4.03 and 8.07 from
the measured bracket. Directly: `max a_flux/dt_eff` among stable runs = 4.03,
`min` among diverged = 8.07.

**Accuracy** — `w_mom ≳ 0.25`. From the `w_mom` sweep on the converged 18×4
solution (`figs/gartling_reattach_vs_wmom.png`), every point a verified fixed
point (`max|v|` p2p 2e-11 … 2e-10):

| `w_mom` | `lo_reatt` | vs 6.1 | `up_sep` | `up_re` |
|---|---|---|---|---|
| 0.1 | 5.819 | **−4.6%** | 4.610 | 10.552 |
| 0.25 | 6.025 | −1.2% | 4.790 | 10.507 |
| 0.5 | 6.086 | −0.2% | 4.848 | 10.501 |
| 1.0 | 6.044 | −0.9% | 4.803 | 10.505 |
| 2.0 | 6.070 | −0.5% | 4.830 | 10.510 |

The curve is a **step, not a slope**: below 0.25 the answer falls off a cliff;
from 0.25 to 2.0 — a factor of eight — everything sits inside ±1.2% of Gartling
with no monotone ordering. `STEADY_FORM_STUDY.md` §4 found monotone behaviour on
a different BFS; that does not reproduce here.

Combining:

    0.25 ≲ a_flux ≲ 6·dt_eff      =>      dt_eff ≳ 0.06

**There is a smallest usable physical time step.** Below it no choice of weights
satisfies both conditions: keep `w_mom` up and you cross the stability limit,
drop it and you cross the accuracy limit. Exactly as observed — dt = 0.025 works
(`a_mass` = 6), dt = 0.0124 at the same weights fails (`a_mass` = 12.1), and
rescuing it with `w_mass` = 0.05 works but puts `a_flux` = 0.05 below the
accuracy plateau.

---

## 8. Why the periodic channel has no such restriction

`TEMPORAL_ACCURACY_STUDY.md` ran the periodic channel at **`a_mass` = 30**
(dt = 0.05, `w_mom = w_mass = 1`) to a bit-exact fixed point with whole-field rms
3.2e-10. The Gartling BFS *diverges* at `a_mass` = 30. Same scheme, same weight,
opposite outcome — so the restriction is a property of the discretisation **on
that flow**, not of the discretisation.

**The weights only matter when the residual cannot be zero.** If the exact
solution makes all four rows vanish simultaneously, the minimiser is that
solution for *any* weights — there is nothing to trade off. Poiseuille and
Stokes decay are exactly representable; `POISEUILLE_DT_STUDY.md` records
`J = 5.94e-27`. Gartling is not: measured `L2(div u) = 2.3e-02`,
`max|div u| = 0.72`, two to three orders above the channel. With a residual that
size the weights decide which row is sacrificed.

Measured, at matched physical time with only dt (hence `a_mass`) differing:

| `a_mass` | L2(div u) | L2(ω+u_y−v_x) |
|---|---|---|
| 1.5 | 2.3015e-02 | 1.8277e-01 |
| 2 | 2.3438e-02 | 1.8279e-01 |
| 3 | 2.5784e-02 | 1.8282e-01 |

Incompressibility degrades monotonically with `a_mass` while the vorticity
residual is flat to four digits — the predicted signature. **Caveat:** a 12%
drift over a 2× change in `a_mass` is far too mild to explain a hard threshold by
itself; the blow-up must involve nonlinear amplification of that error, which has
not been measured directly.

Two aggravating factors specific to the BFS, both circumstantial: every parallel
flow has `u·∇u ≡ 0` (noted as a limitation in `TEMPORAL_ACCURACY_STUDY.md` §5),
so the channel cannot amplify a mass-conservation error at all; and the periodic
channel has no inflow/outflow boundary, whereas the BFS outflow carries modes
measured elsewhere in this repo at ~8,300× softer than generic.

**Practical rule:** the dt floor applies to flows whose discrete residual is not
small — separated, nonlinear, with an outflow boundary. For flows the
discretisation represents exactly there is effectively no restriction, which is
why the order-2.04 measurement reached dt = 6.25e-04 without trouble.

---

## 9. Open: steady form and time-marching disagree

At a fixed point the BDF2 mass term cancels identically
(`fac1·u − Σα_m u^{n-m} = 1.5u − (2u − 0.5u) = 0`), so both formulations reduce
to minimising the same functional and should share a solution. They do not:

| `w_mom` | steady form (`w_mass`=0) | unsteady (`w_mass`=0.1) | gap |
|---|---|---|---|
| 0.1 | 5.392 (`|dU|`=2.8e-10) | 5.819 (p2p=1.6e-10) | **7.9%** |
| 1.0 | 6.158 (`|dU|`=4e-09) | 6.044 (p2p=6.6e-11) | **1.9%** |

Both converged to ~1e-10 in each case. The gap widens as `w_mom` falls, which is
consistent with the pressure block scaling as `a_flux²` and the loose CG
(`cg_tol = 1e-6`) leaving a larger residual in the worse-conditioned system —
`a_mass` = 1.5 supplies diagonal dominance that the steady form's `a_mass` = 0
lacks entirely. **That is a hypothesis, not a result.** The clean test is to
tighten the linear solve on both and see whether they converge toward each other,
but `STEADY_FORM_STUDY.md` §3 warns the steady form *diverges* at
`cgsfac=1e-8, tol=1e-10`, so that experiment may not be available on that side.

A second, milder anomaly: the steady form converges markedly worse at low
`w_mom` — at `w_mom` = 1 these grids reach `|dU|` ~ 1e-9 within 300 iterations,
at `w_mom` = 0.1 the same grid is still at 6.3e-6 after 300 and needs ~2000.

---

## 10. How to reproduce, step by step

All commands from the repo root, using the project venv (`requirements.txt`
documents `uv venv --python 3.12`):

```bash
# 1. grids
cd grids
for N in 5 6 7; do
  uv run --project .. python ../scratch/mesh_gartling.py 17.0 gartling_nx11_N$N\_grid.dat 11 4 $N
  uv run --project .. python ../scratch/mesh_gartling.py 17.0 gartling_nx11g_N$N\_grid.dat 11 4 $N chan
done
uv run --project .. python ../scratch/mesh_gartling.py 17.0 gartling_nx18_N6_grid.dat 18 4 6
cd ..

# 2. figs 3 and 4 -- steady form, w_mass = 0
for N in 5 6 7; do uv run python scratch/gartling_run.py steady 11  $N; done
uv run python scratch/gartling_run.py steady 11g 7
uv run python scratch/gartling_run.py steady 18  6
uv run python scratch/gartling_plot.py          # figs/gartling_steady_{profiles,streamlines}.png
uv run python scratch/gartling_fig4.py          # figs/gartling_fig4_wall_vorticity.png

# 3. figs 5 and 6 -- from rest, w_mom = w_mass = 0.1
#    args: unsteady NX N dt tmax nsub outlet ic ramp w_mom w_mass
uv run python scratch/gartling_run.py unsteady 11 6 0.1 140 3 pz stagnant 0 0.1 0.1
uv run python scratch/gartling_run.py unsteady 18 6 0.1 400 3 pz stagnant 0 0.1 0.1
uv run python scratch/gartling_fig56.py

# 4. the w_mom sweep (continuations from the converged 18x4 field)
REF=scratch/gartling_unsteady_nx18_N6_dt0.1_T400_nsub3_pz_stagnant_wm0.1_ws0.1.npz
for wm in 0.25 0.5 1.0 2.0; do
  uv run python scratch/gartling_run.py unsteady 18 6 0.1 400 3 pz file:$REF 0 $wm 0.1
done
uv run python scratch/gartling_wmom_plot.py     # figs/gartling_reattach_vs_wmom.png

# 5. the permissible region (reads every saved run)
uv run python scratch/gartling_permissible.py   # figs/gartling_permissible_region.png

# 6. the benchmark extraction
uv run python reference/gartling_digitize.py    # reference/gartling_re800_*.csv
```

Every run is saved to a unique `gartling_*.npz`; never re-solve to answer a
follow-up.

> **Filename trap, fixed.** `tmax` was originally absent from the unsteady output
> name, so a t = 10 dt-sweep run silently overwrote the t = 140 fig-5 run that
> shared every other parameter. The name now carries `T<tmax>`, and a
> continuation IC (`file:...`) is tagged `restart` rather than having its path
> interpolated into the filename — which previously produced an unwritable name
> and lost four completed runs at the save step.

---

## 11. Corrections made during this study

Recorded because each was asserted before being checked:

| claim | status |
|---|---|
| "`w_mom` = 0.1 fixes the instability" | **retracted** — `a_mass` is the control variable (§6) |
| "the time integrator is sound; this is a basin problem" | **retracted** — tested only to t = 30; at t = 62 the steady restart diverges (§6) |
| "Chan's grid is 13 elements graded 0.89/1.78" | **retracted** — 11 elements at 1/2 grading; threshold artifact (§1) |
| "grading explains the N=5 `v` discrepancy" | **not supported** — graded is worse at N=5 and N=6 (§4) |
| "the staircase is pixel quantisation" | **wrong** — marker-pitch bias at 38–44 rows (§3) |
| "Family A is not time-accurate" | **imprecise** — it is accurate at `dt_eff`; only the dt label differs (§6) |
| 18×4 limit extrapolated to ~6.03 | **wrong** — converges to 5.819 at t = 400 (§5) |
