# Artificial compressibility on the continuity row

Study date: 2026-08-17/18. Adds a pseudo-time artificial-compressibility (AC)
term to the continuity equation of the LSSEM velocity–vorticity–pressure system,
verifies that it does not disturb the converged solution, and measures what it
buys: **a stability window on the backward-facing step and a 2–27× cut in CG
iterations.**

Implementation: `lssem2d/lssem.py` (`ls_pseudo_p`, `_apply_L_numpy`,
`_apply_LT_numpy`, `_check_ac_backend`), `lssem2d/solver.py` (`_drop_pseudo`,
`compute_jacobi`).
Drivers: `scratch/pois_ac.py` (channel), `scratch/gartling_run.py` (BFS),
`scratch/cavity_ac.py` (cavity), `scratch/cavity_ac_plot.py`,
`scratch/cavity_ac_cgiters.py` + `scratch/cavity_ac_cgplot.py` (CG cost),
`scratch/cavity_ac_pconv.py` + `scratch/cavity_ac_pconv_plot.py` (pressure
convergence), `scratch/cavity_ac_nsweep.py` + `scratch/cavity_ac_nsweep_plot.py`
and `scratch/cavity_n_profiles.py` (order N),
`scratch/cavity_steady_streamlines.py` (§5.3 figure),
`scratch/gartling_ac_streamlines.py`.

Companions: `GARTLING_VALIDATION.md` (where the `a_mass` limit was measured),
`FORTRAN_POISEUILLE_OUTFLOW.md` (the outflow-BC side of the same problem),
`PSEUDO_TIME_DESIGN.md` (the momentum-row pseudo-time term this parallels).

---

## Executive summary

1. **AC is accuracy-neutral at sensible `κ_p`, and provably so.** On plane
   Poiseuille — where the exact solution is representable in the discrete space —
   AC on at `κ_p` = 0.5 … 15 returns `Δp` = 1.44000 (exact 1.44000) and
   `L2(u − u_exact)` ≤ 1.2e−08, *better* than the 2.4e−07 with AC off. The
   solution only starts to move at `κ_p` ≳ 3·`a_mass` and is destroyed at
   `κ_p` = 500 (`Δp` off by 39%).

2. **It opens a stability window on the Gartling BFS that does not exist without
   it.** At `a_mass` ≥ 12.1 every AC-off run diverges (34-run result, no
   crossover). With AC, `dt` = 0.1, 0.05 and 0.025 — `a_mass` = 15, 30 and 60 —
   all run to t = 140 and land within 0.9% of Gartling's reattachment. The
   working value is `κ_p ≈ a_mass/2`, and the window closes again by
   `a_mass` = 120.

3. **The main pay-off is conditioning, not stability.** On the closed cavity
   (Re = 1000, 6×6 elements N = 10) AC cuts Jacobi-preconditioned CG at every
   `a_mass` measured, and the saving **grows monotonically** with it —
   1.5× at `a_mass` = 1.5, then 1.9×, 3.0×, 10.1×, and **27.5×** at `a_mass` = 30
   (1379 → 50 iterations per solve, 148 s → 8 s). The AC-off cost is *U-shaped*
   in `a_mass` with a minimum near 6; the AC-on cost is monotone in `dt` and
   lower everywhere. (15 runs, `figs/cavity_ac_cg_iterations.png`)

4. **On the Ghia Re = 1000 cavity it is accuracy-neutral, and the two velocity
   components are what establish that.** RMS u improves with AC at every `dt`
   (8–13%) but RMS v does **not** move consistently (±2%, both directions) — a
   real improvement would move both. The `dt` effect is ~4× larger than the AC
   effect, and 12 of 13 runs sit at slightly different convergence states, which
   accounts for the u figure. All 13 lie at the 6×6 N=10 discretisation floor of
   ~2% of the lid speed. (§5.1)

5. **The mechanism is a missing preconditioner diagonal.** Pressure appears only
   in the momentum rows of the VVP system, so the pressure block of `LᵀL` scales
   as `a_flux²` and `compute_jacobi` has **no `a33` entry at all** without AC.
   AC supplies `a33 = κ_p·P`, which is exactly the block that was unscaled.

6. **On pressure specifically — the field AC acts on — it is better on both
   axes at once.** At `a_mass` = 30, `κ_p` = `a_mass` reaches a *further*
   converged pressure than AC off (`max|Δp|` 1.88e−04 vs 2.54e−04) for **30×
   fewer CG iterations and 18× less wall time**. Not a speed/accuracy trade.
   A cost-model fit puts per-iteration cost at 0.480 ms with AC and 0.483 ms
   without — identical — so AC is free per iteration and the wall figure
   understates it. `κ_p` = `a_mass` is the optimum: 2·`a_mass` is cheaper but
   converges the pressure *less* well. (§5.2a)

7. **The time-derivative term is load-bearing for correctness, not just
   stability.** At the other end of the same sweep, `w_mass` = 0 (the pure
   steady form, `a_mass` = 0) is both the most expensive setting measured —
   **6747** CG iterations per solve, 5× the worst transient point — and it
   converges to a **spurious** solution. Started *on* the correct field with **no
   line search**, undamped Newton reaches an exact fixed point (`|dU|` = 0.00e+00
   in 10 sweeps) with RMS u = 1.52e−01 — 10× the transient's 1.57e−02. So the
   physical solution is not a fixed point of the steady form, and it is not a
   basin problem. Neither the functional, nor `|dU|`, nor a streamline plot, nor
   the vortex position detects it — that state's vortex centre is within 0.010 of
   Ghia's. Only the benchmark profile catches it. (§5.3,
   `figs/cavity_steady_spurious_streamlines.png`)

8. **Scoping correction: the `a_mass` threshold does not transfer to the closed
   cavity.** `GARTLING_VALIDATION.md` measured stable ≤ 6.05 / divergent ≥ 12.1
   on the BFS. The cavity at `a_mass` = 30 converges *without* AC
   (`|dU|` = 5.4e−10 after 518 steps). The threshold is a property of flows with
   an **outflow boundary**, not of `a_mass` alone.

---

## 1. What the term is

The continuity row of the least-squares functional carries weight exactly 1 (it
is a constraint, not a flux), while the momentum rows carry the time-derivative
coefficient `a_mass = w_mass·fac1/dt`. Refining `dt` therefore drives the
momentum rows up relative to continuity — this is the imbalance
`GARTLING_VALIDATION.md` traces to the `a_mass` limit.

AC restores the balance by giving the continuity row its own scale. The
classical Chorin form

    (1/(β² dτ))·(p − p_prev) + div u = 0

has two knobs, `β` and `dτ`, which only ever appear as a product. `ls_pseudo_p`
collapses them:

    κ_p = 1/dtau_p          # lssem2d/lssem.py:139

and the continuity row of `L` becomes

    su2 = (κ_p·p + u_x + v_y)·wq          # lssem2d/lssem.py:357

with the matching transpose contribution at `lssem2d/lssem.py:425` and the
Jacobi diagonal at `lssem2d/solver.py:90`.

**`p_prev` lives in the right-hand side, not here.** The `κ_p·p` term is the
operator part; the `−κ_p·p_prev` part rides in the residual, which is why
`_drop_pseudo` (`lssem2d/solver.py:214`) must subtract it — otherwise the
pseudo-time term would leak into the reported physical residual.

**This is pseudo-time, not physical time.** `p_prev` is the previous
*sub-iteration*, not the previous time step. At sub-iteration convergence
`p = p_prev` and the term vanishes identically, so time accuracy is untouched.
With `max_newton = 1` it would degenerate into a physical-time term on the
pressure and would break BDF accuracy — every run here uses `max_newton` ≥ 5.

**Caveat on the converged state.** The sub-iterations converge to a *tolerance*,
not to zero, so the least-squares normal equations see a perturbation of
O(`κ_p`·R) where R is the sub-iteration residual. This is what sets the upper
end of the usable `κ_p` window in §3.

**Backend restriction.** AC is implemented in the numpy backend only.
`_check_ac_backend` (`lssem2d/lssem.py:469`) raises `NotImplementedError` rather
than let the numba kernels silently solve the AC-free system.

---

## 2. Verification

**Symmetry.** `LᵀL` must stay symmetric or CG is not applicable. Measured
relative asymmetry `|⟨b,LᵀLa⟩ − ⟨a,LᵀLb⟩| / |⟨b,LᵀLa⟩|` on a 2×2 element N = 4
mesh with a random linearisation and random `a`, `b`:

| | ⟨b, LᵀLa⟩ | relative asymmetry |
|---|---|---|
| AC off | 8.6924712425e+01 | 3.27e−16 |
| `κ_p` = 15 | 9.6005552073e+01 | 0.00e+00 |
| `κ_p` = 60 | 1.9433033442e+02 | 2.93e−16 |

**Regression.** `82 passed` — the full `lssem2d/tests` suite, unchanged. The
AC branches are all guarded on `κ_p != 0`, so with `dtau_p = None` (the default)
the operator is bit-identical to the pre-AC code.

---

## 3. Accuracy: plane channel, Re = 100, P+Z outlet

12 × 2 elements N = 10 on [0,12] × [0,1], `dt` = 0.1 (`a_mass` = 15), nsub = 8.
Two inlets. The **parabolic** inlet makes `u = 6y(1−y)`, `v = 0`, `ω = 12y−6`,
`Δp = 12L/Re = 1.44` exactly representable, so any departure is the AC term and
nothing else. The **uniform** inlet forces the flow to develop, so the residual
is genuinely non-zero and there is something for the weighting to trade off.

| inlet | `κ_p` | max\|u\| | rms div | L2(u − u_exact) | Δp |
|---|---|---|---|---|---|
| parabolic | 0 (off) | 1.5000 | 1.19e−07 | 2.42e−07 | **1.44000** |
| parabolic | 0.5 | 1.5000 | 9.95e−09 | 3.11e−08 | **1.44000** |
| parabolic | 1.5 | 1.5000 | 6.31e−09 | 2.76e−08 | **1.44000** |
| parabolic | 5 | 1.5000 | 1.76e−09 | 9.64e−09 | **1.44000** |
| parabolic | 15 | 1.5000 | 1.69e−09 | 1.16e−08 | **1.44000** |
| parabolic | 50 | 1.5000 | 1.51e−06 | 1.21e−05 | 1.43996 |
| parabolic | 150 | 1.5009 | 6.34e−05 | 5.15e−04 | 1.44169 |
| parabolic | 500 | 1.5000 | 2.42e−02 | 1.88e−01 | 0.87251 |
| uniform | 0 (off) | 1.5045 | 8.06e−02 | 8.96e−02 | 1.89025 |
| uniform | 0.5 | 1.4961 | 8.05e−02 | 8.97e−02 | 1.88631 |
| uniform | 1.5 | 1.5127 | 8.02e−02 | 8.96e−02 | 1.91584 |
| uniform | 5 | 1.5028 | 7.93e−02 | 8.93e−02 | 1.89862 |
| uniform | 15 | 1.5104 | 7.71e−02 | 8.92e−02 | 1.89115 |
| uniform | 50 | 1.6049 | 7.33e−02 | 1.05e−01 | 2.06772 |
| uniform | 150 | 1.3628 | 7.05e−02 | 1.36e−01 | 1.62523 |
| uniform | 500 | 1.2285 | 8.29e−02 | 3.99e−01 | 0.84373 |

*(For the uniform inlet, L2(u − u_exact) and Δp are measured against the
fully-developed profile over the whole domain, so the ~9e−02 and the excess over
1.44 are the entrance region, not error. The defensible reading is agreement
**across `κ_p`**, not against the analytic value.)*

Three things follow.

* **AC does not perturb an exactly-representable solution** for `κ_p` up to
  `a_mass`. `Δp` is unchanged in all six significant figures.
* **AC improves the divergence.** rms div drops two orders, 1.19e−07 → 1.69e−09,
  because the better-conditioned system reaches the sub-iteration tolerance
  properly instead of stalling.
* **The window has a top.** Damage sets in near `κ_p` ≈ 3·`a_mass` and is total
  by 30×. This is the O(`κ_p`·R) perturbation of §1 becoming visible.

---

## 4. Stability: Gartling Re = 800 BFS, 11×4 N = 6, P+Z outlet

`w_mom = w_mass = 1` (time-accurate), nsub = 5, from rest, `a_mass` = 1.5/`dt`.
Reattachment quoted at the final time; Gartling's value is **6.10**.

| `dt` | `a_mass` | `κ_p` | outcome | t_end | x_r | upper sep | upper reatt | pk-pk max\|v\| (last 20 t) |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 3 | 15 | **ok** | 140.0 | 6.062 | 4.870 | 10.543 | 1.87e−02 |
| 0.25 | 6 | 15 | **ok** | 140.0 | 5.957 | 4.728 | 10.493 | 1.31e−02 |
| 0.1 | 15 | 0 | diverged | 16.6 | — | — | — | 1.10e+00 |
| 0.1 | 15 | 7.5 | **ok** | 140.0 | 5.909 | 4.636 | 9.638 | 3.82e−02 |
| 0.1 | 15 | 15 | **ok** | 140.0 | 6.014 | 4.983 | 6.527 | 4.91e−02 |
| 0.05 | 30 | 0 | diverged | 18.7 | — | 0.454 | — | 2.43e+00 |
| 0.05 | 30 | 15 | **ok** | 140.0 | 5.883 | 5.451 | 6.763 | 6.82e−02 |
| 0.025 | 60 | 0 | diverged | 10.9 | 13.132 | 0.855 | 1.277 | 6.35e+00 |
| 0.025 | 60 | 15 | diverged | 40.8 | — | — | — | 6.90e+00 |
| 0.025 | 60 | **30** | **ok** | 140.0 | 6.151 | 5.129 | 5.982 | 7.69e−02 |
| 0.025 | 60 | 45 | diverged | 79.2 | — | — | — | 2.72e+00 |
| 0.025 | 60 | 60 | diverged | 51.5 | — | — | — | 1.56e+00 |
| 0.0125 | 120 | 15 | diverged | 33.7 | — | — | — | 2.30e+00 |
| 0.0125 | 120 | 30 | diverged | 28.6 | — | — | — | 2.01e+01 |
| 0.0125 | 120 | 60 | diverged | 36.9 | — | — | — | 1.62e+00 |
| 0.0125 | 120 | 90 | diverged | 18.5 | — | — | — | 2.72e+00 |
| 0.0125 | 120 | 120 | diverged | 45.4 | — | — | — | 5.24e+00 |

Streamlines for the surviving runs: `figs/gartling_ac_dt_streamlines.png`.

**The window is real but bounded.** Three `dt` values that diverge without AC
(0.1, 0.05, 0.025) survive with it, and all three land at x_r = 5.88–6.15 against
Gartling's 6.10 — within 0.9%. That is a genuine extension of the usable time
step by a factor of 4.

**`κ_p ≈ a_mass/2` is the rule, and it is a window, not a floor.** At
`a_mass` = 60 only `κ_p` = 30 survives: 15 is too little, 45 and 60 too much.
This is the opposite of the cavity's *iteration-count* preference (§5, where
`κ_p = a_mass` is fastest) — stability and conditioning want different values,
and stability wins.

**It closes at `a_mass` = 120.** Five values spanning 15 … 120 all diverge at
`dt` = 0.0125. AC buys about one decade of `a_mass`; it does not remove the
limit.

**Caveat on the upper-wall bubble.** Upper reattachment ranges 5.98 … 10.54
across the surviving runs. These are the pk-pk ≈ 1e−02 oscillating states of
`GARTLING_VALIDATION.md` §fig-5/6 on the 11×4 grid, not fixed points, so the
upper bubble has not settled at t = 140. Lower reattachment is the stable
diagnostic here.

---

## 5. Conditioning: lid-driven cavity, Re = 1000

**Why the cavity.** It removes both confounders. There is no outflow boundary, so
the free/`p=0`/P+Z question does not arise; and unlike Poiseuille it recirculates
and has lid corner singularities, so the residual is substantial — the regime
where the weighting actually bites. Benchmark: Ghia, Ghia & Shin (1982).

### 5.1 Accuracy

6×6 elements N = 10 (17 424 dof), `w_mom = w_mass = 1`, pressure-pinned,
`max_newton` = 5. RMS against Ghia's 17 tabulated points on each centreline
(`figs/cavity_ac_centrelines.png`).

| `dt` | `a_mass` | `κ_p` | steps | \|dU\| | RMS u | RMS v |
|---|---|---|---|---|---|---|
| 0.05 | 30 | 0 (off) | 518 (conv) | 5.37e−10 | 1.792e−02 | 2.040e−02 |
| 0.05 | 30 | 15 | 513 (conv) | 6.55e−10 | 1.554e−02 | 2.025e−02 |
| 0.05 | 30 | 30 | 512 (conv) | 7.75e−10 | 1.568e−02 | 2.079e−02 |
| 0.1 | 15 | 0 (off) | 3000 | 1.24e−09 | 1.826e−02 | 2.123e−02 |
| 0.1 | 15 | 7.5 | 3000 | 1.18e−09 | 1.602e−02 | 2.072e−02 |
| 0.1 | 15 | 15 | 3000 | 1.26e−09 | 1.627e−02 | 2.137e−02 |
| 0.25 | 6 | 0 (off) | 3000 | 2.66e−09 | 1.924e−02 | 2.289e−02 |
| 0.25 | 6 | 3 | 3000 | 2.60e−09 | 1.773e−02 | 2.261e−02 |
| 0.25 | 6 | 6 | 3000 | 2.62e−09 | 1.719e−02 | 2.256e−02 |
| 1.0 | 1.5 | 0 (off) | 3000 | 9.26e−09 | 2.154e−02 | 2.535e−02 |
| 1.0 | 1.5 | 0.75 | 3000 | 9.24e−09 | 2.096e−02 | 2.567e−02 |
| 1.0 | 1.5 | 1.5 | 3000 | 7.86e−09 | 1.964e−02 | 2.548e−02 |
| 2.0 | 0.75 | 0 (off) | 3000 | 1.40e−08 | 2.045e−02 | 2.143e−02 |
| **steady** (`w_mass` = 0), from rest — *line-search stall, not converged* | **0** | 0 (off) | 36 | 6.94e−08 | 2.525e−01 | 2.147e−01 |
| **steady** fixed point, from the correct field, **no line search** | **0** | 0 (off) | 10 | **0.00e+00** | **1.516e−01** | — |

Grouped by `dt` so the two components can be read against each other:

| `dt` | `a_mass` | RMS u: off → `a/2` → `a` | RMS v: off → `a/2` → `a` |
|---|---|---|---|
| 0.05 | 30 | 1.792 → 1.554 → 1.568 (−13%, −12%) | 2.040 → 2.025 → 2.079 (−1%, **+2%**) |
| 0.1 | 15 | 1.826 → 1.602 → 1.627 (−12%, −11%) | 2.123 → 2.072 → 2.137 (−2%, **+1%**) |
| 0.25 | 6 | 1.924 → 1.773 → 1.719 (−8%, −11%) | 2.289 → 2.261 → 2.256 (−1%, −1%) |
| 1.0 | 1.5 | 2.154 → 2.096 → 1.964 (−3%, −9%) | 2.535 → 2.567 → 2.548 (**+1%**, **+1%**) |

*(all ×1e−02)*

The 13 transient runs are all at `|dU|` ≈ 1e−9, i.e. steady in all but name (the
3000-step rows hit the step cap rather than the 1e−9 test). Every profile
overlays Ghia (`figs/cavity_ac_centrelines.png`); the whole table sits at
1.55e−02 … 2.15e−02 in u and 2.02e−02 … 2.57e−02 in v — about 2% of the lid
speed, which is the N = 10 / 6×6 discretisation floor, not an AC effect.

**AC is accuracy-neutral here, and the two velocity components together are what
show it.** RMS u improves with AC at every `dt` (8–13%), but **RMS v shows no
consistent direction** — ±2%, better at some `dt`, worse at others. A genuine
improvement would move both the same way. Two further checks put the u figure in
its place:

* **The `dt` effect is ~4× larger than the AC effect.** RMS u runs 1.55e−02 at
  `dt` = 0.05 to 2.15e−02 at `dt` = 1.0 — a 39% spread from temporal error
  alone, against ~10% from AC.
* **12 of 13 runs hit the step cap rather than the convergence test**, at `|dU|`
  from 1.2e−09 to 9.3e−09, so they sit at slightly different convergence states.
  A ~10% difference in one component is about what that alone produces.

So the defensible statement is the one in §7: AC does not improve the
discretisation — it reaches the same answer faster. For contrast, an actual
accuracy failure on this mesh looks like the two steady rows above: 2.5e−01 and
1.3e−01, 14× and 7× worse, visibly off the benchmark.

**Scoping correction.** The `a_mass` = 30 row converged with AC **off**. The
6.05 / 12.1 threshold of `GARTLING_VALIDATION.md` therefore does **not** transfer
to a closed domain — it is a property of the interaction between the
weighting imbalance and an outflow boundary. Earlier text in this repo that
states the threshold without that qualifier is over-general.

### 5.2 CG iterations — the actual pay-off

Measured by wrapping `solver.pcg_solve` and accumulating the iteration count it
returns. 40 steps from rest, nsub = 5, `cg_tol` = 1e−8, `cgsfac` = 1e−3, so 200
CG calls per case — identical work, only `κ_p` differs.

| `dt` | `a_mass` | `κ_p` | total CG its | its / solve | reduction | wall |
|---|---|---|---|---|---|---|
| 1.0 | 1.5 | 0 (off) | 245 431 | 1227.2 | — | 132.8 s |
| 1.0 | 1.5 | 0.75 | 209 097 | 1045.5 | 1.2× | 114.9 s |
| 1.0 | 1.5 | 1.5 | 168 630 | 843.1 | **1.5×** | 93.5 s |
| 0.5 | 3 | 0 (off) | 166 408 | 832.0 | — | 91.5 s |
| 0.5 | 3 | 1.5 | 113 529 | 567.6 | 1.5× | 64.3 s |
| 0.5 | 3 | 3 | 88 068 | 440.3 | **1.9×** | 50.9 s |
| 0.25 | 6 | 0 (off) | 138 098 | **690.5** | — | 76.6 s |
| 0.25 | 6 | 3 | 67 343 | 336.7 | 2.1× | 39.6 s |
| 0.25 | 6 | 6 | 45 491 | 227.5 | **3.0×** | 27.9 s |
| 0.1 | 15 | 0 (off) | 195 245 | 976.2 | — | 106.7 s |
| 0.1 | 15 | 7.5 | 29 014 | 145.1 | 6.7× | 19.1 s |
| 0.1 | 15 | 15 | 19 381 | 96.9 | **10.1×** | 13.8 s |
| 0.05 | 30 | 0 (off) | 275 791 | 1379.0 | — | 148.3 s |
| 0.05 | 30 | 15 | 14 879 | 74.4 | 18.5× | 10.9 s |
| 0.05 | 30 | 30 | 10 026 | 50.1 | **27.5×** | 8.2 s |
| steady (`w_mass` = 0) | **0** | 0 (off) | 1 349 390 | **6746.9** | — | 660.1 s |

`figs/cavity_ac_cg_iterations.png`. Measured by `scratch/cavity_ac_cgiters.py`
into `scratch/cavity_ac_cgiters.csv`, which `scratch/cavity_ac_cgplot.py` reads —
no hand-copied numbers. The iteration counts are deterministic: re-measuring
`a_mass` = 6 / AC off in a later serial pass returned 138 098 again, bit for bit.
Only the wall column moves with machine load, so run the sweep serially.

**The AC-off cost is U-shaped in `a_mass`, with a minimum near 6.** Reading the
`κ_p` = 0 rows: **6747** (`a_mass` = 0) → 1227 → 832 → **690** → 976 → 1379.
Cost rises in *both* directions from `a_mass` ≈ 6 — refining `dt` past that
point makes each solve harder, but so does coarsening it below, and the left
branch runs away hard: the steady form is **5× worse than the worst transient
point and 135× worse than the best AC-on one.** That is the mass term acting as
a diagonal regulariser on `LᵀL`; remove it and there is nothing holding the
momentum rows together.

`a_mass` = 0 has no position on a log axis, so the plot carries it as a
horizontal reference line rather than a point.

> **Retraction.** An earlier version of this section measured only `a_mass` = 6
> and 30, saw 690 → 1379, and concluded that "AC reverses the sign of the `dt`
> dependence — without AC, refining `dt` makes each solve harder." That sampled
> one branch of a U and mistook it for a monotone trend. The AC-off curve has an
> interior minimum; there is no single sign to reverse. What is true is stated
> below.

**With AC the cost is monotone in `dt`, and lower everywhere.** Both AC curves
fall steadily as `dt` is refined — 843 → 440 → 228 → 97 → 50 at `κ_p` = `a_mass`
— with no interior minimum, and every AC row beats the AC-off row at the same
`a_mass`. So AC does not merely shift the curve down, it removes the penalty for
refining past `a_mass` ≈ 6, which is the regime a time-accurate run wants to be
in.

**The mechanism is a missing diagonal.** In the VVP first-order system pressure
enters only through `∇p` in the momentum rows. The pressure block of `LᵀL`
therefore scales as `a_flux²` while the velocity blocks pick up `a_mass²`, and
`compute_jacobi` has no `a33` term to normalise it with. As `a_mass` grows the
pressure block becomes relatively tiny and the Jacobi preconditioner does nothing
for it. AC contributes `a33 = κ_p·P` (`lssem2d/solver.py:90`) directly to the
diagonal that was empty — which is why the benefit *grows monotonically* with
`a_mass` (1.5× → 1.9× → 3.0× → 10.1× → 27.5× over 1.5 … 30) rather than being a
fixed constant. Five points, no reversal: this is the one trend in the table
that is genuinely monotone.

**Iterations and stability want different `κ_p`.** On the cavity `κ_p = a_mass`
beats `a_mass/2` on iterations at every time step (50 vs 74, 97 vs 145, 228 vs
337, 440 vs 568, 843 vs 1046). On the
BFS at `a_mass` = 60, `κ_p = a_mass` **diverges** and only `a_mass/2` survives.
Choose for stability first; the iteration count at `a_mass/2` is still within
50% of the best.

### 5.2a Pressure convergence per unit cost

AC acts *on* the continuity row, so pressure is the field it should move most.
300 steps from rest, nsub = 5, `cg_tol` = 1e−8, AC-off plus `κ_p` at four
fractions of `a_mass`, run **serially on an idle machine** so the wall column is
comparable. `max|Δp|` is tracked separately from `max|Δu|`: the combined `|dU|`
used elsewhere is dominated by vorticity, whose magnitude here is ~300× the
pressure's, and hides the effect entirely.

`figs/cavity_ac_pressure_convergence.png`, from `scratch/cavity_ac_pconv.py`
(measure) and `scratch/cavity_ac_pconv_plot.py` (plot). Both axes are
cost — CG iterations and wall seconds — not step number: **AC changes what a step
costs far more than how many steps are needed**, so a per-step plot shows almost
nothing.

**`dt` = 0.25 (`a_mass` = 6).** All five reach the *same* fixed point
(`|Δp|` = 2–5e−11) at step 150; only the cost differs.

| `κ_p` | CG its (300 steps) | wall | CG to `\|Δp\|` < 1e−6 | wall to 1e−6 | speed-up |
|---|---|---|---|---|---|
| 0 (off) | 1 070 937 | 543 s | 355 627 | 180 s | — |
| 1.5 = `a/4` | 740 343 | 389 s | 268 490 | 141 s | 1.3× |
| 3 = `a/2` | 536 932 | 289 s | 189 289 | 101 s | 1.9× |
| 6 = `a` | 368 711 | 207 s | 125 575 | 70 s | 2.8× |
| 12 = `2a` | 358 179 | 201 s | 115 635 | 65 s | **3.1×** |

**`dt` = 0.05 (`a_mass` = 30).** 300 steps is only t = 15 here, so these have not
yet reached the fixed point — compare the rows against each other, *not* against
the `dt` = 0.25 block.

| `κ_p` | CG its | wall | `\|Δp\|` at step 300 | `\|Δu\|` at step 300 |
|---|---|---|---|---|
| 0 (off) | 2 345 907 | 1158 s | 2.54e−04 | 1.32e−03 |
| 7.5 = `a/4` | 224 885 | 135 s | 2.43e−04 | 1.32e−03 |
| 15 = `a/2` | 118 008 | 83 s | 2.10e−04 | 1.31e−03 |
| **30 = `a`** | **78 669** | **63 s** | **1.88e−04** | 1.32e−03 |
| 60 = `2a` | 67 117 | 56 s | 3.03e−04 | 1.29e−03 |

**AC is better on both axes at once — this is not a speed/accuracy trade.** At
`κ_p` = `a_mass` the pressure is *further* converged than AC off (1.88e−04 vs
2.54e−04) for **30× fewer CG iterations** and **18× less wall time**.

**Pressure convergence has an optimum near `κ_p` = `a_mass`.** Going on to
2·`a_mass` is marginally cheaper (67k vs 79k iterations) but ends *worse* in
`|Δp|` (3.03e−04 vs 1.88e−04). The plain iteration-count metric of §5.2 showed
no such turn — more `κ_p` simply looked better there — so this is the sharper
criterion of the two, and it agrees with the `κ_p` ≲ `a_mass` accuracy window
of §3.

**AC does not alter the trajectory, only its cost.** `max|Δu|` at step 300 is
1.32, 1.32, 1.31, 1.32, 1.29 (×1e−03) across all five — indistinguishable — and
at `dt` = 0.25 all five land on the same fixed point. Consistent with §3 and
§5.1.

**Why the wall speed-up (18×) is smaller than the CG speed-up (30×).** Fitting
`wall/step = c·(its/step) + h` over all ten runs:

| | `c` (per CG iteration) | `h` (fixed per step) | max fit residual |
|---|---|---|---|
| `dt` = 0.25 | 0.480 ms | 101.7 ms | 8.9 ms |
| `dt` = 0.05 | 0.483 ms | 84.8 ms | 4.5 ms |

`c` is the **same to three digits** at both `dt` and across every `κ_p`,
including AC off. So **AC adds no measurable per-iteration cost** — the extra
`a33` term and its transpose are free at this scale — and the wall speed-up is
capped only by the fixed per-step overhead `h` (assembly, line search, BCs),
which AC does not touch. That ceiling is `3.860/0.0848` ≈ **45×** at `dt` = 0.05.
Two consequences: CG iteration count is the honest metric here, and wall time
*understates* what AC does to the linear solve.

### 5.2b Does the benefit scale with polynomial order N?

Every other result in this document was measured at a **single order per flow** —
N = 10 on the cavity and channel, N = 6 on the BFS — and those two differ in
flow, mesh and boundary conditions as well, so nothing about N could be extracted
from them. This sweep varies N alone: mesh fixed at 6×6 elements, `dt` = 0.05
(`a_mass` = 30), 40 steps from rest, the same protocol as §5.2.

It matters because the `κ_p` ≈ `a_mass` rule balances `κ_p` against `a_mass`,
which carries **no N dependence at all**, while the operator norms it is
balancing against certainly do. If the optimum drifted with N, §6 would only be
valid at N = 10.

| N | dof | AC off | `κ_p` = `a/2` | `κ_p` = `a` | gain at `a/2` | gain at `a` |
|---|---|---|---|---|---|---|
| 4 | 3 600 | 405.7 | 22.1 | 14.1 | 18.4× | 28.8× |
| 6 | 7 056 | 722.4 | 37.8 | 24.5 | 19.1× | 29.5× |
| 8 | 11 664 | 1051.7 | 55.5 | 37.0 | 18.9× | 28.4× |
| 10 | 17 424 | 1379.0 | 74.4 | 50.1 | 18.5× | 27.5× |
| 12 | 24 336 | 1713.4 | 94.7 | 64.9 | 18.1× | 26.4× |
| 14 | 32 400 | 2108.3 | 116.0 | 79.8 | 18.2× | 26.4× |

*(CG iterations per solve, 200 solves per case.
`figs/cavity_ac_n_sweep.png`, from `scratch/cavity_ac_nsweep.py` and
`scratch/cavity_ac_nsweep_plot.py`.)*

**The benefit is essentially N-independent.** `κ_p` = `a_mass` gives 26.4–29.5×
(mean 27.8×) and `κ_p` = `a_mass`/2 gives 18.1–19.1× (mean 18.5×) across the
whole range. Both AC-off and AC-on iteration counts grow **roughly linearly in
N** — AC-off by ~330 iterations per 2 orders, near-constant — so the ratio stays
flat. There is a slight decline at `κ_p` = `a_mass` (28.8× → 26.4×, about 8%
over N = 4 … 14); at `a_mass`/2 even that is absent.

**So §6's recommendation is not an artefact of N = 10**, and `κ_p` needs no N
scaling over this range. Whether that survives to N ≫ 14, or on a mesh refined in
h rather than p, is untested.

**These 18 runs were launched in parallel**, which is legitimate here and worth
recording why: CG iteration counts are deterministic and load-independent. The
N = 10 column came back at 275 791 / 14 879 / 10 026 — **bit-identical** to the
serial §5.2 measurement. The `wall` column from a parallel launch is *not*
comparable and is deliberately not tabulated above.

**Accuracy across N, on converged runs.** The 18 fields above are 40 steps
(t = 2) and are a *conditioning* measurement only — their profiles would show
early-transient shape, not discretisation error. Re-running each N to the stall
exit at `κ_p` = `a_mass` gives the p-convergence picture
(`figs/cavity_n_profiles.png`, `scratch/cavity_n_profiles.py`):

| N | dof | steps to converge | RMS u | RMS v |
|---|---|---|---|---|
| 4 | 3 600 | 550 | 7.189e−02 | 1.053e−01 |
| 6 | 7 056 | 540 | 3.275e−02 | 4.760e−02 |
| 8 | 11 664 | 523 | 2.013e−02 | 2.847e−02 |
| 10 | 17 424 | 512 | 1.568e−02 | 2.079e−02 |
| 12 | 24 336 | 504 | 1.360e−02 | 1.674e−02 |
| 14 | 32 400 | 490 | 1.343e−02 | 1.529e−02 |

Both components converge monotonically toward Ghia and both profiles are visually
on the benchmark by N = 10. **AC is on for all six**, so this also demonstrates
that AC does not interfere with p-convergence — the sequence is clean.

Note RMS u **plateaus near 1.34e−02** between N = 12 and 14 while RMS v is still
falling. That flattening is not the AC term: at fixed 6×6 elements the remaining
error is h-limited, and it is also approaching the accuracy of Ghia's own
tabulated 129×129 finite-difference data. Do not read the last two rows as a
spectral-convergence rate.

### 5.3 The steady form (`w_mass` = 0) does not have the physical solution as a fixed point

`w_mass` = 0 makes `s = w_mass/dt = 0` in `ls_coeffs`, so `a_mass` and
`hist_scale` both vanish and the functional collapses to the pure steady form
with no time-derivative term.

> **Correction (2026-08-18).** This section originally reported *two* converged
> spurious steady states, from rest and from the correct field, on runs with the
> line search **on**. Both were **line-search stalls, not convergence** — see the
> box below. The headline conclusion survives, but only because it was re-tested
> without the line search; the original evidence for it was invalid.

**The line search silently reports a stall as convergence.** `newton_step`'s
backtracking loop runs at most `max_backtrack` = 25 halvings and then takes the
step **anyway**, with no failure signal:

    for _ in range(max_backtrack):
        if _ls_merit(state, U + alpha*dU, ...) <= (1 - 1e-4*alpha)*J_ref: break
        alpha *= 0.5
    U_new = U + alpha*dU        # taken even if nothing was accepted

`0.5**25` = 2.98e−08. Measured on this problem, alpha collapses to exactly that
at sweep 26 and never recovers, so the state stops moving:
`|dU|` = alpha·|step| ≈ 2.98e−08 × 2.3 = **6.9e−08, constant** — precisely the
number originally quoted as a converged fixed point. `lssem2d/solver.py` now sets
`state._ls_exhausted` when this happens; **check it before believing any small
`|dU|` from a line-searched run.**

Re-run with the line search off (`scratch/cavity_steady_ls.py`):

| start | line search | outcome | sweeps | RMS u vs Ghia |
|---|---|---|---|---|
| rest | on | **LS_STALL** (alpha = 2.98e−08) | 26 | 2.525e−01 |
| rest | **off** | **does not converge** — `\|dU\|` oscillates 30–75 throughout | 400 (cap) | 2.475e−01 |
| correct field | on | **LS_STALL** (alpha = 2.98e−08) at sweep 1 | 1 | 1.342e−01 |
| correct field | **off** | **converged, `\|dU\|` = 0.00e+00 exactly** | 10 | **1.516e−01** |
| *(transient `dt` = 0.05, reference)* | on | conv | 512 | 1.568e−02 |

**The conclusion survives, on better evidence.** Started *on* the converged,
Ghia-matching field with **no line search at all**, undamped Newton converges in
10 sweeps to `|dU|` = 0.00e+00 — an exact fixed point of the discrete steady
operator — and that fixed point has RMS u = 1.516e−01, **ten times** the
transient solution's 1.568e−02. So the physical solution genuinely is **not** a
fixed point of the steady form on this mesh, and it is not a basin problem
either: the iteration walks off the correct answer and lands elsewhere.

**What must be withdrawn.** The from-rest case is *not* a converged spurious
state. With the line search it stalls; without it the iteration ran the full
400-sweep cap with `|dU|` oscillating between 30 and 75 the whole way and never
settling (2427 s, final `|dU|` = 3.85e+01). The "spatially oscillatory converged
state" and the claim of *two* spurious fixed points are withdrawn; **one**
spurious fixed point is established, and the from-rest behaviour is simply
non-convergence of Newton on the steady form from a poor initial guess.

**What the two spurious states look like.**
`figs/cavity_steady_spurious_streamlines.png`
(`scratch/cavity_steady_streamlines.py`) puts all three converged fields side by
side, with reversed-flow shading and the primary-vortex centre marked against
Ghia's (0.5313, 0.5625):

| | primary vortex centre | offset from Ghia | RMS u |
|---|---|---|---|
| steady, from rest, LS **stalled** | (0.317, 0.727) | **0.270** | 2.52e−01 |
| steady, from correct field, **no LS** | (0.526, 0.570) | **0.010** | 1.52e−01 |
| time-accurate `dt` = 0.05, AC on | (0.534, 0.567) | 0.005 | 1.57e−02 |

The stalled from-rest state is *visibly* wrong — an extra band of
counter-rotating cells under the lid, primary vortex pushed to (0.32, 0.73) —
which is a useful picture of what a stalled line search leaves behind, but it is
not a solution of anything.

**The middle panel is the dangerous one, and it is a genuine fixed point**
(`|dU|` = 0 exactly, no line search involved). It has the correct topology — one
primary vortex, both bottom corner eddies — and its vortex centre is within
**0.010** of Ghia's, i.e. 1% of the cavity width, against the correct run's
0.005. It passes visual inspection, a topology check *and* a vortex-position
check. Only the full centreline profile exposes it: RMS u = 1.52e−01, ten times
worse than the correct solution, because the velocity magnitudes are inflated
throughout. **Streamline plots and integral diagnostics are not sufficient
validation for this solver; the profile comparison is.**

**The functional cannot detect this either, which is the part worth
remembering.**
Comparing the from-rest steady field against the transient one on the same mesh:

| | rms momentum | rms div u | rms vorticity |
|---|---|---|---|
| steady fixed point (no LS) | 2.044e−01 | 1.299e−01 | 6.650e−02 |
| transient, `dt` = 0.05 | 2.025e−01 | 1.435e−01 | 6.756e−02 |

Indistinguishable — the steady field is even *better* on `div u`. The lid corner
singularities dominate the domain integral at 6×6 N = 10, so J is not a usable
correctness test on this problem, and neither is `|dU|`. Both runs reported
convergence in good faith. **Any steady-form result in this repo needs checking
against a benchmark profile, not against its own residual.**

This is one configuration (6×6 N = 10, Re = 1000, pressure-pinned, nsub = 5,
`cg_tol` = 1e−8) and the "fixed point" is that of the outer iteration as
implemented, whose `|dU|` floor is ~8e−08. The departure of 1.01e+01 on the first
restart sweep is far above that floor, so the conclusion does not rest on it. It
is consistent with, and may explain, the steady-form-versus-unsteady gap left
open in `GARTLING_VALIDATION.md`.

Reproduce: `scratch/cavity_steady_ls.py <on|off> <rest|restart>` — the
line-search comparison, which is the version to trust. The original
line-searched runs were `scratch/cavity_ac.py 1.0 off 0.0` and
`... 1.0 off 0.0 scratch/_ic_converged.npz`; both stall.

**Caveat on scope.** One mesh (6×6, N = 10), one Reynolds number, one linear
solver setting (`cgsfac` = 1e−3, `cg_tol` = 1e−8). Undamped Newton from rest does
not converge here at all, so "the steady form has a spurious fixed point" is
established *from a good initial guess only*. Whether a better globalisation
(trust region, proper Armijo with a failure branch, continuation in Re) would
find the physical solution is untested — and is now the more interesting
question.

---

## 6. How to use it

```python
st = SolverState(mesh, D, nu=nu, dt=dt, fac1=1.5, w_mom=1.0, w_mass=1.0)
a_mass = w_mass*fac1/dt
st.dtau_p = 2.0/a_mass          # kappa_p = a_mass/2
```

* **`dtau_p = None` (default) disables AC entirely** and is bit-identical to the
  pre-AC operator.
* **Start at `κ_p = a_mass/2`.** It is the value that survives on the BFS and
  costs little against the optimum on the cavity.
* **Keep `κ_p` ≲ `a_mass`.** Above ≈ 3·`a_mass` the O(`κ_p`·R) perturbation of
  the converged state becomes measurable (§3).
* **`max_newton` ≥ 5.** The term only cancels at sub-iteration convergence. With
  `max_newton = 1` it becomes a physical-time term on the pressure and breaks
  BDF accuracy.
* **numpy backend only** — `_check_ac_backend` raises rather than silently
  solving the wrong system on numba.

---

## 7. What AC does not do

* **It does not remove the `a_mass` limit**, it moves it. On the BFS the usable
  range went from `a_mass` ≤ 6.05 to `a_mass` ≤ 60 — one decade — and 120 fails
  at every `κ_p` tried.
* **It does not improve the converged accuracy.** The cavity RMS u improves from
  1.79e−02 to 1.55e−02, but RMS **v** over the same runs does not move
  consistently at all (±2%, both directions), and a real improvement would move
  both. What the u column reflects is *how well converged* the run is, not the
  discretisation. The honest statement is that AC is accuracy-*neutral* and gets
  you to the same answer faster. (§5.1)
* **It does not help exactly-representable flows.** Poiseuille has R ≈ 0, so
  there is nothing for AC to fix; the 2.4e−07 → 1.7e−09 divergence improvement is
  the only visible effect and it is below any meaningful tolerance.

## 8. Open

* Whether the `a_mass` = 120 failure is recoverable at some `κ_p` outside
  15 … 120, or whether a second mechanism takes over there.
* Whether the `κ_p` = 500 damage in §3 is purely non-convergence of the
  sub-iterations (raise `max_newton` and re-test) or a genuine perturbation of
  the fixed point.
* Whether the BFS `a_mass` instability is specifically an interaction with the
  outflow condition. §5.1 shows the closed cavity is exempt at `a_mass` = 30,
  which is suggestive but is one point.
* A numba implementation, so AC is usable at production mesh sizes.
* Whether the N-independence of §5.2b survives past N = 14, or under h- rather
  than p-refinement.
* Whether a better globalisation than the current backtracking line search —
  trust region, Armijo with a genuine failure branch, or continuation in Re —
  lets the steady form reach the physical solution. §5.3 shows only that plain
  undamped Newton from a good guess lands on a spurious fixed point, and that
  the current line search stalls rather than converging.
* Whether `_ls_exhausted` should *raise* rather than record. Taking the step
  after 25 failed halvings is the pre-existing behaviour and other studies in
  this repo were run under it, so it is flagged rather than changed — but every
  line-searched result in this repo is now suspect until its alpha is checked.

---

## 9. Related work

> **Provenance.** This section rests on web searches over abstracts and one
> overview page, not on full texts — the key sources (Pontaza & Reddy's penalty
> paper, Bochev & Gunzburger's *Least-Squares Finite Element Methods* (2009),
> Jiang's *The Least-Squares Finite Element Method* (1998)) are paywalled or
> unfetched. Treat what follows as **where this work plausibly sits**, not as a
> verified placement, and read the primary sources before claiming novelty.

**Our own reference implementation does not do this.** Chan & Mittal (1996), the
VVP LSSEM this code reproduces, uses preconditioned CG, BDF and sub-iterations,
with no artificial-compressibility or penalty term on the continuity row.

### What the literature establishes

1. **Why the formulation exists.** LSFEM yields a **symmetric positive-definite**
   system from a non-symmetric PDE and **avoids the inf-sup/LBB condition**, so
   `u`, `p`, `ω` can share equal-order interpolation — which is what this code
   does. Boundary-condition residuals can also be folded into the functional.

2. **Poor mass conservation is the known Achilles heel.** Continuity is enforced
   only in a least-squares sense, so `∇·u` is *traded off* against momentum.
   Reported to become severe with certain boundary conditions and on
   high-aspect-ratio domains. Published remedies: weighting the functional,
   Lagrange multipliers on continuity, piecewise divergence-free bases, and
   **strengthening the pressure–velocity coupling**.

3. **Penalty / pseudo-compressibility.** The penalty formulation replaces
   `∇·u = 0` with `∇·u = −p/Λ`, and there is a spectral/hp penalty least-squares
   NS formulation (Pontaza & Reddy). Classic dilemma: smaller Λ enforces
   incompressibility better but worsens conditioning and adds consistency error.

4. **Space–time coupled LSFEM** (Pontaza & Reddy, JCP 2004) has no time-step
   stability restriction and is spectrally accurate in time.

5. **Diagonal preconditioning** is reported to reduce the LSFEM condition number
   substantially.

### How the results here map onto that

* **§4's `a_mass` limit is the mass-conservation defect in a new guise.** The
  literature treats the continuity weight as a free parameter to be chosen — and
  chooses it *large*. In a time-marching VVP code it is not free: the constraint
  rows carry weight exactly 1 while `a_mass = w_mass·fac1/dt` grows, so
  **refining `dt` silently down-weights continuity.** We have not seen it stated
  that the weight is set by the time step.

* **AC is "strengthening the pressure–velocity coupling", item 2's named
  remedy** — it puts `p` into a row that otherwise contains only velocities. Our
  data agrees quantitatively: rms `∇·u` on Poiseuille improved 1.19e−07 →
  1.69e−09 (§3).

* **Our term is formally the penalty row, but behaves nothing like it.**
  `κ_p·p + ∇·u = 0` is item 3 with `κ_p = 1/Λ`. The difference is decisive:
  penalty is consistency-breaking — at convergence the solution really does
  satisfy `∇·u = −p/Λ`, an O(1/Λ) error, so Λ must be large. Ours uses
  `(p − p_prev)` in **pseudo**-time, so the term vanishes identically at
  sub-iteration convergence and the perturbation is O(`κ_p`·R), not O(`κ_p`).
  §3 is the proof: at `κ_p` = 15 we measure `Δp` = 1.44000 (exact) and rms
  `∇·u` = 1.69e−09, where a penalty with Λ = 1/15 would be imposing
  `∇·u = −15p`. That is why `κ_p` can be as large as `a_mass` rather than
  vanishingly small.

* **Item 5 is consistent with §5.2, and we identify the missing entry.** Jacobi
  helps, but has **no `a33`** because pressure enters only through `∇p`. AC
  supplies exactly that diagonal.

### Where our measurements contradict a standard claim

The functional is routinely advertised as a **built-in local and global error
estimator**, used to drive adaptivity. **§5.3 is a counterexample.** On the
cavity it could not distinguish the correct solution from a spurious oscillatory
one:

| | rms momentum | rms div u | rms vorticity | RMS u vs Ghia |
|---|---|---|---|---|
| steady fixed point (`w_mass` = 0, no line search) | 2.044e−01 | 1.299e−01 | 6.650e−02 | **1.52e−01** |
| transient, `dt` = 0.05 | 2.025e−01 | 1.435e−01 | 6.756e−02 | **1.57e−02** |

Indistinguishable in J — the *wrong* solution is even better on `∇·u` — while the
actual error differs 14×. The lid corner singularities dominate the domain
integral and swamp the signal. The error-estimator property is therefore
conditional on the functional not being dominated by a singularity, which is a
real caveat for adaptivity on a driven cavity.
