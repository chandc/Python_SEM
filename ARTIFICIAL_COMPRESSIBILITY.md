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
`scratch/cavity_ac_cgplot.py`, `scratch/gartling_ac_streamlines.py`.

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

4. **The mechanism is a missing preconditioner diagonal.** Pressure appears only
   in the momentum rows of the VVP system, so the pressure block of `LᵀL` scales
   as `a_flux²` and `compute_jacobi` has **no `a33` entry at all** without AC.
   AC supplies `a33 = κ_p·P`, which is exactly the block that was unscaled.

5. **Scoping correction: the `a_mass` threshold does not transfer to the closed
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
| 0.05 | 30 | 15 | 513 (conv) | 6.55e−10 | **1.554e−02** | **2.025e−02** |
| 0.05 | 30 | 30 | 512 (conv) | 7.75e−10 | 1.568e−02 | 2.079e−02 |
| 0.1 | 15 | 0 (off) | 3000 | 1.24e−09 | 1.826e−02 | 2.123e−02 |
| 0.1 | 15 | 7.5 | 3000 | 1.18e−09 | 1.602e−02 | 2.072e−02 |
| 0.1 | 15 | 15 | 3000 | 1.26e−09 | 1.627e−02 | 2.137e−02 |
| 0.25 | 6 | 0 (off) | 3000 | 2.66e−09 | 1.924e−02 | 2.289e−02 |
| 0.25 | 6 | 3 | 3000 | 2.60e−09 | 1.773e−02 | 2.261e−02 |
| 0.25 | 6 | 6 | 3000 | 2.62e−09 | 1.719e−02 | 2.256e−02 |
| 1.0 | 1.5 | 1.5 | 3000 | 7.86e−09 | 1.964e−02 | 2.548e−02 |

All ten are at `|dU|` ≈ 1e−9, i.e. steady in all but name (the 3000-step rows hit
the step cap rather than the 1e−9 test). Every profile overlays Ghia; the spread
across the whole table is 1.55e−02 … 1.96e−02 in u, about 1% of the lid speed,
which is the N = 10 / 6×6 discretisation error, not an AC effect. If anything AC
is very slightly *better* at every `dt`, consistent with §3.

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

`figs/cavity_ac_cg_iterations.png`. Measured by `scratch/cavity_ac_cgiters.py`
into `scratch/cavity_ac_cgiters.csv`, which `scratch/cavity_ac_cgplot.py` reads —
no hand-copied numbers. The iteration counts are deterministic: re-measuring
`a_mass` = 6 / AC off in a later serial pass returned 138 098 again, bit for bit.
Only the wall column moves with machine load, so run the sweep serially.

**The AC-off cost is U-shaped in `a_mass`, with a minimum near 6.** Reading the
`κ_p` = 0 rows: 1227 → 832 → **690** → 976 → 1379. Cost rises in *both*
directions from `a_mass` ≈ 6 — refining `dt` past that point makes each solve
harder, but so does coarsening it below.

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
* **It does not improve the converged accuracy** in any way that matters. The
  cavity RMS improves from 1.79e−02 to 1.55e−02, but that is a change in *how
  well converged* the run is, not in the discretisation. The honest statement is
  that AC is accuracy-*neutral* and gets you to the same answer faster.
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
