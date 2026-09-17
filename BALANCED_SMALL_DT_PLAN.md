# Demonstrating that balanced weighting makes small `dt` usable in 2D

*Plan, 2026-09-17. Scope: the **2D** solver, where convection is implicit
(Newton). The 3D Fourier code treats convection explicitly with RKW3, making each
stage a Stokes projection — its weighting question is closed (legacy) and nothing
here applies to it or should be re-opened.*

---

## 1. The claim, stated so it can fail

> For 2D time-dependent flows **with an outflow boundary**, balanced weighting
> $w_{\rm mom} = w_{\rm mass} = \sqrt{dt}$ — with artificial compressibility where
> the stability limit requires it — reaches DNS-scale time steps at full temporal
> order, while **legacy** loses the pressure and **$w = 1$** cannot pass
> $dt \approx 0.0125$.

### Why there is anything to demonstrate

At fixed `dt` the ratio is **forced**:

$$\frac{a_{\rm mass}}{a_{\rm flux}} = \frac{\rm fac1}{dt}$$

Only the *split* is a choice, and the two failure modes sit at opposite ends:

| | caps | source |
|---|---|---|
| **stability** | $a_{\rm mass} \le 6.05$ bare, ≈ 120 with AC | `GARTLING_VALIDATION.md` §6, `ARTIFICIAL_COMPRESSIBILITY.md` §4 |
| **accuracy** | $a_{\rm flux}$ must not vanish | `ls_coeffs` docstring (Poiseuille) |

Balanced is the unique split with $a_{\rm mass}\cdot a_{\rm flux} = \mathrm{fac1}$
at *every* `dt` — the geometric mean. That is the whole mechanism.

| `dt` | legacy $a_m/a_f$ | balanced $a_m/a_f$ | $w=1$ $a_m/a_f$ |
|---|---|---|---|
| 0.05 | 1.50 / 5.0e−2 | 6.71 / 2.2e−1 | 30 / 1.0 |
| 0.01 | 1.50 / 1.0e−2 | 15.0 / 1.0e−1 | 150 / 1.0 |
| 8e−4 | 1.50 / 8.0e−4 | 53.0 / 2.8e−2 | 1875 / 1.0 |

Reachable `dt` if AC closes at $a_{\rm mass}\approx120$: **$w=1$ → 1.25e−2**,
**balanced → 1.56e−4**. DNS needs ≈ 8e−4. That factor of 80 is the claim.

---

## 2. What is already established — do not re-run

| result | where | status |
|---|---|---|
| closed-domain temporal order, balanced, `dt` → 7.81e−4, order **1.99**, error 3.97e−09 at N = 12; balanced beats legacy **4.4–5.5×** at N = 8–10 and legacy's order collapses (1.86 → 0.07) | `scratch/mms2d_temporal.log` | **done** |
| $a_{\rm mass}$ is the control variable: 34 runs, no crossover, cutting across grids, initial conditions, sub-iterations and $a_{\rm flux}$ 0.1–1 | `GARTLING_VALIDATION.md` §6 | **done, at $w=1$** |
| AC moves the limit 6.05 → 60 at $\kappa_p = a_{\rm mass}/2$; closes at 120 | `ARTIFICIAL_COMPRESSIBILITY.md` §4 | **done, at $w=1$** |
| threshold does **not** transfer to closed domains (cavity fine at $a_{\rm mass}$=30) | ibid., scope note | **done** |
| cylinder Re=100, balanced, `dt`=0.1 ($a_m$=4.74): St 0.1688, C_D 1.3823, C_L 0.3378 | `CYLINDER_RE100.md` | **done** |
| cylinder, balanced, `dt`=0.05 ($a_m$=6.71) **bare**: diverged at t=157, pressure first (C_L=−3.47 with \|u\|max=1.322) | ibid. | **done** |

**The gap is exactly one thing:** every stability number above was measured at
$w = 1$, and the closed-domain order study has no outflow. Nothing yet shows the
$a_{\rm mass}$ limit behaves the same when reached *via balanced weighting*, at an
outflow, and nothing shows the answer is still *right* there.

---

## 3. The experiments

Each has an answer known before the run. (`SILENT_FAILURES.md`: the failure mode
of this code is a plausible number, so a check that cannot be failed is not a
check.)

### T1 — Temporal order at an outflow *(the core; cheap, exact)*

`scratch/mms2d_temporal.py` on a domain with an outflow edge instead of four
walls, driving it with the same manufactured solution so the outlet data is
consistent. Sweep `dt` = 2.5e−2 … 7.81e−4 × {legacy, balanced, $w=1$}, N = 10, 12.

* **Gate T1a** — balanced holds order ≈ 2 to `dt` = 7.81e−4 where the closed
  domain did, *or* fails at a `dt` the $a_{\rm mass}$ limit predicts.
* **Gate T1b** — legacy's order collapses and its **pressure** error is the first
  thing to go.
* **Gate T1c** — $w=1$ diverges at the predicted `dt` ≈ 0.0125.

This is the only experiment with an exact answer, so it carries the accuracy half
alone. *If T1a fails, the claim is false and the rest is moot — run it first.*

### T2 — Does the $a_{\rm mass}$ limit transfer to balanced? *(closes the gap)*

Gartling BFS, the flow the threshold was measured on, now under **balanced**
weighting: `dt` chosen so $a_{\rm mass}$ = 3, 6, 12, 30, 60, with and without AC at
$\kappa_p = a_{\rm mass}/2$.

* **Gate T2** — the bare limit lands at $a_{\rm mass}$ ≈ 6 and the AC limit near 60,
  as at $w=1$. Reattachment $x_r$ within 1 % of Gartling's **6.10** for every
  surviving run — stability without accuracy is not a pass.

Note $a_{\rm flux}$ differs between the two routes at equal $a_{\rm mass}$
(balanced is lower), and at $a_{\rm mass}$ = 60 balanced gives $a_{\rm flux}$ = 0.025,
*below* the 0.1 floor of Gartling's original sweep. **That is the extrapolation
being tested.**

### T3 — The headline case *(physical, pressure-integral outputs)*

Cylinder Re = 100. Its outputs are pressure integrals over the body, so a
weighting failure appears in numbers people quote rather than in a norm.

| run | `dt` | $a_m$ | config | status |
|---|---|---|---|---|
| baseline | 0.10 | 4.74 | balanced, no AC | **done**: St 0.1688 |
| bare limit | 0.05 | 6.71 | balanced, no AC | **done**: diverged |
| remedy | 0.05 | 6.71 | balanced + AC | **running** |
| refine | 0.01 | 15.0 | balanced + AC | to do |
| control | 0.05 | 30.0 | $w=1$ + AC | to do (expect divergence) |
| control | 0.05 | 1.50 | legacy | to do (expect stable, **wrong forces**) |

* **Gate T3** — St and $\overline{C_D}$ agree between `dt` = 0.1, 0.05 and 0.01
  under balanced + AC (so the answer is `dt`-converged), while the two controls
  fail in their predicted ways.

---

## 4. Order of work, and why

1. **T1** — cheapest, exact, and *decisive*: it is the only test of the accuracy
   half, and if balanced cannot hold order at an outflow the claim dies there.
2. **T2** — closes the $w=1$ → balanced transfer gap on the flow the thresholds
   came from, against a published $x_r$.
3. **T3** — the demonstration a reader will remember, but it rests on 1 and 2.

**T3's two controls are not optional.** "Balanced works" is not a result unless
the alternatives are shown to fail *on the same problem*, and the legacy control
is the more important of the two: it will run to completion and produce forces,
which is precisely the silent failure this project keeps meeting.

---

## 5. Risks

| risk | signal | response |
|---|---|---|
| the MMS outflow does not reproduce the instability | T1 shows no limit for any weighting | the threshold needs a genuine outflow with backflow; fall back to T2's BFS as the mechanism vehicle |
| the limit does **not** transfer to balanced | T2's bare limit ≠ 6 | report the measured balanced limit; the claim becomes quantitative rather than inherited, which is *better* |
| AC's window is narrower under balanced | T2 survives at fewer $\kappa_p$ | map the window; $\kappa_p$ is documented as a window, not a floor |
| DNS `dt` = 8e−4 sits at $a_{\rm mass}$ = 53, close to AC's 120 | margin under 2.5× | state the margin; do not claim headroom that is not measured |
