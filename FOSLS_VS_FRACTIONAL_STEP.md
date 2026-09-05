# FOSLS vs the fractional-step method for DNS — measured on the same problem

*2026-09-04. Evidence from `run01`, the FOSLS-3D minimal channel at Re_τ=180,
seeded from a converged fractional-step field on an identical mesh. Same
geometry, same resolution, same Re — only the formulation differs, which is what
makes the comparison worth anything.*

Companion to `FOSLS_TIME_DEPENDENT.md` (solver behaviour and the `c` regime) and
`3D_STATUS.md` §7S/§7T/§7U.

---

## 1. The headline: pointwise divergence, 3–4 orders of magnitude

![evidence](figs_fosls_vs_fs/evidence.png)

| | rms\|∇·u\| / rms\|u\| | rms\|∇·ω\| / rms\|ω\| |
|---|---|---|
| fractional step (projected) | **3.6e−01** | 5.1e+00 |
| **FOSLS** | **6.6e−05** | **8.5e−05** |

**Panel (a) is the strongest single result in this comparison.** It is the
strong-versus-weak-form distinction made visible. A projection method enforces

$$\int q\,\nabla\!\cdot\!\mathbf u \;=\; 0 \quad \forall q \in Q_h$$

— divergence-free in the **Galerkin** sense — which leaves the *pointwise*
divergence free to be O(0.1–0.5), and the curve shows exactly that, with
**spikes at every element interface** (y⁺ ≈ 20, 40, 60 …) where the weak
constraint is loosest. FOSLS minimises ∫|∇·u|² directly and sits flat at ~5e−05
across the entire channel.

The same holds for ∇·ω, where the gap is larger still (5.1 vs 8.5e−05) because
the fractional-step field has no ω unknown at all — its vorticity is obtained by
differentiating u, and nothing constrains the divergence of the result.

**Why it matters for DNS:** anything that differentiates the velocity field
inherits the pointwise error — vortex identification (Q, λ₂), enstrophy budgets,
Lagrangian particle tracking, and *a priori* SGS model testing. For those,
5,500× is not a cosmetic difference.

## 2. The cost: cond(A) = cond(L)², and it cannot be preconditioned away

FOSLS forms normal equations, so the condition number is **squared**. Measured
consequences (details in `FOSLS_TIME_DEPENDENT.md`):

* ~4700 CG iterations per stage on the production channel, ~60 s/step on a GB10.
* cond(D⁻¹A) ≈ 1.7e3 on a *2×2* mesh, rising to ~3e4 at production size.
* **Seven preconditioner approaches were built and measured; all rejected** —
  exact coarse solve, Galerkin coarse operator, every V-cycle smoothing degree,
  the momentum row weight, node-block Jacobi, per-mode solves, and warm starting
  (1.08×). The V-cycle's total work is *flat* across smoothing degree because it
  degenerates to a polynomial in D⁻¹A, and CG already builds the optimal one.
* Element-block Schwarz does work (6.1× on cond) but abandons the matrix-free
  design, 0.74 GB → 28.3 GB.

The fractional-step method pays one Poisson solve per step instead. **That is
the ~3.4× cost gap, and it is structural rather than an implementation defect.**

## 3. Small scales: a real effect, but weaker than a single snapshot suggests

**This section corrects an over-reading.** From single snapshots it looked like
the streaks had *doubled* in width (Δz⁺ 96 → 180) and ω_x had lost 30%. The time
series in **panel (c)** does not support the first claim:

| streak spacing Δz⁺ over the run | 96, 84, 76, 120, 113, 105, 113, 85, **180** |
|---|---|
| mean 108, sd 29, range 76–180 | canonical ≈ 100 |

**The spacing oscillates around ~108 — essentially canonical — and the t=2.08
value of 180 is 5.4 standard deviations above the earlier mean, i.e. a single
outlier taken at one instant of the bursting cycle, not a trend.** Quoting it as
"the streaks have doubled" was wrong.

What does survive:

* **ω_x rms declines from ~35 to ~26** (mean of the last four samples), with
  large burst-cycle oscillation (range 23.5–36.0). Real, roughly −25%, but not
  the clean monotone decay a single pair of snapshots implied.
* **Panel (b): the intra-element Legendre spectrum is only slightly below the
  fractional-step field**, and only at degrees 6–8. The difference is mild — this
  is *not* a strongly over-dissipative scheme.
* u′⁺ sits 5% below canonical while every other stress is within 2–3% (§4).

So the fair statement is **mild damping of the smallest scales, of order
20–25% in ω_x, with the streak spacing statistically indistinguishable from
canonical.** Consistent with least-squares minimisation penalising
high-frequency content, but far short of the effect a single snapshot suggested.

## 4. What FOSLS gets right: the statistics

![stats](figs_fosls_vs_fs/stats.png)

261 samples to t=2.08, in wall units (u_τ = δ = 1, ν = 1/180 by construction —
nothing rescaled, no constant fitted):

| | FOSLS | KMM Re_τ=180 | off by |
|---|---|---|---|
| −⟨u′v′⟩⁺ peak | **0.719 @ y⁺=30.0** | 0.72 @ 30 | **0.1%** |
| v′⁺ peak | 0.832 | 0.85 | 2.1% |
| w′⁺ peak | 1.074 | 1.05 | 2.3% |
| u′⁺ peak | 2.557 | 2.70 | 5.3% |
| **total-stress balance error** | **0.009** | 0 | — |

**Panel (d) is the decisive check**: for a fully developed channel
−⟨u′v′⟩⁺ + dU⁺/dy⁺ = 1 − y/δ **exactly**, with no fitted constants. It closes to
**under 1%**, having fallen monotonically 0.107 → 0.089 → 0.071 → 0.052 → 0.031
→ 0.009 as samples accumulated. The momentum transport is right.

## 5. Structures

![streaks](figs_fosls_vs_fs/streaks.png)

Near-wall streaks at three heights: elongated ribbons of alternating u′,
strongest at y⁺≈12 where u′ peaks. Lz⁺ = 192 holds roughly one to two streak
pairs — that is what makes the box "minimal".

> **Rendering note.** These are produced by EVALUATING THE SPECTRAL INTERPOLANT
> on a uniform grid (`scratch/semplot.py`), not by triangulating nodal values.
> An earlier version of this figure used `tricontourf` on the raw element-local
> array and showed X-shaped artifacts at regular x intervals, which I wrongly
> described as C⁰ element-boundary features. They were **Delaunay artifacts**:
> the array repeats every interface node once per owning element — 216 points
> carrying only 49 distinct x locations, multiplicity up to 8 — and GLL
> clustering (3.6:1 spacing ratio) turns the triangulation into slivers.
> Interpolating in y and x by the Lagrange basis and refining z by FFT
> zero-padding is exact for this discretisation and removes them entirely.

![omega_x](figs_fosls_vs_fs/omega_x.png)

Streamwise vorticity — a **primary unknown** in FOSLS, not a derivative of u.
Panel (a) resolves the elongated quasi-streamwise vortices; panel (b) is the
x-averaged cross-section.
Panel (c) is the spanwise correlation: the negative lobe locating the partner
vortex is **shallow in both the early and late fields** (−0.118 → −0.055) and
sits outside the canonical 30–50 band. Shallow minima make the *location* noisy,
so the depth is the trustworthy part — the pairing is weaker than canonical.

![three fields](figs_fosls_vs_fs/three_fields.png)

u′, ω_x and p′ on the SAME plane at y⁺≈15 — all three are primary unknowns, so
no post-processing stands between them. The low-speed streak at z⁺≈100–140 runs
the full length of the box; the ω_x vortices sit on its **flanks**, not its axis
(streak-core |ω_x| = 8.5 against 14.3 overall); and p′ is visibly smoother and
larger-scale than either, so the pressure carries the correct elliptic character
even though FOSLS never solves a Poisson equation. corr(p′, |ω_x|) = −0.11 —
vortex cores are low pressure — and the 20% lowest-speed regions sit at
p′ = −0.15 against 0.00 overall.

## 6. Summary

**Pros**

1. **Pointwise incompressibility**, 5,500× better than the projected field —
   the reason to choose it if derivative quantities matter.
2. **Vorticity is a primary unknown**, with ∇·ω enforced to 8.5e−05.
3. **One unified SPD solve** — no splitting error, no fractional-step pressure
   BC ambiguity, no inf–sup condition, CG applies directly (SPD to 5.9e−16).
4. **Second-order in time, verified** (2.00 over three refinements).
5. Statistics converge to canonical: stress balance under 1%, shear stress 0.1%.

**Cons**

1. **cond(A) = cond(L)² is intrinsic** — ~3.4× the cost, and seven
   preconditioner strategies failed to close it.
2. **Mild small-scale damping** — ω_x rms ~−25%, u′⁺ 5% low.
3. **The row weights are free parameters that trade accuracy for conditioning.**
   The system is overdetermined (8 rows, 7 unknowns), so *the discrete answer
   depends on the weighting*: `w_mom`=100 bought 4.28× fewer iterations and 109×
   worse ∇·u. A fractional-step scheme has no equivalent knob.
4. **7 unknowns vs 4** — more work and memory per grid point.
5. This implementation has **no skew-symmetric convective form** (the
   fractional-step code uses one) and does not dealias in (x,y) — measured as not
   currently harmful, but missing insurance. z *is* dealiased by the 3/2 rule.

**For DNS specifically:** FOSLS buys pointwise divergence and a directly computed
vorticity field at the price of ~3.4× cost and mild small-scale damping. If the
science depends on derivative quantities — enstrophy, vortex dynamics, SGS
modelling — that is an attractive trade. If the smallest resolved scales must be
faithful, or long statistics are needed cheaply, fractional step remains the
better instrument.

*Figures regenerated by `scratch/plot_evidence.py`, `scratch/plot_stats.py`,
`scratch/plot_streaks.py`, `scratch/plot_omegax_now.py`.*
