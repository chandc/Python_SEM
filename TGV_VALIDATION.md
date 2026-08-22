# Taylor–Green vortex at Re = 100: the first interacting-vortex validation of the 3D solver

Run date: 2026-08-20. Companion to `3D_STATUS.md` §7E (the Taylor–Green
ladder) and, methodologically, to `ARMALY_VALIDATION.md` /
`GARTLING_VALIDATION.md`. This is the first case in the repo where **vortex
stretching** — the mechanism 2D flow cannot have — is exercised and validated:
modes interact, enstrophy grows, and the run is judged against exact theory,
an internal parameter-free balance, and the published behaviour of the case.

Reproduce: `uv run --quiet python scratch/tgv3d.py run re100` (≈11 h numpy),
then `scratch/tgv3d_movie.py re100` and `scratch/tgv_re100_transient_plot.py`.

---

## 1. The case and the configuration

Classical Taylor–Green vortex on the triply periodic $(2\pi)^3$ box:

$$
\begin{aligned}
u &= \phantom{-}\sin x \cos y \cos z & \omega_x &= -\cos x \sin y \sin z \\
v &= -\cos x \sin y \cos z & \omega_y &= -\sin x \cos y \sin z \\
w &= 0 & \omega_z &= 2 \sin x \sin y \cos z \\
p &= \tfrac{1}{16}(\cos 2x + \cos 2y)(\cos 2z + 2)
\end{aligned}
$$

$\nu = 0.01 \Rightarrow Re = 100$ in the standard convention ($U_0 = k = 1$). No boundaries
of any kind: `periodic_x` + `periodic_y` (SEM seam merging) and Fourier z; the
only constraints are the all-copies pressure pin (`bc.pin_dof` — the seam
corner has multiplicity 4, §7C) and the frozen imaginary halves of the real
modes.

| | |
|---|---|
| resolution | ≈ **24³** — 3×3 elements N = 8 in (x, y), Nz = 24 (13 rfft modes) |
| time integration | RKW3/CN, dt = 0.02 (CFL target 1.0), t → 12 (600 steps) |
| formulation | **legacy row weights, no operator-AC** — the recipe settled by the ν-sweep (`3D_STATUS.md` §7A) |
| solve | batched mode-parallel PCG, tol 1e−8, guarded — **zero capped solves** in 1800 stage solves (~6000 CG/step) |
| wall | 11.1 h (numpy, pre-analytic-Jacobi, pre-tolerance-policy) |

An earlier sizing (4×4, Nz = 32, tol 1e−9) priced at 52 h with the balance
check already at 1.0000 by step 10; $Re = 100$ does not need it. The balance
meter (§3) is the evidence $24^3$ suffices.

## 2. Exact-theory anchors — all hit

| quantity | measured | exact | agreement |
|---|---|---|---|
| $E(0)$ | 31.006277 | $(2\pi)^3/8$ | **5e−16** |
| $\Omega(0)$ | 93.018830 | $3(2\pi)^3/8$ | **2e−16** |
| $E(0)/V$ | 0.125000 | $1/8$ (standard normalisation) | exact |
| $\varepsilon(0)$ | 0.007491 | $2\nu\Omega_0/V = 0.007500$ | 0.1% (finite-difference sampling) |
| early-time $\varepsilon(t)$ | $\varepsilon/\varepsilon_0 \approx 1 + 0.039\,t^2$ | quadratic start (time-analytic solution, even series) | clean quadratic, no linear term |

## 3. The transient, and the two headline numbers

![TGV Re=100 transient](figs/tgv_re100_transient.png)

* **Enstrophy first dips** (viscosity beats stretching until $t \approx 0.3$ at this
  Re — contrast $Re = 400$, where growth starts immediately), **then grows
  1.72× to its peak at $t = 4.84$**, then decays. Energy decays monotonically
  throughout: 2D flow cannot produce the red curve's rise; this is vortex
  stretching, measured.
* **$\varepsilon_{max} = 0.01293$ at $t = 4.84$** in the standard normalisation — the
  literature-comparable pair.
* **The parameter-free energy balance $-dE/dt = 2\nu\Omega$ holds to 0.7% worst-case**
  (ratio $\in [0.993, 1.000]$, worst near/after peak enstrophy, recovering to
  0.997 by $t = 12$). This is the internal referee that needs no reference
  data, and it doubles as the resolution meter: the dip below 1 sits exactly
  where the cascade makes its smallest scales.
* **The balance gap is neither vorticity slack nor divergence** — measured
  from saved frames: $\Omega(\text{state } \omega) = \Omega(\nabla \times u)$ to **four decimals** at every
  sampled time (the weak vorticity definition is effectively exact on this
  flow), and rms $\nabla \cdot u \leq$ 1.4e−4 throughout. Remaining suspects: SEM-plane
  aliasing (no (x, y) dealiasing — the known caveat) and the $O(dt^2)$ energy
  error of the explicit convective half. Small, bounded, open.

Movie and full diagnostics: `figs/tgv_re100_movie.mp4` (|ω| on the three
mid-planes, fixed colour scale, energy/enstrophy cursor),
`figs/tgv_re100_diagnostics.png`.

## 4. Against theory and published results

**What matches, with confidence.** Peak timing $t \approx 4.8$ sits where Brachet et
al. (1983, JFM 130) put the $Re = 100$ member of their dissipation-curve family
(peaks near $t \approx 4$–$5$ at $Re = 100$, drifting to $t \approx 9$ by $Re \gtrsim 800$ — the modern
$Re = 1600$ workshop value is $\varepsilon_{max} \approx 0.0117$ at $t \approx 9$). Peak magnitude $\approx 0.013$
is the right size for the low-Re branch, which peaks *earlier and slightly
higher* than high Re — correct family shape. Enstrophy growth of 1.72× is the
expected modest low-Re value (an order of magnitude at $Re \geq 800$).

**What is deliberately NOT claimed.** A digit-level match to Brachet's
$Re = 100$ curve. That requires digitising the published figure — the
`gartling_digitize.py` treatment; the paper is JFM-paywalled, so per
`reference/README.md` it needs institutional access to fetch. **Open item**,
and the same digitisation serves the $Re = 400$ run in flight, where the
comparison has real teeth.

Uncertainty on our peak: the 0.7% balance floor at peak enstrophy, i.e.
$\varepsilon_{max} = 0.0129 \pm {\sim}0.0001$ from resolution.

## 5. Data inventory

| artefact | content |
|---|---|
| `scratch/tgv_frames_re100/frame_0000..0048.npz` | full complex64 mode-space state every Δt = 0.25 — the movie source, and sufficient for any field post-processing (the §3 curl check was computed from these) |
| `scratch/tgv_frames_re100/chk_*.npz` (6) | float64 checkpoints every 2 t.u. — restart-grade |
| `scratch/tgv_diag_re100.npz` | per-step t, E, Ω, max\|u\|, CG iterations, cap flags |
| `figs/tgv_re100_transient.png` | §3 figure |
| `figs/tgv_re100_movie.mp4`, `figs/tgv_re100_diagnostics.png` | movie + 4-panel diagnostics |

## 6. Caveats and open items

* **Brachet digitisation** (§4) — the outstanding quantitative gate.
* **The 0.7% balance gap's residual cause** — SEM-plane aliasing vs $O(dt^2)$
  convective energy error; a dt-halving rerun of a short window would separate
  them (gap $\propto dt^2$ for the latter, dt-independent for the former).
* This run predates the analytic-Jacobi and tol = 1e−6 improvements; a rerun
  would be ~1.5–2× cheaper. Nothing in it is expected to change — the $Re = 400$
  relaunch reproduced the terminated slow run's $E$ and $\Omega$ to every printed digit
  under exactly that upgrade set.
* Terminology note: the quantity tracked here is **enstrophy** ($\tfrac{1}{2}\int |\omega|^2$), the
  standard companion to energy for this benchmark.

---

## 7. The Re = 400 companion run (2026-08-22)

Same rig at $\nu = 0.0025$, 48³ (6×6 N = 8, $N_z$ = 48), dt = 0.0114,
$t \to 15$ — **9.9 hours** under the row-7 weighting (the pre-fix attempt
priced at ~5 days and matched this trajectory to every printed digit while it
ran, a live cross-weighting consistency check).

![TGV Re=400 transient](figs/tgv_re400_transient.png)

| headline | Re = 400 | Re = 100 |
|---|---|---|
| enstrophy growth | **6.13×** | 1.72× |
| $\varepsilon_{max}$ | **0.01150 at $t = 6.00$** | 0.01293 at $t = 4.84$ |
| energy dissipated by run end | 81.3% | ~50% |
| worst balance ratio | 0.9495 at $t = 9.54$ | 0.993 |

The deep-cascade signatures are all present and correctly ordered against the
Brachet family: enstrophy grows immediately (no initial viscous dip — at 4×
the Reynolds number, stretching outruns viscosity from $t = 0$), the peak
arrives later than Re = 100 and earlier than the published Re ≥ 800 members
($t \approx 9$), and the growth factor jumps from 1.7× to 6.1×. The **5.1%
balance gap** in the post-peak phase is the measured price of 48³ at this
Reynolds number — the honest error bar on the peak ($\pm\sim$0.0006, with the
gap's sign implying the true peak is slightly higher). Tightening it is a 64³
rerun under the numba backend; the digit-level Brachet comparison still awaits
the digitisation of §4.

Data: 31 frames + 6 checkpoints in `scratch/tgv_frames_re400/`, diagnostics in
`scratch/tgv_diag_re400.npz`, movie `figs/tgv_re400_movie.mp4`, ParaView
export `scratch/tgv_vtk_re400/tgv_re400.pvd`.

---

## 8. Placed on a published scale: Gourianov et al. (2022)

`scratch/tgv_vs_gourianov.py`, `reference/2106.05782v3.pdf`. Reference:
N. Gourianov, M. Lubasch, S. Dolgov, Q. Y. van den Berg, H. Babaee, P. Givi,
M. Kiffner, D. Jaksch, *A Quantum Inspired Approach to Exploit Turbulence
Structures*, arXiv:2106.05782v3 / Nature Comp. Sci. **2** (2022).

**Why this reference and not Brachet.** §4 left the digit-level comparison
open because Brachet's dissipation curve needs digitising from a paywalled
figure. This paper does something better: its 3-D TGV study adopts the *same*
diagnostic this project arrived at independently. For incompressible periodic
flow the INSE imply

$$\zeta(t) \equiv \nu\!\int_V |\nabla\times V|^2\,dr \;=\; \varepsilon(t) \equiv -\tfrac{1}{2}\frac{d}{dt}\!\int_V |V|^2\,dr ,$$

and the paper states plainly that restricting the number of variables
"results in numerical diffusion violating this equality" — our balance meter,
independently motivated. Their Eq. (20) integrates the violation into one
number,

$$e = \frac{1}{E_0}\int_0^{2T_0} \bigl|\zeta(t) - \varepsilon(t)\bigr|\,dt ,
\qquad T_0 = L_\mathrm{box}/u_0 ,$$

and **Table 1 publishes $e$ for three schemes at $Re = 800$**: DNS on $256^3$
(8th-order central finite differences, RK2, Chorin projection), their
tensor-network MPS algorithm at three compressions, and under-resolved DNS
(URDNS, the same solver on coarse grids). That is a published yardstick our
runs can be placed on directly.

![TGV vs Gourianov](figs/tgv_vs_gourianov.png)

| scheme | grid | $Re$ | $e$ |
|---|---|---|---|
| **ours (LSSEM/VVP)** | $24^3$ | 100 | **0.0028** |
| **ours (LSSEM/VVP)** | $48^3$ | 400 | **0.0172** |
| their DNS | $256^3$ | 800 | 0.0020 |
| their MPS 1:25 / 1:49 / 1:78 | — | 800 | 0.0385 / 0.0844 / 0.2618 |
| their URDNS | $\approx88^3$ / $70^3$ / $60^3$ | 800 | 0.1599 / 0.2133 / 0.4563 |

**Reading it honestly.** Our $Re$ is lower than theirs, and lower $Re$ is
easier to resolve, so $e$ alone flatters us. The like-for-like axis is
**resolution adequacy**, $k_\mathrm{max}\eta$ at peak dissipation (the standard
DNS criterion; $\gtrsim 1$ is resolved):

| run | $k_\mathrm{max}\eta$ | $e$ |
|---|---|---|
| ours, $Re$ = 100, $24^3$ | 1.13 (resolved) | 0.0028 |
| ours, $Re$ = 400, $48^3$ | **0.82** (marginal) | **0.0172** |
| their DNS, $Re$ = 800, $256^3$ | 2.57 (amply resolved) | 0.0020 |
| their URDNS 1:25, $\approx88^3$ | **0.88** (marginal) | **0.1599** |

*(their $\eta$ from $\nu = 1/800$ and $\varepsilon_\mathrm{max} \approx 0.012$,
the literature value for TGV in this $Re$ range — an estimate, flagged as such)*

**The one comparison that controls for Reynolds number**: at essentially the
same resolution adequacy — $k_\mathrm{max}\eta$ = 0.82 for us, 0.88 for them —
our least-squares spectral-element solver carries **$e$ = 0.017 against their
URDNS's 0.160, about 9× less numerical diffusion**, and beats even their best
*compressed* MPS result (0.0385) by 2.2×. Our well-resolved $Re$ = 100 run at
$e$ = 0.0028 sits alongside their $256^3$ DNS reference value of 0.0020.

**Caveats, stated because they bound the claim.**

* ~~**$E_0$ ambiguity**~~ — **RESOLVED by digitising their figure** (§8.1).
  The paper's text states $E_0 = u_0^2/2$ while the TGV initial condition
  carries mean kinetic energy density $u_0^2/8$, a factor of 4. The $t = 0$
  intercept of their own Fig. 3b settles it: measured 0.0472 against the
  analytic $2\nu\Omega_0/V \big/ (E_0/T_0) = 0.0471$ for $E_0 = u_0^2/8$,
  versus 0.0118 for $u_0^2/2$. **Their normalisation is the actual initial
  kinetic energy — the same one we used — so the $e$ values above are directly
  comparable with no factor of four.**
* **Window coverage**: the $Re$ = 100 run ends at $t = 12.0$ against the
  window's $2T_0 = 12.57$ (95%); the $Re$ = 400 run covers it fully.
* Different $Re$, different flow states: this compares *numerical fidelity at
  comparable resolution adequacy*, not identical physics. The Brachet
  digitisation (§4) remains the outstanding same-$Re$ check.

**What it settles.** The §3/§7 balance gaps — 0.7% at $Re$ = 100, 5.1% at
$Re$ = 400 — were reported as an unbenchmarked resolution price. On the
published scale they are **DNS-class and better-than-compressed-MPS
respectively**, which is the first external, quantitative corroboration that
the LSSEM/VVP formulation dissipates energy physically rather than numerically.


### 8.1 Curve-level comparison: their Fig. 3b, digitised

`reference/gourianov_digitize.py` → `reference/gourianov_fig3b_tgv_re800.csv`.
Fig. 3b is **vector art**, so the curves and the axis ticks are read straight
out of the PDF content stream — nothing is estimated by pixel analysis, and the
accuracy is limited by the authors' plotting rather than by our extraction.
Extracted: $\varepsilon(t)/(E_0/T_0)$ for their DNS ($256^3$) and their MPS at
$\chi$ = 192 / 128 / 96, all at $Re$ = 800. (Their enstrophy $\zeta(t)$ is
drawn as markers; for the DNS curve it coincides with $\varepsilon$ to their
own $e$ = 0.002, and the $\zeta-\varepsilon$ gap is exactly what Table 1
already quantifies.)

![dissipation vs Gourianov](figs/tgv_dissipation_vs_gourianov.png)

**The $t = 0$ intercept is a parameter-free calibration.** For the TGV initial
condition $\Omega_0/V = 3/8$ exactly, so
$\varepsilon(0)/(E_0/T_0) = 2\nu(3/8)/(E_0/T_0)$ — a pure function of $\nu$.
Extraction reproduces it to 0.1%, which validates the digitisation *and* fixes
their $E_0$ convention (above). Our runs land on their own analytic intercepts
by construction: 0.375 at $Re$ = 100, 0.094 at $Re$ = 400, 0.047 at $Re$ = 800.

**The Reynolds-number family.** The three curves are *not* expected to
coincide — different $Re$, different flows — and what the overlay tests is
whether ours extend the published one in the right direction:

| run | $Re$ | peak $\varepsilon/(E_0/T_0)$ | $t_\mathrm{peak}/T_0$ |
|---|---|---|---|
| ours, $24^3$ | 100 | 0.650 | 0.77 |
| ours, $48^3$ | 400 | 0.578 | 0.96 |
| **their DNS, $256^3$** | **800** | **0.589** | **1.43** |
| literature (Brachet / workshop) | 1600 | ≈0.588 | ≈1.43 |

Both trends are the textbook ones: the peak **moves monotonically later** with
$Re$ (0.77 → 0.96 → 1.43), and its **height flattens onto the high-$Re$
plateau** — elevated at $Re$ = 100 where viscosity dissipates the large scales
directly, then essentially $Re$-independent from 400 upward (0.578, 0.589,
0.588), which is the dissipation anomaly. Our two points sit on the published
family, on both axes.

**What is still not a same-$Re$ test.** Nothing here compares our solver with
theirs *on the same flow*. That needs a run at $Re$ = 800 — and now that the
row-7 fix and the numba backend have cut the cost ~40×, the decisive experiment
is affordable: **repeat their exact URDNS grids ($60^3$, $70^3$, $88^3$) at
$Re$ = 800** and compare $e$ against their published 0.4563 / 0.2133 / 0.1599 at
identical resolution and Reynolds number. Estimated cost with numba: ~7 h at
$60^3$, ~1.5 days at $88^3$. That is the experiment this section makes possible
and does not yet contain.
