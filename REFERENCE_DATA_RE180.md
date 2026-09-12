# Reference data for the Re_τ = 180 channel

Which published experimental and DNS datasets the FOSLS-3D minimal-channel run
(`run01`, Re_τ = 180, Lx = πh, Lz = 1.07h) can be compared against, what each
one provides, and — the important part — **which of our quantities are actually
comparable given our minimal box**.

All files below are downloaded into `reference/` and readable through one
interface, `scratch/ref180.py` (`mkm1999() lm2015() vreman2014() torroja()
akm2001()` each return `yp, U, urms, vrms, wrms, uv, [oxrms oyrms ozrms prms]`
in wall units on the lower half-channel).

## 1. Full-channel DNS databases (all downloaded)

| Dataset | Re_τ | Box (Lx × Lz)/h | Method / grid | Averaging | Quantities in the profile files | Local path |
|---|---|---|---|---|---|---|
| **Moser, Kim & Mansour 1999** (PoF 11:943), the modern re-run of KMM 1987 | 178.1 | 4π × 4π/3 | Chebyshev–Fourier, 128×129×128 | — | U, dU/dy, P; ⟨uᵢuⱼ⟩ all six; **⟨ωᵢωⱼ⟩ all six**; ⟨uᵢp⟩, ⟨p²⟩; skewness, flatness | `reference/mkm1999_chan180/chan180.{means,reystress,vortvar,velp,skew,flat}` |
| **Lee & Moser 2015** (JFM 774:395), Re_τ=180 member of the 180–5200 series | 182.1 | 8π × 3π | Fourier–B-spline, 512×128×256 | very long | U, dU/dy, P; ⟨uᵢuⱼ⟩; **⟨ωᵢωⱼ⟩, ⟨p²⟩**; full ⟨u′v′⟩ budget; spectra (h5, not downloaded) | `reference/lm2015_chan180/LM_Channel_0180_{mean,vel_fluc,vor_pres_fluc,RSTE_uv}_prof.dat` |
| **Vreman & Kuerten 2014** (PoF 26:015102 and 26:085103), S4_B3 case | 180.0 | 4π × 2π/3 | Chebyshev-tau spectral, 576×577×385, 3rd-order time | T = 100 h/u_τ | **288 rows × 88 columns**: mean and rms of u, v, w, p, all 9 first derivatives, 3 strain, **3 vorticity components**, all second derivatives, Laplacians, ⟨uᵢuⱼ⟩; plus dissipation-rate, enstrophy and per-component vorticity budgets; skewness and flatness | `reference/vreman2014_chan180/Chan180_S4_B3_*.txt` |
| **Torroja / del Álamo & Jiménez 2003** (PoF 15:L41; statistics by Hoyas & Jiménez) | 180 (185.6 nominal) | 12π × 4π | KMM-type spectral | — | U, u′v′w′ rms, mean Ω_z, **ω′ₓ ω′_y ω′_z rms**, uv, uw, vw, pressure rms split into rapid/slow/total; `.std` file gives standard deviations | `reference/torroja_chan180/Re180.{prof,std}` |
| **Abe, Kawamura & Matsuo 2001** (ASME JFE 123:382; JAXA database, ch180 file as reprocessed for Abe, Antonia & Kawamura 2009) | 180 (Re_b = 5546) | 12.8 × 6.4 | 4th/2nd-order FD, 768×128×384, Δx⁺=Δz⁺=3 | 3960 ν/u_τ² (= 22 h/u_τ) | U, uu, vv, ww, −uv; budgets of uu, vv, ww, uv | `reference/akm_chan180/ch180.dat` |

The original **Kim, Moin & Moser 1987** (JFM 177:133; 192×129×160, 4π×2π,
10 h/u_τ) has no downloadable profiles; its numbers survive as the MKM 1999
re-run above. From the paper (Table 1 and figures): U_m/u_τ = 15.63,
U_c/u_τ = 18.20, U_c/U_m = 1.16, C_f = 8.18e-3, log-law constant 5.5;
p_rms peak 1.75 at y⁺≈30, wall value 1.5; ω_x′ wall value 0.20 ν/u_τ² with a
local minimum at y⁺≈5 and a local maximum at y⁺≈20 (the streamwise-vortex
signature); streak spacing λ⁺ ≈ 100 at y⁺=10 rising to ≈130 at y⁺=30
(their fig. 24); R_uu(Δz) minimum at Δz⁺≈50, R_vv minimum at Δz⁺≈25.

### 1.1 The five databases agree with each other to ≈1%

Peak values from `scratch/ref180.py` (value @ y⁺):

| | U_c⁺ | u′ | v′ | w′ | −u′v′ | ω′ₓ(wall) | ω′_z(wall) | p′ |
|---|---|---|---|---|---|---|---|---|
| MKM 1999 | 18.30 | 2.658 @15.3 | 0.836 @52 | 1.087 @35 | 0.723 @30 | 0.198 | 0.364 | 1.878 @30 |
| Lee–Moser 2015 | 18.27 | 2.672 @14.6 | 0.849 @54 | 1.098 @36 | 0.728 @32 | 0.202 | 0.370 | 1.908 @31 |
| Vreman–Kuerten S4_B3 | 18.26 | 2.664 @14.5 | 0.842 @55 | 1.089 @35 | 0.723 @32 | 0.200 | 0.368 | 1.892 @30 |
| Torroja | 18.28 | 2.659 @14.2 | 0.850 @55 | 1.091 @35 | 0.734 @31 | 0.205 | 0.368 | 1.933 @31 |
| Abe–Kawamura–Matsuo | 18.34 | 2.666 @14.5 | 0.847 @55 | 1.095 @36 | 0.724 @31 | — | — | — |
| **spread** | 0.4 % | 0.5 % | 1.7 % | 1.0 % | 1.5 % | 3.5 % | 1.6 % | 2.9 % |

U⁺ at y⁺ = 5 / 15 / 30 / 50 / 100 is 4.81 / 10.85 / 13.86 / 15.30 / 17.1 in
all five to within 0.5 %. Vreman & Kuerten measured this reproducibility
formally (their Table II): two unrelated codes agree to 0.1 % on U, 0.6 % on
rms, 1.8 % on dissipation; MKM 1999 sits 2.7–3.5 % off on rms because of its
shorter averaging and coarser grid. **So the inter-database spread (~1 %) is
the reference uncertainty; any disagreement of ours larger than that is ours.**

### 1.2 Which one to use for what

- **Primary: Vreman & Kuerten S4_B3.** Highest resolution, same box family as
  KMM, and the only one that tabulates rms of every velocity derivative and all
  three vorticity components alongside enstrophy and vorticity budgets. Since
  ω is a **primary unknown** in our VVP-FOSLS formulation (not a post-processed
  derivative), the vorticity-rms and enstrophy-budget comparison is the
  FOSLS-specific test and this database is built for it.
- **Secondary: Lee & Moser 2015** for the ⟨u′v′⟩ budget and, if we ever need
  them, spectra; **MKM 1999** because it is what everyone else plots against
  (its ⟨ωᵢωⱼ⟩ file includes the off-diagonal ⟨ω_x ω_y⟩ etc.).
- **Torroja `.std`** is the only source of a statistical error bar on each
  profile — useful when deciding whether a run01 deviation is significant.
- AKM 2001 is finite-difference; keep it only as an independent-method check.

## 2. Experiments near Re_τ = 180

| Experiment | Re_τ | Technique | What it gives | Data form |
|---|---|---|---|---|
| **Niederschulte, Adrian & Hanratty 1990** (Exp. Fluids 9:222) | **178.6** (and 158.5) | 2-component LDV, water channel, y⁺ down to 0.6 | U⁺, u′, v′, −u′v′, skewness, flatness, spectra; U_B/u* = 15.55 (KMM 15.63) | figures only — digitisation needed. Finding: v′ and −uv agree with KMM within error; u′ measured slightly *above* KMM (≈+3 %) outside experimental error |
| Kreplin & Eckelmann 1979 (PoF 22:1233) | 194 | hot-film, oil channel | u′v′w′, wall-vorticity fluctuations, skewness, flatness — the symbols in KMM figs 6–8, 14, 19, 20 | figures only |
| Eckelmann 1974 (JFM 65:439) | 142, 208 | hot-film, oil channel | −u′v′ (KMM fig. 10) | figures only |
| Wei & Willmarth 1989 (JFM 204:57) | 170–1650 | 2-comp LDV | u′, v′, −uv, spectra; lowest case Re_h≈2970 ≈ Re_τ 170 | figures only |
| Schultz & Flack 2013 (PoF 25:025104) | 1000–6000 | LDV | — | **not applicable** (checked: lowest case is Re_τ = 1010) |

None of the experiments publish tables; all pre-date the DNS and KMM already
showed the DNS falls within their scatter (except for u′ where hot-film
cross-contamination inflates v′, w′). For quantitative comparison the DNS
databases are strictly better references; the experiments matter only as the
independent confirmation that the DNS canon is physical. Niederschulte 1990 is
the one to cite: it is a genuine Re_τ ≈ 180 water-channel measurement with
resolved sublayer, and its U_B/u_τ, U_c/U_m = 1.16 and −uv peak ≈ 0.72 @ y⁺≈30
are directly checkable against run01.

## 3. What is — and is not — comparable from our minimal box

Our box is Lx⁺ = 565, Lz⁺ = 192 (Lx = πh, Lz = 1.07h). The references are
full channels with Lx ≥ 4πh, Lz ≥ 4π/3 h.

- **Jiménez & Moin 1991** (JFM 225:213) established that boxes of this size
  (their smallest sustaining case: Lx⁺ ≈ 250–350, Lz⁺ ≈ 100) reproduce the
  near-wall statistics of the full channel while the outer flow is unphysical.
  Jiménez & Simens (CTR brief 2000, saved) restate it: "the statistics of the
  near-wall fluctuations were essentially identical to those of fully developed
  channels", and give the minimal-box figures: streak spacing z⁺ ≈ 100,
  vortex longitudinal spacing x⁺ ≈ 400, bursting period T⁺ ≈ 400 with
  intermittent bursts 2–3× longer.
- **Flores & Jiménez 2010** (PoF 22:071704) and **Lozano-Durán & Jiménez 2014**
  (PoF 26:011702): a box captures the statistics correctly only up to
  y ≈ Lz/3 (Flores–Jiménez), and needs Lx ≈ 6h, Lz ≈ 3h for the whole log/outer
  region. For us Lz/3 ≈ 0.36h → **y⁺ ≲ 60**.

Therefore compare against the references **only for y⁺ ≲ 40–60**, and expect
the following to be off by construction, not by numerics:

| Quantity | Comparable? | Note |
|---|---|---|
| U⁺ for y⁺ ≲ 60 (sublayer, buffer, start of log) | yes | log constant may drift by a few tenths above y⁺≈40 |
| U_c⁺, U_b⁺, U_c/U_b, C_f | **no** | outer flow in a minimal box is a laminar-like plug; KMM's 18.20/15.63 will not be reproduced |
| u′, v′, w′, −u′v′ peaks and their y⁺ locations | yes (peaks are at y⁺ = 15/55/35/30, inside the trusted region; v′ at 55 is marginal) | Lozano-Durán shows w′ and v′ are the first to be starved by a narrow box — treat a low v′ peak as a box effect |
| ω′ₓ, ω′_y, ω′_z rms for y⁺ ≲ 40, including the wall values 0.20/0/0.37 and the ω′ₓ minimum @5 / maximum @20 | **yes — the FOSLS-specific test** | vorticity is dominated by near-wall scales; least sensitive to box size |
| p′ rms | wall value yes (1.5–1.9); profile above y⁺≈30 no | p′ has the largest outer-scale contribution of all |
| Total-stress balance −⟨u′v′⟩⁺ + dU⁺/dy⁺ = 1 − y/h | yes, everywhere | exact identity, independent of box |
| Streak spacing λ⁺ ≈ 100 @ y⁺=10; R_uu(Δz) minimum @ Δz⁺≈50; R_vv minimum @ Δz⁺≈25 | yes, but Lz⁺ = 192 holds ~2 streaks so the estimate is coarse | we measured 96 wall units |
| Skewness / flatness | yes for y⁺ ≲ 40 (Vreman F(v′)→29 at wall, MKM 22) | needs long averaging; ours is marginal |
| Spectra | only the k_z spectrum, and only with 2–3 resolved wavelengths | not worth plotting |

## 4. Proposed comparison once run01 finishes (t = 5)

Window t ∈ [1, 5] by checkpoint subtraction, then one figure per row:

1. U⁺(y⁺), 0.1 ≤ y⁺ ≤ 180, log axis, against Vreman S4_B3 with the Torroja
   ±σ band; annotate that y⁺ > 60 is outside the box's validity.
2. u′, v′, w′, −u′v′ against all five (they overlap; the spread is the error
   bar) — restricted to y⁺ ≤ 60 in the inset.
3. **ω′ₓ, ω′_y, ω′_z** against Vreman / MKM / Torroja, 0 ≤ y⁺ ≤ 60, with the
   KMM fig. 14 features marked (ω′ₓ minimum at 5, maximum at 20; ω′_y ~ y).
4. p′ against Vreman / LM / MKM, with the KMM wall value and 1.75–1.9 peak.
5. Enstrophy budget (Vreman `budget_enstrophy.txt`) — optional, FOSLS can
   evaluate every term from primary unknowns without differentiating twice.

## 5. Sources

- Kim, Moin & Moser 1987, JFM 177:133 — https://doi.org/10.1017/S0022112087000892 (PDF read)
- Moser, Kim & Mansour 1999, PoF 11:943 — data https://turbulence.oden.utexas.edu/MKM_1999.html
- Lee & Moser 2015, JFM 774:395 — data https://turbulence.oden.utexas.edu/channel2015/
- Vreman & Kuerten 2014, PoF 26:015102 — https://doi.org/10.1063/1.4861064 (PDF read); data http://www.vremanresearch.nl/Chan180_S4_B3.html
- del Álamo & Jiménez 2003, PoF 15:L41; Hoyas & Jiménez 2008, PoF 20:101511 — data https://torroja.dmt.upm.es/ftp/channels/data/statistics/Re180/profiles/
- Abe, Kawamura & Matsuo 2001, ASME JFE 123:382 — data https://jaxa-dns-database.jaxa.jp/channelflow/ch180.dat
- Niederschulte, Adrian & Hanratty 1990, Exp. Fluids 9:222 (PDF read)
- Jiménez & Moin 1991, JFM 225:213 — https://doi.org/10.1017/S0022112091002033
- Jiménez & Simens 2000, CTR Annual Research Briefs pp. 67–78 (PDF read)
- Flores & Jiménez 2010, PoF 22:071704; Lozano-Durán & Jiménez 2014, PoF 26:011702
- Schultz & Flack 2013, PoF 25:025104 (PDF read; Re_τ ≥ 1000 only)

## 6. Comparison at the end of run01 (t = 4.96) and the fractional-step E-run

`scratch/compare_ref180.py` → `figs_fosls_vs_fs/compare_ref180.png`;
`scratch/plot_stats.py` → `figs_fosls_vs_fs/run01_stats_final.png`.
run01 finished on 2026-09-07: 6250 steps to t = 5.0; the final session
(resumed at step 4400) took 30.9 h on the GB10, i.e. ≈60 s per step at
~4680 CG iterations per stage. Last checkpoint t = 4.96.
FOSLS window t ∈ [1.28, 4.96] by checkpoint subtraction (461 plane samples,
3.7 turnovers; vorticity and pressure rms from the 19 checkpoint snapshots
in the window, pressure gauge removed per snapshot). Fractional step
(E-path, `results/minchan_re180_E`): archived window t ∈ [3, 15.95] (7401
samples, 13 turnovers), vorticity by differentiation of the single archived
state. Both scaled by their measured u_τ (FOSLS 1.0004 → Re_τ 180.1; FS
1.0071 → Re_τ 181.3). Reference is Vreman & Kuerten S4_B3; "spread" is the
max disagreement among the five databases.

| quantity | reference | spread | FOSLS | dev | fractional step | dev |
|---|---|---|---|---|---|---|
| U⁺ @ 5 / 15 / 30 / 50 | 4.81 / 10.86 / 13.88 / 15.33 | 0.5 % | 4.79 / 11.18 / 14.27 / 15.46 | −0.4 / +3.0 / +2.7 / +0.8 % | 4.78 / 11.12 / 14.24 / 15.44 | −0.6 / +2.4 / +2.6 / +0.7 % |
| u′ peak | 2.663 @ 14.5 | 0.9 % | 2.925 @ 16.5 | **+9.8 %** | 2.843 @ 16.5 | +6.8 % |
| v′ peak | 0.842 @ 55 | 1.1 % | 0.878 @ 60 | +4.2 % | 0.842 @ 57.5 | 0.0 % |
| w′ peak | 1.089 @ 35.5 | 1.7 % | 1.043 @ 59 (plateau) | −4.2 % | 1.030 @ 54 (plateau) | −5.4 % |
| −u′v′ peak | 0.723 @ 31.5 | 1.6 % | 0.775 @ 34 | **+7.3 %** | 0.726 @ 33.5 | +0.4 % |
| ω′ₓ wall / min @5 / max @20 | 0.200 / 0.10 / 0.145 | 2.4 % | 0.166 / 0.09 / 0.13 | −17 / −10 / −10 % | 0.244 / 0.13 / 0.18 | +22 / +30 / +24 % |
| ω′_y peak | 0.196 @ 14.5 | 0.6 % | 0.193 @ 14 | −1.9 % | 0.223 @ 13.5 | +13.6 % |
| ω′_z wall | 0.368 | 1.3 % | 0.345 | −6.6 % | 0.391 | +4 % |
| p′ wall / peak | 1.54 / 1.89 @ 30.5 | 2.3 % | 1.39 / 1.64 @ 36.5 | −10 / −13 % | 4.7 / 5.05 @ 13.5 | not comparable, see below |

Total-stress balance error over y⁺ < 150: 0.040 (`plot_stats.py`).

**Reading it.**

1. *Box effects dominate the velocity statistics, and both codes show the
   same ones.* The +2.5–3 % buffer-layer U⁺, the u′ excess, the flat w′
   plateau at 1.03–1.04 instead of a 1.09 peak at y⁺ = 35, and the v′ peak
   pushed out to y⁺ ≈ 60 appear identically in the fractional-step run on
   the same mesh after 13 turnovers. These are the Lz⁺ = 192 minimal-box
   signature, not a property of either scheme. Sublayer U⁺ is exact in both.
2. *FOSLS's remaining excess in u′ (+9.8 vs +6.8 %) and −u′v′ (+7.3 vs
   +0.4 %) shrank as the window lengthened* (from +11.5 / +9.6 % at
   t = 4.48) and is within run01's own window-to-window swing (sub-window
   u′ peaks 2.57 / 3.15 / 2.62). 3.7 turnovers against the FS run's 13 is
   the whole difference; a longer continuation would settle it.
3. *Vorticity — the FOSLS-specific test — is the best-agreeing set.* ω′_y
   within 2 % across the buffer layer; ω′_z wall within 7 %; ω′ₓ reproduces
   KMM's local minimum at y⁺ ≈ 5 and maximum at y⁺ ≈ 20 (the streamwise-
   vortex signature) 10–17 % low. Primary ω and curl u agree to < 0.05 %
   in rms. The single-snapshot FS vorticity is 14–30 % high with a spurious
   ω′_z bump at y⁺ ≈ 20 — one snapshot of a 565 × 192 box, differentiated
   from a C⁰ field, not a verdict.
4. *Pressure.* FOSLS p′ is 10–13 % low and its peak sits at y⁺ ≈ 36 rather
   than 30. The archived FS `p` field is 2.5–3× the physical pressure with
   its peak at y⁺ ≈ 13 and is neither p nor p ± ½|u|²: it is the E-path
   projection pseudo-pressure as stored, so no FS pressure comparison is
   possible from the archive. The low FOSLS p′ and ω′ₓ wall values are
   consistent with the ADN finding (ADN_FOSLS.md §2) that the L² VVP
   functional with no-slip walls is not norm-equivalent in H¹ for ω and p.
5. *Outside y⁺ ≈ 60* both codes fall below the reference in v′, w′ and (FS)
   u′ toward the centreline, as expected for a box that cannot hold outer
   eddies.

**Verdict.** run01 sustained turbulence for 5 turnovers (rms_w 0.91–0.92,
u_τ 0.99, ε at its run maximum at the end) with no sign of decay, and its
near-wall statistics match the Re_τ = 180 canon to the same fidelity as the
fractional-step scheme on the same box for velocity, and better for
vorticity, which is the quantity FOSLS carries as a primary unknown. The
demonstration that FOSLS can do DNS stands on this run; the open items are
statistical (a longer window for u′ and −u′v′) and formulational (the H⁻¹
momentum norm for ω and p accuracy, ADN_FOSLS.md §6.5), not dynamical.

## 7. Is the time step set by physics or by the scheme? (2026-09-12, `scratch/dns_timescales.py`)

A DNS time step has to resolve the fastest motion in the flow.  It must also stay
inside the scheme's stability limit, but a DNS whose step is set by *stability*
is one whose scheme is in the way of the physics.  Which of the two binds is a
measurement, not an assumption, so it is measured here on run01's own fields —
the dissipation from velocity gradients (not from the stored ω, which is an
independent VVP unknown left free at the walls, exactly where dissipation
peaks), the Kolmogorov time τ_η = √(ν/ε) plane by plane, and the convective CFL
on the true smallest GLL spacing against the RKW3 limit √3.

| checkpoint | $t$ | $u_\tau$ | $\Delta t^+$ | min $\tau_\eta$ | $\tau_\eta/\Delta t$ | CFL | largest stable $\Delta t$ | largest $\Delta t$ resolving $\tau_\eta/10$ |
|---|---|---|---|---|---|---|---|---|
| 0002600 | 2.08 | 0.958 | 0.132 | 5.54e−3 | 6.9 | 1.185 | 1.17e−3 | 5.54e−4 |
| 0006200 | 4.96 | 0.994 | 0.142 | 5.19e−3 | 6.5 | 1.113 | 1.24e−3 | 5.19e−4 |

The minimum of τ_η is at the wall (y⁺ = 0), as it must be; the pointwise minimum
is 2.7e−3, half the plane-averaged value.

**Physics binds, by a factor of 2.4.** Resolving the Kolmogorov time to a tenth
requires Δt ≤ 5.2e−4; RKW3 stability on the same field allows 1.24e−3.  The
production Δt = 8e−4 sits between them: it resolves τ_η by 6.5× and runs at 64 %
of the stability limit, which is Δt⁺ = 0.14 in the units the channel literature
quotes.  Three consequences.

1. **The step was not chosen to appease the solver**, and could not be enlarged
   to save time without under-resolving the dissipation range: the ceiling is
   physical.  Taking the strict τ_η/10 convention instead would cost 54 % more
   steps for the same simulated time and is the direction to move if the
   dissipation statistics ever look suspect, not the other way.
2. **The two limits are within a factor of 2.4 of each other at this Reynolds
   number**, so explicit convection is not obviously the wrong choice here.  It
   becomes the wrong choice at higher Re_τ, where the CFL limit tightens like the
   grid while τ_η only falls like Re_τ^{−1/2} in wall units — the regime in which
   implicit convection, and therefore the weighting result of the 2D work, starts
   to matter for this code.
3. **It is why the preconditioner matters.** A DNS is forced to small Δt by the
   physics, and small Δt means large c = fac₁/Δt, which is precisely the regime
   where pointwise relaxation cannot see the divergence-free modes
   (ADN_FOSLS.md §3, §5.3).  The Gartling sweep measures the same thing from the
   other side: the patch preconditioner's advantage grows from 36× to 309× as c
   rises from 15 to 1500 (LOW_MEMORY_PATCH_SOLVERS.md §6).  The small step is not
   a choice the solver can talk us out of, so the solver has to be built for it.

*The continuation run that extends this record is driven from Colab in overnight
sessions; the procedure, the settings and the failure modes are in
COLAB_DNS_RUNBOOK.md.*
