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
