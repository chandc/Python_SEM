# Development plan: 3D LSSEM via Fourier expansion in z

Plan date: 2026-08-18. Target: extend the 2D VVP LSSEM solver to 3D with a
Fourier basis in a periodic `z`, validated against turbulent channel flow DNS at
`Re_τ` = 180.

Decisions taken (2026-08-18):

| | choice |
|---|---|
| convection | **explicit** — `u·∇u` formed in physical space, FFT'd, carried in the RHS |
| backend | **NumPy first, numba second**, behind the existing `set_backend()` |
| target case | **turbulent channel DNS, `Re_τ` = 180** (Kim–Moin–Moser) |

Mathematics is already written up and is not repeated here:
[3d_vvp_fourier_expansion.md](./3d_vvp_fourier_expansion.md) (the 7-variable
first-order system and its per-mode form) and
[3d_fourier_sem_expansion.md](./3d_fourier_sem_expansion.md) (Helmholtz
decoupling). This document covers **what to build, in what order, how to keep the
data movement sane, and how to know each step works.**

---

## 0. Read this before starting: two things that do not carry over

**0.1 The stability results in this repo do not apply to the new formulation.**
`GARTLING_VALIDATION.md` measures `a_mass = w_mass·fac1/dt` as the stability
control variable, with a hard threshold (stable ≤ 6.05, divergent ≥ 12.1 on a
flow with an outflow boundary), and `ARTIFICIAL_COMPRESSIBILITY.md` §4 shows AC
moves it to ~60 and fails by 120. **All of that was measured with convection
*inside* the least-squares functional**, multiplied by `a_flux`. Moving `u·∇u` to
the right-hand side removes it from `L` entirely: the momentum row becomes

    a_mass·u + a_flux·(∇p + ν∇×ω)  =  RHS(history, convection)

which is a *linear Stokes-like operator*, not the linearised Navier–Stokes
operator whose conditioning we characterised. The `a_mass` threshold, the `κ_p ≈
a_mass/2` rule, and the 27× AC speed-up all have to be **re-measured**, not
assumed. Treat §5 of this plan as mandatory, not optional.

**0.2 Explicit convection collides with the `a_mass` floor — this is the main
technical risk.** Explicit convection imposes a CFL limit. For `Re_τ` = 180
channel DNS a typical step is `dt` ≈ 1e−3 … 1e−2 in wall units-normalised time.
With `w_mass` = 1 and BDF2, `a_mass = 1.5/dt` gives **150 … 1500** — one to two
orders above anything that has ever been stable in this code, and above the
`a_mass` = 120 at which AC failed on the BFS.

**Do not take reassurance from the 2D channel.** The periodic channel ran
happily at `a_mass` = 1.5, 3 and **30** (`TEMPORAL_ACCURACY_STUDY.md`, bit-exact
fixed points) and at 15 (`ARTIFICIAL_COMPRESSIBILITY.md` §3, exact `Δp`), with no
sign of the instability that kills the BFS above 12.1. That looks like evidence
the 3D channel will be fine. **It is not.** `GARTLING_VALIDATION.md` §8 gives the
reason for the exemption: laminar Poiseuille/Stokes are *exactly representable*
(`J` = 5.94e−27), so all four rows vanish together and the weighting has nothing
to trade off. The BFS sits at `L2(div u)` = 2.3e−02 and there the weights decide
which row is sacrificed.

A **turbulent** channel at `Re_τ` = 180 is not exactly representable and its
residual is nowhere near zero — it belongs to the BFS regime, not the laminar
one. Two further gaps: the largest `a_mass` ever measured on any channel here is
**30**, five to fifty times below the 150–1500 that CFL implies; and the
`pois_ac.py` sweep that nominally reaches `a_mass` = 60 only ever had its
`dt` = 0.1 runs saved, so that range is unmeasured rather than clean.

### 0.3 Measured 2026-08-18: the exemption does not survive, but AC rescues it

The 2D proxy for the Stage 5 question, run before writing any 3D code
(`scratch/chan_amass_sweep.py`). Same 12×2 N=10 grid, `Re` = 100, P+Z outlet,
`w_mom` = `w_mass` = 1, nsub = 5, to t = 15 from rest. Only the inlet differs:
**parabolic** = exact solution representable, residual ≈ 0; **uniform** = flow
must develop, rms `div u` ≈ 8e−02, i.e. the BFS regime.

| inlet | `a_mass` | `κ_p` | outcome | max\|u\| | wall |
|---|---|---|---|---|---|
| uniform | 60 | 0 (off) | **BLEWUP @ t = 0.83** (33 steps, max\|u\| 22.2) | 22.2 | 503 s |
| uniform | 60 | 30 | ok to t = 15 | 2.28 | 58 s |
| uniform | 120 | 60 | ok to t = 15 | 2.26 | 97 s |
| uniform | 300 | 150 | **ok to t = 15** | 2.24 | 219 s |

**Two results, and they point in opposite directions.**

1. **The laminar exemption is confirmed to be about the residual, and it does not
   transfer.** The same grid that is perfectly stable at `a_mass` = 30 with a
   parabolic inlet blows up at `a_mass` = 60 within 33 steps once the inlet is
   uniform and the residual becomes non-zero. §0.2's concern is real and now
   measured, not inferred.

2. **AC extends the window to at least `a_mass` = 300 on this flow** — further
   than on the BFS, where `a_mass` = 120 failed at *every* `κ_p` tried
   (`ARTIFICIAL_COMPRESSIBILITY.md` §4). 300 is inside the 150–1500 band that CFL
   implies at `Re_τ` = 180. That is materially better news for the plan than §0.2
   assumed.

> **Scope, stated plainly.** "ok" here means *did not diverge by t = 15 from
> rest*, not *converged to the right answer*: max\|u\| ≈ 2.24–2.28 against ≈ 1.5
> for the developed profile, so these are still transient. This is a stability
> result only. It also does not test the Stokes-like operator of §0.1 — convection
> is still inside the functional here. Stage 5 remains a gate.

Three ways out, none free, and the plan must pick one *with measurements* at
Stage 5:

| option | effect | cost |
|---|---|---|
| Scale `w_mass` with `dt` | `a_mass = w_mass·fac1/dt` held fixed | `dt_eff = dt·w_mom/w_mass` — the scheme takes steps of size `w_mom` regardless of nominal `dt`, the trap documented in `ls_coeffs`. Time accuracy must be re-verified. |
| AC with re-tuned `κ_p` | supplies the missing `a33`; worked to `a_mass` = 60 in 2D | untested in the Stokes-like operator; AC is numpy-only today |
| Accept a floor on `dt` | stay under the threshold | may violate CFL → unstable convection. **May be infeasible**; this is the case to test first. |

If none works, the fallback is semi-implicit convection (option 3 of the original
question), which restores mode coupling and costs the decoupling. **Decide this
at Stage 5, before writing any turbulence machinery.**

---

## 1. Architecture

### 1.1 Array layout — decided by the two operations that must both be fast

Two access patterns compete:

* the **FFT** in `z` wants `z` contiguous;
* the **2D SEM contractions** (`D @ u` on the `(i,j)` axes) want the element
  interior adjacent and a large trailing batch dimension for BLAS.

Both are satisfied by putting `z` **last**:

    U[e, i, j, var, k]        e   element        (nelem)
                              i,j GLL nodes      (n = N+1)
                              var 7 fields       (u,v,w,ωx,ωy,ωz,p)
                              k   z-mode or z-plane

* `np.fft.rfft(U, axis=-1)` is then a contiguous, stride-1 transform.
* The `(i,j)` contractions become `np.einsum('pi,eijvk->epjvk', D, U)` — strided
  on `(i,j)` but with `var*k` as a fat contiguous inner block, which is what
  makes the batched matmul efficient.
* Every mode rides along as a batch dimension **for free**: the existing 2D
  operator code generalises by adding one trailing axis rather than looping.

> **Do not loop over modes in Python.** The decoupling is what buys performance,
> and a `for k in range(Nz)` loop around the 2D solver throws it away — that is
> the single most likely way this implementation ends up slow.

### 1.2 Real-to-complex, and the real-valued CG

The physical fields are real, so only `Nz/2 + 1` modes are independent
(`np.fft.rfft`). **Use `rfft`/`irfft`, not `fft`** — it halves both memory and
per-mode solve count. This is a factor-of-two on the entire run and is easy to
get wrong by reaching for `np.fft.fft` out of habit.

Each mode's 2D problem is complex. Two options:

| | pro | con |
|---|---|---|
| `complex128` arrays | compact; natural | `LᵀL` becomes **Hermitian**, so the CG inner products, the adjoint test, and `compute_jacobi` all need conjugation. Every one of those is a place the existing real-valued code is silently wrong. |
| **split real/imag into 2 real fields** | reuses the existing real CG, Jacobi, adjoint test, and the 82-test suite **unchanged** | 14 fields instead of 7; slightly more memory traffic |

**Recommend the split-real form.** The `i·k_z` terms couple real and imaginary
parts into one real system of twice the size, and everything downstream — the
symmetry check in `ARTIFICIAL_COMPRESSIBILITY.md` §2, the Jacobi diagonal, the
line search merit — keeps working without modification. Given how much of this
project's time has gone into diagnostics that turned out to be measuring the
wrong thing, reusing verified machinery is worth 2× the field count.

The `k_z` = 0 mode has zero imaginary part and, for `Nz` even, so does the
Nyquist mode. Both should be **solved as genuinely real** (half the work) and
their imaginary parts asserted zero in debug builds — a nonzero imaginary part at
`k_z` = 0 is the classic symptom of a botched transform.

### 1.3 Module layout

    lssem3d/
      fourier.py     rfft/irfft wrappers, wavenumbers, dealias padding,
                     Hermitian-symmetry assertions
      lssem3d.py     apply_L / apply_LT for the 7-field (14 real) per-mode system
      convect.py     physical-space u·grad u, 3/2-rule dealiasing, CFL estimate
      solver3d.py    BDF + per-mode batched PCG, reusing lssem2d.solver where it
                     is dimension-agnostic
      stats.py       channel statistics: <u>(y), rms, Reynolds stress, u_tau

Keep `lssem2d` untouched. The Stage-1 test below depends on being able to compare
against it *exactly*.

---

## 2. Data-movement plan

The FFT/iFFT pair is the only place data is reorganised, and the convective term
is the only place we go to physical space. Everything else stays in mode space.

**Per time step:**

```
1.  irfft(Û)                 modes  -> physical   (z axis, contiguous)
2.  form N = u·∇u            physical space, dealiased (§2.2)
3.  rfft(N)                  physical -> modes
4.  build RHS                history (BDF) + N̂        [mode space]
5.  batched PCG over all modes at once               [mode space]
```

Two transforms per step, both stride-1. Steps 4–5 never touch `z` in physical
space.

### 2.1 Rules

* **One buffer per role, allocated once.** The 2D code already suffers from
  `apply_L`/`apply_LT` returning persistent state buffers (a bug I hit earlier in
  this project, needing `.copy()` in the adjoint test). In 3D the arrays are
  `Nz`× larger, so allocation churn is `Nz`× more expensive. Pre-allocate and
  document which buffers alias.
* **No `z`-axis transposes.** With `z` last, none are needed. If a transpose
  appears in a profile, the layout has been violated somewhere.
* **Batch the preconditioner too.** `compute_jacobi` must produce a diagonal for
  all modes in one vectorised call. Per-mode Jacobi construction in a loop would
  dominate cost — the diagonal depends on `k_z` only through the `i·k_z` terms,
  so it vectorises cleanly.
* **`float64` throughout for validation.** Revisit precision only after Stage 6
  passes, and re-run the full ladder if it changes.

### 2.2 Dealiasing

Convection is quadratic, so `u·∇u` in a `Nz`-mode basis aliases. Use the **3/2
rule in `z`**: pad to `3Nz/2` modes, transform, multiply, transform back,
truncate.

This is not optional for a turbulence run. The 2D code has no dealiasing (GLL
quadrature is exact to degree `2N−1` and the aliasing question was raised but not
resolved earlier in this project); in `z` the Fourier basis makes the aliasing
exact and unavoidable, and `Re_τ` = 180 DNS with aliased convection is a known
route to spurious pile-up at high `k_z`.

Cost: the transforms in step 1/3 above are done at `3Nz/2`. Budget for it.

**Verification:** Stage 3 below includes an explicit aliasing test — a product of
two single modes must produce *exactly* the two sum/difference modes and nothing
else. Without dealiasing that test fails, which makes it a real test of the
padding rather than a formality.

---

## 3. Test ladder

Each stage has a **pass criterion that can fail**. Do not proceed past a stage
whose test is merely "looks reasonable".

### Stage 0 — Fourier utilities (no solver)

| test | criterion |
|---|---|
| `irfft(rfft(f)) == f` | ≤ 1e−14 relative, random real field |
| `∂f/∂z` for `f = sin(2πmz/Lz)` | spectral accuracy, ≤ 1e−13 for `m < Nz/2` |
| wavenumber array | matches `2πn/Lz`; Nyquist handled explicitly |
| Hermitian symmetry | `Im(û)` = 0 at `k` = 0 and Nyquist, ≤ 1e−15 |

### Stage 1 — `k_z` = 0 must reproduce 2D **exactly** ⚑

**The single most valuable test in this plan.** With one mode and no
`z`-dependence, the 3D system must collapse to the 2D one: `w = ωx = ωy = 0`, and
`(u, v, ωz, p)` must satisfy exactly the 2D equations.

| test | criterion |
|---|---|
| operator | `apply_L3D` at `k_z`=0 vs `apply_L2D` — **bit-identical**, or ≤ 1e−15 |
| adjoint | `⟨b, LᵀLa⟩ = ⟨a, LᵀLb⟩` to ≤ 1e−15 (as `ARTIFICIAL_COMPRESSIBILITY.md` §2) |
| Kovasznay | reproduces `KOVASZNAY_VALIDATION.md` to the digits published there |
| **Ghia cavity Re=1000** | RMS u = **1.568e−02**, matching `ARTIFICIAL_COMPRESSIBILITY.md` §5.1 |

That last one is the anchor: the number is already measured, on a mesh we have,
against a benchmark. If the 3D code at `k_z` = 0 does not reproduce it, stop.

> This stage is also why `lssem2d` must not be refactored "while we're in there".
> A shared-code refactor makes the comparison vacuous.

### Stage 2 — single non-zero mode, analytic

Manufactured solution with one `k_z`, e.g. `u = û(x,y)·cos(k_z z)`, chosen so
`∇·u = 0` exactly.

| test | criterion |
|---|---|
| all 8 residual rows | machine zero at the exact solution, ≤ 1e−12 |
| mode isolation | energy in every other mode ≤ 1e−14 — **no leakage** |
| `k_z` convergence | error ~ constant in `Nz` (a single resolved mode is exact) |

Mode leakage here means the transform or the `i·k_z` terms are wrong, and it will
be invisible later.

### Stage 3 — dealiasing

| test | criterion |
|---|---|
| `cos(k₁z)·cos(k₂z)` | produces **only** `k₁±k₂`; all other modes ≤ 1e−14 |
| same, dealiasing off | test **fails** (confirms the test has power) |
| 3/2 padding round trip | truncation loses nothing below `Nz/2` |

### Stage 4 — full 3D MMS

Method of manufactured solutions with all fields `z`-dependent and a non-trivial
forcing.

| test | criterion |
|---|---|
| spatial convergence in `N` | spectral in `(x,y)`: error falls faster than any algebraic rate |
| spectral convergence in `Nz` | exponential until round-off |
| temporal order | BDF2 slope 2.0 ± 0.1 on `dt` refinement, as `TEMPORAL_ACCURACY_STUDY.md` |

### Stage 5 — the `a_mass` / CFL collision ⚑ **decision gate**

Before any turbulence. Laminar 3D channel (Poiseuille + a decaying `z`
perturbation), which has a known answer.

Run it **laminar first, then with a finite-amplitude perturbation**, because the
laminar case is exactly the one §0.2 says will look deceptively healthy.

| measure | why |
|---|---|
| stability vs `a_mass`, laminar | expected to be *clean* up to at least 30 — this reproduces the known 2D channel result and is a control, not evidence of feasibility |
| stability vs `a_mass`, perturbed | the case that matters: a non-zero residual is what activates the `a_mass` mechanism |
| stability vs `a_mass` | re-derive the threshold **for the Stokes-like operator**; §0.1 says the 2D number does not transfer |
| CFL limit of the explicit convection | measured, not assumed |
| do the two windows overlap? | **if not, the formulation is infeasible as specified** |
| AC sweep `κ_p` | does AC extend the window here as it did in 2D? |
| `w_mass ∝ dt` variant | does holding `a_mass` fixed preserve BDF2 order? (test against Stage 4's temporal-order harness) |

**Exit criterion: a documented, measured `(dt, w_mass, κ_p)` operating point that
is simultaneously CFL-stable and `a_mass`-stable at the resolution Stage 6 needs.**
If no such point exists, escalate to semi-implicit convection and re-plan.

### Stage 6 — turbulent channel, `Re_τ` = 180

Domain `4πδ × 2δ × (4/3)πδ`, periodic in `x` (SEM connectivity) and `z`
(Fourier), walls in `y`. Reference: Kim, Moin & Moser (1987) / Moser, Kim &
Mansour (1999).

Sub-stages, each a gate:

1. **Laminar Poiseuille in 3D** — recovers the 2D result; `w` stays at round-off.
2. **Transition** — perturbed initial field sustains turbulence rather than
   relaminarising; monitor `u_τ` for 20+ flow-through times.
3. **Statistics** — after discarding transients:

| quantity | criterion |
|---|---|
| `u_τ` | within 2% of 1.0 (normalised) |
| `⟨u⁺⟩(y⁺)` | log-law region within 3% of KMM |
| `u'⁺, v'⁺, w'⁺` rms | peak values and locations within 5% |
| Reynolds stress `⟨u'v'⟩⁺` | linear away from the wall; total stress balance closes to 1% |
| spectra | no pile-up at high `k_z` — **the dealiasing check that matters** |

---

## 4. Performance work — after Stage 4, not before

Correctness first; this project has repeatedly shown that fast wrong answers are
expensive. Once Stage 4 passes:

1. **Profile before optimising.** Expect the split to be PCG matvec ≫ FFT >
   convection. If FFT dominates, the layout is wrong.
2. **Numba kernels** for `apply_L`/`apply_LT`, mirroring the 2D backend split.
   Note `_check_ac_backend` already guards the numpy-only AC path — the 3D code
   needs the same guard, or AC will silently vanish in numba runs.
3. **Batched-mode efficiency**: measure iterations-per-solve as a function of
   `k_z`. High-`k_z` modes are better conditioned (the `k_z²` term is
   diagonally dominant), so a *uniform* iteration count across modes is evidence
   the preconditioner is not using `k_z`.
4. **Only then** consider MLX. Re-run the full ladder on any backend change.

### Scaling targets to record

| metric | why |
|---|---|
| wall/step vs `Nz` | should be ~linear; super-linear means a Python mode loop |
| CG its/solve vs `k_z` | should *fall* with `k_z`; flat means a bad preconditioner |
| memory high-water | `Nz`× the 2D footprint, ×2 for split-real, ×1.5 for dealias padding |
| **CG iteration counts** | deterministic and load-independent (verified in this project) — the honest cross-machine metric. Wall time is not. |

---

## 5. Risk register

| risk | likelihood | mitigation |
|---|---|---|
| **`a_mass`/CFL windows do not overlap** | **medium** (was high) | §0.3 measures AC holding to `a_mass` = 300 on a non-zero-residual 2D channel, inside the 150–1500 band. Stage 5 still a gate — that test used the full NS operator, not the Stokes-like one |
| AC-off at large `a_mass` is unusably slow even when stable | high | §0.3: 503 s for 33 steps at `a_mass` = 60. AC is not optional at these `a_mass`, it is the enabling technology |
| 2D stability results assumed to transfer | high | §0.1; Stage 5 re-measures from scratch |
| Python loop over modes | medium | §1.1; Stage 4 scaling test catches it |
| `fft` used where `rfft` belongs | medium | Stage 0 asserts Hermitian symmetry; 2× cost otherwise |
| Aliasing pile-up at high `k_z` | medium | Stage 3 test with a deliberate negative control |
| Complex/Hermitian bugs in CG | medium | split-real form avoids the class entirely |
| Buffer aliasing (`apply_L` returns state) | medium | known 2D bug; pre-allocate and document |
| AC silently off on numba | low | mirror `_check_ac_backend` |

---

## 6. Milestones

| # | deliverable | gate |
|---|---|---|
| M1 | `fourier.py` + Stage 0 | transform and derivative tests |
| M2 | `lssem3d.py` operator + Stage 1 | **reproduces Ghia RMS u = 1.568e−02 at `k_z`=0** |
| M3 | Stages 2–3 | single-mode exact; dealiasing with negative control |
| M4 | `solver3d.py` + Stage 4 | MMS spectral + BDF2 order 2.0 |
| M5 | **Stage 5 decision gate** | documented feasible `(dt, w_mass, κ_p)` |
| M6 | numba backend | Stage 1–4 re-pass, scaling targets met |
| M7 | Stage 6 | `Re_τ`=180 statistics within tolerance of KMM |

---

## 7. Open questions for later

* Subgrid model: not needed for `Re_τ` = 180 DNS, but Chan & Mittal's BFS is an
  LES. If that becomes a target, the SGS term's placement in the least-squares
  functional is a fresh design question.
* Whether `x`-periodicity via SEM connectivity is already supported well enough,
  or needs work (the 2D code has periodic-channel cases; confirm at M2).
* Whether the split-real doubling can be dropped later for a Hermitian CG once
  the real version is trusted — a performance option, not a correctness one.
