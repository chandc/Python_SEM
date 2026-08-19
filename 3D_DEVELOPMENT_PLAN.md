# Development plan: 3D LSSEM via Fourier expansion in z

Plan date: 2026-08-18. Target: extend the 2D VVP LSSEM solver to 3D with a
Fourier basis in a periodic `z`, validated against turbulent channel flow DNS at
`Re_τ` = 180.

Decisions taken (2026-08-18):

| | choice |
|---|---|
| convection | **explicit**, 3-stage low-storage **RKW3** (Spalart–Moser–Rogers) |
| viscous / linear | **implicit, Crank–Nicolson**, per mode |
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
With RKW3/CN the momentum row carries `c_k = 1/(β_k·dt)` and the worst stage has
`β` = 1/6, so `c = 6/dt` gives **6000 … 60000** — three to four orders above
anything that has ever been stable in this code, and far above the `a_mass` = 300
to which AC was measured to hold in §0.3.

> **Do not use `1.5/dt` here.** That is the BDF2 coefficient. RKW3/CN's worst
> stage is **4× larger at the same `dt`** (§0.4). An implementation that budgets
> against `1.5/dt` will under-estimate its own `a_mass` by a factor of four.

**On paper the windows do NOT currently overlap.** `a_mass_worst = 6/dt` needs
`dt` ≥ 0.02 to stay under the `a_mass` = 300 that AC was measured to reach
(§0.3), while CFL implies `dt` ≈ 1e−3 … 1e−2 — a shortfall of **2× to 20×**.
That ceiling was measured for the *linearised Navier–Stokes* operator though,
and §0.1 says the Stokes-like operator must be re-measured. **The cheapest
decisive experiment is in 2D and needs no 3D code and no change to `lssem2d`:
zero the linearisation (`fu = fv = 0`) so `apply_L` becomes the Stokes operator,
and measure its `a_mass` threshold.** Do that before committing to explicit
convection.

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

| inlet | residual | `a_mass` | `κ_p` | outcome | max\|u\| | wall |
|---|---|---|---|---|---|---|
| uniform | ~8e−02 | 60 | 0 (off) | **BLEWUP @ t = 0.83** (33 steps) | 22.2 | 503 s |
| uniform | ~8e−02 | 120 | 0 (off) | **BLEWUP @ t = 0.58** (46 steps) | 20.4 | 731 s |
| uniform | ~8e−02 | 300 | 0 (off) | **BLEWUP @ t = 0.35** (71 steps) | 23.9 | 913 s |
| **parabolic** | **≈ 0** | 60 | 0 (off) | **BLEWUP @ t = 0.83** (33 steps) | 22.9 | 531 s |
| **parabolic** | **≈ 0** | 120 | 0 (off) | **BLEWUP @ t = 0.58** (46 steps) | 20.6 | 616 s |
| **parabolic** | **≈ 0** | 300 | 0 (off) | **BLEWUP @ t = 0.29** (59 steps) | 45.1 | 659 s |
| uniform | ~8e−02 | 60 | 30 | ok to t = 15 | 2.28 | 58 s |
| uniform | ~8e−02 | 120 | 60 | ok to t = 15 | 2.26 | 97 s |
| uniform | ~8e−02 | 300 | 150 | **ok to t = 15** | 2.24 | 219 s |

**The residual is NOT the discriminator — the outflow boundary is.** The
parabolic inlet, whose exact solution *is* representable and whose residual is
≈ 0, blows up at `a_mass` = 60 **at the same step and the same time as the
uniform inlet** (t = 0.83, 33 steps, in both). The two inlets are
indistinguishable in failure. Whatever protects the periodic channel at
`a_mass` = 30 (`TEMPORAL_ACCURACY_STUDY.md`), it is not the smallness of its
residual.

What these runs share, and what the periodic channel lacks, is a **P+Z outflow
boundary**. That matches `ARTIFICIAL_COMPRESSIBILITY.md` §5.1's scoping
correction — the closed cavity converges at `a_mass` = 30 with no remedy at all,
and the threshold was already flagged there as *"a property of flows with an
outflow boundary, not of `a_mass` alone."* These runs are direct evidence for
that reading and against the residual reading in `GARTLING_VALIDATION.md` §8.

> `GARTLING_VALIDATION.md` §8 attributes the periodic channel's exemption to its
> near-zero residual. That is consistent with its own evidence (periodic channels
> have no outflow *and* a zero residual, so the two explanations were
> confounded), but the parabolic runs above separate them: zero residual **with**
> an outflow gives no protection whatsoever. §8 should be read as unresolved on
> this point.

**AC extends the window to at least `a_mass` = 300** — further than on the BFS,
where `a_mass` = 120 failed at *every* `κ_p` tried
(`ARTIFICIAL_COMPRESSIBILITY.md` §4). 300 is inside the 150–1500 band that CFL
implies at `Re_τ` = 180. That is materially better news than §0.2 assumed, and it
is the finding the plan actually depends on.

> **Scope, stated plainly.** "ok" here means *did not diverge by t = 15 from
> rest*, not *converged to the right answer*: max\|u\| ≈ 2.24–2.28 against ≈ 1.5
> for the developed profile, so these are still transient. This is a stability
> result only. It also does not test the Stokes-like operator of §0.1 — convection
> is still inside the functional here. Stage 5 remains a gate.

### 0.4 RKW3/CN does not relieve `a_mass` — it mildly aggravates it

Worth stating plainly, because the opposite is the intuitive guess: a scheme with
a bigger CFL limit permits a bigger `dt`, and `a_mass ∝ 1/dt`, so RK "should"
help. It does not, because the implicit stage coefficient works the other way.

| | implicit coefficient | at matched CFL |
|---|---|---|
| BDF2 + AB2 | `fac1/dt` = **1.50**/dt | `1.50/dt_AB2` |
| RKW3/CN, worst stage (`β` = 1/6) | `1/(β·dt)` = **6.00**/dt | `6.00/(3.46·dt_AB2)` = **1.73**/dt_AB2 |

`1/β` = (4.324, 4.800, **6.000**), so at the *same* `dt` RKW3/CN is 4× worse. The
CFL gain (√3 ≈ 1.73 against AB2's ≈ 0.5, i.e. a 3.46× larger step) recovers most
but not all of that: the net is **≈ 15% worse than BDF2**, not better.

So the case for RKW3/CN rests elsewhere, and it is still a good case:

* **AB2 has no imaginary-axis stability interval at all** — it is unstable for
  pure advection at any `dt`, surviving only on viscous damping. RKW3 has a
  genuine one. On fine DNS grids, where convective eigenvalues are nearly
  imaginary, this is the argument that decides it.
* 3rd order on convection against AB2's 2nd.
* **2 storage registers.** At `Nz`× the 2D footprint this is the binding
  constraint; classical RK4 needs 4.
* ~13% *fewer* implicit solves per unit physical time — 3 solves per step, but
  the step is 3.46× larger.

**Consequence for Stage 5:** the quantity to test against the measured stability
window is `max_k 1/(β_k·dt)`, and the window must be found for *that*, not for
`1.5/dt`. `lssem3d/timestep.py` exposes `a_mass_worst(dt)` for exactly this, and
`lssem3d/tests/test_fourier.py` pins the 4× and the 15% so the correction cannot
be quietly lost.

Three ways out, none free, and the plan must pick one *with measurements* at
Stage 5:

| option | effect | cost |
|---|---|---|
| Scale `w_mass` with `dt` | `a_mass = w_mass·fac1/dt` held fixed | `dt_eff = dt·w_mom/w_mass` — the scheme takes steps of size `w_mom` regardless of nominal `dt`, the trap documented in `ls_coeffs`. Time accuracy must be re-verified. |
| AC with re-tuned `κ_p` | supplies the missing `a33`; worked to `a_mass` = 60 in 2D | untested in the Stokes-like operator; AC is numpy-only today |
| Accept a floor on `dt` | stay under the threshold | may violate CFL → unstable convection. **May be infeasible**; this is the case to test first. |

### 0.6 RESOLVED: a periodic channel is already measured stable to `a_mass` = 2400

The evidence was in the repo the whole time. `TEMPORAL_ACCURACY_STUDY.md`, via
`scratch/pois_temporal.py`, runs **startup plane Poiseuille on a
streamwise-periodic channel** — `build_channel(..., bcs=(0,0,1,1))` with
`m.periodic_x = LX`, so **no outflow plane anywhere** — at
`w_mom = w_mass = 1`, the identical weighting to every run that fails above:

| `dt` | 0.01 | 0.005 | 0.0025 | 0.00125 | 0.000625 |
|---|---|---|---|---|---|
| **`a_mass`** | 150 | 300 | 600 | 1200 | **2400** |

Across that whole range, at N = 10, 14 and 18, the scheme is not merely stable —
it is **time-accurate to second order, fitted slope 2.04**, on a genuinely
unsteady solution. `a_mass` = 2400 is 40× the value at which the *same code* on
the *same equations* diverges within 33 steps when an outflow boundary is present
(§0.3, §0.5).

**This closes the Stage 5 feasibility question in the affirmative.** The 3D
target — `Re_τ` = 180 channel, periodic in `x` and `z`, walls in `y` — has no
outflow plane, and the RKW3/CN requirement of `a_mass` = 600 … 6000 is *directly
covered by measured data* at 600, 1200 and 2400. Only the top of the band
(6000, i.e. `dt` = 1e−3) sits above what has been measured, and nothing in the
150 → 2400 sequence shows any degradation approaching it.

**Revised conclusion on the whole `a_mass` story:** it is an **outflow-boundary
phenomenon**, not a property of the least-squares weighting in general. Three
causes have now been excluded by measurement — non-zero residual (§0.3),
convection (§0.5), and small `dt` per se (this section) — and the outflow
boundary is the only factor that has ever separated a stable run from an unstable
one. Options 2–4 of §0.5's ladder (AC, `w_mass ∝ dt`, fractional step) are
therefore **not needed for the target case**, and in particular there is no
reason to abandon the least-squares VVP formulation for a projection method.

> **Remaining exposure, stated so it is not forgotten.** (i) `a_mass` = 6000 is
> extrapolated, not measured — check it at Stage 5, it is one cheap run. (ii) The
> periodic evidence is 2D and laminar; a turbulent 3D field is a different
> dynamical regime even if the boundary treatment is the same. (iii) If any
> variant of the 3D problem ever grows an outflow — an inflow/outflow BFS, say —
> every constraint in §0.3 applies again in full.

### 0.5 Measured 2026-08-18: it is not convection, so semi-implicit is not a fallback

`scratch/stokes_amass_probe.py` zeroes the linearisation (`fu = fv = 0`) so that
`apply_L` becomes exactly the Stokes-like operator that explicit convection
leaves behind — without modifying `lssem2d`. Same channel, parabolic inlet
(Poiseuille *is* a Stokes solution, so the exact answer is unchanged):

| operator | `a_mass` | outcome |
|---|---|---|
| linearised Navier–Stokes (convection **in** the functional) | 60 | BLEWUP @ step 33 |
| **Stokes-like** (convection **removed**) | 60 | **BLEWUP @ step 29** |

**Removing convection changes nothing.** Together with §0.3 — where a residual of
≈ 0 failed at the same step as a residual of 8e−02 — three candidate causes are
now excluded by measurement:

| candidate | verdict |
|---|---|
| non-zero residual | **excluded** (§0.3: parabolic ≡ uniform, same step) |
| convective term | **excluded** (§0.5: Stokes ≡ full NS, same step) |
| **outflow boundary** | **only surviving explanation** |

**Consequence: the fallback ladder in this plan was wrong and is replaced.**
Semi-implicit convection cannot fix a failure that persists with no convection at
all. The remaining options, in order of disruption:

1. **Do nothing — if closed/periodic domains are exempt.** The `Re_τ` = 180
   channel is periodic in `x` and `z` with walls in `y`: it has **no outflow
   plane**. `ARTIFICIAL_COMPRESSIBILITY.md` §5.1 already showed the closed cavity
   converging at `a_mass` = 30 where the BFS diverges at 12.1. If that exemption
   extends to 600–6000, the entire problem is an outflow-BC artefact that the
   target case never encounters. **This is the measurement in flight and it
   decides everything below.**
2. **AC.** Already extends the window; a channel run at `a_mass` = 600 with
   `κ_p` = 300 survived 200 steps where AC-off fails at 60.
3. **`w_mass ∝ dt`,** holding `a_mass` fixed, at the cost of `dt_eff = w_mom`
   and a re-verification of temporal order.
4. **Fractional-step / projection.** The real fallback if (1) fails. It does not
   *have* the `a_mass` mechanism: continuity is enforced by a projection rather
   than traded off against the momentum rows inside a functional, so there is no
   weighting to get wrong. The cost is that it is a **different method, not an
   extension** — it abandons the coupled least-squares VVP formulation and with
   it every result validated in this repo, and brings its own splitting-error and
   pressure-boundary-condition problems. Do not reach for it before (1) is
   measured.

**Decide at Stage 5, before writing any turbulence machinery.**

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
                     Hermitian-symmetry assertions          [BUILT, tested]
      timestep.py    RKW3/Crank-Nicolson coefficients, implicit_coeff(dt, stage),
                     a_mass_worst(dt)                       [BUILT, tested]
      deriv.py       (x,y) derivatives + adjoints carrying arbitrary trailing
                     axes.  lssem2d.operators is shape-locked to (nelem,n,n)
                     -- facx[:,None,None] -- so it cannot batch over modes;
                     pinned bitwise to it on 3-D input   [BUILT, tested]
      operator.py    per-mode L / L^T, 7 complex fields -> 8 rows, split-real
                     facade so the real CG applies unchanged [BUILT, tested]
      convect.py     explicit u.grad u with 3/2 dealiasing in z, CFL estimate
                     and max_dt_for_cfl                     [BUILT, tested]
      solver3d.py    batched per-mode normal_op / PCG / probed Jacobi, and the
                     RKW3-CN stage driver                   [BUILT, tested]

**Standard array layout, fixed:** `U[e, i, j, var, k]` — the field axis is
**second to last** and z-modes are **last**. Two separate bugs came from
indexing `U[..., f]` (which selects a *mode*) instead of `U[..., f, :]`, in
`convect.py` and again in `operator.py`'s split-real facade. Both were silent —
they produce correctly-shaped arrays.
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
| temporal order | **RKW3 slope 3.0 ± 0.15** on `dt` refinement (not 2.0 — the scheme is 3rd order; measuring 2.0 means the `α/β/γ/ζ` table is mis-transcribed or Crank–Nicolson is limiting) |

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
| **`a_mass`/CFL windows do not overlap** | **low** (was high, then medium) | §0.6: a *periodic* 2D channel is already measured stable **and second-order accurate** to `a_mass` = 2400, covering 600–2400 of the required 600–6000. The failure is an outflow phenomenon and the target case has no outflow. Residual (§0.3) and convection (§0.5) are both excluded as causes |
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
| M4 | `solver3d.py` + Stage 4 | MMS spectral + **RKW3 order 3.0** (solver core built and tested; MMS still to do) |
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
