# 3D expansion — status, the M2 gate, and where the cores go

*Companion to `3D_DEVELOPMENT_PLAN.md`. That document is the plan; this one
records what was measured against it.*

`lssem3d/` is a **new module**. `lssem2d/` is untouched, as required — the 3D
code reuses `lssem2d.mesh`, `lssem2d.lgl` and `lssem2d.assembly.gather_scatter`
by calling them, never by editing them. Every place the 2D API did not fit was
worked around on the 3D side (see §2.4).

**Suite: 144 tests passing** (`uv run --quiet python -m pytest lssem3d/tests -q`).

| milestone | state | evidence |
|---|---|---|
| M1 `fourier.py` + Stage 0 | **done** | transform/derivative tests, Hermitian assertions |
| M2 operator + Stage 1 | **PASSED**\* | §1, `figs/cavity3d_kz0_profiles.png` |
| M3 Stages 2–3 | **done** | analytic single-mode; dealiasing with negative control |
| M4 Stage 4 MMS | **PASSED** | §4 — spectral in `N` and `Nz`; temporal gate restated. **PDE-level order 2.00 confirmed** (§7A.5) |
| M5 Stage 5 gate | **PASSED in 3D**\* | §7 — stable to `a_mass` = 6000; window spans ~66× in `dt` |
| M6 numba backend | not started | §3 changes what this should target |
| M7 `Re_τ` = 180 | not started | — |

**Five of seven milestones complete** (M1–M5). M6 (numba) and M7 (`Re_τ` = 180)
remain.

\* **M2 and M5 were measured before the row-weight fix** (§7A.2) and so ran on a
mis-scaled least-squares functional. Neither verdict is expected to flip — M2
compared 3D against 2D at matched settings, and M5 was a stability result — but
both should be re-run (§8.3), and until then they carry this asterisk.

M5 had been recorded as done on **2D evidence only** — the `a_mass` sweeps, the
periodic channel to `a_mass` = 2400, the AC studies were all 2D, while the plan's
risk register rates "2D stability results assumed to transfer" as **high** risk
and requires Stage 5 to re-measure from scratch. It has now been measured in 3D
(§7) and it passes, so the conclusion stands — but it stands on 3D data, which it
did not before.

---

## Lessons so far — the transferable part

Details live in the numbered sections; this is what generalises beyond this
project.

### L1. Tests that compare the operator to itself cannot find a wrong operator

Four missing factors of `A = M Qᵀ Q L₀ᵀ W L₀ M` (§2.1) survived a full suite of
symmetry, adjointness and convergence tests, because **`L₀ᵀL₀` is symmetric
whether or not `W` is there**, and CG converges happily on a mis-weighted inner
product. Only two kinds of test caught them: comparison against an *independent
implementation* (Stage 1, vs `lssem2d`) and against a *hand-derived analytic
forcing* (Stage 2). Every project should own at least one of each.

### L2. Nine bugs, zero exceptions

Not one of the nine (§2, §5) raised an error — each produced a correctly-shaped,
plausible array. Three returned a *sensible number for field 0* and were found
only when an index finally went out of range. The mitigations that actually work
are cheap: **assert the layout on entry** to anything taking the 5-D state, and
**write the negative control** (a test that must fail when the thing under test
is disabled).

### L3. Four gates and diagnostics were wrong *as written* — and each would have
### condemned correct code

This is the most surprising pattern in the project, and the reason gates are
worth re-deriving rather than inheriting:

| stated criterion | what measurement showed |
|---|---|
| "RKW3 temporal slope 3.0 ± 0.15" | Unachievable. Explicit-only **3.025**, CN-only **2.002**, mixed **2.189** — the table is right and CN is the limiter. No correct implementation could pass (§4.2) |
| "iterations should fall with `k_z`; flat means a bad preconditioner" | Flat is *correct* at production `a_mass`: `ν·k_z²` = 1.42 against `a_mass` = 1200. Competing would need `k_z` ≈ 465; the largest available is 64 (§6) |
| "M5 feasibility closed in the affirmative" | Closed on **2D** evidence, for a plan whose own risk register rates that transfer as *high* risk. Re-measured in 3D, it does pass — but it did not before (§7) |
| "AC is the enabling technology" | True for the 2D **outflow** case, and true for the 3D cavity (25 CG/step with AC against 12320 without). **Not** true for the ν = 1 Stokes benchmark under legacy weighting, where the AC-free system solves in 22047. AC's value is Reynolds-number dependent (§7A.2b) |
| "`κ_p` = `a_mass`" (inherited from the 2D steady studies) | Cost a **12.5% error in the Stokes decay rate** at every `dt`. Part initial-condition artifact, the rest a symptom of the scaling bug — not a fundamental accuracy/cost trade (§7A.3) |

### L4. Measure the mechanism, not just the symptom

A symptom is usually consistent with several stories. Three times the ambiguity
was resolved by finding a knob that separates them:

- **flat `div u` under `dt` refinement** could be an AC error *or* an ordinary
  spatial residual (also `dt`-independent). Fixing `κ_p` instead of scaling it as
  1/`dt` made it fall — O(1) vs O(`dt`), diagnosis settled (§7A.1).
- **threads plateauing at 6.7×** could be the GIL *or* memory bandwidth.
  Processes are immune to the GIL and tied threads exactly ⇒ bandwidth (§3.3).
- **flat `k_z` iteration profile** could be a bad preconditioner *or* `a_mass`
  swamping the `k_z²` term. Sweeping `a_mass` made the trend appear (§6).

### L5. Know your floor before calling something an error

`div u` is never zero in a least-squares formulation — continuity is a weighted
row, not a constraint. The meaningful quantity was **2.39× the AC-off floor**,
not the raw 1.46e−02. Likewise the M2 gate's target was the *2D curve*, not
Ghia's numbers, because 2D and 3D share a discretisation error that Ghia does not.

### L7. A symptom investigated long enough starts to look like a law

The AC work produced four correct measurements — 2.39× the divergence floor,
order 0.92, 12.5% at every `dt`, `nsub` buying nothing — and an elaborate,
self-consistent theory of AC as a fundamental accuracy/solvability trade. All of
it was downstream of one missing feature: **row weights in the least-squares
functional** (§7A.2). The measurements were right; the framing was invented to
explain them.

What eventually broke it was not a better measurement but a **comparison
question**: *at `k_z` = 0 this is the 2D system, so why can 2D solve what 3D
cannot?* The 2D code was sitting there the whole time as an oracle. The rule that
falls out: when a scheme behaves badly, first ask whether a working
implementation of the same equations disagrees — before theorising about the
equations.

### L6. Discard confounded results out loud — and re-check the diagnosis too

The Richardson temporal-order run returned negative orders, and was discarded
rather than reported. But the *first* explanation offered for it — an
inconsistent `p` = 0 initial condition — was **also wrong**. The real cause was
holding `κ_p` fixed while refining `dt`, which makes the pressure evolve at a rate
∝ 1/`dt` (§7A.3). A wrong diagnosis of a discarded result is still a wrong
diagnosis, and it survived a step longer than the result did.

Earlier casualties of the same kind: "AC reverses the sign of the `dt`
dependence" (a two-point sample of a U-shaped curve) and "the exemption is about
the residual" (refuted by a parabolic control failing at the identical step).

---

## 1. The M2 gate: 3D at `k_z` = 0 reproduces the 2D solution

The gate is *not* "does it match Ghia". It is **"does the 3D code at `k_z` = 0
reproduce the 2D code"**, because only that separates a 3D bug from
discretisation error the two share. Both were compared, and in **both velocity
components** — precedent from `ARTIFICIAL_COMPRESSIBILITY.md` §5.1, where RMS
`u` improved at every `dt` while RMS `v` did not move at all, and it was the `v`
column that showed AC is accuracy-*neutral* rather than better. A gate on `u`
alone can be passed by a solution that is wrong in `v`.

Cavity Re = 1000, 6×6 elements, N = 10, RKW3/CN with AC:

| | RMS `u` | RMS `v` |
|---|---|---|
| 3D at `k_z` = 0 | 1.812e−02 | 2.220e−02 |
| 2D reference (converged) | 1.568e−02 | 2.079e−02 |
| **ratio** | **1.16** | **1.07** |

Both centreline profiles lie on the 2D curve and pass through Ghia's points
(`figs/cavity3d_kz0_profiles.png`).

**The residual 16%/7% is un-converged transient, not discretisation.** The run
stopped at `t` = 25 on the step cap (14371 steps, 3717 s) with `|dU|` = 3.3e−03
and RMS still falling monotonically over the final 400 steps:
2.29e−02 → 2.04e−02 → 1.81e−02. It was still heading toward the 2D value when
the cap cut it off. Reporting this as a converged 16% discrepancy would have
been wrong in the direction that matters — it would have sent us looking for a
bug that is not there.

---

## 2. What the gate actually caught — a taxonomy of silent failures

This is the part worth keeping. **Eleven distinct bugs so far, and not one
raised an exception.** Every one produced a plausible array of the right shape.
Eight are below; the ninth is §5, and the tenth and eleventh are §2.5.

### 2.1 Four missing pieces of the operator

The assembled operator is `A = M Qᵀ Q L₀ᵀ W L₀ M`. Each omission below drops one
factor, and each was invisible to the tests that existed at the time:

| omission | symptom | why the tests missed it |
|---|---|---|
| **quadrature weights `W`** | wrong operator, converged fine | *every* symmetry and adjointness test passed — `L₀ᵀL₀` is symmetric with or without `W`. Caught only by the Stage 1 comparison against 2D |
| **gather-scatter `Qᵀ Q`** | BC solve failed to converge in 20000 CG iterations | the only one that announced itself, and only because it was catastrophic |
| **multiplicity weighting** in CG inner products | interface nodes counted twice (four times at corners) | inner products are still symmetric when mis-weighted; CG still "converges" |
| **inhomogeneous BCs** | converged to a **motionless cavity**, RMS 3.27e−01 | solving `A U = LᵀWf` with a masked `A` is a perfectly well-posed problem — just not the right one. Fixed with defect correction |

The lesson generalises: **symmetry, adjointness and convergence tests cannot
detect a consistently wrong operator.** They all test the operator against
itself. Only Stage 1 — comparison against an independent implementation — and
Stage 2 — comparison against a hand-derived analytic forcing — can.

### 2.2 Four field-vs-mode layout slips

The standard layout is `U[e, i, j, var, k]`: **field axis −2, mode axis −1**.
Four places indexed the wrong one:

- `convect.convective` — wrong axis for the velocity components
- the `operator` split-real facade
- `cfl()` velocities — `U_phys[..., OP.V_]` indexes the *mode* axis, not the field
- `cfl()` polynomial order — read `shape[-2]`, which is `NVAR` = 7, making the
  CFL limit **independent of `N`**: it reported the same `dt` at N = 6, 8 and 10,
  a silently over-permissive time step

Three of these return a plausible number for field 0 and are therefore
undetectable by a smoke test. The `cfl` velocity bug was noticed only because a
single-mode array finally made the index go out of bounds. That is luck, not
method — so `test_convect.py` now carries explicit tests that each of `u`, `v`,
`w` contributes, that a non-velocity field contributes nothing, that the limit
*falls* with `N`, and that a wrong-shaped array raises rather than guesses.

**Mitigation now in place:** `cfl()` asserts its layout on entry
(`U.ndim == 5 and shape[1] == shape[2]`). Layout assertions are cheap and this
class of bug is otherwise silent.

### 2.3 Two more, for completeness

- **NaN at step 36** — backward Euler with explicit convection is forward Euler
  *on the convective term*, whose imaginary-axis stability limit is exactly
  **zero**. Fixed by RKW3 (limit √3). Worth stating plainly because "implicit
  scheme" reads as "stable" and here it was not.
- **Unreachable `|dU| < 1e-9` exit** — the floor was 6.937e−08, so the
  convergence test could never fire. Added a stagnation exit.

A ninth is in §5 — it needed the integrator under test, which came later.

### 2.5 Two more, found by comparison and by review

**10. No row weights in the least-squares functional** (§7A.2). The momentum rows
outweighed the constraints by `c²` ≈ 1.4×10⁶, so the minimiser ignored `div u`
and the AC-free system could not be solved at all. Invisible to every test in the
suite: the operator is *correct*, it is the **functional** that was mis-weighted,
and Stage 1 compared against 2D at matched `a_mass` rather than matched row
scaling. Found only by asking why `lssem2d` could solve what `lssem3d` could not.

**11. The Jacobi probe was 1.4% wrong at every interface node.** The probe set
local index `(i,j)` in *every* element, which is **discontinuous** at an
interface — one copy of a shared node is 1 while its twin is 0 — so `gs()` folded
intra-element off-diagonal couplings into the reading. Probing the *unassembled*
operator and keeping the gather is exact (0.000e+00 over all 441 free dofs).

The second is the more embarrassing one, and the more instructive. The test that
"confirmed" the earlier assembly fix **spot-checked five nodes on the velocity
field** — and velocity contamination shrinks like 1/`c²`, so it is invisible at
production `a_mass`; the error lives on the `c`-independent pressure and vorticity
rows. That is exactly L1 — a test comparing the operator against itself — written
by the author of L1, in the same file that states it. The replacement sweeps every
free dof of every field and carries a negative control asserting the contaminated
probe really is contaminated, so it cannot pass for free.

### 2.4 Reusing `lssem2d` without touching it

The constraint is that `lssem2d/` must not be modified. Two places the 2D API
genuinely did not fit, and how each was handled **on the 3D side**:

| 2D API | why it did not fit | 3D workaround |
|---|---|---|
| `lssem2d.operators.dUdx/dUdy` | shape-locked to `(nelem, n, n)` by `facx[:, None, None]` — it cannot broadcast over trailing `(var, mode)` axes | `lssem3d/deriv.py` re-derives the same contractions with `einsum('pi,eij...->epj...')` and a rank-agnostic `_fac`, so any number of trailing axes works |
| `lssem2d.assembly.gather_scatter` | accepts 3-D or 4-D arrays only; the 3D state is 5-D | `solver3d.gs` folds `(var, mode)` into one trailing axis and calls it unchanged. `Q` acts on the spatial index alone, so this is exact — and it reuses the connectivity rather than reimplementing it |

The second is the important pattern: **fold, don't fork.** Reimplementing the
gather-scatter connectivity would have been a fresh source of the §2.1
class of bug.

---

## 3. Parallelism: measured, and the answer is not the obvious one

The algorithm *is* embarrassingly parallel across `k_z` — modes never interact
inside the implicit solve. But *where* to spend cores was measured, not assumed
(`scratch/prof3d.py`, `prof3d_modes.py`, `prof3d_procs.py`), and two of the four
results contradict the natural guess.

### 3.1 Only one thing costs anything

6×6 elements, N = 10, Nz = 32:

| | time | share of a step |
|---|---|---|
| **`normal_op` (one matvec)** | **95.5 ms** | **99.4%** |
| `convective` (dealiased) | 26.2 ms | 0.6% |
| gather-scatter | 0.62 ms | — |
| `rfft`+`irfft` pair | 0.55 ms | — |

The plan (§4) expected "PCG matvec ≫ FFT > convection". Confirmed, but far more
lopsided than anticipated — 99.4/0.6. **The FFT and the gather-scatter are not
worth optimising at all**, and the layout choice in §1.1 is vindicated.

There are two candidate parallel axes and this settles which matters:
convection is parallel over *z-planes*, the solve over *`k_z` modes*. Convection
is 0.6% of the work, so the mode axis is the only one worth exploiting.

### 3.2 BLAS threading buys nothing

95.51 ms → 94.84 ms going from 1 to 8 BLAS threads — 0.7%, i.e. noise. The
contractions are too small for threaded BLAS to amortise. Every driver in this
project pins `OMP_NUM_THREADS=1`; that had never been *tested*, and it turns out
to cost nothing. So: **one BLAS thread per worker, parallelism across modes.**

### 3.3 Threads tie processes — so the ceiling is bandwidth, not the GIL

Speedup of a single matvec, chunked across the mode axis:

| workers | Nz=32 (17 modes) | Nz=64 (33) | Nz=128 (65) |
|---|---|---|---|
| 4 | 3.38× | 3.48× | 3.29× |
| 8 | 3.80× | **5.68×** | 6.05× |
| 12 | 3.68× | 4.84× | **6.69×** |
| 16 | 3.08× | 4.13× | 6.10× |

Processes (fork, no pickling) reached 4.57× / 5.33× / 6.47× — **statistically
the same as threads.** Processes are immune to the GIL; if the GIL were the
constraint they would have won outright. They did not, so the ceiling is
**memory bandwidth**. Two consequences:

1. **Use threads.** No pickling, no fork, shared arrays, simpler code — and no
   performance cost. Do not "improve" this to multiprocessing later; it was
   tried and it tied.
2. **More cores will not help.** ~6.7× on a 12P+4E machine is near the ceiling.
   The next real gain must come from *reducing memory traffic per matvec*, not
   from adding workers — which redirects M6 (numba) toward fusing the operator's
   passes over the data rather than merely compiling them.

The ceiling **rises with problem size** (3.8× → 5.7× → 6.7× as modes go
17 → 33 → 65), because each worker gets more modes per chunk. Production `Nz`
should do at least as well as 6.7×; small mode counts should not bother.

### 3.4 `lssem3d/parallel.py`

Parallelises the **whole PCG** per mode-chunk, not each matvec. Two reasons:

1. one thread dispatch per solve instead of one per CG iteration — ~45× less
   dispatch overhead;
2. serial `pcg` exits on `np.all(rn < target)`, so a mode that converged long
   ago **keeps iterating until the worst mode catches up**. Chunked, each chunk
   exits on its own modes. High-`k_z` modes are strongly damped and converge
   fast, so this is not a rounding-level effect — it is real work removed.

Because of (2) the parallel `pcg` is **not bitwise identical** to the serial one:
converged modes take a different number of extra iterations. `test_parallel.py`
therefore asserts what is actually true rather than what would be convenient:

- `apply_op` **is** bitwise identical to serial at every worker count — the mode
  axis carries no cross-mode work, so chunking it is exact. This has teeth only
  because a negative control confirms `kz` actually changes the answer;
  otherwise a mis-sliced `kz` would be invisible.
- `pcg` meets the **same per-mode residual tolerance**, checked against the
  operator directly rather than against serial output, and agrees with the
  serial solution to solver tolerance.
- `workers=1` is a genuine bitwise passthrough, so the parallel path can be left
  on without perturbing a reference run.
- chunking never *costs* iterations.
- `mode_chunks` tiles the mode axis exactly — a dropped or duplicated mode would
  silently corrupt one wavenumber and leave the rest of the field looking right.

`M_inv` and `x0` both carry a mode axis and must be sliced *with* the data;
slicing one and not the other mismatches mode to operator, silently. Tested.

### 3.5 End-to-end, on a whole solve

`scratch/prof3d_endtoend.py`, Nz = 64 (33 modes), tol 1e−8:

| workers | 2 | 4 | 6 | **8** | 12 | 16 |
|---|---|---|---|---|---|---|
| speedup | 2.02× | 3.71× | 4.51× | **5.24×** | 4.88× | 4.18× |

against 718 s serial, with `max|dx|` = 5.9e−09 from the serial solution — i.e.
at the solver tolerance, as it should be. The peak at 8 workers and the ~5×
plateau match the microbenchmark's 5.68× at the same size, so the whole-solve
overhead (per-chunk `multiplicity_weight`, load imbalance) is not eating the
gain.

**One caveat, stated because it limits what this run proves:** every
configuration hit `max_iter = 4000`, serial included — a random RHS is far
harder than the real one (~45 iterations). So the "iterations saved" benefit of
§3.4(2) is *not* measured here; the iteration column reads 1.00× only because
every run was capped. The speedup above is pure parallel throughput. The
converging case is `test_integration_multimode.py`.

---

## 4. Stage 4 (M4): spectral in space, and a temporal gate that was wrong

### 4.1 Spatial and z convergence — passed

`test_stage4_mms.py`. Stage 2 deliberately used potentials that are
**polynomials of degree ≤ N**, so the GLL derivative is *exact* and the operator
could be pinned to ~1e−12. That is right for finding bugs and exactly wrong for
measuring a rate — with no truncation error there is nothing to converge. Stage 4
keeps the same vector-potential construction (so all five constraint rows still
vanish identically, giving a forcing-free probe) and swaps in **trigonometric**
potentials.

Convergence is checked **as a rate, not a tolerance**, because a fixed tolerance
is passed by any merely high-order scheme:

- **in `N`**: the late log-log slope must beat the early one by a wide margin —
  an `h^p` scheme has a *constant* slope, spectral convergence steepens without
  bound. At N = 12 the error is at round-off (< 1e−9) for both `k_z` = 0 and 2.
- **in `Nz`**: exponential convergence is a straight line in `log(err)` vs `Nz`;
  algebraic is a straight line in `log(err)` vs `log Nz`. Both fall steeply, so
  smallness cannot distinguish them — **both models are fitted and the
  exponential one is required to win**. Measured: 5.86e−2 → 1.70e−3 → 2.82e−5 →
  2.25e−9 over Nz = 8 → 24, reaching round-off (5.0e−14) at Nz = 32.

A negative control guards the whole file: at N = 4 the error must exceed 1e−6.
If the manufactured solution were accidentally representable, every error would
be ~1e−15, there would be no rate to measure, and the convergence test would
still pass on noise.

### 4.2 The temporal gate was unachievable as written

The plan asked for **RKW3 slope 3.0 ± 0.15**, adding that measuring 2.0 means
"the `α/β/γ/ζ` table is mis-transcribed **or** Crank–Nicolson is limiting". One
convergence run on the PDE cannot separate those two diagnoses, so the scalar
model `u' = (λ_e + λ_i)u` — precisely the problem the coefficients were designed
for — was run three ways:

| configuration | measured order |
|---|---|
| explicit only — the `γ/ζ` table alone | **3.025** |
| implicit only — Crank–Nicolson alone | **2.002** |
| **mixed — as actually run** | **2.189** |

**The table is correct; CN is the limiter.** The consequence matters: *no correct
implementation can score 3.0 on the mixed scheme*, so the original gate would
have failed working code and sent us to re-derive a coefficient table that was
right all along. Restated gate: **explicit-only 3.0 ± 0.15**, mixed ≈ 2.

This does not weaken the case for RKW3. Its argument was never overall 3rd-order
accuracy — it is the imaginary-axis stability interval (√3) that AB2 lacks
*entirely* (plan §0.4). CN's unconditional stability is exactly what the stiff
`a_mass` term needs, and viscous error is not what limits a DNS at these steps.

A negative control keeps the order test honest: corrupting `γ₂` from 3/4 to 0.7
must collapse the explicit-only order below 2.5, otherwise the test is not
actually sensitive to the coefficient table.

### 4.3 A trap pinned in passing

`solver3d.rkw3_step` builds `rhs = U + dt(γ N + ζ N_prev)` — the **`α_k L^{k-1}`
term is not in it**, and is left to `solve_stage`. A `solve_stage` that forgets
`α` loses an order silently, with no shape error and a plausible field. That
interface is now pinned by test rather than left as an accident.

---

## 5. A ninth silent bug, found by finally testing the driver

Everything in §2 was found before the integrator was under test. The driver
itself lives in `scratch/cavity3d_kz0.py` — a *script*, not the library — so the
assembled sequence (explicit convection → defect-corrected stage RHS → solve →
update, ×3) had never run under test. And the M2 gate that did exercise it ran at
`k_z` = 0, where every `i·k_z` term vanishes and there is exactly one mode. **No
test had ever run the driver with the mode axis populated.**

`test_integration_multimode.py` does, and immediately found this:

**The imaginary half of the Nyquist mode was unconstrained.** `fourier.py`
already states the invariant — `real_mode_indices` returns k = 0 *and* the
Nyquist mode for even `Nz`, and `assert_hermitian_ok` checks both. But that
invariant was only ever **asserted in tests, never enforced in the solve**:
`build_mask` froze real and imaginary halves *at boundaries*, and nothing froze
the imaginary half of the real modes in the interior.

`irfft` silently discards those components, so anything the solver puts there is
invisible in physical space — an unconstrained, non-physical direction that CG
fills happily. Measured after three steps: Nyquist imaginary part **1.5e−03**
against a real part of **6.1e−03** — comparable, not a rounding artefact. The
solver's own state would have failed `assert_hermitian_ok`.

Why nothing caught it earlier: at `k_z` = 0 the real and imaginary halves
decouple (no `i·k` term), so an imaginary half that starts at zero *stays* zero.
The whole class is invisible to every `k_z` = 0 test, which is all the driver had.

Fixed in `bc.py` via `real_mode_columns`, a single source of truth shared by
`build_mask`, `apply_values` and `prescribed_entries` — those three must agree on
which DOFs are prescribed, and a disagreement between the mask and the
value-writer is precisely the original 2D bug. Correctness only: the arrays keep
their full width, so the matvec costs what it did before.

**A second, self-inflicted lesson from the same test.** Its first version ran
800 unpreconditioned iterations and stopped at residual 1.1e−01 — unconverged —
which quietly turned a serial-vs-parallel comparison into a comparison of two
*different unconverged states*. The tell was that the discrepancy was **identical
at every solver tolerance** (6.310e−05 at 1e−8, 1e−10 and 1e−12); a genuine
tolerance effect scales with tolerance. The test now uses the Jacobi
preconditioner and asserts it did not hit `max_iter`, so it cannot pass
vacuously again.

---

## 6. Conditioning: `k_z` is not what limits the solve

`scratch/kz_iterations.py`. The plan (§4) states an expectation and a diagnosis
together: iterations-per-solve should *fall* with `k_z`, because the `k_z²` term
makes the operator more diagonally dominant, and **"a flat profile is evidence
the preconditioner is not using `k_z`"**. Measured before optimising the
preconditioner — because if the profile were flat, the Jacobi diagonal would be
the wrong object to optimise.

Each mode solved as its own single-mode problem (the batched `pcg` reports only
the worst mode, so per-mode counts are invisible there by construction),
`Re` = 180, `dt` = 5e−3 ⇒ `a_mass` = 1200, AC on with `κ_p` = `a_mass`:

| `k_z` | 0 | 2 | 8 | 15 | 16 (Nyquist) |
|---|---|---|---|---|---|
| iterations | 1001 | 1398 | 946 | 621 | 906 |

Not falling — it *rises* to a peak near `k_z` = 2, then declines, ending at
**0.905×** the `k_z` = 0 count. By the plan's stated criterion that is "flat",
which would condemn the preconditioner. **That verdict would have been wrong.**

### Why: `a_mass` dominates the diagonal by three orders of magnitude

The `k_z²` term enters scaled by viscosity. At `Re` = 180 and the largest
wavenumber here, `ν·k_z²` = **1.42**, against `c` = `a_mass` = **1200**. It
cannot possibly shift the conditioning. Sweeping `a_mass` with everything else
fixed confirms the mechanism directly:

| `a_mass` | iterations at `k_z`max ÷ at `k_z` = 0 |
|---|---|
| 1200 (production) | 0.905 — flat |
| 100 | 0.997 — flat |
| 10 | 0.763 |
| **1** | **0.249 — falls steeply** |

The `k_z` trend is real and appears exactly when `a_mass` stops swamping it. For
the `k_z²` term to compete at production settings one would need
`k_z ~ √(c/ν)` = √(1200·180) ≈ **465**; at `Nz` = 128 the largest `k_z` is 64.
**The regime the plan's diagnostic assumes is unreachable in this formulation.**

### Consequences

1. **The diagnostic is restated**: a flat `k_z` profile at production `a_mass` is
   *expected and correct*, not evidence of a defect. It only carries information
   at small `a_mass`.
2. **Tuning the preconditioner's `k_z` handling would buy nothing** — `k_z` is
   not what limits conditioning. `a_mass` is. This removes the conditioning
   motivation for the analytic Jacobi diagonal; that work is now justified only
   as a setup-cost saving (§8.2), which is a much smaller prize.
3. The Nyquist mode costs **46% more** iterations than its neighbour (906 vs
   621) because its imaginary half is frozen (§5), leaving a different — and
   evidently worse-conditioned — system. Worth knowing before it is mistaken for
   a bug.

**Caveat on absolute counts.** These use a random RHS, which is far harsher than
the smooth defect-correction RHS a real step produces (~46 iterations with AC in
the M2 run). The *trend across `k_z`* is the trustworthy part, since the same
spatial pattern is used at every mode; the ~1000 figures are not a forecast of
production cost.

---

## 7. Stage 5 (M5) measured in 3D: the windows overlap, by a wide margin

`scratch/channel3d.py` (rig) and `scratch/channel3d_stage5.py` (sweep). This is
the plan's ⚑ **decision gate** — "if the windows do not overlap, the formulation
is infeasible as specified" — and until now it had only ever been answered with
2D numbers the plan itself says do not transfer.

### 7.1 The collision, and why the sweep runs `dt` downward

With RKW3/CN the two constraints pull in opposite directions:

```
a_mass = 1/(β₂·dt) = 6/dt        (worst stage)
CFL    ∝ dt                       (limit √3 for RKW3)
```

so the feasible window is `6/a_max < dt < dt_CFL`, and it is non-empty iff
`a_max > 6/dt_CFL`. **`a_mass` instability therefore appears at SMALL `dt`** —
the opposite of the usual intuition, and the reason the sweep decreases `dt`.

### 7.2 The rig, and its control

Walls in `y`, periodic in `x` (SEM connectivity, `mesh.periodic_x`) and `z`
(Fourier). Base flow `u = 6y(1−y)` held by a body force `f_x = 12ν`. Perturbation
is an analytic, divergence-free roll pair from a streamfunction in `(y,z)`,
`ψ = A sin²(πy)cos(kz)`, which puts energy in `w` and the transverse
vorticities — the components every `k_z` = 0 test is blind to.

Validated before use: the base flow is held to **2.2e−15** over 5 steps, which
confirms the periodicity, the body-force normalisation (an unnormalised `rfft`
means a physical constant `C` has mode-0 coefficient `C·nz`; getting this wrong
rescales the flow by `nz` and is otherwise invisible), and — importantly —
that **Poiseuille is exactly representable**, which is the premise of plan §0.2.
That makes the laminar case a *control*, not evidence: a zero residual is
precisely the condition under which the `a_mass` mechanism stays hidden.

### 7.3 Result — stable at every `a_mass` tested, up to 6000

Re = 180, N = 6, 3×3 elements, Nz = 16, 200 steps (2D failures appeared by step
~33, so ~6× margin), AC on with `κ_p` = `a_mass`:

| case | `dt` | `a_mass` | CFL | status | `E_pert/E₀` | mean-profile err |
|---|---|---|---|---|---|---|
| laminar (control) | 0.001 | 6000 | 0.021 | **OK** | — | 1.1e−15 |
| perturbed | 0.05 | 120 | 1.303 | **OK** | 0.245 | 1.1e−02 |
| perturbed | 0.02 | 300 | 0.521 | **OK** | 0.731 | 1.7e−02 |
| perturbed | 0.01 | 600 | 0.261 | **OK** | 0.762 | 1.0e−02 |
| perturbed | 0.005 | 1200 | 0.130 | **OK** | 0.807 | 4.6e−03 |
| perturbed | 0.0025 | 2400 | 0.065 | **OK** | 0.856 | 1.6e−03 |
| perturbed | 0.001 | **6000** | 0.026 | **OK** | 0.927 | 3.5e−04 |

The perturbation **decays in every case**, which is the known answer for laminar
plane Poiseuille far below `Re_crit` ≈ 5772. The `E/E₀` column falls with `dt`
only because 200 steps is less physical time at smaller `dt` (t = 10 down to
t = 0.2); the decay *rate* is consistent throughout.

### 7.4 Verdict

`dt_CFL` = **0.0665** on this grid, so the constraints are

```
CFL:     dt < 0.0665   ⇒  a_mass > 90
a_mass:  clean to 6000 ⇒  dt > 0.001
```

**The window spans a factor of ~66 in `dt`.** The gate passes, and not
marginally. The plan's worst-case fear — that no operating point exists and the
formulation would need semi-implicit convection and a re-plan — is resolved
against.

**Scaling to M7.** The requirement is `a_mass > 6/dt_CFL`, and `dt_CFL` shrinks
with resolution:

| grid | `dt_CFL` | `a_mass` required |
|---|---|---|
| N=6, 3×3, Nz=16 | 0.0665 | 90 |
| N=8, 4×4, Nz=32 | 0.0294 | 204 |
| N=10, 6×6, Nz=64 | 0.0129 | 466 |

Against a measured ceiling of at least 6000 this leaves ~13× headroom at the
finest grid tested. Turbulence will tighten `dt_CFL` further (larger velocity
gradients), so this margin should be re-checked at the M7 resolution rather than
extrapolated — but there is no sign of a collision.

### 7.5 AC buys affordability here, not stability — a correction

AC-off at the same settings costs **~10× more per step** (≈1 min against 5.3 s).
But the completed AC-off sweep does **not** support the stronger claim:

| `dt` | `a_mass` | CFL | AC-off outcome |
|---|---|---|---|
| 0.05 | 120 | 1.303 | **DIVERGED at step 184** |
| 0.01 | 600 | 0.261 | OK, 200 steps |
| 0.0025 | 2400 | 0.065 | OK, 200 steps |
| 0.001 | 6000 | 0.026 | OK, 200 steps |
| 0.001 (laminar) | 6000 | 0.021 | OK, 200 steps |

**AC-off is stable across the whole `a_mass` range**, and the single failure is at
the *largest* step, where CFL = 1.303 sits near the RKW3 limit of √3 — i.e. it
points at CFL, not at `a_mass`.

So in this 3D periodic channel **AC is required for affordability, not for
stability.** The "AC is the enabling technology" framing came from the 2D
*outflow* case (plan §0.3), where AC-off genuinely blew up, and it should not be
carried over to the periodic channel unqualified. This does not change the Stage
5 verdict, which passes either way; it changes *why* AC is in the scheme.

---

## 7A. The AC investigation — and the scaling bug underneath all of it

This section reads chronologically on purpose. Everything in §7A.1 was measured
and is correct as data; §7A.2 is the root cause that reinterprets all of it. The
symptoms were real; the diagnosis attached to them was not.

### 7A.1 The symptoms, in the order they appeared

Artificial compressibility is a **steady-state** device unless sub-iterated. The
continuity row solves

```
κ_p·p + div u = κ_p·p_prev     ⇒     div u = −κ_p·(p − p_prev)
```

At a steady state `p = p_prev` and it is exact — which is why the M2 cavity gate
could never see a problem. The driver had **no sub-iterations at all**, a
configuration never tested in this project, 2D or 3D. Four measurements followed:

| measurement | result |
|---|---|
| `div u` vs the AC-off floor, `dt` = 0.008 | AC `nsub`=1: **2.39×** the floor; `nsub`=3: 1.75× for 1.9× the CG cost |
| `κ_p` scaling | flat under refinement when `κ_p` ∝ 1/`dt`; falls when `κ_p` is fixed |
| Richardson self-convergence, real 3D stepper | order **~0.92**, where Stage 4 predicts ~2 |
| **Stokes decay** (exact answer, `figs/chan_fig1_pref.png`) | **12.5% error in σ at every `dt` — zeroth order**; `nsub`=10 no better than `nsub`=1 |

The Stokes case was decisive because it has an **analytic** rate, so the error is
absolute. It exposed what self-convergence structurally cannot: the ~0.92 was a
true *rate* toward the scheme's own limit, and that limit was simply wrong (L5).

**Two contributions to the 12.5% were then separated.** The initial condition set
`p` = 0, which is inconsistent — for the Stokes mode
`p = σ·(A sinh(αy) + B cosh(αy))·sin(αx)`, derived by hand from
`−σu = −p_x + ν∇²u`, where the hyperbolic terms cancel. Supplying it cut the
error **12.5% → 1.0%**. So most of the headline number was an initial-condition
artifact, and only ~1% was attributable to AC.

### 7A.2 CAUSE: `lssem3d` hard-coded one row weighting, and it is the wrong one for this benchmark

**This subsection was originally written as "ROOT CAUSE: the functional had no
row weights", i.e. as a bug. That was overreach, corrected below in §7A.2b after
the cavity re-run contradicted it.**

The question that broke it open: at `k_z` = 0 the 3D system **is** the 2D system
(Stage 1 proves the operator), so why can `lssem2d` solve the AC-free problem
routinely while `lssem3d` cannot solve it at all?

`lssem2d` writes the momentum row as `a_mass·u + a_flux·N(u)` with the continuity
and vorticity rows at weight 1 — so **`a_flux` is the least-squares weight of
momentum against the constraints**. Its legacy setting, the one the Chan (1996)
validation runs, is `a_mass = fac1 = 1`, `a_flux = dt`: **every row O(1)** in the
velocity.

**`lssem3d` had no row weighting whatsoever.** Its momentum rows carried
`c = 1/(β_k·dt)` against constraint rows of O(1). At `dt` = 5e−3 that is
`c` = 1200, and the functional *squares* the rows:

```
J₃D = ∫ [ (c·u + p_x + ν∇×ω)²  +  (div u)²  +  (ω-definitions)² ]
J₂D = ∫ [ (u − u_old + dt·N)²   +  (div u)²  +  (ω-definitions)² ]
```

Momentum outweighed continuity by **c² ≈ 1.4×10⁶**. The minimiser therefore
essentially ignored `div u`, and the normal operator was hopelessly conditioned.

Stokes decay, AC **off**, `dt` = 5e−3, σ_exact = 9.3137399:

| | σ | rel err | CG |
|---|---|---|---|
| no row weights | 9.31809 | 4.68e−04 | **600000** (cap saturated) |
| **row weights** | **9.31413** | **4.19e−05** | **22047** (converged) |

**27× fewer CG iterations and 11× more accurate — on this benchmark.** The
qualifier is not decorative; see §7A.2b.

### 7A.2b CORRECTION: the weighting is a problem-dependent choice, not a bug

Re-running the M2 cavity with row weights **contradicted the framing above** and
forced a re-reading of `lssem2d`.

`lssem2d` supports **two** weightings, not one:

| setting | `a_mass` | `a_flux` | used by |
|---|---|---|---|
| legacy (both `None`) | `fac1` = 1 | `dt` | the **Chan (1996) Stokes-decay validation** |
| `w_mom` = 1 | `fac1/dt` | 1 | the cavity / BFS / `a_mass` studies |

**`lssem3d`'s original scaling is exactly the `w_mom` = 1 setting** — a
legitimate, well-used 2D configuration, not a defect. What `lssem3d` lacked was
the *option*: it hard-coded one of the two, and the one it hard-coded is not the
one the Stokes benchmark uses. That is why the 3D code could not reproduce a 2D
result the 2D code produces routinely.

**Which weighting wins is problem-dependent**, and the two benchmarks disagree
sharply. Cavity, Re = 1000, `dt` = 1.74e−3, CG per step:

| row weights (legacy) | `κ_p` | CG/step |
|---|---|---|
| **off** (`w_mom` = 1, original) | `a_mass` | **25** |
| on | `a_mass` | 688 |
| on | 10 | 1028 |
| on | 1 | 5519 |
| on | 0 | 12320 |

Legacy weighting is **27× worse at best** on the cavity, while it is what makes
the Stokes case solvable at all. The plausible discriminant is viscosity: legacy
scales the momentum row to `u + β·dt·(p_x + ν∇×ω)`, and at ν = 1e−3 with
β·dt ≈ 3e−4 the vorticity coupling is ~3e−7 — effectively absent. At ν = 1
(Stokes) it is not.

**Consequences for the claims in §7A.3:**

* "AC was never a solver requirement" — **true only for the ν = 1 Stokes case.**
  On the cavity, AC is worth 25 CG/step against 12320 without it. AC's value is
  Reynolds-number dependent and it is **not** dispensable.
* "correctly weighted, AC is slower as well as less accurate" — **withdrawn.**
  That was measured on Stokes alone and does not generalise.
* The order-2.00 result (§7A.5) stands: it is a statement about the Stokes
  benchmark under legacy weighting, which is the configuration it was measured in
  and the one the 2D reference uses.

**What is actually established:** `rw` gives `lssem3d` the second weighting it was
missing, the Stokes benchmark now reproduces at its design order, and **choosing
the weighting per problem is an open design question** — exactly as it is in 2D.

### 7A.3 What this does to the AC findings

**AC was never a solver requirement.** It was compensating for the row scaling:
adding `κ_p·p` to the continuity row lifts that row toward the momentum rows.
One fact explains the whole tangle —

* the **63× "conditioning benefit"** of AC: it was repairing a scaling bug, not
  preconditioning a well-posed system;
* why **AC-off was unsolvable in 3D but routine in 2D**: 2D never had the bug;
* why **AC cost accuracy**: it rebalanced by *changing the equation* rather than
  by rescaling it, so incompressibility was traded away;
* why **`nsub` could not rescue it**: with `κ_p` large,
  `κ_p(p−p_prev) + div u = 0` is satisfiable by a tiny pressure change for *any*
  `div u`, so the sub-iteration that would restore incompressibility converges
  ever more slowly as `κ_p` grows.

The κ_p sweep (12.5% → 2.5% as `κ_p` fell 1000×, for 1.36× the CG cost) was
therefore measuring how much of the scaling damage could be undone by backing AC
off — not a fundamental accuracy/cost trade. **With row weights the question is
no longer "how small can `κ_p` be" but "is AC needed at all".**

`rw` is optional and defaults to `None`, so every earlier result remains
reproducible.

### 7A.4 Retractions

Three, recorded because the sequence is instructive:

1. **"κ_p fixed is the inconsistent scaling"** (§7A.3, previous version) — the
   sign was backwards. With `κ_p` fixed, `p − p_prev` → 0 as `dt` → 0, so
   `div u` → 0 and the incompressible limit **is** recovered; with `κ_p` ∝ 1/`dt`
   it is not. I had also used this to overturn a *correct* earlier diagnosis (the
   `p` = 0 initial condition), which the Stokes test then vindicated. A wrong
   diagnosis of a discarded result outlived the result.
2. **"AC-off gives 0.76% error, so the 12.5% is AC's fault."** Both AC-off runs
   saturated the CG cap (1.2e6 and 1.8e6 iterations) and returned mutually
   inconsistent rates (9.384 and 14.279). Withdrawn. We now know *why* they could
   not converge: §7A.2.
3. **"AC is the enabling technology"** — true for the 2D *outflow* case, and true
   of `lssem3d` only because of the scaling bug.

### 7A.5 RESOLVED: with row weights and no operator-AC, the scheme is exactly second order

`scratch/stokes3d.py`, N = 10, 2×4 elements, Stokes decay against the analytic
σ = 9.3137399. Row weights **on**. No solve hit `max_iter` (guard armed, and it
matters — a capped solve promotes the preconditioner into the scheme, which is
how the earlier AC-off "controls" returned two different wrong answers).

| `dt` | 0.01 | 0.005 | 0.0025 | 0.00125 | **order** |
|---|---|---|---|---|---|
| **AC off** (`κ_p` = 0) | 1.680e−04 | 4.189e−05 | 1.046e−05 | **2.613e−06** | **2.00, 2.00, 2.00** |
| AC on, `κ_p` = `a_mass` | 6.477e−03 | 6.063e−03 | 6.062e−03 | 6.072e−03 | 0.10, 0.00, −0.00 |

**Exactly the design order.** RKW3/CN is second order by construction — RK3's
third order applies to the explicit convective half alone (measured 3.025 on the
coefficient table), and Crank–Nicolson caps the mixed scheme at 2 (measured 2.189
on the scalar model). The 3D PDE stepper now delivers that 2, to two decimal
places, three refinements running.

This is the **first PDE-level demonstration** that the 3D scheme achieves its
design order. Stage 4's temporal gate ran on a scalar model with no pressure, no
constraint rows, no assembly and no BCs; the AC-off PDE controls never converged.
"AC is the whole gap" was a hypothesis until this table.

**And the contrast is total.** Operator-AC at `κ_p` ∝ 1/`dt` is pinned at
6.1e−03 with zeroth order — 2300× the AC-off error at the finest `dt`, and
refining time does nothing. The entire accuracy pathology of §7A.1 was
operator-AC standing on a mis-scaled functional.

#### The consistency ladder, completed

`div u = −κ_p·(p − p_prev) ≈ −κ_p·ṗ·dt`, so the scaling of `κ_p` sets the order
the scheme is *permitted* to reach:

| `κ_p` scaling | AC error | order permitted | status |
|---|---|---|---|
| ∝ 1/`dt` (was production) | O(1) | zeroth | **measured: 0.00** |
| fixed | O(`dt`) | first | measured: error falls, ~0.4 |
| ∝ `dt` | O(`dt²`) | second | untested — now moot |
| **0 (AC off)** | **none** | **second** | **measured: 2.00** |

With row weights the last row is simply available, so the ∝ `dt` compromise is
unnecessary for production. It remains the fallback if AC is ever wanted back for
speed on a harder problem.

### 7A.6 What is still open

* **Preconditioner-only AC** — `κ_p` in `jacobi_diagonal`/`M_inv` only, zero in
  the operator. Zero bias by construction. Much less urgent now that AC-off
  solves in 17k–147k iterations, but it is the right form if AC is ever wanted
  back for speed.
* **Is AC wanted at all?** Cost at `dt` = 0.01: AC-off 17203 CG against AC-on
  24471 — AC is now *slower as well as less accurate*. The 63× benefit was
  entirely an artifact of the mis-scaled functional.
* **The spatial floor** has not been mapped. The 2D figure shows error flattening
  at fine `dt` as spatial error takes over, and `p`-refinement lowering it. At
  N = 10 nothing has flattened by `dt` = 1.25e−3 (still 2.00), so the floor is
  below 2.6e−06 — worth locating before M7 sets its resolution.
* **`nsub` non-monotonicity**: `nsub` = 10 was worse than `nsub` = 3. Unexplained,
  and possibly a defect in the sub-iteration. Re-check with row weights on.
* **Row weights everywhere**: only the Stokes rig passes `rw` today. The cavity
  and channel drivers, and the M2/Stage 5 results, still run unweighted.

## 8. What is next

Reordered by §7A.2. The AC accuracy programme is largely dissolved: it was
measuring a scaling bug. What replaces it is re-validation with the functional
correctly weighted.

### 8.1 DONE — the 2D Stokes result is reproduced

Order **2.00, 2.00, 2.00** with row weights and AC off (§7A.5), error 2.6e−06 at
`dt` = 1.25e−3. The remaining piece is the *p*-refinement half of
`figs/chan_fig1_pref.png` — locating the spatial floor by sweeping `N`, which
M7 needs anyway to choose its resolution.

### 8.2 Decide AC's future — the case for it has collapsed

At `dt` = 0.01, AC-off costs **17203** CG against AC-on's **24471**: with the
functional correctly weighted AC is *slower as well as less accurate*. Unless a
harder problem revives it, the production configuration is **row weights, no
operator-AC**. If it is ever wanted back, use **preconditioner-only AC**
(`κ_p` in `M_inv`, zero in the operator), which cannot bias the answer.

### 8.3 Propagate row weights to every driver

Only the Stokes rig passes `rw`. The cavity and channel drivers still run
unweighted, so **M2 and Stage 5 were both measured on the mis-scaled functional**.
Neither verdict is likely to flip — M2 compared 3D against 2D at matched settings,
and Stage 5 was a stability result — but both should be re-run once 8.1 passes,
and until then their numbers carry an asterisk.

### 8.4 Hardening carried over from review

* ~~Port the true-residual safeguard~~ **DONE.** `pcg` now verifies the
  recursive residual against `b − A x` when it claims convergence, restarts the
  recursion from the true residual on drift, and **reports the true residual**
  rather than the recursive one. Costs one matvec per solve, only at the
  convergence check. Per-mode restart is safe because the recurrences are
  independent.
* ~~Standardise `M_inv`~~ **DONE.** `solver3d.jacobi_inverse` — zero on
  prescribed dofs, and it *raises* on a negative diagonal instead of clamping a
  bug into a 1e30 multiplier on a live dof. All drivers and tests converted; the
  `1.0/np.maximum(d, 1e-30)` idiom is gone from the 3D code.
* **Convergence guards on every order study.** A capped solve promotes the
  preconditioner into the scheme — that is how two AC-off runs returned σ = 9.384
  and σ = 14.279. Assert `max_iter` was not reached, and that the solver
  tolerance sits well below the `dt`-to-`dt` difference being measured; at
  `dt` = 1.25e−3 the signal is ~2.6e−06, so 1e−12 leaves six decades, but a
  1e−8 tolerance would have left only two.

* `nsub` non-monotonicity (`nsub` = 10 worse than 3) — recheck with row weights.
  *(Two stale duplicates of the DONE items above were removed here in the
  closure review.)*

### 8.5 Then the original queue

M6 (numba, aimed at *fusing* passes — the matvec is bandwidth-bound, §3.3), the
M7 step-cost model, the Stage 6 forcing decision (constant mass flux vs constant
pressure gradient — undecided, and it changes the per-step constraint), and
closing M2 by restarting from the saved field.

### Noted, not owned — CLOSED by the OBC session (commit 7ddcda5)

Two defects in `lssem2d`, handed to the OBC session and since fixed there:

* ~~`newton_step` builds `M_inv` at `U/2` but solves at full `U`~~ **fixed** —
  the diagonal is now built from the same linearisation the CG solves.
  Measured impact on the convection-dominated Re = 1000 cavity: **1.2%** fewer
  CG iterations (18925 → 18696). Small by structure: the GLL differentiation
  matrix has zero diagonal at interior nodes, so first-derivative convection
  barely reaches a *diagonal* preconditioner — which is also why the defect
  survived every study unnoticed.
* ~~the dead per-step `state.M_inv` build~~ **removed** — numerics-neutral
  (identical iteration counts), one full Jacobi build per time step saved.
  The `M_inv` parameter of `newton_step` is kept for call-site compatibility
  and documented as ignored.

Converged fixed points are unaffected by both (a preconditioner is invisible
in a converged solve); the 2D and 3D suites pass, `test_stage1_vs_2d` included.

## 9. Inventory

### Modules (`lssem3d/`)

| file | role |
|---|---|
| `fourier.py` | `rfft`/`irfft` with z last, wavenumbers, 3/2-rule dealiasing, Hermitian assertions |
| `deriv.py` | batched (x, y) derivatives — §2.4 |
| `operator.py` | the 7-unknown / 8-row VVP operator, `L₀`, `W·L₀`, `L₀ᵀ`, split-real facades |
| `convect.py` | dealiased `u·∇u`, CFL with a layout assertion |
| `timestep.py` | RKW3/CN coefficients, exact-arithmetic consistency check at import |
| `bc.py` | mask and prescribed values, incl. `real_mode_columns` (§5) |
| `solver3d.py` | gather-scatter, multiplicity weight, `normal_op`, batched `pcg`, `rkw3_step` |
| **`parallel.py`** | **mode-parallel `apply_op` and `pcg` (§3.4)** |

### Tests — 144, all passing

```
uv run --quiet python -m pytest lssem3d/tests -q
```

`test_fourier` · `test_operator` · `test_convect` · `test_solver3d` · `test_bc` ·
`test_deriv` · `test_stage1_vs_2d` (vs. 2D) · `test_stage2_mms` (analytic) ·
**`test_stage4_mms`** (spectral rates) · **`test_stage4_temporal`** (order) ·
**`test_parallel`** · **`test_integration_multimode`** (the driver, many modes)

### Benchmarks (`scratch/`)

| script | answers |
|---|---|
| `prof3d.py [nz] [threads]` | where the time goes; does BLAS threading help |
| `prof3d_modes.py [nz] [maxw]` | thread-parallel modes: speedup and bitwise equality |
| `prof3d_procs.py [nz]` | threads vs processes → GIL or bandwidth |
| `prof3d_endtoend.py [nz]` | speedup on a whole PCG, not one matvec |
| `kz_iterations.py` | CG iterations per `k_z` mode (§6) |
| `channel3d.py` / `channel3d_stage5.py` | Stage 5 rig and the `a_mass`/CFL sweep (§7) |
| **`stokes3d.py`** | **3D Stokes decay against the analytic rate — the accuracy gate (§7A)** |
| `ac_subiter.py`, `ac_unsteady_divergence.py`, `ac_temporal2.py` | the AC investigation (§7A.1) |

### Using the parallel solver

Drop-in for `solver3d.pcg`, plus `workers` (default: performance-core count,
overridable with `LSSEM3D_WORKERS`):

```python
from lssem3d import parallel as PAR
dU, iters, resid = PAR.pcg(b, D, mesh.facx, mesh.facy, kz, NU, c,
                           mesh=mesh, mask=mask, M_inv=Minv[k],
                           tol=1e-10, wq=mesh.wq, kap=kap)   # workers=None -> auto
```

`workers=1` is a bitwise passthrough to the serial version, so it can be left on
in a reference run. Call `PAR.shutdown()` to release the thread pool. **It does
nothing for a single-mode (`k_z` = 0) run** — see §8.2.
