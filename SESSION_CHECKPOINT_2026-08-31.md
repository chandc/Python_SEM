# Session checkpoint — 2026-08-31

State at reboot. Everything below is committed and pushed to
`github.com/chandc/Python_SEM` on `main`.

---

## What this session did

Two workstreams, both in 2D. **`lssem3d` was not touched.**

### A. FOSLS study — finished (`FOSLS_2D_PLAN.md`)

Tested McCormick's claims on `lssem2d`. **C1 (H¹ norm-equivalence) and C3
(h-independent AMG) both CONFIRMED; C2 (weak BCs) still untested.** The
strongest result: F1 measured `c₂/c₁ = 1.55e4`, predicting `√(c₂/c₁) ≈ 124`
iterations *before* F2 measured AMG at 113–138.

Extended this session:

| | |
|---|---|
| **F2e** | Every earlier F2 number was at **N=4**. Re-measured to N=12: ρ flat (0.9710 → 0.9643), ω fraction invariant at 0.977–0.978, but AMG is **not p-independent** (2.16× growth). |
| **F2f** | **AmgX 2.5.0 on the GB10** (`/tmp/AMGX`, sm_121). Pipeline validated exactly — PCG+Jacobi reproduced scipy's 671/1245/1611/2103. Compiled V-cycle beats compiled Jacobi **2.20× under h**, break-even under p. |
| **F2g** | Both AmgX schemes **collapse** at high p (aggregation stalls from N=6, classical from N=8); pyamg's energy-minimised SA does not. **LOR stays refuted at N=12.** |
| **F2h** | **Prior art: Heys, Manteuffel, McCormick & Olson (2005)** did this study — AMG on FOSLS Stokes, GLL spectral elements, p=1…8. Four of our findings are confirmations, including that ρ ≈ 0.95 with CG carrying it is the *known* behaviour, not a defect. |
| **F2h(ii)** | Read Pazner properly: **LOR trades density for anisotropy**, and AMG is *not* p-robust on the LOR operator — he uses ILU-smoothed geometric multigrid. |

### B. p-multigrid coarse solver — new (`PMG_ALGORITHM.md`)

Wrote the algorithm reference (V-cycle, Chebyshev recurrence, transfer
operators, coarse solvers) and added **`DirectCoarse`** to `lssem2d/precond.py`:
assemble the coarse operator by probing `apply_A`, factorise, reuse.

| test | result |
|---|---|
| **T1** Poiseuille, linear solve | 1.25–1.40× fewer CG iterations, ~2× less wall |
| **T2** Poiseuille, path dependence | spread **1.26e-06 → 1.07e-08**, a **117×** reduction |
| **T3** Gartling BFS | drift **17×** slower; reattachment spread 3.0% → 1.3%; **2.6× wall** |
| **T4** short BFS | **not run** — case characterised only (§6.5) |

Also fixed: PMG2 never propagated the Dong OBC coefficients to the coarse level
(`obc_D0` 1.0 → 0.0), the same class of bug `precond.py` already documents for
`w_mom`/`w_mass`.

---

## Resume queue, highest value first

1. **The 3D minimal channel.** Stalled since **2026-08-24**; Spark GPU at 0% for
   a week. The probe `amp_roll=2.0, amp_noise=1.0, dt=8e-4` was specified and
   **never launched**. This is the actual scientific goal — everything above is
   solver research that F1 already scoped away from the time-stepper.
2. **Log `J` in `scratch/minchan.py`.** F4′ validated it as an error estimator
   (effectivity to 1.40×; rose 8.6e9 on the `minchan_001` defect). It is already
   computed every step and never read. ~1 hour.
3. **The T3 drift.** No Gartling configuration reaches a fixed point — it drifts
   at constant rate along the soft direction (`|U(k)−U(k−2)|` is exactly 2×
   `|U(k)−U(k−1)|`). That is a *formulation* problem, arguably more important
   than the preconditioner comparison that found it.
4. **T4** — short BFS under genuine backflow. `obc_D0 ≠ 0` there, so the coarse
   propagation fix is live, unlike T3.
5. **Element-local coarse assembly.** `DirectCoarse` is `O(elements)`; this is
   what blocks `coarse_solver='direct'` as a default.
6. **F3 — weak boundary conditions**, the last untested McCormick claim.

---

## Environment notes

* **AmgX 2.5.0** is built on Spark at `/tmp/AMGX/build/libamgxsh.so` for
  `sm_121`. **`/tmp` does not survive a reboot** — if Spark is rebooted too, it
  must be rebuilt (CUDA 13.1 container; the host's CUDA 12.0 *cannot* target
  `sm_121`).
* 2D sweeps run 12-way parallel on Spark via `~/pmg2d/` + the PyTorch container;
  Spark's host Python has no numpy.
* `lssem2d` is **CPU-only** (`VALID = ('numpy', 'numba')`) — no GPU backend, and
  at ~4–9k DOF it would not benefit from one.
* Two more hardcoded Mac paths fixed (`gartling_run.py`, `fgrid.py`); ~288 remain.

## Not mine, left uncommitted

`TGV_VALIDATION.md`, `BFS_VALIDATION_LADDER.md`, `DIVERGENCE_AND_CONSISTENCY.md`,
`KIM_MOIN_REVIEW.md`, `SCHEME_COMPARISON.md`, `SESSION_CHECKPOINT_2026-08-30.md`,
`reference/`, `results/`, and the TGV plots are from the parallel OBC/validation
session and were deliberately not touched. They survive a reboot as files.
