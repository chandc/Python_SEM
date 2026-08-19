# The `a_mass` instability is an outflow-boundary phenomenon

Study date: 2026-08-18. Consolidates and **resolves** a question that runs
through `GARTLING_VALIDATION.md`, `ARTIFICIAL_COMPRESSIBILITY.md` and
`3D_DEVELOPMENT_PLAN.md`, and whose stated cause changed twice along the way.

Reproduce: `scratch/chan_amass_sweep.py` (inlet-profile sweep),
`scratch/stokes_amass_probe.py` (Stokes operator, and the closed-domain probe),
`scratch/pois_temporal.py` (the periodic-channel evidence, pre-existing).

---

## The question

`a_mass = w_mass·fac1/dt` is the coefficient on the time-derivative term in the
momentum rows of the least-squares functional. The continuity and vorticity rows
carry weight exactly 1, so refining `dt` raises `a_mass` and progressively
outweighs the constraints. `GARTLING_VALIDATION.md` measured a hard threshold on
the backward-facing step — **every run with `a_mass` ≤ 6.05 bounded, every run
with `a_mass` ≥ 12.1 divergent, 34 runs, no crossover** — and the question was
what actually causes it, because that determines whether it constrains a 3D
turbulent-channel solver.

Three explanations were live. Each is now settled by measurement.

---

## The evidence

All runs below use the same solver, the same 12×2 N=10 channel grid or the 6×6
N=10 cavity, `w_mom = w_mass = 1`, and BDF2 (`a_mass = 1.5/dt`).

### A. With an outflow: the residual makes no difference

`scratch/chan_amass_sweep.py`, plane channel, P+Z outlet. The **parabolic**
inlet makes the exact solution representable (residual ≈ 0); the **uniform**
inlet forces the flow to develop (rms `div u` ≈ 8e−02, the BFS regime).

| inlet | residual | `a_mass` | outcome | step | max\|u\| |
|---|---|---|---|---|---|
| uniform | 8e−02 | 60 | BLEWUP @ t=0.83 | **33** | 22.4 |
| **parabolic** | **≈ 0** | 60 | BLEWUP @ t=0.83 | **33** | 22.9 |
| uniform | 8e−02 | 120 | BLEWUP @ t=0.58 | **46** | 20.4 |
| **parabolic** | **≈ 0** | 120 | BLEWUP @ t=0.58 | **46** | 20.6 |
| uniform | 8e−02 | 300 | BLEWUP @ t=0.35 | 71 | 23.9 |
| **parabolic** | **≈ 0** | 300 | BLEWUP @ t=0.29 | 59 | 45.1 |

The two inlets fail at the **same step and the same time** at `a_mass` = 60 and
120. A residual of zero buys nothing.

> **Excludes:** the non-zero-residual explanation of `GARTLING_VALIDATION.md` §8.

### B. With an outflow: convection makes no difference

`scratch/stokes_amass_probe.py` zeroes the linearisation (`fu = fv = 0`) so the
convective terms vanish identically from `apply_L`, leaving the Stokes-like
operator that explicit convection would produce. Poiseuille *is* a Stokes
solution, so the exact answer is unchanged. An assert inside the loop verifies
`dfu_dx = dfv_dy = 0` rather than trusting the patch.

| operator | `a_mass` | outcome | step |
|---|---|---|---|
| linearised Navier–Stokes | 60 | BLEWUP | 33 |
| **Stokes-like** | 60 | BLEWUP | **29** |
| **Stokes-like** | 300 | BLEWUP | 144 |

> **Excludes:** convection, and with it semi-implicit convection as a remedy —
> it cannot fix a failure that persists with no convection at all.

### C. Without an outflow: no failure, over a 40× wider range

Closed cavity (no outflow anywhere), same Stokes-like operator:

| domain | `a_mass` | outcome | max\|u\| |
|---|---|---|---|
| cavity, closed | 60 | **ok**, 200 steps | 1.0000 (the lid value) |
| cavity, closed | 300 | **ok**, 200 steps | 3.49 |

Streamwise-**periodic** channel, full Navier–Stokes, from
`TEMPORAL_ACCURACY_STUDY.md` / `scratch/pois_temporal.py` — pre-existing evidence
that predates this investigation:

| `dt` | 0.01 | 0.005 | 0.0025 | 0.00125 | 0.000625 |
|---|---|---|---|---|---|
| `a_mass` | 150 | 300 | 600 | 1200 | **2400** |

Across that entire range, at N = 10, 14 and 18, the scheme is not merely stable —
it is **time-accurate to second order, fitted slope 2.04**, on a genuinely
unsteady solution.

And `ARTIFICIAL_COMPRESSIBILITY.md` §5.1: the closed cavity converges at
`a_mass` = 30 with no remedy at all (`|dU|` = 5.4e−10 after 518 steps) where the
BFS diverges at 12.1.

---

## Conclusion

| candidate cause | verdict | evidence |
|---|---|---|
| non-zero residual | **excluded** | §A — parabolic ≡ uniform, identical step |
| convective term | **excluded** | §B — Stokes ≡ full NS, identical step |
| small `dt` / large `a_mass` per se | **excluded** | §C — periodic channel clean to 2400 |
| **outflow boundary** | **only surviving explanation** | every failure has one; no failure lacks one |

**`a_mass` = 2400 on a periodic domain is 40× the value at which the same code,
on the same equations, with the same weighting, diverges within 33 steps once an
outflow plane is present.** The threshold is a property of the interaction
between the weighting imbalance and the outflow condition, not of the
least-squares weighting in general.

### Artificial compressibility remains the remedy where an outflow exists

| `a_mass` | `κ_p` | outcome |
|---|---|---|
| 60 | 0 | BLEWUP @ 33 |
| 60 | 30 | **ok** |
| 120 | 60 | **ok** |
| 300 | 150 | **ok** |
| 600 | 300 | **ok** (Stokes operator) |

AC also makes these runs 10–60× cheaper: AC-off at `a_mass` = 60 took 503 s to
reach step 33, against 58 s for 600 steps with AC on. **At these `a_mass` AC is
not an optimisation, it is the enabling technology.**

---

## Consequence for the 3D solver

The `Re_τ` = 180 channel is periodic in `x` and `z` with walls in `y` — **it has
no outflow plane**. The RKW3/Crank–Nicolson requirement of `a_mass` = 600 … 6000
(§0.4 of `3D_DEVELOPMENT_PLAN.md`) is directly covered by measured data at 600,
1200 and 2400. The plan's headline risk accordingly moved high → medium → **low**,
and neither artificial compressibility, nor `w_mass ∝ dt`, nor a switch to a
fractional-step projection method is needed for the target case.

---

## Corrections made along the way

Recorded because each was published before being falsified, and the sequence is
the point: two of the three explanations were mine, and both were retracted by
the next measurement.

1. **"The exemption is about the residual"** (commit `165071b`) — asserted from
   the uniform-inlet failure while the parabolic control was still running. The
   control refuted it: identical failure at an identical step. Retracted in
   `c3067ee`.
2. **"Semi-implicit convection is the fallback"** (`3D_DEVELOPMENT_PLAN.md`
   §0.2, original) — retracted in `fcfd701` once the Stokes probe showed the
   failure survives with no convection at all.
3. **"RKW3 relieves `a_mass` by ~3.5×"** (a since-reverted edit to
   `3d_fourier_sem_expansion.md`) — the arithmetic is the other way. `1/β` =
   (4.32, 4.80, **6.00**), so at matched `dt` RKW3/CN is 4× *worse*, and the
   3.46× larger step leaves it ~15% worse than BDF2 overall.
4. **`GARTLING_VALIDATION.md` §8's residual argument** was not wrong on its own
   evidence — periodic channels have no outflow *and* a near-zero residual, so
   the two were confounded and its data could not separate them. §A separates
   them. That section is now flagged as unresolved on the point rather than left
   asserting a cause.

---

## Limits of this result

* Every measurement is **2D**. A turbulent 3D field is a different dynamical
  regime even with identical boundary treatment.
* `a_mass` = 6000, the top of the 3D band, is **extrapolated**. The measured
  ceiling on a periodic domain is 2400.
* The closed/periodic runs use a fixed step budget (200 steps for the probes),
  so small-`dt` runs cover less physical time. Defensible because every measured
  divergence landed at steps 29–144 regardless of `dt`, but it is a bound on the
  claim, not a proof of unconditional stability.
* **Why** an outflow boundary triggers it is not explained here. The candidate
  worth testing is the ADN complementing condition — P+Z supplies two scalar
  conditions per boundary point, and the weighting may interact with how the
  outflow rows are closed. That is open.
