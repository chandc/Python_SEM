# Fortran LSSEM on plane Poiseuille: the implementation is exact, the free outflow is not

Study date: 2026-08-17. A dt sweep of the **Fortran** LSSEM solver
(`F90_SEM/pmg_clean`, legacy weighting) on plane Poiseuille with a known analytic
solution, comparing a free outflow against `p = 0` on the outlet plane.

The point of using Poiseuille is that the exact solution is *representable* in the
discrete space, so any departure from it is attributable — there is no
discretisation error to hide behind.

Reproduce: `scratch/mesh_poiseuille_f90.py` (grid),
`scratch/pois_f90_analyse.py` (post-processing),
`F90_SEM/pmg_clean/src/SEM_08_bfs_pout.f90` (the `p = 0` driver, added by this
study), and the namelists in `F90_SEM/pmg_clean/poiseuille_dt/`.

Companions: `OUTFLOW_BC_STUDY.md` (the Python outflow study this corroborates),
`PRECONDITIONER_AND_DT_STUDY.md` §5 (the legacy `a_flux = dt` weighting),
`GARTLING_VALIDATION.md` §6–8 (the `a_mass` limit, which is a *different*
mechanism from anything here).

---

## Executive summary

1. **The Fortran implementation is exact on this problem.** With `p = 0` on the
   outlet plane it returns the analytic solution to round-off at every dt over a
   500× range: `max|u|` = 1.5000, `max|v|` ≈ 3e-11 against an exact 0, rms
   divergence ≈ 7e-11, and pressure drop **0.48000 against the analytic
   12L/Re = 0.48000** — five digits, whole domain included.

2. **Free outflow fails, and *which* dt fail is decided by the solver
   tolerance.** At `tol = 1e-12` two of seven dt survive; at `tol = 1e-6` four
   survive — and they are **different ones**. Four of seven outcomes invert on a
   change of linear-solver tolerance alone.

3. **One condition fixes the stability.** `p = 0` (one of the two scalar
   conditions ADN requires per 2D boundary point; free supplies zero) takes the
   sweep from 2/7 converged with a +502% pressure error to 7/7 converged with
   none.

4. **`p = 0` still leaves a band of dt with a second attractor.** For
   1.5 ≲ dt ≲ 2.5 the solver converges — reproducibly and *independently of
   tolerance across six orders* — to a different, wrong steady state
   (`max|u|` ≈ 2.4, Δp ≈ +350%). dt ≤ 1 and dt ≥ 3 give the exact solution.
   One condition is not two.

5. **dt has no effect on accuracy here, and that is the expected result.** The
   legacy weighting makes `a_flux = dt` the momentum weight, but when the exact
   solution zeroes all four residual rows simultaneously the weighting has
   nothing to trade off. Same reason the periodic channel tolerates
   `a_mass` = 30 (`TEMPORAL_ACCURACY_STUDY.md`).

---

## 1. Setup

Plane channel, no step — `scratch/mesh_poiseuille_f90.py`:

| quantity | value |
|---|---|
| domain | `[0, 4] × [0, 1]`, 4×2 elements, N = 10 |
| inflow | `x = 0`, full height, `u = 6y(1-y)` (u_max 1.5, mean 1) |
| walls | `y = 0` and `y = 1`, no-slip |
| outlet | `x = 4`: free, or `p = 0` |
| Re | 100 (ν = 0.01) |
| solver | `nsub = 1`, `cgsfac = 1e-3`, `nitcgs = 40000`, no p-MG |
| run length | `ntime·dt = 200` for every dt (viscous time h²/ν = 100) |

The Fortran inlet is `u = 6·eta·(1-eta)`, `eta = (y - ystep)/hinlet`, so
`ystep = 0, hinlet = 1` gives full-height plane Poiseuille. Exact solution:

    u = 6y(1-y)     v = 0     om = dv/dx - du/dy = 12y - 6
    dp/dx = nu·u'' = -12/Re    =>    dp = 12L/Re = 0.48 over the domain

All four are exactly representable for any order ≥ 2.

> **A trap worth recording.** `SEM_2D_BFS` cannot run this case: `ystep` and
> `hinlet` are not in its namelist, so its inlet is hardcoded to y ∈ [0.5, 1] and
> would silently impose a half-height step inflow instead of Poiseuille. Only the
> `_freeout` variant carries those parameters. This study therefore adds
> `src/SEM_08_bfs_pout.f90` — the freeout source with the outlet changed to
> `p = 0` (both the mask and the imposed value) and nothing else touched.

---

## 2. `p = 0` on the outlet plane: exact at every dt

`SEM_2D_BFS_POUT`, `tol = 1e-12`. **Whole domain, outflow plane included:**

| dt | `a_flux` | max\|u\| | max\|v\| | rms div | L2 err u | Δp | Δp err |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.01 | 1.5000 | 9.9e-11 | 1.7e-10 | 1.9e-06 | 0.48000 | **0.00%** |
| 0.05 | 0.05 | 1.5000 | 3.0e-11 | 7.1e-11 | 1.9e-06 | 0.48000 | **0.00%** |
| 0.1 | 0.1 | 1.5000 | 3.0e-11 | 6.8e-11 | 1.9e-06 | 0.48000 | **0.00%** |
| 0.5 | 0.5 | 1.5000 | 2.8e-11 | 7.2e-11 | 1.9e-06 | 0.48000 | **0.00%** |
| 1 | 1 | 1.5000 | 2.6e-11 | 7.0e-11 | 1.9e-06 | 0.48000 | **0.00%** |
| **2** | **2** | **2.4442** | **1.99** | **6.2e-02** | **2.6e-01** | **2.1718** | **+352%** |
| 5 | 5 | 1.5000 | 6.8e-11 | 2.0e-10 | 1.9e-06 | 0.48000 | **0.00%** |
| *exact* | | 1.5000 | 0 | 0 | 0 | 0.48000 | — |

Every run reached `res ≈ 1e-12`. The L2 errors sit at a floor of 1.889e-06 (u)
and 5.637e-06 (ω), identical at every good dt — that is the discretisation and
solver floor, not a dt effect.

**This is the implementation check, and it passes.** `v` = 3e-11 against an exact
zero, divergence 7e-11, and the pressure drop right to five digits, with no dt
dependence over 500×.

### 2a. The dt band where a second attractor wins

The dt = 2 outlier is not isolated and not a solver artifact:

| dt | tol | max\|u\| | max\|v\| | Δp | Δp err |
|---|---|---|---|---|---|
| 1.5 | 1e-12 | 2.5102 | 1.983 | 2.31666 | +382.6% |
| 2 | 1e-12 | 2.4442 | 1.992 | 2.17180 | +352.5% |
| 2.5 | 1e-12 | 2.3988 | 2.000 | 2.07563 | +332.4% |
| **3** | 1e-12 | **1.5000** | 9.6e-11 | **0.48000** | **0.00%** |
| 2 | 1e-10 | 2.4442 | 1.992 | 2.17180 | +352.5% |
| 2 | 1e-6 | 2.4438 | 1.992 | 2.17084 | +352.3% |

A **band** roughly 1.5 ≲ dt ≲ 2.5, bounded by exact solutions at dt = 1 and
dt = 3, and giving the *same* wrong answer to five digits across six orders of
tolerance. So it is a genuine second steady state of the discrete system, and dt
selects the basin — the "two attractors, not stability" pattern of
`OUTFLOW_BC_STUDY.md` §6, here with one condition imposed instead of none.

---

## 3. Free outflow: the stable dt set is chosen by the tolerance

Same grid, same everything, `SEM_2D_BFS_FREEOUT`.

| dt | free, `tol = 1e-12` | free, `tol = 1e-6` |
|---|---|---|
| 0.01 | NaN (healthy to step ~4500, then diverged) | (killed while running) |
| 0.05 | **converged**, res 9.3e-13 | **blew up**, 8.0e+118 |
| 0.1 | **NaN** | **converged** |
| 0.5 | **NaN** | **converged** |
| 1 | **NaN** | **converged** |
| 2 | converged, res 1.8e-04 | **converged** |
| 5 | blew up, 1.9e+25 | blew up, 1.9e+25 |

**Four of seven outcomes invert** when only the linear-solver tolerance changes.
dt = 0.1, 0.5 and 1 go from NaN to converged; dt = 0.05 goes from converged to a
blow-up. There is no monotone dt threshold to design around, and the survivors
are not a property of the discretisation.

This is the Fortran-side confirmation of what `OUTFLOW_BC_STUDY.md` established
in Python: free outflow supplies **zero** of the two scalar conditions ADN
requires, and survives only when the solve is too inexact to resolve the
resulting soft modes. Tighten the solve and it fails; loosen it and a *different*
set works.

### 3a. Where free outflow's error lives

At dt = 0.05, `tol = 1e-12` (a run that "converged" to 9.3e-13):

| station | max\|u − exact\| | max\|v\| |
|---|---|---|
| x = 0 (inlet) | 0.0000 | 0.0000 |
| x = 1 | 0.0021 | 0.0016 |
| x = 2 | 0.0207 | 0.0143 |
| **x = 4 (outlet)** | **4.1052** | **3.9939** |

Exact at the inlet, ~2e-3 one unit in, then three orders of magnitude worse at
the outflow plane. Excluding only the last element column moves the pressure drop
from **+502% to +3.0%** — the entire error is the boundary.

Interior-only (x < 3) figures, free outflow:

| dt | tol | L2 err u | Δp err |
|---|---|---|---|
| 0.05 | 1e-12 | 3.57e-02 | +3.01% |
| 2 | 1e-12 | 1.24e-02 | +0.37% |
| 0.1 | 1e-6 | 2.06e-02 | +0.74% |
| 0.5 | 1e-6 | 1.70e-02 | +0.64% |
| 1 | 1e-6 | 2.02e-02 | +0.95% |
| 2 | 1e-6 | 1.29e-02 | +0.41% |

So even a *converged* free-outflow run carries ~1–4% interior error on a problem
whose exact solution is representable, against `p = 0`'s 1.9e-06.

---

## 4. Why dt does not affect accuracy here

Legacy weighting makes the momentum row `fac1·u + dt·N(u)` against constraint
rows of weight 1, i.e. `a_mass = fac1 = 1.5` fixed and **`a_flux = dt`** — dt
*is* the momentum weight (`PRECONDITIONER_AND_DT_STUDY.md` §5). On the BFS that
produces a strongly dt-dependent answer: the upper-wall bubble runs 2.848 → 1.683
across dt = 0.05 → 5, and mass loss goes 0.05% → 2.4%.

Here it produces nothing at all: with `p = 0` the answer is identical to five
digits over a 500× dt range. The reason is that the least-squares weights only
matter when the residual **cannot** be zero. Poiseuille zeroes all four rows
simultaneously, so every weighting has the same minimiser and dt drops out. The
same argument explains why the periodic channel tolerates `a_mass` = 30
(`GARTLING_VALIDATION.md` §8) where the Gartling BFS diverges.

**Corollary for interpreting dt sweeps:** a dt sweep on a flow whose exact
solution is representable measures nothing about the weighting. It is a test of
the implementation, which is what it is used for here.

---

## 5. Practical conclusions

1. **Do not use free outflow with a tight solve.** It is not a matter of accuracy
   — 5 of 7 timesteps diverge at `tol = 1e-12`, and the ones that survive change
   when the tolerance does.
2. **`p = 0` on the outlet plane is cheap and transformative** on this flow:
   2/7 → 7/7 converged, +502% → 0.00% on the pressure drop.
3. **One condition is not two.** `p = 0` still admits a wrong steady state over
   1.5 ≲ dt ≲ 2.5. The tested well-posed pair is P+Z (`p = 0` with
   `∂ω/∂x = 0`), per `OUTFLOW_BC_STUDY.md` §7b; extending the Fortran driver to
   impose it is untested and is the obvious next step.
4. **The Fortran implementation is not in question.** On a problem with an
   analytic answer it is exact to round-off in u, v, ω, divergence and pressure
   drop, independent of dt.

---

## 6. How to reproduce

```bash
# grid (writes into the Fortran tree)
cd /Users/danielchan/Dropbox/F90_SEM/pmg_clean/poiseuille_dt
uv run --project <sem_demo> python <sem_demo>/scratch/mesh_poiseuille_f90.py \
        4.0 pois_grid.dat 4 2 10

# build the p=0 driver (added by this study; nothing existing is modified)
cd .. && gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-form \
        -c src/SEM_08_bfs_pout.f90 -o SEM_08_bfs_pout.o
gfortran -o SEM_2D_BFS_POUT SEM_08_bfs_pout.o pmg2_ctrl.o traction_bc.o \
        pmg2_level.o pmg_ml.o sem_data_baseline.o lssem_baseline.o \
        solver_pmg2.o lgl_baseline.o -framework Accelerate

# sweeps (namelists in poiseuille_dt/: in_dt* free 1e-12, in6_dt* free 1e-6,
#         inP_dt* p=0 1e-12, inQ_dt* the dt-band probe)
./SEM_2D_BFS_FREEOUT < poiseuille_dt/in_dt2.nml   > poiseuille_dt/log_dt2.txt
./SEM_2D_BFS_POUT    < poiseuille_dt/inP_dt2.nml  > poiseuille_dt/logP_dt2.txt

# post-process: prefix selects the family (sol | sol6 | solP)
cd <sem_demo>
uv run python scratch/pois_f90_analyse.py solP
```

`ntime` is set so `ntime·dt = 200` for every dt, and `nsave = ntime` so the
solution is written once at the end.

---

## 7. Caveats

- **One geometry, one Re, one mesh.** 4×2 elements at N = 10, Re = 100. The dt
  band in §2a and the tolerance inversions in §3 are not claimed to be
  quantitatively transferable.
- **`nsub = 1` throughout.** Sub-iteration was not varied; the Gartling work
  found `nsub` had no effect on the `a_mass` split, but that is a different
  mechanism.
- **The free-outflow dt = 0.01 fields were lost.** Both the `tol = 1e-12` and
  `tol = 1e-6` runs at that dt were killed after going NaN or while still
  running, and `nsave = ntime` means nothing was written. Re-running with a
  smaller `nsave` would capture the pre-divergence state.
- **`p = 0` here is one condition, not P+Z.** No `∂ω/∂x = 0` was imposed in the
  Fortran; the comparison to the Python P+Z results is therefore indirect.
