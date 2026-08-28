# Python_SEM — Spectral Element Navier-Stokes on Laptop-Class Hardware

What began as a matrix-free 2D SEM Poisson benchmark is now a validated 3D
incompressible Navier-Stokes solver: spectral elements in x-y, Fourier in z,
running production DNS on a MacBook (NumPy/Accelerate) and desk-side GPUs
(CuPy on NVIDIA GB10; A100-ready).

## Two solver paths

| | VVP LSSEM | Fractional step (RK3-CN) |
|---|---|---|
| formulation | velocity-vorticity-pressure least squares | projection, per-RKW3-substage |
| strengths | tight coupling, robust BCs | 30-130x cheaper per step |
| status | validated (TGV, Kovasznay, BFS, cavity...) | production path for periodic + wall flows |

Convection uses the skew-symmetric form (Horiuti); z is 3/2-rule dealiased.
A consistent P_N-P_N pressure operator E = G^T M^-1 G (`lssem3d/epmg.py`,
`project.py`) zeroes the enforced divergence identically where required.

## Headline validations

- **Taylor-Green Re=800, 160^3** (30 h on a GB10): peak dissipation within
  0.12% of Gourianov et al. (2022) 256^3 DNS, 0.22% mean-curve error —
  tighter than the two published references agree with each other; energy
  balance within [0.9988, 1.0076] for the whole run.  `TGV_VALIDATION.md`.
- **Minimal channel Re_tau=180** (Jimenez-Moin box, tripped from rest by a
  shaped solenoidal low-k disturbance): statistically stationary over 15
  eddy-turnover units; kappa = 0.408, B = 5.69, U+_c = 18.38, second-order
  statistics within a few percent of Kim-Moser-Moin 1987; burst periods
  t+ ~ 139/450 match the literature.  Data: `results/minchan_re180_K/`
  (in the Dropbox tree).
- **A-priori quality metrics that predicted, not explained**: the energy-
  enstrophy balance -dE/dt / 2nu*Omega and the k_max*eta >= 1.5 criterion
  priced every run in advance.  `DIVERGENCE_AND_CONSISTENCY.md`.

## Performance (measured)

- GPU (CuPy): fused kernels, GEMM inner products, device-resident p-multigrid
  with disk-cached setup — `CUPY_BACKEND.md`.
- CPU (Apple Silicon): threaded-GEMM derivatives + per-mode adaptive freezing
  + mode-space multiprocessing = 3.6x, channel step 4.6 -> 1.27 s
  (`lssem3d/modepar.py`).

## Key documents

`DIVERGENCE_AND_CONSISTENCY.md` (progress report + operator decision map),
`SCHEME_COMPARISON.md` (RK4-CN Kim-Moin vs RK3-CN, equations side by side),
`KIM_MOIN_REVIEW.md`, `TGV_VALIDATION.md`, `CHANNEL_VALIDATION.md`,
`CUPY_BACKEND.md`, `FRACTIONAL_STEP_PLAN.md`, `3D_STATUS.md` (full log).

## The original 2D benchmark

The matrix-free 2D Poisson solver and its NumPy/MLX/PyTorch/Fortran
benchmark suite remain in place: `python sem_2d.py` sweeps p = 3..15 and
reproduces the convergence plot.  See git history for that chapter.
