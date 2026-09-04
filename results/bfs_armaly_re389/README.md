# BFS Gate 3: Armaly Re = 389 (projection path, quasi-2D) -- PASSED

2026-08-29.  RK3-CN substage + skew + outflow OBC v1 (do-nothing) +
outflow-aware p-multigrid.  Mesh: parametric L-shape (lssem2d.build_bfs),
inlet 3x3 + outlet 2x(16x3) elements, N = 7, Nz = 4 (Lz = 1), ER = 1.94
(h = 1.0, S = 0.94), Re = U_m*2h/nu = 389, dt = 4e-3, impulsive start.

## Result
x_r/S transient: 5.98 (t=10) -> peak 8.71 (t=20) -> plateau 8.110 (t=50),
still creeping at ~0.005/unit toward ~8.10.

| source | x_r/S |
|---|---|
| this gate (N=7, uniform x) | 8.11 |
| in-house 2D validated (P=10, graded) | 8.145 |
| Armaly 1983 experiment | 8.05 +/- 0.7 |

0.4% from the fine-grid 2D value on a deliberately economical grid; CG = 48
per step throughout (outflow-PMG; one-level FDM needed ~1300).

## Files
state_t50.npz (U, p at t = 50), console.log (x_r trace),
streamlines3d.png (t = 15 development), obc_vortex_streamlines.png (Gate 2).

## Next (BFS_VALIDATION_LADDER.md)
Grid-refinement pass (graded x, N 9-10); Barkley Re_c/lambda_z ladder;
Dong Theta stabiliser when exit backflow first appears.

## Refined-run re-validation (N=9, xpow=1.6, 2026-08-29)

`final_N9refined_t60.npz`, `crossings_N9refined.csv`.  Same case at
polynomial order 9 on the graded mesh, WITH the multiblock corner-mask
fix (the original gate run predated it).  Reattachment approach fitted
to x_r(t) = A - B exp(-t/tau) over t=35-60 (rms 4e-4):

    A = 8.108,  tau = 38.0

vs coarse gate 8.11, 2D reference 8.145, Armaly 8.05 +/- 0.7.
Grid-converged at the 0.1% level and re-validates Gate 3 after the
mask fix.  Startup eddy exited cleanly through the OBC at t~30.
