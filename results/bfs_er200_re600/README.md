# BFS ER=2.0 Re=600 (Erturk comparison; L_out=32, N=7 graded, tend=100)

RK3-CN + skew convection, outflow OBC v1, mesh-CFL dt=2.26e-3, ramp 3.
Startup eddy exited OBC at t~48; single wall-shear crossing thereafter.
x_r(t) fitted to A - B exp(-t/tau) (windows t>50/60/70 agree):

    x_r/h = 10.2 +/- 0.1   (tau ~ 70)

Reference: Erturk (2008) 2D steady, x_r/h = 10.05 at Re=600 -> +1.5%.
Consistent with the residual difference being finite-tend extrapolation
plus inlet-channel/formulation differences.
`final_t100.npz` (t=100 field), `crossings.csv`, figures at t=41.3.
