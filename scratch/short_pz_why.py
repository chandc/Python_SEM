"""Why does the ER-1.94 short/P+Z run report x_r/S = 5.174 while the earlier
cnos short/P+Z run reported 'none in domain'?  Compare the two on the ONE
diagnostic that decides it: bottom-wall shear du/dy(x), in step heights."""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.lgl import diff_matrix

CASES = [('scratch/bfs_pz_state.npz',    'EARLY  cnos ER 2.00 short/P+Z', 0.5),
         ('scratch/armaly_short_pz.npz', 'NEW    ER 1.94 short/P+Z',      0.94),
         ('scratch/bfs_long_pz.npz',     'ref    cnos ER 2.00 long/P+Z',  0.5),
         ('scratch/armaly_long_pz.npz',  'ref    ER 1.94 long/P+Z',       0.94)]

for f, lab, S in CASES:
    d = np.load(f, allow_pickle=True)
    U, xn, yn, hy, N = d['U'], d['xnod'], d['ynod'], d['hy'], int(d['N'])
    D = diff_matrix(N); n = N+1
    xs, tw = [], []
    for e in range(U.shape[0]):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i]); tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    # element rows/cols
    nex = len(set(np.round(xn[:, 0], 9))); ney = len(set(np.round(yn[:, 0], 9)))
    xmax = xn.max()
    print(f"\n{lab}")
    print(f"   elements {U.shape[0]} = {nex} x-stations x {ney} y-stations, N = {N}")
    print(f"   x in [{xn.min():.2f},{xmax:.2f}]  ->  outlet x/S = {xmax/S:.3f},"
          f"  downstream length = {xmax/S:.2f} S")
    print(f"   wall shear at outlet     du/dy = {tw[-1]:+.5f}")
    print(f"   min wall shear in bubble       = {tw.min():+.5f} at x/S = {xs[np.argmin(tw)]/S:.3f}")
    xr = None
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            xr = xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k]); break
    print(f"   detected x_r/S           = {xr/S if xr else float('nan'):.3f}"
          if xr else "   detected x_r/S           = none in domain")
    # how close does the shear come to zero near the exit?
    m = xs > 0.6*xmax
    print(f"   last 40% of domain: shear range [{tw[m].min():+.5f}, {tw[m].max():+.5f}]")
