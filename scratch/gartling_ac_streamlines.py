"""Streamlines from the Gartling BFS AC sweep, one panel per dt.

    uv run --quiet python scratch/gartling_ac_streamlines.py

All runs: 11x4 N=6, parabolic inlet, P+Z outlet, from rest, w_mom = w_mass = 1
(time-accurate), kappa_p = 15, t = 140 unless the run failed earlier.  The 11x4
grid is Chan's fig-5 case, so the flow is PERIODIC -- each panel is one instant
of a limit cycle, not a converged state, and panels at different dt are at
different phases.  Time-averaged features are in the accuracy table.
"""
import os, sys, glob, re
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

FILES = sorted(glob.glob(f'{SC}/gartling_unsteady_nx11_N6_dt*_T*_nsub5_*ac15.npz'),
               key=lambda s: -float(re.search(r'dt([\d.]+)_T', s).group(1)))
GX = np.linspace(0, 17, 1400); GY = np.linspace(-0.5, 0.5, 190)
MX, MY = np.meshgrid(GX, GY)

fig, axs = plt.subplots(len(FILES), 1, figsize=(15.0, 2.45*len(FILES)))
for ax, f in zip(np.atleast_1d(axs), FILES):
    d = np.load(f, allow_pickle=True)
    U, xn, yn, h = d['U'], d['xnod'], d['ynod'], d['hist']
    dt = float(d['dt']); n = U.shape[1]
    px, py, q = [], [], [[], []]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                q[0].append(U[e, i, j, 0]); q[1].append(U[e, i, j, 1])
    tri = Triangulation(np.array(px), np.array(py))
    ui = np.array(LinearTriInterpolator(tri, np.array(q[0]))(MX, MY).filled(np.nan))
    vi = np.array(LinearTriInterpolator(tri, np.array(q[1]))(MX, MY).filled(np.nan))
    ax.contourf(MX, MY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.5)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(MX))
    ax.contourf(MX, MY, rev, levels=[.5, 1.5], colors=['red'], alpha=.26)
    ax.streamplot(GX, GY, np.nan_to_num(ui), np.nan_to_num(vi), density=2.7,
                  color='w', linewidth=.6, arrowsize=.6)
    for xv in (6.1, 4.8, 10.5):
        ax.axvline(xv, color='gold', lw=1.3, ls=':', zorder=6)
    ax.set_xlim(0, 17); ax.set_ylim(-0.5, 0.5); ax.set_ylabel('y')
    t = h[:, 0]; m = t > t[-1]-40
    ok = h[-1, 0] > 100
    ax.set_title(f"dt = {dt:g}   a_mass = {1.5/dt:.4g}   "
                 f"{'t = %.1f' % h[-1,0] + ('' if ok else '  BLEW UP')}   |   "
                 f"max|u| = {h[-1,1]:.4f}   |   "
                 f"<lo_re> = {np.nanmean(h[m,3]):.3f}  <up_re> = {np.nanmean(h[m,5]):.3f}"
                 f"   |   p2p|v| = {h[m,2].max()-h[m,2].min():.2e}", fontsize=9.5)
np.atleast_1d(axs)[-1].set_xlabel('x')
fig.suptitle('Gartling BFS Re = 800, 11x4 N=6, parabolic inlet, P+Z, from rest, '
             'time-accurate (w_mom = w_mass = 1), kappa_p = 15.\n'
             'Gold dotted = Gartling 6.1 / 4.8 / 10.5;  red = reversed flow.  '
             'This grid is Chan fig-5: the flow is PERIODIC, so each panel is one '
             'instant of a limit cycle at an arbitrary phase.', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('figs/gartling_ac_dt_streamlines.png', dpi=118, bbox_inches='tight')
print('figs/gartling_ac_dt_streamlines.png')
for f in FILES:
    d = np.load(f, allow_pickle=True)
    print(f"  dt = {float(d['dt']):<7g} t_end = {d['hist'][-1,0]:>6.1f}  {str(d['status'])}")
