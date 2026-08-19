"""Streamlines of the Dong-outlet prediction on the SHORT-domain BFS,
against the P+Z reference on the same grid.

    uv run --quiet python scratch/dong_bfs_plot.py

Fields from scratch/dong_bfs.py (Re = 389, dt = 1, cold start, converged
bit-exact).  Red shading = reversed flow (u < 0): on this domain the
recirculation CROSSES the outlet plane, which is why free outflow blows up
on step 1 here.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SC))
os.chdir(os.path.dirname(SC))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

CASES = [(f'{SC}/dong_bfs_cold_D02_off.npz',
          'Dong OBC (bc = 6, D0 = 2, switch off): conv 136 steps, |dU| = 0'),
         (f'{SC}/dong_bfs_pzref.npz',
          'P+Z (p = 0 and dω/dx = 0): conv 246 steps, |dU| = 0')]

fig, axs = plt.subplots(2, 1, figsize=(12.5, 6.4), sharex=True)
for ax, (f, title) in zip(axs, CASES):
    d = np.load(f, allow_pickle=True)
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    n = U.shape[1]
    xmin, xmax = xn.min(), xn.max()
    GX = np.linspace(xmin, xmax, 900)
    GY = np.linspace(0.0, 1.0, 260)
    MX, MY = np.meshgrid(GX, GY)
    px, py, qu, qv = [], [], [], []
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                qu.append(U[e, i, j, 0]); qv.append(U[e, i, j, 1])
    tri = Triangulation(np.array(px), np.array(py))
    ui = np.array(LinearTriInterpolator(tri, np.array(qu))(MX, MY).filled(np.nan))
    vi = np.array(LinearTriInterpolator(tri, np.array(qv))(MX, MY).filled(np.nan))
    # blank the solid block upstream of the step (x < 0, y < 0.5)
    solid = (MX < 0.0) & (MY < 0.5)
    ui[solid] = np.nan; vi[solid] = np.nan
    sp = np.ma.masked_invalid(np.hypot(ui, vi))
    ax.contourf(MX, MY, sp, levels=40, cmap='viridis', vmin=0, vmax=1.5)
    rev = np.ma.masked_where(~(np.nan_to_num(ui, nan=1.0) < 0), np.ones_like(MX))
    ax.contourf(MX, MY, rev, levels=[.5, 1.5], colors=['red'], alpha=.30)
    ax.streamplot(GX, GY, np.nan_to_num(ui), np.nan_to_num(vi), density=2.2,
                  color='w', linewidth=.7, arrowsize=.7)
    ax.plot([xmin, 0, 0], [0.5, 0.5, 0.0], 'w-', lw=2)      # step outline
    ax.axvline(xmax, color='orange', lw=2)
    umin_out = min(U[e, -1, :, 0].min() for e in range(U.shape[0])
                   if abs(xn[e, -1] - xmax) < 1e-9)
    ax.set_xlim(xmin, xmax); ax.set_ylim(0, 1); ax.set_ylabel('y')
    ax.set_title(f'{title}   |   min u on outlet = {umin_out:.3f}', fontsize=10)
axs[-1].set_xlabel('x')
fig.suptitle('SHORT-domain BFS, Re = 389, dt = 1, cold start.  '
             'Red = reversed flow (u < 0); orange = outflow boundary.\n'
             'The recirculation crosses the outlet (backflow through the '
             'boundary) -- free outflow blows up on step 1 here.',
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = f'{SC}/dong_bfs_streamlines.png'
fig.savefig(out, dpi=150)
print('wrote', out)
