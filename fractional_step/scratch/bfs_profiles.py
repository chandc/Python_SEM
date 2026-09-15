"""Axial velocity profiles u(y) at streamwise stations, for whichever BFS states
are on disk.

Stations are chosen relative to the physics: x/h with h = 0.5, so x = 0.5, 1, 2
sit inside the recirculation, x = 4 is near reattachment (x_r ~ 4.1), and x = 6, 8
are downstream recovery -- the last three exist only on the long domain (x to
8.5); the short one ends at 2.5.

Reads whatever npz files exist:
    bfs_pz_state.npz    short domain, P+Z          (saved)
    bfs_long_free.npz   long domain, free outflow  (saved by the amended
    bfs_long_pz.npz     long domain, P+Z            bfs_long_vs_short.py)

so it can be re-run as more become available without re-solving anything.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

H = 0.5
STATIONS = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
CASES = [('bfs_pz_state.npz', 'SHORT, P+Z', 'tab:blue', '-'),
         ('bfs_long_free.npz', 'LONG, free outflow', 'tab:red', '--'),
         ('bfs_long_pz.npz', 'LONG, P+Z', 'tab:green', '-.')]


def interp(U, xn, yn):
    n = U.shape[1]
    px, py, pu = [], [], []
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j]); pu.append(U[e, i, j, 0])
    px, py, pu = map(np.array, (px, py, pu))
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    return LinearTriInterpolator(tri, pu), px.max()


have = []
for f, lab, c, ls in CASES:
    p = f'{SC}/{f}'
    if not os.path.exists(p):
        print(f"  (missing: {f} — {lab} not plotted)")
        continue
    d = np.load(p)
    have.append((interp(d['U'], d['xnod'], d['ynod']), lab, c, ls, d))
    print(f"  loaded {f}: {lab}")

if not have:
    sys.exit("no states on disk")

ncol = min(len(STATIONS), 4)
nrow = int(np.ceil(len(STATIONS)/ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(3.5*ncol, 3.7*nrow), sharey=True)
axs = np.atleast_1d(axs).ravel()
yy = np.linspace(0, 1, 400)

for k, xs in enumerate(STATIONS):
    ax = axs[k]
    for (fu, xmax), lab, c, ls, d in have:
        if xs > xmax + 1e-9:
            continue
        uu = np.array(fu(np.full_like(yy, xs), yy).filled(np.nan))
        ax.plot(uu, yy, ls, color=c, lw=1.9, label=lab)
    ax.axvline(0, color='k', lw=.8, ls=':')
    ax.axhline(0.5, color='0.6', lw=.8, ls=':')
    ax.set_title(f'x = {xs:g}   (x/h = {xs/H:g})', fontsize=10)
    ax.set_xlabel('u'); ax.grid(alpha=.3)
    if k % ncol == 0:
        ax.set_ylabel('y')
for k in range(len(STATIONS), len(axs)):
    axs[k].axis('off')
axs[0].legend(fontsize=8, loc='upper left')

fig.suptitle('BFS Re = 389, dt = 1, w_mom = w_mass = 1 — axial velocity profiles.\n'
             'u < 0 (left of the dotted line) is reversed flow;  y = 0.5 is the '
             'step height.  Short domain ends at x = 2.5.', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f'{SC}/../figs/bfs_profiles.png', dpi=125, bbox_inches='tight')
print('\nfigs/bfs_profiles.png')

# numeric comparison where domains overlap
if len(have) > 1:
    print(f"\n{'x':>6}  " + "  ".join(f"{lab:>22}" for _, lab, _, _, _ in have))
    for xs in STATIONS:
        row = []
        for (fu, xmax), lab, c, ls, d in have:
            if xs > xmax + 1e-9:
                row.append(f"{'--':>22}")
            else:
                uu = np.array(fu(np.full_like(yy, xs), yy).filled(np.nan))
                row.append(f"{'min %.4f max %.4f' % (np.nanmin(uu), np.nanmax(uu)):>22}")
        print(f"{xs:>6g}  " + "  ".join(row))
