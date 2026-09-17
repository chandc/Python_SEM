"""Snapshot of the running Re = 100 cylinder: fields and force history.

    uv run python scratch/curvi_cyl_snapshot.py [chk.npz]

Reads whatever checkpoint is on disk, so it can be run at any time during the
campaign without disturbing it.  The vorticity panel is free: omega is a SOLVED
VARIABLE in the velocity-vorticity-pressure formulation, not a post-processed
derivative of the velocity, so what is plotted is the discrete unknown itself.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from lssem2d import curvi

OUT = os.path.join(_R, 'figs', 'curvi_cyl_re100_snapshot.png')
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _R, 'scratch', '_cyl_re100', 'chk_latest.npz')


def main():
    z = np.load(SRC, allow_pickle=True)
    U = z['U0']
    t = float(z['t'])
    hist = np.asarray(z['hist']) if 'hist' in z.files else np.zeros((0, 3))
    m = curvi.build_cylinder_box()
    X, Y = m.X, m.Y

    # mask triangles inside the body: Delaunay spans the convex hull and would
    # otherwise paint across the cylinder as if it were fluid
    tri = mtri.Triangulation(X.ravel(), Y.ravel())
    cx = X.ravel()[tri.triangles].mean(axis=1)
    cy = Y.ravel()[tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(cx, cy) < m.r_cyl*1.02)

    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.30, wspace=0.20)

    for k, (fld, lab, cmap) in enumerate(
            ((np.hypot(U[..., 0], U[..., 1]), r'$|\mathbf{u}|$', 'viridis'),
             (U[..., 3], r'$\omega$  (a solved variable, not a derivative)', 'RdBu_r'))):
        ax = fig.add_subplot(gs[k, 0])
        v = fld.ravel()
        lim = np.percentile(np.abs(v), 99.5)
        kw = dict(vmin=-lim, vmax=lim) if k else dict()
        sc = ax.tripcolor(tri, v, shading='gouraud', cmap=cmap, **kw)
        fig.colorbar(sc, ax=ax, shrink=0.9)
        ax.add_patch(plt.Circle((0, 0), m.r_cyl, color='0.6', zorder=6))
        ax.set_xlim(-3, 14); ax.set_ylim(-4, 4)
        ax.set_aspect('equal'); ax.tick_params(labelsize=8)
        ax.set_title(f'{lab}   at $t = {t:.2f}$', fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    if len(hist):
        ax.plot(hist[:, 0], hist[:, 1], '-', color='C0', lw=1.6)
        ax.axhspan(1.32, 1.35, color='C2', alpha=0.20,
                   label='published mean 1.32-1.35')
        ax.legend(fontsize=8)
    ax.set_xlabel('$t$'); ax.set_ylabel('$C_D$')
    ax.set_title('drag: impulsive-start transient', fontsize=10)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    if len(hist):
        ax.plot(hist[:, 0], hist[:, 2], '-', color='C3', lw=1.6)
        ax.axhline(0.0, color='C7', ls=':', lw=1)
    ax.set_xlabel('$t$'); ax.set_ylabel('$C_L$')
    ax.set_title('lift: the shedding mode growing from the seed', fontsize=10)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Cylinder Re = 100 on the curvilinear box mesh — '
                 f'snapshot at $t = {t:.2f}$ of 150', fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f't = {t:.3f}, {len(hist)} force samples')
    if len(hist):
        print(f'  C_D last {hist[-1,1]:.4f}   C_L last {hist[-1,2]:+.4f}   '
              f'|C_L| max {np.abs(hist[:,2]).max():.4f}')
    print(f'  |u|max {np.abs(U[...,0:2]).max():.4f}   '
          f'|omega|max {np.abs(U[...,3]).max():.2f}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
