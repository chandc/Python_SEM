"""Vorticity across the P&G ladder, all four rungs on one colour scale.

Plotted in a COMMON WINDOW (|y| <= 20) rather than each rung's own extent, so
the wakes are directly comparable and the eye is not fooled by four different
axis scalings.  Each rung's lateral boundary is drawn where it falls inside the
window; H = 80 and H = 160 have theirs off-picture, which is the point.

If the nesting and the protocol are working, panels (a)-(d) should be
indistinguishable in the wake and differ only in where the boundary sits.  A
visible difference in the near wake would mean the lateral boundary is still
reaching in, i.e. we are not yet in the asymptotic regime the extrapolation
assumes.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import lssem2d
from lssem2d import curvi
from curvi_cyl_vort import refine, panel          # same spectral interpolation

N, XU, XD = 8, 30.0, 50.0
RUNGS = (20, 40, 80, 160)
VMAX = 1.0


def build(H):
    kw = dict(N=N, xs_upstream=curvi.nested_upstream_edges(XU, 3),
              xs_downstream=curvi.nested_downstream_edges(XD, 3))
    if H > 20:
        kw['ys_side'] = curvi.chained_lateral_edges(H/2, 2)
    m = curvi.build_cylinder_box(**kw)
    m.compute_global_indices()
    return m


def main():
    lssem2d.set_backend('numpy')
    fig, axes = plt.subplots(len(RUNGS), 1, figsize=(13.0, 12.0))
    got = 0
    for ax, H in zip(axes, RUNGS):
        f = f'scratch/_cyl_pg_H{H}/chk_latest.npz'
        if not os.path.exists(f):
            ax.text(0.5, 0.5, f'H = {H}: no checkpoint yet', ha='center',
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        m = build(H)
        z = np.load(f, allow_pickle=True)
        t = float(z['t'])
        Xf, Yf, Of = refine(m, z['U0'][..., 3])
        norm = panel(ax, Xf, Yf, Of, VMAX)
        ax.add_patch(plt.Circle((0, 0), 0.5, fc='0.8', ec='k', lw=0.9, zorder=5))
        if H/2 <= 20:
            for s in (+1, -1):
                ax.axhline(s*H/2, color='0.15', lw=1.2, ls='--', zorder=4)
            ax.text(49, H/2 + 0.8, f'boundary y = {H/2:.0f}', fontsize=8,
                    color='0.15', ha='right')
        else:
            ax.text(49, 17, f'boundary y = {H/2:.0f} (off picture)', fontsize=8,
                    color='0.15', ha='right')
        ax.set_xlim(-30, 50); ax.set_ylim(-20, 20); ax.set_aspect('equal')
        ax.set_ylabel('y')
        ax.set_title(f'H = {H}   t = {t:.0f}   max|omega| = '
                     f'{np.abs(z["U0"][..., 3]).max():.1f}',
                     fontsize=10, loc='left')
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), ax=ax,
                     fraction=0.016, pad=0.01, label='omega')
        got += 1
        print(f'  H={H:4d}  t={t:6.1f}  max|omega|={np.abs(z["U0"][...,3]).max():.2f}',
              flush=True)
    axes[-1].set_xlabel('x')
    fig.suptitle('Posdziech & Grundmann ladder, intermediate vorticity  '
                 f'(N = {N}, Xu = {XU:.0f}, Xd = {XD:.0f}, dt = 0.1, +AC)\n'
                 'common window |y| <= 20 and one colour scale: the wakes should '
                 'be indistinguishable, only the boundary moves', fontsize=11.5)
    out = 'figs/curvi_cyl_pg_vorticity.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print('wrote', out, f'({got}/{len(RUNGS)} rungs)')


if __name__ == '__main__':
    main()
