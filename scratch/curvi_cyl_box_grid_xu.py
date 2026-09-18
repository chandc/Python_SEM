"""The upstream extension: Xu 10 -> 20, drawn against the current box.

The streamwise counterpart of `curvi_cyl_box_grid.py`, and it exists for the
same reason: `_geom_edges` normalises its widths to span its endpoints, so
moving the inlet by raising Lu and nx_up together would rescale every division
between the inlet and the body -- an Xu study run that way measures the
upstream RESOLUTION as much as the upstream extent.  `nested_upstream_edges`
prepends elements instead, leaving the reference divisions untouched.

Xu is the last untested domain parameter.  The lateral test (H 20 -> 40) moved
St by 0.0004 against a predicted 0.0028, so lateral blockage explains under 20 %
of the offset from the literature.  Our Xu = 10 is half Posdziech & Grundmann's
recommended minimum, and an inlet imposing uniform u = 1 that close to the body
pins the stagnation streamline where it should still be adjusting -- which would
raise both St and C_D, the signature that survives.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)
sys.path.insert(0, os.path.join(_R, 'scratch'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lssem2d
from lssem2d import curvi
from curvi_cyl_box_grid import draw, dof, GREY, ORANGE, BLUE

N, LU_REF, LU_NEW, N_EXTRA = 6, 10.0, 20.0, 2


def main():
    lssem2d.set_backend('numpy')
    xs = curvi.nested_upstream_edges(LU_NEW, N_EXTRA, Lu=LU_REF)
    mn = curvi.build_cylinder_box(N=N)
    mw = curvi.build_cylinder_box(N=N, xs_upstream=xs)
    for m in (mn, mw):
        m.compute_global_indices()
    key = lambda m: {tuple(r) for r in
                     np.round(np.stack([m.X.ravel(), m.Y.ravel()], 1), 9)}
    shared = len(key(mn) & key(mw))/len(key(mn))

    fig = plt.figure(figsize=(13.8, 8.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.62], hspace=0.42)
    ax0, ax1, axl = (fig.add_subplot(gs[i]) for i in range(3))

    inner = lambda lo, hi: lo >= -LU_REF - 1e-9
    for ax, m, Lu, tag in (
            (ax0, mn, LU_REF, f'(a) current    Xu = {LU_REF:.0f}   '
                              f'{mn.nelem} elements   {dof(mn):,} dof'),
            (ax1, mw, LU_NEW, f'(b) proposed   Xu = {LU_NEW:.0f}   '
                              f'{mw.nelem} elements   {dof(mw):,} dof   '
                              f'(+{dof(mw)/dof(mn) - 1:.0%})')):
        draw(ax, m, keep=lambda lo, hi: True, colour=GREY)
        # colour only the elements the extension adds
        for e in range(m.nelem):
            if m.X[e].min() < -LU_REF - 1e-9:
                for sl in ((np.s_[0, :]), (np.s_[-1, :]),
                           (np.s_[:, 0]), (np.s_[:, -1])):
                    ax.plot(m.X[e][sl], m.Y[e][sl], '-', color=ORANGE,
                            lw=0.7, zorder=2)
        ax.add_patch(plt.Circle((0, 0), 0.5, fc='0.80', ec='k', lw=0.8, zorder=3))
        ax.axvline(-Lu, color=BLUE, lw=1.3, ls='--', zorder=4)
        ax.set_xlim(-21.5, 26.5); ax.set_ylim(-11.5, 11.5)
        ax.set_aspect('equal'); ax.set_ylabel('y')
        ax.set_title(tag, fontsize=10.5, loc='left')
        ax.text(-Lu + 0.4, 9.4, f'inlet  u = 1  at x = {-Lu:.0f}', fontsize=8,
                color=BLUE)
        ax.text(26.2, 9.4, 'Dong outflow  x = 25', fontsize=8, color=BLUE,
                ha='right')
    ax1.set_xlabel('x')

    naive = curvi._geom_edges(-LU_NEW, -1.5, 4 + N_EXTRA, 1.0/1.45)
    ref = curvi._geom_edges(-LU_REF, -1.5, 4, 1.0/1.45)
    for k, (lab, e, c, mk) in enumerate((
            (f'current, Xu = {LU_REF:.0f}', ref, GREY, 'o'),
            ('proposed: nested', xs, ORANGE, 's'),
            ('naive: raise Lu, nx_up', naive, '#6b7280', 'x'))):
        axl.plot(e, np.full_like(e, -k), mk + '-', color=c, ms=5, lw=1.2,
                 mfc='none' if k else c)
        axl.text(-21.0, -k, lab, ha='right', va='center', fontsize=8.5, color=c)
    axl.axvspan(-LU_REF, -1.5, color=GREY, alpha=0.10)
    axl.text((-LU_REF - 1.5)/2, 0.5, 'shared with the current mesh',
             ha='center', fontsize=8, color=GREY)
    axl.set_xlim(-34, 1.5); axl.set_ylim(-2.7, 0.95)
    axl.set_yticks([]); axl.set_xlabel('x of the upstream element edges')
    axl.set_title('(c) the nested edges keep every current division; the naive '
                  'widening moves all of them', fontsize=9.5, loc='left')
    axl.spines[['left', 'right', 'top']].set_visible(False)

    fig.suptitle('Cylinder in a box: upstream extension, Xu 10 -> 20   '
                 f'(N = {N}, lateral H_full = 20 and Xd = 25 untouched)   '
                 f'current-mesh nodes retained: {shared:.1%}', fontsize=12)
    out = 'figs/curvi_cyl_box_grid_Xu20.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('wrote', out)
    print('nested upstream edges:', np.round(xs, 4))
    print('naive  upstream edges:', np.round(naive, 4))
    print(f'{mn.nelem} -> {mw.nelem} elements, {dof(mn):,} -> {dof(mw):,} dof, '
          f'nodes retained {shared:.2%}')


if __name__ == '__main__':
    main()
