"""The Posdziech & Grundmann ladder: four nested domains, one near field.

Their protocol isolates the lateral height H by fixing everything else far
away -- L_in >= 30D, L_out >= 50D -- so that the measured quantities depend on
H alone.  This draws what that means for our mesh, and the point to check by
eye is that the four rungs differ ONLY in how far they reach: the near field,
the wake resolution and the streamwise divisions are shared, and each rung's
elements are a strict subset of the next.

COLOUR IS THE NESTING.  Grey is the H = 20 box; each colour is the annulus a
doubling adds.  If any colour appeared inside a smaller box's region, the rungs
would not be nested and the ladder would be measuring lateral resolution as
well as lateral extent -- which is what `chained_lateral_edges` exists to
prevent and what the earlier `nested_lateral_edges` construction got wrong
(78 % node retention between consecutive rungs instead of 100 %).
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
import lssem2d
from lssem2d import curvi

N, XU, XD = 8, 30.0, 50.0
RUNGS = (20, 40, 80, 160)
COL = {20: '0.45', 40: '#2563eb', 80: '#c2410c', 160: '#15803d'}


def build(H):
    kw = dict(N=N, xs_upstream=curvi.nested_upstream_edges(XU, 3),
              xs_downstream=curvi.nested_downstream_edges(XD, 3))
    if H > 20:
        kw['ys_side'] = curvi.chained_lateral_edges(H/2, 2)
    m = curvi.build_cylinder_box(**kw)
    m.compute_global_indices()
    return m


def draw(ax, m, lo, hi, colour, lw):
    """Element edges for elements whose |y| extent lies in [lo, hi)."""
    for e in range(m.nelem):
        ay = np.abs(m.Y[e])
        if not (ay.max() > lo + 1e-9 and ay.max() <= hi + 1e-9):
            continue
        X, Y = m.X[e], m.Y[e]
        for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
            ax.plot(X[sl], Y[sl], '-', color=colour, lw=lw, zorder=2)


def shells(ax, m, lw=0.4):
    draw(ax, m, 0.0, 10.0, COL[20], lw)
    for a, b, H in ((10.0, 20.0, 40), (20.0, 40.0, 80), (40.0, 80.0, 160)):
        draw(ax, m, a, b, COL[H], lw)
    ax.add_patch(plt.Circle((0, 0), 0.5, fc='0.8', ec='k', lw=0.8, zorder=5))


def main():
    lssem2d.set_backend('numpy')
    ms = {H: build(H) for H in RUNGS}
    big = ms[160]
    dof = lambda m: (int(m.gidx.max()) + 1)*4

    fig = plt.figure(figsize=(14.5, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.75],
                          height_ratios=[1.0, 0.85], hspace=0.26, wspace=0.18)
    axL = fig.add_subplot(gs[:, 0])
    axM = fig.add_subplot(gs[0, 1])
    axN = fig.add_subplot(gs[1, 1])

    # (a) the whole ladder, drawn on the largest mesh
    shells(axL, big, lw=0.35)
    for H in RUNGS:
        for s in (+1, -1):
            axL.axhline(s*H/2, color=COL[H], lw=1.4, ls='--', zorder=6)
        axL.text(49, H/2 + 1.5, f'H = {H}', color=COL[H], fontsize=9, ha='right')
    axL.set_xlim(-31, 51); axL.set_ylim(-84, 84); axL.set_aspect('equal')
    axL.set_xlabel('x'); axL.set_ylabel('y')
    axL.set_title('(a) the ladder: every rung is the previous one plus an annulus',
                  fontsize=10.5, loc='left')

    # (b) the H = 20 rung at its own scale -- the box the protocol starts from
    shells(axM, ms[20], lw=0.5)
    axM.set_xlim(-31, 51); axM.set_ylim(-11, 11); axM.set_aspect('equal')
    axM.set_xlabel('x'); axM.set_ylabel('y')
    axM.set_title(f'(b) H = 20 rung: Xu = {XU:.0f}, Xd = {XD:.0f}  '
                  f'({ms[20].nelem} elements, {dof(ms[20]):,} dof)',
                  fontsize=10.5, loc='left')

    # (c) near field, all four overlaid thickest-first: no colour should show
    for H in (160, 80, 40, 20):
        draw(axN, ms[H], 0.0, 10.0, COL[H], 3.0 - 0.7*RUNGS.index(H))
    axN.add_patch(plt.Circle((0, 0), 0.5, fc='0.8', ec='k', lw=1.0, zorder=5))
    axN.set_xlim(-3.4, 6.5); axN.set_ylim(-3.2, 3.2); axN.set_aspect('equal')
    axN.set_xlabel('x'); axN.set_ylabel('y')
    axN.set_title('(c) near field, all four overlaid — grey on top means identical',
                  fontsize=10.5, loc='left')

    key = lambda m: {tuple(r) for r in
                     np.round(np.stack([m.X.ravel(), m.Y.ravel()], 1), 9)}
    chain = [len(key(ms[a]) & key(ms[b]))/len(key(ms[a]))
             for a, b in zip(RUNGS[:-1], RUNGS[1:])]
    fig.suptitle(
        f'Posdziech & Grundmann ladder, N = {N}, Xu = {XU:.0f}, Xd = {XD:.0f}, '
        f'dt = 0.1, +AC\n'
        + '   '.join(f'H={H}: {ms[H].nelem} el, {dof(ms[H]):,} dof' for H in RUNGS)
        + f'\nnode retention rung-to-rung: '
        + ', '.join(f'{f:.0%}' for f in chain), fontsize=11.5)
    out = 'figs/curvi_cyl_pg_ladder.png'
    fig.savefig(out, dpi=135, bbox_inches='tight')
    print('wrote', out)
    for H in RUNGS:
        print(f'  H={H:4d}  {ms[H].nelem:4d} elements  {dof(ms[H]):8,d} dof')
    print('  rung-to-rung node retention:', [f'{f:.1%}' for f in chain])


if __name__ == '__main__':
    main()
