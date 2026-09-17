"""The grid used for the Ghia Re=1000 comparison, affine and deformed.

    uv run python scratch/curvi_ghia_grid.py

6x6 elements at N = 10 -- 61x61 GLL nodes, the resolution the repo's existing
cavity runs use.  The point of the middle panel is that the DOMAIN is identical:
`curvi.deform` perturbs by a product of sines that vanishes on the boundary, so
the cavity is the same unit square with the same lid, and the only difference
between the two runs is whether the elements are rectangles.

The right panel is the reason the comparison needs an inverse map.  Ghia's
centreline x = 0.5 is an element interface on the affine mesh, so the profile can
be read straight off a node line.  On the deformed mesh that interface is CURVED
and crosses x = 0.5 at one point only -- the GLL nodes (dots) sit to either side
of the dashed line, never on it.

WHY kx = 1 AND NOT 2.  `deform` perturbs by sin(kx*pi*x)*sin(ky*pi*y), and the
first attempt used kx = 2 -- which vanishes at x = 0, 0.5 AND 1.  Ghia's vertical
centreline was therefore a nodal line OF THE DEFORMATION: all 132 of its nodes
stayed exactly where the affine mesh put them, and the u(y) comparison would have
been read off an undeformed line while looking like a curvilinear result.  The
count printed below is the check that catches it.
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

from lssem2d import curvi
from lssem2d.mesh import build_channel

OUT = os.path.join(_R, 'figs', 'curvi_ghia_grid.png')
EX, N = 6, 10


def build(deformed):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    if deformed:
        curvi.deform(m, amp=0.10, kx=1, ky=1)
    else:
        curvi.attach(m, *curvi.affine_coords(m))
    return m


def draw(ax, m, nodes=True, lw=1.1, ms=1.4):
    n = m.nterm
    for e in range(m.nelem):
        for idx in (0, n-1):
            ax.plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=lw)
            ax.plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=lw)
    if nodes:
        ax.plot(m.X.ravel(), m.Y.ravel(), '.', ms=ms, color='C0', alpha=0.55)


def main():
    ma, md = build(False), build(True)
    ng = int(ma.gidx.max()) + 1
    disp = np.hypot(md.X - ma.X, md.Y - ma.Y)

    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95],
                          height_ratios=[1, 1], hspace=0.34, wspace=0.26)

    axA = fig.add_subplot(gs[:, 0])
    draw(axA, ma, ms=2.6)
    axA.set_aspect('equal'); axA.tick_params(labelsize=9)
    axA.set_xlabel('$x$'); axA.set_ylabel('$y$')
    axA.set_title(f'affine control\n{EX}x{EX} elements, $N = {N}$, '
                  f'{EX*N+1}x{EX*N+1} = {ng} nodes', fontsize=11)

    axB = fig.add_subplot(gs[:, 1])
    draw(axB, md, ms=2.6)
    axB.axvline(0.5, color='C3', ls='--', lw=1.8, zorder=5)
    axB.axhline(0.5, color='C2', ls='--', lw=1.8, zorder=5)
    axB.set_aspect('equal'); axB.tick_params(labelsize=9)
    axB.set_xlabel('$x$')
    axB.set_title(f'deformed 10 %  —  same domain, same lid\n'
                  f'max node displacement {disp.max():.4f} '
                  f'({100*disp.max()/(1.0/EX):.0f} % of an element), '
                  f'boundary {_bnd(ma, md):.0e}', fontsize=11)

    axC = fig.add_subplot(gs[0, 2])
    draw(axC, md, ms=9)
    draw(axC, ma, nodes=False, lw=0.6)
    axC.axvline(0.5, color='C3', ls='--', lw=2.2, zorder=6)
    axC.set_xlim(0.42, 0.58); axC.set_ylim(0.42, 0.58)
    axC.set_aspect('equal'); axC.tick_params(labelsize=8)
    axC.set_title(r"zoom: Ghia's centreline $x = 0.5$ (red)" '\n'
                  'thin = affine interface, thick = deformed', fontsize=10)

    axD = fig.add_subplot(gs[1, 2])
    sc = axD.tripcolor(md.X.ravel(), md.Y.ravel(), md.jacq.ravel(),
                       shading='gouraud')
    fig.colorbar(sc, ax=axD, shrink=0.88)
    axD.set_aspect('equal'); axD.tick_params(labelsize=8)
    axD.set_title(f'Jacobian $J$ over the deformed mesh\n'
                  f'{md.jacq.min():.5f} to {md.jacq.max():.5f} '
                  f'(ratio {md.jacq.max()/md.jacq.min():.3f})', fontsize=10)

    fig.suptitle('The mesh used for the Ghia Re = 1000 comparison — '
                 'the deformation moves interior nodes only', fontsize=13)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')

    onx = int((np.abs(md.X - 0.5) < 1e-12).sum())
    print(f'{EX}x{EX} elements, N = {N}: {ma.nelem} elements, '
          f'{ng} global nodes ({EX*N+1} x {EX*N+1})')
    print(f'max interior node displacement : {disp.max():.6f} '
          f'({100*disp.max()/(1.0/EX):.1f} % of an element)')
    print(f'max BOUNDARY node displacement : {_bnd(ma, md):.3e}  '
          f'(the domain is unchanged)')
    print(f'Jacobian  min {md.jacq.min():.5f}  max {md.jacq.max():.5f}  '
          f'ratio {md.jacq.max()/md.jacq.min():.3f}')
    print(f'nodes exactly on x = 0.5: deformed {onx}, affine '
          f'{int((np.abs(ma.X - 0.5) < 1e-12).sum())}   '
          f'-- the guard against a deformation whose nodal line is the '
          f'extraction line')
    print(f'wrote {OUT}')


def _bnd(ma, md):
    n = ma.nterm
    d = 0.0
    for e in range(ma.nelem):
        for idx in (0, n-1):
            for A, B in ((ma.X, md.X), (ma.Y, md.Y)):
                d = max(d, np.abs(A[e, idx, :] - B[e, idx, :]).max()
                        if ma.bc[e, 0 if idx == 0 else 1] else 0.0)
    # simpler and exact: any node on the domain boundary
    onb = ((np.abs(ma.X) < 1e-12) | (np.abs(ma.X - 1) < 1e-12)
           | (np.abs(ma.Y) < 1e-12) | (np.abs(ma.Y - 1) < 1e-12))
    return float(np.hypot(md.X - ma.X, md.Y - ma.Y)[onb].max())


if __name__ == '__main__':
    sys.exit(main())
