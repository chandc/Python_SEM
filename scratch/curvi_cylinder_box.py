"""Body-fitted O-ring inside a rectangular box — the cylinder mesh that makes
free-stream and outflow conditions trivial.

    uv run python scratch/curvi_cylinder_box.py

WHY NOT A PLAIN O-GRID.  `build_cylinder` puts the far field on a circle, which
is fine for steady Re = 40 with Dirichlet all round, but an outflow condition on
a curved arc needs the boundary NORMAL -- machinery `obc.py` does not yet carry
on a curved mesh (plan step 8).  Here the ring's OUTER edge is a square and
everything beyond it is rectangular, so every boundary that carries a condition
is axis aligned and the existing codes apply unchanged.  Step 8 leaves the
critical path for G6.

THE ENABLING FACT IS THAT TOPOLOGY IS FREE.  `compute_global_indices` hashes
PHYSICAL COORDINATES, so a conforming collection of elements merges correctly no
matter what order they are stored in or what each block's own indexing means.
The O-ring is indexed radial x azimuthal, its neighbours are indexed x x y, and
nothing needs to reconcile the two.  Conformity is the only requirement.

CONFORMITY IS CHECKED HERE, NOT ASSUMED.  Two blocks that abut without merging
give a mesh that looks perfect, has the exactly correct area, and is
hydrodynamically a SLIT -- flow cannot cross the interface, and the solution
would be smooth, plausible and wrong.  Area cannot detect it (it is a sum over
elements either way).  The test that can: every node on an element edge that is
not a domain boundary must be shared by at least two elements.
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
from lssem2d.lgl import diff_matrix

OUT = os.path.join(_R, 'figs', 'curvi_cylinder_box.png')
RE = 40.0


def conformity(m):
    """Every interior-edge node must be shared.  Returns (n_bad, n_checked)."""
    ng = int(m.gidx.max()) + 1
    mult = np.bincount(m.gidx.ravel(), minlength=ng)
    edges = ((np.s_[0, :], 0), (np.s_[-1, :], 1), (np.s_[:, 0], 2), (np.s_[:, -1], 3))
    bad = checked = 0
    for e in range(m.nelem):
        for sl, d in edges:
            if m.bc[e, d] != 0:
                continue                       # a domain boundary: single copy
            g = m.gidx[e][sl]
            checked += len(g)
            bad += int((mult[g] < 2).sum())
    return bad, checked


def quality(m):
    D = diff_matrix(m.N)
    xr = np.matmul(D, m.X); xs = np.matmul(m.X, D.T)
    yr = np.matmul(D, m.Y); ys = np.matmul(m.Y, D.T)
    a = np.stack([xr, yr], -1); b = np.stack([xs, ys], -1)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cos = (a*b).sum(-1)/np.maximum(na*nb, 1e-300)
    skew = np.degrees(np.abs(np.arccos(np.clip(cos, -1, 1)) - np.pi/2))
    return float(skew.max()), float(np.percentile(skew, 99))


def main():
    m = curvi.build_cylinder_box()
    m.compute_global_indices()
    ng = int(m.gidx.max()) + 1
    x0, x1, y0, y1 = m.box
    Ax = (x1 - x0)*(y1 - y0) - np.pi*m.r_cyl**2
    bad, checked = conformity(m)
    sk, sk99 = quality(m)
    nb = {c: int((m.bc == c).sum()) for c in (1, 3, 4, 5)}

    print(f'O-ring (r = {m.r_cyl}) inside a box '
          f'[{x0:g}, {x1:g}] x [{y0:g}, {y1:g}], square interface at '
          f'a = {m.a_sq:g}')
    print(f'  {m.nelem} elements, {ng} global nodes, {ng*4:,} dof')
    print(f'  area {m.wq.sum():.6f}  exact {Ax:.6f}  rel {abs(m.wq.sum()-Ax)/Ax:.2e}')
    print(f'  min J {m.jacq.min():.3e} (> 0)   max skew {sk:.1f} deg '
          f'(99th pct {sk99:.1f})')
    print(f'\n  CONFORMITY: {checked - bad} of {checked} interior-edge nodes '
          f'shared -> {"OK" if bad == 0 else f"{bad} UNMERGED -- the mesh is slit"}')
    print(f'\n  boundary edges by condition:')
    for c, lab in ((1, 'no-slip (cylinder, curved)'), (3, 'free stream inlet'),
                   (4, 'outflow p = 0'), (5, 'symmetry top/bottom')):
        print(f'    bc {c}  {nb[c]:3d}  {lab}')
    print(f'  every one of these except bc 1 is AXIS ALIGNED, so no boundary '
          f'normal is needed')

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(16.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.35, 1, 1],
                          hspace=0.30, wspace=0.22)
    n = m.nterm

    def draw(ax, nodes=False, lw=0.6, ms=2.0):
        for e in range(m.nelem):
            for idx in (0, n-1):
                ax.plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=lw)
                ax.plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=lw)
        if nodes:
            ax.plot(m.X.ravel(), m.Y.ravel(), '.', ms=ms, color='C0', alpha=0.55)

    ax = fig.add_subplot(gs[:, 0])
    draw(ax, lw=0.5)
    ax.add_patch(plt.Circle((0, 0), m.r_cyl, color='0.7', zorder=6))
    col = {3: ('C2', 'inlet, free stream'), 4: ('C3', 'outflow $p=0$'),
           5: ('C0', 'symmetry')}
    for e in range(m.nelem):
        for sl, d in ((np.s_[0, :], 0), (np.s_[-1, :], 1),
                      (np.s_[:, 0], 2), (np.s_[:, -1], 3)):
            c = m.bc[e, d]
            if c in col:
                ax.plot(m.X[e][sl], m.Y[e][sl], color=col[c][0], lw=3.2, zorder=5)
    for c, (cc, lab) in col.items():
        ax.plot([], [], color=cc, lw=3.2, label=lab)
    ax.plot([], [], color='0.7', lw=4, label='no-slip (curved)')
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'{m.nelem} elements, {ng} nodes — every condition-carrying\n'
                 'boundary is axis aligned', fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    draw(ax, nodes=True, lw=0.9, ms=1.6)
    ax.add_patch(plt.Circle((0, 0), m.r_cyl, color='0.7', zorder=6))
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.set_title('the O-ring meets the box on a square\n'
                 'radial×azimuthal indexing meets x×y', fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    draw(ax, nodes=True, lw=1.0, ms=3.5)
    ax.add_patch(plt.Circle((0, 0), m.r_cyl, color='0.7', zorder=6))
    d = 1.0/np.sqrt(RE)
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot((m.r_cyl+d)*np.cos(th), (m.r_cyl+d)*np.sin(th), '--', color='C2',
            lw=1.6, label=r'$\delta\sim D/\sqrt{Re}$')
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.95, 0.95)
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.set_title('body-fitted ring, packed to the wall', fontsize=10)

    ax = fig.add_subplot(gs[:, 2])
    draw(ax, nodes=True, lw=0.8, ms=1.3)
    ax.add_patch(plt.Circle((0, 0), m.r_cyl, color='0.7', zorder=6))
    ax.set_xlim(-2, 12); ax.set_ylim(-5, 5)
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.set_title('the wake region, graded downstream', fontsize=10)

    fig.suptitle('Cylinder mesh with a FREE topology: a curved body-fitted ring '
                 'inside rectangular blocks\n'
                 'nine blocks, three different indexings, merged by coordinate '
                 'hashing alone', fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'\nwrote {OUT}')
    return 0 if bad == 0 and m.jacq.min() > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
