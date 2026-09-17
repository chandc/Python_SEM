"""An O-grid spectral-element mesh around a circular cylinder — the G6 geometry.

    uv run python scratch/curvi_cylinder_grid.py [E_r E_th N]

This is the mesh the Cartesian code cannot represent at all: the body is a
circle, every element touching it has an arc for an edge, and the no-slip
condition is applied on the true surface rather than on a staircase.

WHAT TO CHECK ON A MESH LIKE THIS, and none of it is visible in the picture:

  * min J > 0 everywhere, or it is not a mesh.
  * the theta seam actually closed.  A cylinder O-grid is periodic in theta, and
    if the two sides fail to merge the domain is silently a slit annulus with
    free edges -- the flow would look plausible and be wrong.  The global node
    count is the check: (E_r*N+1)*(E_th*N) if closed, one column more if not.
  * near-wall resolution against the boundary layer, delta ~ D/sqrt(Re).  What
    matters for a spectral element is the FIRST GLL NODE SPACING, not the element
    size -- the nodes cluster like 1/N^2 at the edges, so an element far thicker
    than the layer can still resolve it.
  * aspect ratio, which geometric radial spacing holds nearly constant by
    construction because azimuthal size grows like r too.
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
from lssem2d.lgl import lgl_nodes, diff_matrix

OUT = os.path.join(_R, 'figs', 'curvi_cylinder_grid.png')
RE = 40.0
R_CYL, R_FAR = 0.5, 25.0


def quality(m):
    D = diff_matrix(m.N)
    xr = np.matmul(D, m.X); xs = np.matmul(m.X, D.T)
    yr = np.matmul(D, m.Y); ys = np.matmul(m.Y, D.T)
    a = np.stack([xr, yr], -1); b = np.stack([xs, ys], -1)
    cos = (a*b).sum(-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
    skew = np.degrees(np.abs(np.arccos(np.clip(cos, -1, 1)) - np.pi/2))
    ar = np.linalg.norm(a, axis=-1)/np.maximum(np.linalg.norm(b, axis=-1), 1e-300)
    return dict(skew=float(skew.max()), armin=float(ar.min()),
                armax=float(ar.max()), jmin=float(m.jacq.min()),
                jrat=float(m.jacq.max()/m.jacq.min()))


def main(E_r=8, E_th=24, N=8):
    m = curvi.build_cylinder(R_CYL, R_FAR, E_r, E_th, N)
    m.compute_global_indices()
    ng = int(m.gidx.max()) + 1
    closed = ng == (E_r*N + 1)*(E_th*N)
    q = quality(m)

    # near-wall GLL spacing along a radial line in the first element
    xi = lgl_nodes(N)
    r0 = m.redges[0]*(m.redges[1]/m.redges[0])**((xi + 1)/2)
    dwall = r0 - R_CYL
    delta = 1.0/np.sqrt(RE)                 # laminar BL thickness, D = 1

    print(f'O-grid cylinder: D = 1, far field {R_FAR/1.0:.0f} D, '
          f'{E_r} x {E_th} elements at N = {N}')
    print(f'  {m.nelem} elements, {ng} global nodes, '
          f'{ng*4:,} degrees of freedom')
    print(f'  theta seam: {"CLOSED" if closed else "NOT CLOSED -- domain is slit"}'
          f'  ({ng} vs {(E_r*N+1)*(E_th*N)} expected closed)')
    print(f'  area {m.wq.sum():.6f}, exact {np.pi*(R_FAR**2-R_CYL**2):.6f}, '
          f'rel err {abs(m.wq.sum()-np.pi*(R_FAR**2-R_CYL**2))/(np.pi*R_FAR**2):.2e}')
    print(f'  min J {q["jmin"]:.5f} (> 0, no fold),  max/min J {q["jrat"]:.1f}')
    print(f'  max skew {q["skew"]:.2f} deg   aspect ratio {q["armin"]:.2f} '
          f'to {q["armax"]:.2f}')
    print(f'\n  boundary layer at Re = {RE:.0f}: delta ~ D/sqrt(Re) = {delta:.4f}')
    print(f'  first radial element spans {dwall[-1]:.4f} '
          f'({dwall[-1]/delta:.2f} delta)')
    print(f'  first GLL node off the wall at {dwall[1]:.5f} '
          f'({dwall[1]/delta:.3f} delta)')
    nin = int((dwall < delta).sum())
    print(f'  GLL nodes inside the layer, first element: {nin} of {N+1}')

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.24)
    n = m.nterm

    def draw(ax, nodes=False, lw=0.7):
        for e in range(m.nelem):
            for idx in (0, n-1):
                ax.plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=lw)
                ax.plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=lw)
        if nodes:
            ax.plot(m.X.ravel(), m.Y.ravel(), '.', ms=2.2, color='C0', alpha=0.6)

    ax = fig.add_subplot(gs[0, 0])
    draw(ax, lw=0.5)
    ax.add_patch(plt.Circle((0, 0), R_CYL, color='0.75', zorder=5))
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.set_title(f'full domain: far field at {R_FAR/1.0:.0f} $D$\n'
                 f'{E_r}x{E_th} elements, $N = {N}$, {ng} nodes', fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    draw(ax, nodes=True, lw=0.9)
    ax.add_patch(plt.Circle((0, 0), R_CYL, color='0.75', zorder=5))
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.set_title('near field, GLL nodes shown\n'
                 'every element edge on the body is an arc', fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    draw(ax, nodes=True, lw=1.1)
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(R_CYL*np.cos(th), R_CYL*np.sin(th), '-', color='C3', lw=2.5,
            label='cylinder surface')
    ax.plot((R_CYL+delta)*np.cos(th), (R_CYL+delta)*np.sin(th), '--',
            color='C2', lw=1.8, label=r'$\delta \sim D/\sqrt{Re}$')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'wall detail: {nin} of {N+1} GLL nodes inside '
                 r'$\delta$' '\nfirst node at '
                 f'{dwall[1]/delta:.3f}' r'$\delta$', fontsize=10)

    fig.suptitle('Spectral-element O-grid for the 2D cylinder (gate G6) — '
                 r'the $\theta$ seam closes on itself by coordinate hashing',
                 fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'\nwrote {OUT}')
    return 0 if closed and q['jmin'] > 0 else 1


if __name__ == '__main__':
    a = [int(x) for x in sys.argv[1:4]] if len(sys.argv) > 3 else []
    sys.exit(main(*a) if a else main())
