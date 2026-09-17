"""What a 10 %, 20 % and 30 % deformation of the cavity mesh actually looks like.

    uv run python scratch/curvi_deform_amp.py

`curvi.deform(m, amp, kx, ky)` displaces every node along the diagonal by
amp * h * sin(kx pi x) sin(ky pi y), with h the smallest element size -- so `amp`
is in units of an ELEMENT, not of the domain, and the perturbation vanishes on
the domain boundary so the cavity stays the unit square with the same lid.

FOUR MEASURES, because they say different things and the picture says none of
them:

  * max node displacement.  The largest-looking number and the least meaningful:
    a uniform translation of the whole interior would score high and distort
    nothing.
  * max J / min J.  What the DISCRETISATION feels -- the element is stretched in
    one place and compressed in another, and the effective resolution follows.
    This is the one that costs accuracy.
  * skew, the departure of the two coordinate directions from orthogonal.  What
    the METRIC CROSS TERMS feel: on an affine axis-aligned mesh it is exactly
    zero and r_y = s_x = 0, so any nonzero value means the cross terms are live
    and the curvilinear path is being exercised rather than merely entered.
  * edge bow, how far an element edge departs from the straight chord joining its
    ends.  Small whenever the deformation is smooth on the scale of an element --
    such elements are sheared and stretched far more than they are curved.

And one hard limit: min J > 0.  The mapping folds when the Jacobian reaches zero,
and a folded element is not a mesh -- `collocation_metrics` raises rather than let
one through.  The fold amplitude is found here by bisection.
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
from lssem2d.lgl import diff_matrix

OUT = os.path.join(_R, 'figs', 'curvi_deform_amp.png')
EX, N = 6, 10
AMPS = [0.0, 0.30, 0.50, 1.00]
TABLE = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]


def build(amp):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    if amp > 0:
        curvi.deform(m, amp=amp, kx=1, ky=1)
    else:
        curvi.attach(m, *curvi.affine_coords(m))
    return m


def measures(m, ma, D):
    """The four distortion measures, all geometric and all mesh-independent."""
    disp = float(np.hypot(m.X - ma.X, m.Y - ma.Y).max())
    xr = np.matmul(D, m.X); xs = np.matmul(m.X, D.T)
    yr = np.matmul(D, m.Y); ys = np.matmul(m.Y, D.T)
    a = np.stack([xr, yr], -1); b = np.stack([xs, ys], -1)
    cos = (a*b).sum(-1)/(np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1))
    skew = float(np.degrees(np.abs(np.arccos(np.clip(cos, -1, 1)) - np.pi/2)).max())
    bow = elen = 0.0
    for e in range(m.nelem):
        for P in (np.stack([m.X[e, 0, :], m.Y[e, 0, :]], 1),
                  np.stack([m.X[e, -1, :], m.Y[e, -1, :]], 1),
                  np.stack([m.X[e, :, 0], m.Y[e, :, 0]], 1),
                  np.stack([m.X[e, :, -1], m.Y[e, :, -1]], 1)):
            c0, c1 = P[0], P[-1]
            t = c1 - c0
            L = float(np.linalg.norm(t))
            t = t/L
            nv = np.array([-t[1], t[0]])
            bow = max(bow, float(np.abs((P - c0) @ nv).max()))
            elen = max(elen, L)
    return dict(disp=disp, jrat=float(m.jacq.max()/m.jacq.min()),
                jmin=float(m.jacq.min()), skew=skew, bow=bow,
                bowpc=100*bow/elen)


def fold_amplitude(lo=0.3, hi=4.0, tol=1e-4):
    """Smallest amp at which the mapping folds, by bisection on min J <= 0."""
    def ok(a):
        try:
            build(a)
            return True
        except ValueError:
            return False
    if ok(hi):
        return None
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return hi


def main():
    ma = build(0.0)
    D = diff_matrix(N)
    h = 1.0/EX
    print(f'cavity {EX}x{EX} elements, N = {N}, element size h = {h:.4f}, '
          f'deform(kx=1, ky=1)\n')
    print(f'{"amp":>5s} {"node shift":>11s} {"%h":>6s} | {"min J":>9s} '
          f'{"J max/min":>10s} {"skew deg":>9s} | {"edge bow":>9s} {"%edge":>7s}')
    for a_ in TABLE:
        q = measures(build(a_), ma, D)
        print(f'{a_*100:4.0f}% {q["disp"]:11.4f} {100*q["disp"]/h:6.1f} | '
              f'{q["jmin"]:9.5f} {q["jrat"]:10.3f} {q["skew"]:9.2f} | '
              f'{q["bow"]:9.4f} {q["bowpc"]:7.2f}')

    fa = fold_amplitude()
    print(f'\nthe mapping folds (min J reaches 0) at amp = '
          + (f'{fa*100:.1f} %' if fa else '> 400 %'))

    fig, ax = plt.subplots(2, len(AMPS), figsize=(17, 8.6),
                           gridspec_kw=dict(height_ratios=[1, 0.85],
                                            hspace=0.30, wspace=0.26))
    for k, a_ in enumerate(AMPS):
        m = build(a_)
        q = measures(m, ma, D)
        n = m.nterm
        for e in range(m.nelem):
            for idx in (0, n-1):
                ax[0, k].plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=1.0)
                ax[0, k].plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=1.0)
        ax[0, k].plot(m.X.ravel(), m.Y.ravel(), '.', ms=1.5, color='C0', alpha=0.5)
        ax[0, k].set_aspect('equal'); ax[0, k].tick_params(labelsize=8)
        ax[0, k].set_title(('affine (0 %)' if a_ == 0 else f'deformed {a_*100:.0f} %')
                           + f'\nnode shift {100*q["disp"]/h:.0f} % of $h$, '
                             f'skew {q["skew"]:.1f}$^\\circ$', fontsize=10)

        sc = ax[1, k].tripcolor(m.X.ravel(), m.Y.ravel(), m.jacq.ravel(),
                                shading='gouraud')
        fig.colorbar(sc, ax=ax[1, k], shrink=0.85)
        ax[1, k].set_aspect('equal'); ax[1, k].tick_params(labelsize=8)
        ax[1, k].set_title(f'Jacobian, max/min = {q["jrat"]:.3f}', fontsize=10)

    fig.suptitle('Deformation amplitude for the 2D cavity — `amp` is in units of '
                 'an ELEMENT, and the boundary never moves\n'
                 f'the mapping folds at amp = '
                 + (f'{fa*100:.0f} %' if fa else '> 400 %'), fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
