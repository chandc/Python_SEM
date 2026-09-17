"""What a 10 %, 20 % and 30 % deformation of the cavity mesh actually looks like.

    uv run python scratch/curvi_deform_amp.py

`curvi.deform(m, amp, kx, ky)` displaces every node along the diagonal by
amp * h * sin(kx pi x) sin(ky pi y), with h the smallest element size -- so `amp`
is in units of an ELEMENT, not of the domain, and the perturbation vanishes on
the domain boundary so the cavity stays the unit square with the same lid.

Two numbers decide whether an amplitude is usable, and neither is the picture:

  * min J > 0.  The mapping folds when the Jacobian reaches zero, and a folded
    element is not a mesh -- `collocation_metrics` raises rather than let one
    through.  The fold amplitude is found here by bisection.
  * max J / min J.  This is what the discretisation actually feels: the element
    is stretched in one place and compressed in another, and the effective
    resolution follows.  A large ratio costs accuracy long before anything folds.
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

OUT = os.path.join(_R, 'figs', 'curvi_deform_amp.png')
EX, N = 6, 10
AMPS = [0.0, 0.10, 0.20, 0.30]


def build(amp):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    if amp > 0:
        curvi.deform(m, amp=amp, kx=1, ky=1)
    else:
        curvi.attach(m, *curvi.affine_coords(m))
    return m


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
    fig, ax = plt.subplots(2, len(AMPS), figsize=(17, 8.4),
                           gridspec_kw=dict(height_ratios=[1, 0.85],
                                            hspace=0.30, wspace=0.24))
    print(f'cavity {EX}x{EX} elements, N = {N}, deform(kx=1, ky=1)\n')
    print(f'{"amp":>6s} {"max disp":>10s} {"% of elem":>10s} {"min J":>10s} '
          f'{"max J":>10s} {"max/min":>9s}')
    for k, a in enumerate(AMPS):
        m = build(a)
        disp = np.hypot(m.X - ma.X, m.Y - ma.Y).max()
        rat = m.jacq.max()/m.jacq.min()
        print(f'{a*100:5.0f}% {disp:10.4f} {100*disp/(1.0/EX):10.1f} '
              f'{m.jacq.min():10.5f} {m.jacq.max():10.5f} {rat:9.3f}')

        n = m.nterm
        for e in range(m.nelem):
            for idx in (0, n-1):
                ax[0, k].plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=1.0)
                ax[0, k].plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=1.0)
        ax[0, k].plot(m.X.ravel(), m.Y.ravel(), '.', ms=1.6, color='C0', alpha=0.5)
        ax[0, k].set_aspect('equal'); ax[0, k].tick_params(labelsize=8)
        ax[0, k].set_title(('affine (0 %)' if a == 0 else f'deformed {a*100:.0f} %')
                           + f'\nmax node shift {100*disp/(1.0/EX):.0f} % '
                             f'of an element', fontsize=10)

        sc = ax[1, k].tripcolor(m.X.ravel(), m.Y.ravel(), m.jacq.ravel(),
                                shading='gouraud')
        fig.colorbar(sc, ax=ax[1, k], shrink=0.85)
        ax[1, k].set_aspect('equal'); ax[1, k].tick_params(labelsize=8)
        ax[1, k].set_title(f'Jacobian, max/min = {rat:.3f}', fontsize=10)

    fa = fold_amplitude()
    print(f'\nthe mapping folds (min J reaches 0) at amp = '
          + (f'{fa*100:.1f} %' if fa else '> 400 %'))
    print(f'so 30 % is a factor {fa/0.30:.1f} inside the fold'
          if fa else '30 % is far inside the fold')
    fig.suptitle('Deformation amplitude for the 2D cavity — `amp` is in units of '
                 'an ELEMENT, and the boundary never moves\n'
                 f'the mapping folds at amp = '
                 + (f'{fa*100:.0f} %' if fa else '> 400 %'), fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
