"""C_D and C_L over the whole run: transient, growth, and the limit cycle.

    uv run python scratch/curvi_cyl_history.py [chk.npz]

Both coefficients are recorded EVERY step (dt = 0.1), not only at the log
interval, so the record resolves the ~5.9 time-unit shedding period with ~59
samples -- enough that the Strouhal number comes from zero crossings rather than
from an FFT of a coarse series.

The phase portrait is the sharpest statement that the solution has reached a
LIMIT CYCLE rather than a slowly drifting state: a genuine limit cycle closes on
itself, and successive orbits lie on top of one another.  A drifting solution
spirals.  C_D traces two loops per C_L cycle because drag is forced at twice the
shedding frequency -- both the upper and the lower vortex pull it the same way.
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

OUT = os.path.join(_R, 'figs', 'curvi_cyl_re100_history.png')
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _R, 'scratch', '_cyl_re100', 'chk_latest.npz')


def main():
    z = np.load(SRC, allow_pickle=True)
    h = np.asarray(z['hist'], float)
    t, cd, cl = h[:, 0], h[:, 1], h[:, 2]
    print(f'{len(t)} samples at dt = {t[1]-t[0]:.2f}, t = {t[0]:.2f} to {t[-1]:.2f}')
    print(f'  samples per shedding period (~5.92): {5.92/(t[1]-t[0]):.0f}')

    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 1], hspace=0.32,
                          wspace=0.26)
    sat = t > 60.0
    zoomw = (t > t[-1] - 30) if t[-1] > 90 else sat

    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(t, cd, '-', color='C0', lw=1.0)
    ax.axhspan(1.32, 1.35, color='C2', alpha=0.18,
               label='published, unbounded 1.32-1.35')
    ax.axhspan(1.36, 1.40, color='C1', alpha=0.15,
               label='with 5 % blockage 1.36-1.40')
    if sat.any():
        ax.axhline(cd[sat].mean(), color='C3', ls='--', lw=1.2,
                   label=f'mean over $t>60$: {cd[sat].mean():.4f}')
    ax.set_ylim(1.15, 1.45); ax.set_xlabel('$t$'); ax.set_ylabel('$C_D$')
    ax.set_title('drag: impulsive start (off scale), relaxation to the unstable '
                 'steady state, then growth with the wake', fontsize=10)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(t, cl, '-', color='C3', lw=1.0)
    ax.axhline(0, color='C7', ls=':', lw=1)
    for s in (+1, -1):
        ax.axhline(s*0.33, color='C2', ls='--', lw=1.0,
                   label='published amplitude 0.32-0.34' if s > 0 else None)
    ax.set_xlabel('$t$'); ax.set_ylabel('$C_L$')
    ax.set_title('lift: exponential growth of the shedding mode, then saturation',
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    if zoomw.any():
        ax.plot(t[zoomw], cl[zoomw], '-', color='C3', lw=1.3, label='$C_L$')
        ax.plot(t[zoomw], (cd[zoomw]-cd[zoomw].mean())*6, '-', color='C0',
                lw=1.3, label=r'$6(C_D-\overline{C_D})$')
    ax.set_xlabel('$t$'); ax.legend(fontsize=8)
    ax.set_title('the last 30 time units\n$C_D$ at TWICE the $C_L$ frequency',
                 fontsize=9)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    if zoomw.any():
        ax.plot(cd[zoomw], cl[zoomw], '-', color='C4', lw=0.9)
    ax.set_xlabel('$C_D$'); ax.set_ylabel('$C_L$')
    ax.set_title('phase portrait: orbits closing\non themselves = limit cycle',
                 fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Cylinder Re = 100 — force coefficients recorded every step, '
                 f't = {t[0]:.1f} to {t[-1]:.1f}', fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    if sat.any():
        print(f'  mean C_D over t > 60 : {cd[sat].mean():.4f}')
        print(f'  C_D range            : {cd[sat].min():.4f} to {cd[sat].max():.4f}')
        print(f'  C_L amplitude        : {0.5*(cl[sat].max()-cl[sat].min()):.4f}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
