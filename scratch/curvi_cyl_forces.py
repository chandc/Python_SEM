"""Strouhal number and mean force coefficients from the C_L / C_D history.

    uv run python scratch/curvi_cyl_forces.py [chk.npz] [--from T]

MEASURED OVER WHOLE PERIODS, from zero crossings of C_L, and only after
saturation.  Both details matter: a mean taken over a partial cycle inherits the
phase, and a mean taken during growth is not a mean of anything.  The Strouhal
number comes from the zero crossings rather than an FFT because the record is
short -- a dozen periods gives an FFT bin spacing comparable to the effect being
measured, while crossings use every cycle directly.

C_D oscillates at TWICE the shedding frequency (both vortices push it the same
way), so its period is half C_L's; averaging C_D over a C_L period covers two of
its own and is correct either way.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np

SRC = os.path.join(_R, 'scratch', '_cyl_re100', 'chk_latest.npz')
T0 = None
for i, a in enumerate(sys.argv[1:]):
    if a == '--from':
        T0 = float(sys.argv[i + 2])
    elif not a.startswith('--'):
        SRC = a


def crossings(t, y):
    """Upward zero crossings, linearly interpolated."""
    s = np.signbit(y)
    idx = np.flatnonzero(s[:-1] & ~s[1:])
    return np.array([t[i] + (t[i+1] - t[i])*(-y[i])/(y[i+1] - y[i]) for i in idx])


def main():
    z = np.load(SRC, allow_pickle=True)
    h = np.asarray(z['hist'], float)
    t, cd, cl = h[:, 0], h[:, 1], h[:, 2]
    print(f'{SRC}\n  {len(t)} samples, t = {t[0]:.2f} to {t[-1]:.2f}')

    xc = crossings(t, cl)
    if len(xc) < 3:
        print('  not enough cycles yet')
        return
    per = np.diff(xc)
    # saturation: take cycles whose amplitude is within 5 % of the last one
    amps = []
    for a, b in zip(xc[:-1], xc[1:]):
        k = (t >= a) & (t < b)
        amps.append(0.5*(cl[k].max() - cl[k].min()))
    amps = np.array(amps)
    sat = amps > 0.95*amps[-1]
    first = int(np.argmax(sat)) if sat.any() else len(amps) - 1
    print(f'\n  cycle    period      St      C_L amp    mean C_D')
    for k in range(len(per)):
        m = (t >= xc[k]) & (t < xc[k+1])
        flag = ' <- saturated' if k >= first else ''
        print(f'  {k:4d}  {per[k]:8.3f}  {1/per[k]:7.4f}  {amps[k]:9.4f}  '
              f'{cd[m].mean():9.4f}{flag}')
    use = slice(first, None)
    tm = (t >= xc[first])
    print(f'\n  SATURATED, {len(per)-first} cycles from t = {xc[first]:.2f}:')
    print(f'    Strouhal   St      = {np.mean(1/per[use]):.4f} '
          f'+- {np.std(1/per[use]):.4f}')
    print(f'    mean drag  C_D     = {cd[tm].mean():.4f}')
    print(f'    drag range         = {cd[tm].min():.4f} to {cd[tm].max():.4f}')
    print(f'    lift amplitude     = {np.mean(amps[use]):.4f} '
          f'+- {np.std(amps[use]):.4f}')
    print(f'\n  published Re = 100 (unbounded): St 0.164-0.167, '
          f'mean C_D 1.32-1.35, C_L amp 0.32-0.34')
    print(f'  this domain has 5 % blockage (D/2H = 1/20), which raises C_D '
          f'~2-4 %:\n    expected here C_D ~ 1.36-1.40')


if __name__ == '__main__':
    main()
