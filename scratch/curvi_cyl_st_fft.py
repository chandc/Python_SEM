"""Strouhal number by FFT, cross-checked against zero crossings.

    uv run python scratch/curvi_cyl_st_fft.py

WHY THE RAW FFT IS NOT ENOUGH.  The record is ~40 time units and the period is
~5.9, so the bin spacing 1/T = 0.025 is 15 % of St itself.  argmax on the
spectrum would quantise the answer to something useless.  Three standard steps
fix that and are all applied here:

  * DETREND -- remove the mean, or the DC bin leaks into the first few bins.
  * HANN WINDOW -- a rectangular window on a non-integer number of periods
    leaks energy across many bins and biases the peak.  Hann trades a factor 2
    in resolution for ~30 dB less leakage, which is the right trade when the
    peak position, not its width, is the measurement.
  * QUADRATIC PEAK INTERPOLATION -- fit a parabola through the peak bin and its
    two neighbours in LOG magnitude.  For a windowed sinusoid this recovers the
    true frequency to far better than a bin; it is what makes the estimate
    comparable with the crossing method rather than 15 % worse.

THE HARMONIC IS A FREE CHECK.  C_D is forced at TWICE the shedding frequency
(both the upper and lower vortex pull downstream), so its spectrum must peak at
2*St.  That is a property of the physics, not of the fit, and it costs nothing
to verify -- if the C_D peak is not at 2x the C_L peak, something is wrong with
the signal rather than with the estimator.

Two independent estimators are reported side by side on purpose.  They share no
code and fail differently: crossings are local and noise-sensitive, the FFT is
global and leakage-sensitive.  Agreement is evidence; disagreement means the
record is not a clean limit cycle.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np


def st_fft(t, y, pad=32):
    """Peak frequency of y(t) by windowed FFT with quadratic interpolation."""
    y = y - y.mean()
    n = len(y)
    dt = t[1] - t[0]
    w = np.hanning(n)
    Y = np.abs(np.fft.rfft(y*w, n=pad*n))
    f = np.fft.rfftfreq(pad*n, dt)
    k = int(np.argmax(Y[1:])) + 1
    # quadratic interpolation in log magnitude on the padded grid
    if 0 < k < len(Y) - 1:
        a, b, c = np.log(Y[k-1] + 1e-300), np.log(Y[k] + 1e-300), np.log(Y[k+1] + 1e-300)
        d = 0.5*(a - c)/(a - 2*b + c) if (a - 2*b + c) != 0 else 0.0
        return f[k] + d*(f[1] - f[0]), Y, f
    return f[k], Y, f


def st_cross(t, y):
    s = np.signbit(y)
    idx = np.flatnonzero(s[:-1] & ~s[1:])
    xc = np.array([t[i] + (t[i+1]-t[i])*(-y[i])/(y[i+1]-y[i]) for i in idx])
    per = np.diff(xc)
    return (1/per).mean(), (1/per).std(), len(per)


def saturated(t, cl, frac=0.95):
    """First index of the window whose cycle amplitude is within frac of the last."""
    s = np.signbit(cl); idx = np.flatnonzero(s[:-1] & ~s[1:])
    if len(idx) < 3:
        return 0
    xc = np.array([t[i] + (t[i+1]-t[i])*(-cl[i])/(cl[i+1]-cl[i]) for i in idx])
    amp = [0.5*(cl[(t >= a) & (t < b)].max() - cl[(t >= a) & (t < b)].min())
           for a, b in zip(xc[:-1], xc[1:])]
    amp = np.array(amp)
    sat = amp > frac*amp[-1]
    return int(np.searchsorted(t, xc[int(np.argmax(sat))]))


RUNS = (('dt 0.10  no AC', 'scratch/_cyl_re100/final.npz'),
        ('dt 0.10  +AC',   'scratch/_cyl_dt0.1ac/chk_latest.npz'),
        ('dt 0.05  +AC',   'scratch/_cyl_dt0.05ac/final.npz'),
        ('dt 0.025 +AC',   'scratch/_cyl_dt0.025ac/chk_latest.npz'),
        ('dt 0.0125 +AC',  'scratch/_cyl_dt0.0125ac/chk_latest.npz'))


def main():
    print(f'{"run":16s} {"span":>13s} {"cyc":>4s} {"St (FFT)":>10s} '
          f'{"St (cross)":>18s} {"diff":>8s} {"C_D peak / C_L":>15s}')
    for tag, f in RUNS:
        if not os.path.exists(f):
            print(f'{tag:16s} {"(not yet)":>13s}')
            continue
        h = np.asarray(np.load(f, allow_pickle=True)['hist'], float)
        t, cd, cl = h[:, 0], h[:, 1], h[:, 2]
        i0 = saturated(t, cl)
        t, cd, cl = t[i0:], cd[i0:], cl[i0:]
        if len(t) < 64:
            print(f'{tag:16s} {"too short":>13s}')
            continue
        sf, _, _ = st_fft(t, cl)
        sc, sd, n = st_cross(t, cl)
        fd, _, _ = st_fft(t, cd)
        print(f'{tag:16s} {t[0]:6.1f}-{t[-1]:6.1f} {n:4d} {sf:10.4f} '
              f'{sc:10.4f} +-{sd:.4f} {sf-sc:+8.4f} {fd/sf:15.3f}')
    print('\n  the last column must be 2.000: C_D is forced at twice the shedding')
    print('  frequency, which is physics and not a property of the estimator')


if __name__ == '__main__':
    main()
