"""Has the channel reached a statistically stationary state, and from when?

    uv run python scratch/stationarity.py <stats or checkpoint npz> [--out fig.png]
    uv run python scratch/stationarity.py <dir>          # newest file in it

Recording $u_\\tau(t)$ is not the same as showing stationarity, and Section 10 of
the paper needs the second.  This reduces the record to the three things that
decide it.

WHERE THE DATA IS.  `utau_series` (in `stats*.npz` and inside every checkpoint as
`stats_series`) holds (t, u_tau) every 10 steps and RIDES WITH THE CHECKPOINT, so
it is cumulative across restarts and across nights -- run01's covers t = 0.0008
to 4.96 in 622 samples.  `diag.npz` carries twelve quantities at the same cadence
but is re-initialised on restart, so it holds the CURRENT SESSION only.  Use the
series for anything spanning a restart; use diag.npz for within-session detail.

THE THREE TESTS.

 1. TREND.  Least squares on the analysis window, with the slope's standard error
    computed from the EFFECTIVE sample count rather than the nominal one:
    successive samples 0.008 apart are heavily correlated, and using N would
    declare a trend significant that is only autocorrelation.  N_eff = T/(2*T_int)
    with T_int the integral time scale from the autocorrelation's first zero.
 2. TWO HALVES.  Split the window and compare the means against the same
    effective standard error.  A drift the trend test smooths over shows here.
 3. RUNNING MEAN.  Plotted, so the eye can see it settle -- the check a referee
    does first.

The forcing prescribes u_tau = 1, so the mean is a known target rather than a
fitted one, and a deviation is an error bar on the whole simulation.
"""
import argparse
import glob
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)


def load_series(path):
    if os.path.isdir(path):
        c = sorted(glob.glob(os.path.join(path, 'stats*.npz'))
                   + glob.glob(os.path.join(path, 'checkpoint_*.npz')))
        if not c:
            raise SystemExit(f'no stats or checkpoint npz in {path}')
        path = c[-1]
    with np.load(path, allow_pickle=True) as z:
        k = 'utau_series' if 'utau_series' in z.files else 'stats_series'
        if k not in z.files:
            raise SystemExit(f'{path} carries no u_tau series')
        s = np.asarray(z[k], dtype=float)
    return s[s[:, 0] > 0], path


def integral_scale(x, dt):
    """Integral time scale from the autocorrelation up to its first zero."""
    x = x - x.mean()
    n = len(x)
    if n < 8 or x.std() == 0:
        return dt
    ac = np.correlate(x, x, 'full')[n-1:]
    ac /= ac[0]
    z = np.flatnonzero(ac <= 0)
    k = z[0] if z.size else len(ac)
    return max(dt, float(np.trapezoid(ac[:k], dx=dt)) if hasattr(np, 'trapezoid')
               else float(np.trapz(ac[:k], dx=dt)))


def assess(t, u, label=''):
    dt = float(np.median(np.diff(t)))
    T = t[-1] - t[0]
    tint = integral_scale(u, dt)
    neff = max(2.0, T/(2*tint))
    mu, sd = u.mean(), u.std(ddof=1)
    se = sd/np.sqrt(neff)
    # trend
    A = np.vstack([np.ones_like(t), t - t.mean()]).T
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    slope = coef[1]
    # slope standard error with the effective sample count
    sxx = np.sum((t - t.mean())**2)*(neff/len(t))
    se_slope = sd/np.sqrt(max(sxx, 1e-30))
    drift = slope*T                                   # change across the window
    half = len(t)//2
    m1, m2 = u[:half].mean(), u[half:].mean()
    se_half = sd/np.sqrt(max(neff/2, 1.0))
    print(f'{label}window t = {t[0]:.2f} .. {t[-1]:.2f}  ({T:.2f} turnovers, '
          f'{len(t)} samples)')
    print(f'  u_tau        = {mu:.4f} +- {sd:.4f} (sd), standard error {se:.4f}')
    print(f'  integral scale {tint:.3f} turnovers -> {neff:.1f} effective samples')
    print(f'  trend        = {drift:+.4f} across the window '
          f'({abs(drift)/max(2*se_slope*T, 1e-30):.1f} sigma)  '
          f'{"NO significant drift" if abs(drift) < 2*se_slope*T else "** DRIFT **"}')
    print(f'  two halves   = {m1:.4f} vs {m2:.4f}, difference {m2-m1:+.4f} '
          f'({abs(m2-m1)/max(se_half*np.sqrt(2), 1e-30):.1f} sigma)  '
          f'{"consistent" if abs(m2-m1) < 2*se_half*np.sqrt(2) else "** INCONSISTENT **"}')
    print(f'  vs prescribed u_tau = 1: {100*(mu-1):+.2f} %'
          f'  ({abs(mu-1)/max(se, 1e-30):.1f} standard errors)')
    return dict(mu=mu, sd=sd, se=se, tint=tint, neff=neff, drift=drift, T=T)


def figure(t, u, t0, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = np.cumsum(u)/np.arange(1, len(u)+1)
    k = t >= t0
    runw = np.cumsum(u[k])/np.arange(1, k.sum()+1)
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(t, u, lw=0.8, color='C0', label=r'$u_\tau(t)$')
    ax[0].plot(t, run, lw=1.8, color='k', label='running mean, from $t=0$')
    ax[0].plot(t[k], runw, lw=1.8, color='C3', label=f'running mean, from $t={t0:g}$')
    ax[0].axhline(1.0, color='C7', ls='--', lw=1, label='prescribed')
    ax[0].axvline(t0, color='C3', ls=':', lw=1)
    ax[0].set_ylabel(r'$u_\tau$'); ax[0].legend(fontsize=8, ncol=2); ax[0].grid(alpha=0.3)
    ax[0].set_title('friction velocity and its convergence')
    err = 100*(runw - 1.0)
    ax[1].plot(t[k], err, lw=1.5, color='C3')
    ax[1].axhline(0, color='C7', ls='--', lw=1)
    for lev in (1, -1):
        ax[1].axhline(lev, color='C7', ls=':', lw=0.8)
    ax[1].set_xlabel('$t$ (eddy turnovers)')
    ax[1].set_ylabel(r'running mean error in $u_\tau$  [%]')
    ax[1].grid(alpha=0.3)
    ax[1].set_title(f'convergence of the window average (dotted: $\\pm1\\%$)')
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--from-t', type=float, default=None,
                    help='start of the analysis window (default: discard the first third)')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    s, path = load_series(a.path)
    t, u = s[:, 0], s[:, 1]
    print(f'{path}: {len(t)} samples, t = {t[0]:.3f} .. {t[-1]:.3f}\n')
    t0 = a.from_t if a.from_t is not None else t[0] + (t[-1] - t[0])/3
    assess(t, u, label='FULL RECORD: ')
    print()
    k = t >= t0
    assess(t[k], u[k], label=f'ANALYSIS WINDOW (from t = {t0:.2f}): ')
    if a.out:
        figure(t, u, t0, a.out)


if __name__ == '__main__':
    main()
