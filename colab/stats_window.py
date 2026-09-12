"""Reynolds-stress profiles over a chosen time window, by differencing two
accumulated statistics files.

    python colab/stats_window.py A.npz B.npz [--out fig.png]
    python colab/stats_window.py --list DIR          # what windows are available

`PlaneStats` carries running SUMS, so the average over [t_A, t_B] is

    <q>_[A,B] = (sums_B - sums_A) / (nsamp_B - nsamp_A)

exactly.  That is the whole point of keeping the snapshots: the start-up
transient is excluded after the fact, by choosing A past it, with no need to
have known where it ended while the run was going.  Passing a single file (or
omitting A) averages from t = 0, which is what the run's own `stats.npz` is.

Accepts either a stats file (`sums`, `nsamp`) or a checkpoint (`stats_sums`,
`stats_nsamp`) at either end.

The total-stress balance is printed because it needs no reference data: for a
fully developed channel

    -<u'v'>^+  +  dU^+/dy^+  =  1 - y/delta

exactly, so a departure indicts u_tau, the forcing balance or the averaging
rather than the physics.  Against the literature at Re_tau = 180 the marks are
u'_rms peak 2.65-2.75 at y+ ~ 15, w' ~ 1.05 at y+ ~ 40, v' ~ 0.85 at y+ ~ 70.
"""
import argparse
import glob
import os
import sys

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=True)
    k = 'sums' if 'sums' in z.files else 'stats_sums'
    n = 'nsamp' if 'nsamp' in z.files else 'stats_nsamp'
    ser = 'utau_series' if 'utau_series' in z.files else 'stats_series'
    return dict(sums=np.asarray(z[k], dtype=float), nsamp=int(z[n]),
                y=np.asarray(z['y']) if 'y' in z.files else None,
                series=np.asarray(z[ser], dtype=float) if ser in z.files else np.zeros((0, 2)),
                nu=float(z['nu']) if 'nu' in z.files else 1/180.0,
                t=float(z['t']) if 't' in z.files else np.nan,
                path=path)


def fill_geometry(*ds):
    """Checkpoints carry the sums but not the y grid (it lives in the stats
    files), so borrow it: from the other end of the window, else from any stats
    file beside them."""
    y = next((d['y'] for d in ds if d is not None and d['y'] is not None), None)
    if y is None:
        for d in ds:
            if d is None:
                continue
            for c in sorted(glob.glob(os.path.join(os.path.dirname(d['path']) or '.', 'stats*.npz'))):
                y = load(c)['y']
                if y is not None:
                    break
            if y is not None:
                break
    if y is None:
        raise SystemExit('no y grid found: give a stats*.npz at one end of the window, '
                         'or keep one beside the checkpoints')
    for d in ds:
        if d is not None and d['y'] is None:
            d['y'] = y


def window(b, a=None):
    """Profiles from the window (a, b].  `a` None means from the start."""
    sums, n = b['sums'].copy(), b['nsamp']
    t0 = 0.0
    if a is not None:
        sums -= a['sums']
        n -= a['nsamp']
        t0 = a['t']
        if n <= 0:
            raise SystemExit(f'empty window: {a["path"]} has {a["nsamp"]} samples, '
                             f'{b["path"]} has {b["nsamp"]}')
    U, uu, vv, ww, uv = sums/n
    ser = b['series']
    if a is not None and ser.size:
        ser = ser[ser[:, 0] > t0]
    utau = float(ser[:, 1].mean()) if ser.size else 1.0
    nu, y = b['nu'], b['y']
    return dict(y=y, yp=y*utau/nu, Up=U/utau,
                urms=np.sqrt(np.maximum(uu - U**2, 0))/utau,
                vrms=np.sqrt(np.maximum(vv, 0))/utau,
                wrms=np.sqrt(np.maximum(ww, 0))/utau,
                uv=uv/utau**2, utau=utau, nsamp=n, t0=t0, t1=b['t'],
                utau_rms=float(ser[:, 1].std()) if ser.size else 0.0,
                Re_tau=utau/nu)


def report(p):
    half = p['yp'] <= p['yp'].max()/2 + 1e-9          # lower half, wall units
    yp, up = p['yp'][half], p['Up'][half]
    print(f'window t = {p["t0"]:.2f} .. {p["t1"]:.2f}  ({p["t1"]-p["t0"]:.2f} turnovers, '
          f'{p["nsamp"]} samples)')
    print(f'u_tau  = {p["utau"]:.4f} +- {p["utau_rms"]:.4f}   ->  Re_tau = {p["Re_tau"]:.1f} '
          f'(target 180)')
    for key, lit in (('urms', '2.65-2.75 @ y+ ~ 15'), ('wrms', '~1.05 @ y+ ~ 40'),
                     ('vrms', '~0.85 @ y+ ~ 70')):
        q = p[key][half]
        i = int(np.argmax(q))
        print(f'{key:>5s} peak {q[i]:.3f} at y+ = {yp[i]:5.1f}     [KMM {lit}]')
    # total stress: -<u'v'>+ + dU+/dy+ = 1 - y/delta
    o = np.argsort(yp)
    dUdy = np.gradient(up[o], yp[o])
    tot = -p['uv'][half][o] + dUdy
    lin = 1 - p['y'][half][o]/1.0
    m = (yp[o] > 5) & (yp[o] < 0.9*yp.max())
    print(f'total-stress balance: max |(-uv+ + dU+/dy+) - (1-y)| = '
          f'{np.abs(tot[m]-lin[m]).max():.3f} over 5 < y+ < {0.9*yp.max():.0f}')
    print(f'U+ at y+ = {yp[o][-1]:.0f}: {up[o][-1]:.2f}   '
          f'[log law 2.44 ln y+ + 5.2 = {2.44*np.log(yp[o][-1])+5.2:.2f}]')
    return p


def figure(p, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    half = p['yp'] <= p['yp'].max()/2 + 1e-9
    o = np.argsort(p['yp'][half])
    yp = p['yp'][half][o]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogx(yp, p['Up'][half][o], 'k-', lw=1.6, label='FOSLS')
    yv = np.logspace(np.log10(max(yp[1], 0.3)), np.log10(yp.max()), 60)
    ax[0].semilogx(yv[yv < 12], yv[yv < 12], 'C7--', lw=1, label=r'$U^+=y^+$')
    ax[0].semilogx(yv[yv > 10], 2.44*np.log(yv[yv > 10]) + 5.2, 'C7:', lw=1.2,
                   label=r'$2.44\ln y^++5.2$')
    ax[0].set_ylim(0, 1.15*float(np.nanmax(p['Up'][half])))
    ax[0].set_xlabel(r'$y^+$'); ax[0].set_ylabel(r'$U^+$'); ax[0].legend(fontsize=8)
    ax[0].set_title(f'mean profile, t = {p["t0"]:.1f}-{p["t1"]:.1f}')
    for k, c, lab in (('urms', 'C0', "u'"), ('vrms', 'C1', "v'"), ('wrms', 'C2', "w'")):
        ax[1].plot(yp, p[k][half][o], c, lw=1.5, label=lab + '_rms')
    ax[1].plot(yp, -p['uv'][half][o], 'C3', lw=1.5, label=r"$-\langle u'v'\rangle$")
    for ypk, v, c in ((15.0, 2.70, 'C0'), (70.0, 0.85, 'C1'), (40.0, 1.05, 'C2')):
        ax[1].plot([ypk], [v], 'o', color=c, mfc='none', ms=7)
    ax[1].set_xlabel(r'$y^+$'); ax[1].set_ylabel('wall units')
    ax[1].set_title(f'Reynolds stresses (circles: KMM peaks), $Re_\\tau$ = {p["Re_tau"]:.0f}')
    ax[1].legend(fontsize=8)
    for b in ax:
        b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='[A] B  (stats or checkpoint npz)')
    ap.add_argument('--out', default='')
    ap.add_argument('--list', default='', help='list the stats snapshots in a directory')
    a = ap.parse_args()
    if a.list:
        rows = []
        for p in sorted(glob.glob(os.path.join(a.list, 'stats*.npz'))
                        + glob.glob(os.path.join(a.list, 'checkpoint_*.npz'))):
            d = load(p)
            rows.append((d['t'], d['nsamp'], p))
        for t, n, p in sorted(rows):
            print(f'  t = {t:8.3f}  {n:6d} samples  {os.path.basename(p)}')
        if len(rows) >= 2:
            print(f'\nwindow over the last stretch:\n  python colab/stats_window.py '
                  f'{rows[-2][2]} {rows[-1][2]} --out window.png')
        return 0
    if not a.files:
        ap.error('give one or two npz files, or --list DIR')
    b = load(a.files[-1])
    aa = load(a.files[0]) if len(a.files) > 1 else None
    fill_geometry(b, aa)
    p = report(window(b, aa))
    if a.out:
        figure(p, a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
