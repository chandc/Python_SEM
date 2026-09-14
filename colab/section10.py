"""Section 10 of the paper from a finished run: profiles, comparison, verdict.

    python colab/section10.py A.npz B.npz [--fs results/minchan_re180_K/stats_MERGED.npz]
                              [--out DIR] [--label run02]

Takes the two accumulated-statistics files that bound the averaging window (see
colab/stats_window.py for why differencing them is exact), reduces them to wall
units, and puts the result beside the five published Re_tau = 180 databases in
`reference/` and beside our own fractional-step run on the SAME box and mesh.

THE BOX-VALIDITY LINE MATTERS MORE THAN ANY SINGLE NUMBER.  This is a minimal
channel, Lx+ = 565 and Lz+ = 192.  Jimenez & Moin's argument is that such a box
reproduces the near-wall cycle and NOT the outer layer, and the usual limit
quoted for it is y+ ~ Lz+/3 ~ 64.  Comparisons above that line are not expected
to hold and are shown shaded, so that a centreline discrepancy is read as what
it is -- the box -- rather than as a defect of the scheme.  The quantitative
table is therefore computed BELOW the line; the profile above it is plotted for
honesty, not scored.

Outputs `section10_profiles.png`, `section10_table.md` (paste-ready) and prints
the same table.
"""
import argparse
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
sys.path.insert(0, os.path.join(_R, 'colab'))
os.chdir(_R)

import numpy as np

import ref180
from stats_window import load, window, fill_geometry

YV = 60.0                      # box-validity limit, y+ ~ Lz+/3
KAPPA, B_LOG = 0.41, 5.2


def fold(y, q, odd=False):
    """Fold a full-channel profile onto the lower half (channel symmetry)."""
    yw = np.round(np.minimum(y, 2.0 - y), 8)
    u = np.unique(yw)
    out = np.array([np.mean(np.where(y[yw == v] > 1.0, (-q if odd else q)[yw == v],
                                     q[yw == v])) for v in u])
    return u, out


def profiles(p):
    """Window dict -> folded half-channel profiles in wall units."""
    y, ut, nu = p['y'], p['utau'], p['nu']
    yp, U = fold(y, p['Up'])
    _, ur = fold(y, p['urms']); _, vr = fold(y, p['vrms']); _, wr = fold(y, p['wrms'])
    _, uv = fold(y, -p['uv'], odd=True)          # -<u'v'>+, odd about the centreline
    return dict(yp=yp*ut/nu, y=yp, U=U, urms=ur, vrms=vr, wrms=wr, uv=uv,
                utau=ut, Re_tau=ut/nu)


def peak(yp, q, lo=1.0, hi=YV):
    m = (yp >= lo) & (yp <= hi)
    i = int(np.argmax(q[m]))
    return float(q[m][i]), float(yp[m][i])


def stress_balance(pr):
    """-<u'v'>+ + dU+/dy+ - (1 - y/delta): exact for a developed channel."""
    o = np.argsort(pr['yp'])
    yp, U, uv, y = pr['yp'][o], pr['U'][o], pr['uv'][o], pr['y'][o]
    tot = uv + np.gradient(U, yp)
    m = (yp > 5) & (yp < 0.9*yp.max())
    return float(np.abs(tot[m] - (1 - y[m])).max())


def table(pr, fs, refs, label):
    rows = []
    up, uy = peak(pr['yp'], pr['urms'])
    vp, vy = peak(pr['yp'], pr['vrms'])
    wp, wy = peak(pr['yp'], pr['wrms'])
    got = {"u'_rms peak": up, "v'_rms peak": vp, "w'_rms peak": wp,
           "-<u'v'>+ max": peak(pr['yp'], pr['uv'])[0],
           "U+ at y+=30": float(np.interp(30.0, pr['yp'], pr['U'])),
           "Re_tau": pr['Re_tau']}
    ref = {}
    for k, f in refs.items():
        d = f()
        ref[k] = {"u'_rms peak": peak(d['yp'], d['urms'])[0],
                  "v'_rms peak": peak(d['yp'], d['vrms'])[0],
                  "w'_rms peak": peak(d['yp'], d['wrms'])[0],
                  "-<u'v'>+ max": peak(d['yp'], d['uv'])[0],
                  "U+ at y+=30": float(np.interp(30.0, d['yp'], d['U'])),
                  "Re_tau": d['Re_tau']}
    fsv = None
    if fs is not None:
        fsv = {"u'_rms peak": peak(fs['yp'], fs['urms'])[0],
               "v'_rms peak": peak(fs['yp'], fs['vrms'])[0],
               "w'_rms peak": peak(fs['yp'], fs['wrms'])[0],
               "-<u'v'>+ max": peak(fs['yp'], fs['uv'])[0],
               "U+ at y+=30": float(np.interp(30.0, fs['yp'], fs['U'])),
               "Re_tau": fs['Re_tau']}

    keys = list(got)
    names = list(ref)
    hdr = f'| quantity | {label} | ' + (' fractional step | ' if fsv else '') + \
          ' | '.join(names) + ' | spread of the databases | our deviation |'
    sep = '|' + '---|'*(2 + (1 if fsv else 0) + len(names) + 2)
    lines = [hdr, sep]
    for k in keys:
        vals = np.array([ref[n][k] for n in names])
        mu, sp = vals.mean(), (vals.max() - vals.min())/vals.mean()*100
        dev = (got[k] - mu)/mu*100
        cells = [f'{got[k]:.3f}']
        if fsv:
            cells.append(f'{fsv[k]:.3f}')
        cells += [f'{ref[n][k]:.3f}' for n in names]
        lines.append(f'| {k} | ' + ' | '.join(cells) + f' | {sp:.1f} % | **{dev:+.1f} %** |')
    lines.append(f'| peak locations $y^+$ | u {uy:.0f}, v {vy:.0f}, w {wy:.0f} | ' +
                 ('| '*(1 if fsv else 0)) + '| '*len(names) + ' | |')
    return '\n'.join(lines)


def figure(pr, fs, refs, out, label, win):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    R = {k: f() for k, f in refs.items()}

    a = ax[0, 0]
    # the wall node folds to y+ = 0 (to round-off), which a log axis renders as
    # a decade axis running to 1e-14; drop the sub-0.2 points from this panel only
    def _lg(d, q='U'):
        m = d['yp'] > 0.2
        return d['yp'][m], d[q][m]
    for k, d in R.items():
        a.semilogx(*_lg(d), lw=1, alpha=0.6, label=d['name'])
    if fs is not None:
        a.semilogx(*_lg(fs), color='C7', ls='--', lw=1.4, label='fractional step, same box')
    a.semilogx(*_lg(pr), 'k-', lw=2.2, label=f'{label} (FOSLS)')
    a.set_xlim(0.2, 200)
    yv = np.logspace(np.log10(11), np.log10(200), 40)
    a.semilogx(yv, np.log(yv)/KAPPA + B_LOG, ':', color='C3', lw=1.2,
               label=r'$\ln y^+/0.41+5.2$')
    a.set_xlabel(r'$y^+$'); a.set_ylabel(r'$U^+$'); a.legend(fontsize=7)
    a.set_title('mean velocity')

    a = ax[0, 1]
    for q, c in (('urms', 'C0'), ('vrms', 'C1'), ('wrms', 'C2')):
        for d in R.values():
            a.plot(d['yp'], d[q], color=c, lw=0.9, alpha=0.45)
        if fs is not None:
            a.plot(fs['yp'], fs[q], color=c, ls='--', lw=1.2)
        a.plot(pr['yp'], pr[q], color=c, lw=2.2,
               label={'urms': "u'", 'vrms': "v'", 'wrms': "w'"}[q] + '_rms')
    a.set_xlim(0, 180); a.set_xlabel(r'$y^+$'); a.set_ylabel('wall units')
    a.legend(fontsize=8); a.set_title('velocity fluctuations (thin: databases, dashed: FS)')

    a = ax[1, 0]
    for d in R.values():
        a.plot(d['yp'], d['uv'], lw=0.9, alpha=0.5)
    if fs is not None:
        a.plot(fs['yp'], fs['uv'], 'C7--', lw=1.2, label='fractional step')
    a.plot(pr['yp'], pr['uv'], 'k-', lw=2.2, label=f'{label}')
    a.set_xlim(0, 180); a.set_xlabel(r'$y^+$'); a.set_ylabel(r"$-\langle u'v'\rangle^+$")
    a.legend(fontsize=8); a.set_title('Reynolds shear stress')

    a = ax[1, 1]
    o = np.argsort(pr['yp'])
    tot = pr['uv'][o] + np.gradient(pr['U'][o], pr['yp'][o])
    a.plot(pr['yp'][o], tot, 'k-', lw=2, label=r"$-\langle u'v'\rangle^++\mathrm{d}U^+/\mathrm{d}y^+$")
    a.plot(pr['yp'][o], 1 - pr['y'][o], 'C3--', lw=1.4, label=r'$1-y/\delta$ (exact)')
    a.set_xlim(0, 180); a.set_ylim(0, 1.15); a.set_xlabel(r'$y^+$')
    a.legend(fontsize=8)
    a.set_title(f'total stress balance (max error {stress_balance(pr):.3f})')

    for b in ax.ravel():
        b.axvspan(YV, 200, color='y', alpha=0.10)
        b.grid(alpha=0.3)
    fig.suptitle(f'minimal channel $Re_\\tau=180$ ($L_x^+=565$, $L_z^+=192$), '
                 f'{label}: window {win}\nshaded $y^+>{YV:.0f}$ is outside the '
                 f'minimal box\'s validity and is not expected to match', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=140)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='A.npz B.npz bounding the window (or just B)')
    ap.add_argument('--fs', default='results/minchan_re180_K/stats_MERGED.npz')
    ap.add_argument('--out', default='.')
    ap.add_argument('--label', default='run02')
    a = ap.parse_args()

    b = load(a.files[-1])
    aa = load(a.files[0]) if len(a.files) > 1 else None
    fill_geometry(b, aa)
    w = window(b, aa)
    pr = profiles(w)
    win = f"t = {w['t0']:.1f}-{w['t1']:.1f} ({w['t1']-w['t0']:.1f} turnovers, {w['nsamp']} samples)"

    fs = None
    if a.fs and os.path.exists(a.fs):
        fz = load(a.fs)
        fs = profiles(window(fz))
        print(f'fractional-step twin: {a.fs}, {fz["nsamp"]} samples\n')

    os.makedirs(a.out, exist_ok=True)
    tb = table(pr, fs, ref180.ALL, a.label)
    hdr = (f"**{a.label}**, {win}; $u_\\tau$ = {pr['utau']:.4f}, "
           f"$Re_\\tau$ = {pr['Re_tau']:.1f}; total-stress balance closes to "
           f"{stress_balance(pr):.3f}.  Peaks measured below $y^+={YV:.0f}$.")
    print(hdr + '\n'); print(tb)
    with open(os.path.join(a.out, 'section10_table.md'), 'w') as f:
        f.write(hdr + '\n\n' + tb + '\n')
    figure(pr, fs, ref180.ALL, os.path.join(a.out, 'section10_profiles.png'),
           a.label, win)


if __name__ == '__main__':
    main()
