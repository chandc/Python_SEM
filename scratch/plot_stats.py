"""run01: time-averaged mean profile and Reynolds stresses, in wall units.

Uses the ACCUMULATED statistics carried in the checkpoint (81 plane-averaged
samples over t = 0.0008..0.64), not a single snapshot.

u_tau = delta = 1 and nu = 1/180 by construction, so U and y ARE wall units --
nothing is rescaled and no constant is fitted anywhere in this figure.

sums[0]=<u> sums[1]=<uu> sums[2]=<vv> sums[3]=<ww> sums[4]=<uv>, so the
FLUCTUATIONS need the mean removed: u'^2 = <uu> - <u>^2.  <v> and <w> are zero
by symmetry but are not assumed -- only <u> is subtracted, which is the only one
that matters.

THE TOTAL-STRESS BALANCE is the strongest check available and needs no
reference data: for a fully developed channel, exactly

    -<u'v'>^+  +  dU^+/dy^+   =   1 - y/delta

so the two measured curves must sum to a straight line from 1 to 0.  Any error
in u_tau, in the forcing balance, or in the averaging shows up as a departure.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'scratch', 'run01_stats.png')
RT = 180.0

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    import minchan as MC
    s = MC.setup(); m = s['m']; n = s['N']+1
    Y = np.empty((m.nelem, n, n))
    for e in range(m.nelem):
        Y[e] = m.ynod[e][None, :]
    yk = np.round(Y, 10).ravel()
    order = np.argsort(yk, kind='stable')
    splits = np.flatnonzero(np.diff(yk[order]) > 1e-9) + 1
    groups = np.split(np.arange(yk.size), splits)
    y = np.array([yk[order][g[0]] for g in groups])

    cks = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))
    f = cks[-1]
    d = np.load(f); S = d['stats_sums']/int(d['stats_nsamp'])
    t = float(d['t']); ns = int(d['stats_nsamp'])
    # An EARLIER accumulation, overlaid, so the figure shows whether the
    # statistics are still moving.  Averages that have stopped drifting are
    # converged; ones still walking are not, whatever they agree with.
    prev = None
    for g in reversed(cks[:-1]):
        dg = np.load(g)
        if int(dg['stats_nsamp']) < ns - 10:
            prev = (dg['stats_sums']/int(dg['stats_nsamp']), float(dg['t']),
                    int(dg['stats_nsamp']))
            break
    U, uu, vv, ww, uv = S
    up = np.sqrt(np.maximum(uu - U**2, 0.0)); vp = np.sqrt(np.maximum(vv, 0.0))
    wp = np.sqrt(np.maximum(ww, 0.0)); uvp = uv                      # <v> = 0

    # fold onto y+ using channel symmetry (u,v,w,uv all even/odd appropriately)
    # ROUND BEFORE unique().  y and 2-y differ in the last floating-point bits,
    # so raw np.unique keeps BOTH copies of every mirrored station -- 118 entries
    # with duplicated values, zero spacing between neighbours, and np.gradient
    # then returns 0 for dU+/dy+.  That silently zeroed the viscous stress.
    yp = np.round(np.minimum(y, 2.0 - y)*RT, 8)
    ypu = np.unique(yp)
    fold = lambda a, sign=1: np.array(
        [np.mean(np.where(y[yp == v] > 1.0, sign*a[yp == v], a[yp == v]))
         for v in ypu])
    Uf, upf, vpf, wpf = (fold(U), fold(up), fold(vp), fold(wp))
    uvf = fold(uvp, -1)                       # <u'v'> is ODD about the centreline

    fig, ax = plt.subplots(1, 4, figsize=(21.5, 4.7))
    if prev is not None:
        Sp, tp, nsp = prev
        Up_, uup, vvp, wwp, uvp_ = Sp
        upp = np.sqrt(np.maximum(uup - Up_**2, 0.0))
        foldp = lambda a, sign=1: np.array(
            [np.mean(np.where(y[yp == v] > 1.0, sign*a[yp == v], a[yp == v]))
             for v in ypu])
        ax[0].semilogx(ypu[1:], foldp(Up_)[1:], '-', color='0.65', lw=1.2,
                       label=f't={tp:.2f} ({nsp} samp)', zorder=1)
        for a_, col in ((upp, 'C0'), (np.sqrt(np.maximum(wwp, 0)), 'C2'),
                        (np.sqrt(np.maximum(vvp, 0)), 'C1')):
            ax[1].plot(ypu, foldp(a_), '--', color=col, lw=1.0, alpha=.55, zorder=1)
        ax[2].plot(ypu, -foldp(uvp_, -1), '--', color='0.55', lw=1.2, alpha=.85,
                   zorder=1, label=f't={tp:.2f} ({nsp} samp)')

    yy = np.logspace(-1, np.log10(180), 300)
    ax[0].semilogx(ypu[1:], Uf[1:], 'k-', lw=2.2, label=f'run01, t={t:.2f}')
    ax[0].semilogx(yy, yy, 'b--', lw=1.1, label='$U^+=y^+$')
    lg = yy > 8
    ax[0].semilogx(yy[lg], np.log(yy[lg])/0.41 + 5.2, 'r--', lw=1.1,
                   label=r'$\frac{1}{0.41}\ln y^++5.2$')
    ax[0].set(xlabel='$y^+$', ylabel='$U^+$', ylim=(0, 21),
              title=f'(a) mean profile ({ns} samples)')
    ax[0].grid(alpha=.3, which='both'); ax[0].legend(loc='upper left', fontsize=9)

    # KMM Re_tau=180 peak (y+, value).  These are LITERATURE reference points
    # quoted from the standard tabulations, not recomputed here -- shown as
    # single markers so the comparison is a point-to-point one rather than a
    # line that invites reading agreement where none was measured.
    KMM = {"u": (15.0, 2.70, 'C0'), "w": (40.0, 1.05, 'C2'), "v": (70.0, 0.85, 'C1')}
    for a_, lab, col, key in ((upf, "$u'^+$", 'C0', 'u'), (wpf, "$w'^+$", 'C2', 'w'),
                              (vpf, "$v'^+$", 'C1', 'v')):
        ax[1].plot(ypu, a_, '-', color=col, lw=2, label=lab)
        yk_, vk_, ck_ = KMM[key]
        ax[1].plot([yk_], [vk_], marker='*', ms=22, mfc=ck_, mec='k', mew=1.4,
                   ls='none', zorder=5)
        # and mark where OUR peak actually falls, for the same quantity
        k = int(np.argmax(a_))
        ax[1].plot([ypu[k]], [a_[k]], marker='o', ms=9, mfc='none', mec=ck_,
                   mew=2.0, ls='none', zorder=5)
    ax[1].plot([], [], marker='*', ms=18, mfc='0.6', mec='k', ls='none',
               label='KMM peak (lit.)')
    ax[1].plot([], [], marker='o', ms=9, mfc='none', mec='0.4', mew=2, ls='none',
               label='run01 peak')
    ax[1].set(xlabel='$y^+$', ylabel='rms', xlim=(0, 180), ylim=(0, 3.0),
              title='(b) Reynolds normal stresses')
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=9, ncol=2, loc='upper right')

    dUdy = np.gradient(Uf, ypu)
    # (c) the Reynolds SHEAR stress on its own -- the term that carries the
    # turbulent momentum flux, and the one that must peak near y+ ~ 30 and go to
    # zero at both the wall (no-slip) and the centreline (symmetry).
    ax[2].plot(ypu, -uvf, 'k-', lw=2.2, label=r"$-\langle u'v'\rangle^+$")
    ax[2].plot([30.0], [0.72], marker='*', ms=22, mfc='k', mec='k', mew=1.2,
               ls='none', zorder=5, label='KMM peak (lit.)')
    k = int(np.argmax(-uvf))
    ax[2].plot([ypu[k]], [-uvf[k]], marker='o', ms=10, mfc='none', mec='k',
               mew=2.2, ls='none', zorder=5, label='run01 peak')
    ax[2].axhline(0, color='0.7', lw=0.8)
    ax[2].set(xlabel='$y^+$', ylabel=r"$-\langle u'v'\rangle^+$", xlim=(0, 180),
              title="(c) Reynolds shear stress")
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=9, loc='lower right')

    # (d) the balance, which must close exactly for a developed channel
    ax[3].plot(ypu, -uvf, 'k-', lw=1.6, label=r"$-\langle u'v'\rangle^+$ (turbulent)")
    ax[3].plot(ypu, dUdy, 'b-', lw=1.6, label=r'$dU^+/dy^+$ (viscous)')
    ax[3].plot(ypu, -uvf + dUdy, 'r-', lw=2.4, label='total')
    ax[3].plot(ypu, 1 - ypu/RT, 'g--', lw=1.6, label=r'$1-y/\delta$ (exact)')
    ax[3].set(xlabel='$y^+$', ylabel='stress', xlim=(0, 180), ylim=(-0.05, 1.15),
              title='(d) total-stress balance')
    ax[3].grid(alpha=.3); ax[3].legend(fontsize=9)

    fig.suptitle(f'run01 FOSLS-3D minimal channel $Re_\\tau$=180 — '
                 f'{ns} samples to t={t:.2f}', y=1.02)
    fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches='tight')

    tot = -uvf + dUdy
    ex = 1 - ypu/RT
    i = ypu < 150
    print(f'{"y+":>7} {"U+":>7} {"u+":>6} {"v+":>6} {"w+":>6} {"-uv+":>7} {"total":>7} {"1-y/d":>7}')
    for tgt in (1, 5, 10, 15, 20, 30, 50, 100, 170):
        k = np.argmin(np.abs(ypu - tgt))
        print(f'{ypu[k]:7.1f} {Uf[k]:7.2f} {upf[k]:6.3f} {vpf[k]:6.3f} {wpf[k]:6.3f} '
              f'{-uvf[k]:7.3f} {tot[k]:7.3f} {ex[k]:7.3f}')
    print(f'\n  peaks:  u+ {upf.max():.3f} at y+={ypu[np.argmax(upf)]:.1f}  [KMM 2.65-2.75 @ ~15]')
    print(f'          v+ {vpf.max():.3f}  [~0.85]     w+ {wpf.max():.3f}  [~1.05]')
    print(f'          -<uv>+ {(-uvf).max():.3f} at y+={ypu[np.argmax(-uvf)]:.1f}  [~0.72 @ ~30]')
    print(f'  total-stress balance: max |total-(1-y/d)| over y+<150 = '
          f'{np.max(np.abs(tot-ex)[i]):.3f}')
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
