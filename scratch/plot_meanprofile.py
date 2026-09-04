"""Mean velocity profile of run01, in wall units, against the law of the wall.

u_tau = delta = 1 and nu = 1/180 by construction, so U and y ARE already in wall
units -- no rescaling, and no fitted constants anywhere in this plot.

Three references, in increasing order of how much they can be argued with:
  U+ = y+                     viscous sublayer, exact as y+ -> 0
  U+ = (1/0.41) ln y+ + 5.2   log law, standard constants
  Reichardt                   a single formula covering the whole profile
At Re_tau=180 there is barely a log layer at all -- the canonical centreline is
U+ ~ 18.2 -- so agreement in the sublayer and buffer region is the real test.
"""
import os, sys
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'scratch', 'run01_meanprofile.png')
RT = 180.0

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR
    import minchan as MC
    s = MC.setup(); m = s['m']; nz = s['nz']
    Y = np.empty((m.nelem, s['N']+1, s['N']+1))
    for e in range(m.nelem):
        Y[e] = m.ynod[e][None, :]
    yp = np.minimum(Y, 2.0 - Y)*RT
    yk = np.round(yp, 8).ravel(); ed = np.unique(yk)

    import glob
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cks = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))
    for f in cks:
        d = np.load(f); C = OP.to_complex(d['U'])
        u = FR.to_physical(np.ascontiguousarray(C[..., OP.U_:OP.U_+1, :]), nz)[..., 0, :]
        Um = np.array([u.reshape(-1, nz)[yk == e].mean() for e in ed])
        t = float(d['t'])
        last = (f == cks[-1])
        ax[0].semilogx(ed[1:], Um[1:], '-' if last else '-', lw=2.2 if last else 0.9,
                       color='k' if last else '0.7',
                       label=f't={t:.2f}' if last else None, zorder=3 if last else 1)
        if last:
            Ulast, tlast = Um, t

    yy = np.logspace(-1, np.log10(180), 300)
    ax[0].semilogx(yy, yy, 'b--', lw=1.2, label='$U^+=y^+$')
    lg = yy > 8
    ax[0].semilogx(yy[lg], np.log(yy[lg])/0.41 + 5.2, 'r--', lw=1.2,
                   label=r'$\frac{1}{0.41}\ln y^+ + 5.2$')
    reich = (np.log(1 + 0.4*yy)/0.41
             + 7.8*(1 - np.exp(-yy/11) - (yy/11)*np.exp(-yy/3)))
    ax[0].semilogx(yy, reich, 'g-', lw=1.0, alpha=.8, label='Reichardt')
    ax[0].set(xlabel='$y^+$', ylabel='$U^+$', ylim=(0, 21),
              title=f'(a) mean profile, grey = earlier checkpoints')
    ax[0].grid(alpha=.3, which='both'); ax[0].legend(loc='upper left', fontsize=9)

    # deviation from Reichardt, and the diagnostic function
    rl = (np.log(1 + 0.4*ed[1:])/0.41
          + 7.8*(1 - np.exp(-ed[1:]/11) - (ed[1:]/11)*np.exp(-ed[1:]/3)))
    ax[1].semilogx(ed[1:], Ulast[1:] - rl, 'k-', lw=1.8)
    ax[1].axhline(0, color='g', lw=1)
    ax[1].fill_between([0.1, 200], -0.5, 0.5, color='g', alpha=.12,
                       label='$\\pm0.5$ wall units')
    ax[1].set(xlabel='$y^+$', ylabel='$U^+ - U^+_{Reichardt}$', xlim=(0.5, 200),
              title='(b) deviation from Reichardt')
    ax[1].grid(alpha=.3, which='both'); ax[1].legend(fontsize=9)

    fig.suptitle(f'run01 FOSLS-3D minimal channel, $Re_\\tau$=180, t={tlast:.3f} '
                 f'(single snapshot, plane-averaged)', y=1.01)
    fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches='tight')

    print(f'{"y+":>7} {"U+ FOSLS":>9} {"Reichardt":>10} {"diff":>7}')
    for tgt in (1, 2, 5, 10, 15, 20, 30, 50, 100, 180):
        i = np.argmin(np.abs(ed - tgt))
        r = (np.log(1 + 0.4*ed[i])/0.41
             + 7.8*(1 - np.exp(-ed[i]/11) - (ed[i]/11)*np.exp(-ed[i]/3)))
        print(f'{ed[i]:7.1f} {Ulast[i]:9.2f} {r:10.2f} {Ulast[i]-r:+7.2f}')
    sub = ed[1:] < 5
    print(f'\n  sublayer  U+/y+ mean = {np.mean(Ulast[1:][sub]/ed[1:][sub]):.3f}  (exact: 1.000)')
    print(f'  centreline U+ = {Ulast[-1]:.2f}   [canonical Re_tau=180: ~18.2]')
    print(f'  max |U+ - Reichardt| over y+<100 = '
          f'{np.max(np.abs((Ulast[1:]-rl)[ed[1:]<100])):.2f} wall units')
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
