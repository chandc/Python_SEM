"""Near-wall streamwise vorticity at the CURRENT state, with the t=0.32 field
alongside -- because the streaks have roughly doubled in width since then, and
the quasi-streamwise vortices are what set that width.

omega_x is a PRIMARY UNKNOWN in FOSLS (field OX_), so this is the solver's own
vorticity, not a finite difference of u.

Three views:
  (a) wall-parallel plane at y+ ~ 20, where omega_x' peaks
  (b) cross-stream (z, y+) slice, the classic view of counter-rotating pairs
  (c) spanwise two-point correlation of omega_x -- the negative lobe locates the
      partner vortex, so its position IS the vortex pair spacing, measured
      rather than eyeballed.  Canonical: negative minimum near dz+ ~ 30-50.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import semplot
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'scratch', 'run01_omegax_now.png')
RT = 180.0

def load(f):
    import lssem3d
    from lssem3d import operator as OP, fourier as FR
    d = np.load(f); C = OP.to_complex(d['U'])
    ox = FR.to_physical(np.ascontiguousarray(C[..., OP.OX_:OP.OX_+1, :]), 32)[..., 0, :]
    return ox, float(d['t']) if 't' in d.files else 0.0

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    import minchan as MC
    s_ = MC.setup(); m = s_['m']; nz = s_['nz']; N = s_['N']; s = s_
    z = (s['lz']/nz)*np.arange(nz)
    X = np.empty((m.nelem, N+1, N+1)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    yp = np.minimum(Y, 2.0 - Y)*RT
    dzp = (s['lz']/nz)*np.arange(nz//2+1)*RT

    latest = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))[-1]
    ox, t = load(latest)

    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])

    # (a) wall-parallel at y+ ~ 20
    ax = fig.add_subplot(gs[0, :])
    xf, zf, img = semplot.plane(ox, m, N, 20.0, RT=RT, nz=nz)
    lim = np.percentile(np.abs(img), 99)
    sc = ax.pcolormesh(xf*RT, zf*s_['lz']*RT, img, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       shading='gouraud', rasterized=True)
    ax.contour(xf*RT, zf*s_['lz']*RT, img, levels=[-0.5*lim, 0.5*lim], colors='k',
               linewidths=0.35, alpha=.45)
    ax.set(xlabel='$x^+$', ylabel='$z^+$', title='(a) $\\omega_x$ at $y^+\\approx20$')
    ax.set_aspect('equal'); plt.colorbar(sc, ax=ax, fraction=0.020, pad=0.01)

    # (b) cross-stream slice: build it from y-planes through the same
    # interpolant, so it is smooth in both directions rather than triangulated.
    ax = fig.add_subplot(gs[1, 0])
    ypl = np.linspace(1.0, 95.0, 120)
    cols = []
    for yv in ypl:
        _, zf2, im = semplot.plane(ox, m, N, yv, RT=RT, nx_per_elem=2, nz=nz)
        cols.append(im.mean(axis=1))          # average over x -> (nz',)
    im2 = np.array(cols)                       # (ny, nz')
    lim2 = np.percentile(np.abs(im2), 99)
    sc = ax.pcolormesh(zf2*s_['lz']*RT, ypl, im2, cmap='RdBu_r', vmin=-lim2, vmax=lim2,
                       shading='gouraud', rasterized=True)
    ax.set(xlabel='$z^+$', ylabel='$y^+$',
           title='(b) $\\omega_x$, $x$-averaged cross-section')
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)

    # (c) spanwise correlation, now vs early
    ax = fig.add_subplot(gs[1, 1])
    early = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))[0]
    for fname, lab, col in ((early, None, 'C1'), (latest, None, 'k')):
        a, tt = load(fname)
        sub = a[np.abs(yp - 20.0) < 4.0]; sub = sub - sub.mean()
        c = np.array([np.mean(sub*np.roll(sub, k, axis=1)) for k in range(nz//2+1)])
        c /= c[0]
        i = int(np.argmin(c))
        ax.plot(dzp, c, '-', color=col, lw=2,
                label=f't={tt:.2f}: min {c[i]:+.2f} at $\\Delta z^+$={dzp[i]:.0f}')
        ax.plot([dzp[i]], [c[i]], 'o', color=col, ms=8, mfc='none', mew=2)
        print(f'  t={tt:5.2f}  omega_x R min {c[i]:+.3f} at dz+={dzp[i]:5.1f}   '
              f'rms={a[np.abs(yp-20.0)<4.0].std():6.2f}')
    ax.axhline(0, color='0.7', lw=0.8)
    ax.axvspan(30, 50, color='g', alpha=.12, label='canonical pair spacing')
    ax.set(xlabel='$\\Delta z^+$', ylabel='$R_{\\omega_x\\omega_x}$',
           title='(c) spanwise correlation of $\\omega_x$ at $y^+\\approx20$')
    ax.grid(alpha=.3); ax.legend(fontsize=9)

    fig.suptitle(f'run01 FOSLS-3D minimal channel $Re_\\tau$=180, t={t:.2f} — '
                 f'streamwise vorticity', y=0.995)
    fig.tight_layout(); fig.savefig(OUT, dpi=125, bbox_inches='tight')
    print(f'\n  canonical: counter-rotating pair -> negative minimum at dz+ ~ 30-50')
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
