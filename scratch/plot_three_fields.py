"""u', omega_x and p' on the SAME near-wall plane, aligned for comparison.

All three are PRIMARY UNKNOWNS in FOSLS -- the vorticity is not a derivative of
u, and the pressure is not recovered from a Poisson solve.  That is the whole
point of the velocity-vorticity-pressure formulation, and it is why these three
panels can be put side by side without any post-processing between them.

y+ ~ 15 is the buffer layer: streaks strongest, quasi-streamwise vortices
strongest, and the pressure signature of the vortex cores clearest.

Fluctuations (mean removed on the plane) for u and p -- the mean shear dominates
u entirely, and the FOSLS pressure carries an arbitrary constant (it is pinned
at one node), so only its fluctuation is meaningful.

WHAT TO LOOK FOR.  Vortex cores are LOW-pressure, so blue in (c) should sit on
the sign changes of omega_x in (b), which is where a vortex axis lies -- and
those vortices pump slow fluid away from the wall, making the low-speed (blue)
streaks in (a).
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import semplot
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'scratch', 'run01_three_fields.png')
RT, YP = 180.0, 15.0

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR
    import minchan as MC
    s = MC.setup(); m = s['m']; nz = s['nz']; N = s['N']
    f = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))[-1]
    d = np.load(f); t = float(d['t']); C = OP.to_complex(d['U'])
    F = lambda i: FR.to_physical(np.ascontiguousarray(C[..., i:i+1, :]), nz)[..., 0, :]
    u, ox, p = F(OP.U_), F(OP.OX_), F(OP.P_)
    z = (s['lz']/nz)*np.arange(nz)
    X = np.empty((m.nelem, N+1, N+1)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    yp = np.minimum(Y, 2.0 - Y)*RT

    panels = [('u',  "$u'^+$  (streamwise velocity fluctuation)", u,  'RdBu_r'),
              ('ox', "$\\omega_x$  (streamwise vorticity)",       ox, 'RdBu_r'),
              ('p',  "$p'$  (pressure fluctuation)",              p,  'PuOr_r')]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.2), sharex=True)
    store = {}
    for ax, (key, title, fld, cmap) in zip(axes, panels):
        # EVALUATE THE INTERPOLANT on a uniform grid -- see semplot.py for why
        # tricontourf on the raw nodal points is wrong here.
        xf, zf, img = semplot.plane(fld, m, N, YP, RT=RT, nz=nz)
        if key in ('u', 'p'):
            img = img - img.mean()
        store[key] = img.ravel()
        lim = np.percentile(np.abs(img), 99)
        sc = ax.pcolormesh(xf*RT, zf*s['lz']*RT, img, cmap=cmap,
                           vmin=-lim, vmax=lim, shading='gouraud', rasterized=True)
        ax.contour(xf*RT, zf*s['lz']*RT, img, levels=[-0.5*lim, 0.5*lim],
                   colors='k', linewidths=0.35, alpha=.45)
        ax.set(ylabel='$z^+$', title=f'{title}    rms {img.std():.3g}')
        ax.set_aspect('equal'); plt.colorbar(sc, ax=ax, fraction=0.020, pad=0.01)
    axes[-1].set_xlabel('$x^+$')
    fig.suptitle(f'run01 FOSLS-3D minimal channel $Re_\\tau$=180, t={t:.2f} — '
                 f'$y^+\\approx{YP:.0f}$   (all three are PRIMARY unknowns)', y=0.995)
    fig.tight_layout(); fig.savefig(OUT, dpi=125, bbox_inches='tight')

    up_, ox_, p_ = store['u'], store['ox'], store['p']
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print(f't={t:.2f}, y+={YP:.0f}, {len(up_)} points\n')
    print(f"  corr(p', |omega_x|)  = {c(p_, np.abs(ox_)):+.3f}   "
          f"(negative => vortex cores are LOW pressure)")
    print(f"  corr(p', u')         = {c(p_, up_):+.3f}")
    print(f"  corr(u', |omega_x|)  = {c(up_, np.abs(ox_)):+.3f}")
    lo = up_ < np.percentile(up_, 20)
    print(f"\n  in the 20% LOWEST-speed streak regions:")
    print(f"    mean p'      {p_[lo].mean():+.4f}  (vs {p_.mean():+.4f} overall)")
    print(f"    mean |om_x|  {np.abs(ox_)[lo].mean():7.2f}  (vs {np.abs(ox_).mean():7.2f} overall)")
    print(f'\nsaved -> {OUT}')

if __name__ == '__main__':
    main()
