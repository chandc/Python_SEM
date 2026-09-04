"""Near-wall streamwise velocity fluctuation u' -- the low/high-speed STREAKS.

The single most recognisable structure in wall turbulence: long streamwise
ribbons of alternating u', with a spanwise spacing of ~100 wall units that is
remarkably universal across Reynolds number.  They are the footprint of the
quasi-streamwise vortices (see scratch/plot_omegax_wall.py) sweeping slow fluid
up and fast fluid down.

Plotted as the FLUCTUATION u - <u>_plane, because the mean shear dominates the
raw field entirely.  In wall units, since u_tau = 1 by construction.

This box holds Lz+ = 192, so only about TWO streak pairs fit across the span --
that is precisely what makes it a "minimal" channel, and why the structures
should look crowded rather than statistically rich.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'scratch', 'run01_streaks.png')
RT = 180.0
TARGETS = (5.0, 12.0, 30.0)

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR
    import minchan as MC
    s = MC.setup(); m = s['m']; nz = s['nz']; N = s['N']
    f = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))[-1]
    d = np.load(f); t = float(d['t'])
    C = OP.to_complex(d['U'])
    u = FR.to_physical(np.ascontiguousarray(C[..., OP.U_:OP.U_+1, :]), nz)[..., 0, :]
    z = (s['lz']/nz)*np.arange(nz)
    X = np.empty((m.nelem, N+1, N+1)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    yp = np.minimum(Y, 2.0 - Y)*RT

    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11.5, 9.2))
    for ax, tgt in zip(axes, TARGETS):
        sel = np.abs(yp - tgt) < max(2.0, 0.18*tgt)
        # fluctuation: remove the plane mean at this y+
        up = u[sel] - u[sel].mean()
        xs = np.concatenate([X[sel]]*nz)
        zs = np.concatenate([np.full(sel.sum(), zz) for zz in z])
        vs = np.concatenate([up[:, k] for k in range(nz)])
        lim = np.percentile(np.abs(vs), 98)
        sc = ax.tricontourf(xs*RT, zs*RT, vs, levels=np.linspace(-lim, lim, 33),
                            cmap='RdBu_r', extend='both')
        ax.set(ylabel='$z^+$', title=f"$u'^+$ at $y^+\\approx{tgt:.0f}$   "
                                     f"(rms {vs.std():.2f}, range ±{lim:.2f})")
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, fraction=0.020, pad=0.01)
        # spanwise spacing from the two-point correlation at this height
        sub = up - up.mean()
        c = np.array([np.mean(sub*np.roll(sub, k, axis=1)) for k in range(nz//2+1)])
        c /= c[0]
        dzp = (s['lz']/nz)*np.arange(nz//2+1)*RT
        neg = np.argmin(c)
        print(f"  y+={tgt:4.0f}: u' rms {vs.std():5.2f}, R_uu minimum {c[neg]:+.3f} "
              f"at dz+={dzp[neg]:5.1f}  ->  streak spacing ~{2*dzp[neg]:.0f} wall units")
    axes[-1].set_xlabel('$x^+$')
    fig.suptitle(f"run01 FOSLS-3D minimal channel $Re_\\tau$=180, t={t:.2f} — "
                 f"near-wall streaks  ($L_x^+$={np.pi*RT:.0f}, $L_z^+$={s['lz']*RT:.0f})",
                 y=0.995)
    fig.tight_layout(); fig.savefig(OUT, dpi=125, bbox_inches='tight')
    print(f'\n  canonical streak spacing: ~100 wall units')
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
