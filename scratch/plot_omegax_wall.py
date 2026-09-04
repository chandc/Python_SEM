"""Streamwise (axial) vorticity near the wall, from run01's own state.

omega_x is a PRIMARY UNKNOWN in FOSLS (field OX_), so this is the solver's
vorticity, not a post-hoc finite difference of u.

Near-wall streamwise vortices in the buffer layer (y+ ~ 15-30) are THE coherent
structure of wall turbulence: counter-rotating pairs, diameter ~30 wall units,
spanwise spacing ~100 wall units, which for Re_tau=180 and Lz+ = 192 means only
about two pairs fit across this minimal box -- that is what makes it minimal.
If FOSLS is doing DNS rather than carrying a decaying field, they must be there.
"""
import os, sys
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CK = os.path.join(_R, 'scratch', 'run01_ck', 'checkpoint_0000400.npz')
OUT = os.path.join(_R, 'scratch', 'run01_omegax.png')
RE_TAU = 180.0

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR
    import minchan as MC
    s = MC.setup(); m = s['m']; nz = s['nz']
    d = np.load(CK); U = d['U']; t = float(d['t'])
    C = OP.to_complex(U)
    ox = FR.to_physical(np.ascontiguousarray(C[..., OP.OX_:OP.OX_+1, :]), nz)[..., 0, :]
    u  = FR.to_physical(np.ascontiguousarray(C[..., OP.U_:OP.U_+1, :]), nz)[..., 0, :]
    z = (s['lz']/nz)*np.arange(nz)
    X = np.empty((m.nelem, s['N']+1, s['N']+1)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    yplus = np.minimum(Y, 2.0 - Y)*RE_TAU            # distance to nearest wall
    print(f't={t:.4f}, step={int(d["step"])}; max|omega_x|={np.abs(ox).max():.1f}, '
          f'rms={ox.std():.2f}')

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))

    # (a) wall-parallel plane at y+ ~ 15 (buffer layer)
    tgt = 15.0
    sel = np.abs(yplus - tgt) < 4.0
    xs, zs, vs = [], [], []
    for k in range(nz):
        xs.append(X[sel]); zs.append(np.full(sel.sum(), z[k])); vs.append(ox[sel, k])
    xs, zs, vs = np.concatenate(xs), np.concatenate(zs), np.concatenate(vs)
    lim = np.percentile(np.abs(vs), 99)
    ax = axes[0]
    sc = ax.tricontourf(xs, zs, vs, levels=np.linspace(-lim, lim, 31),
                        cmap='RdBu_r', extend='both')
    ax.set(xlabel='x', ylabel='z', title=f'(a) $\\omega_x$ at $y^+\\approx{tgt:.0f}$')
    plt.colorbar(sc, ax=ax)

    # (b) cross-stream plane, near-wall zoom
    e_mid = X[:, 0, 0].argsort()[m.nelem//2]
    xcut = X[e_mid, s['N']//2, 0]
    cut = np.abs(X - xcut) < 1e-9
    ys, zs2, vs2 = [], [], []
    for k in range(nz):
        ys.append(Y[cut]); zs2.append(np.full(cut.sum(), z[k])); vs2.append(ox[cut, k])
    ys, zs2, vs2 = np.concatenate(ys), np.concatenate(zs2), np.concatenate(vs2)
    keep = ys*RE_TAU < 90
    ax = axes[1]
    lim2 = np.percentile(np.abs(vs2[keep]), 99)
    sc = ax.tricontourf(zs2[keep], ys[keep]*RE_TAU, vs2[keep],
                        levels=np.linspace(-lim2, lim2, 31), cmap='RdBu_r', extend='both')
    ax.set(xlabel='z', ylabel='$y^+$', title=f'(b) $\\omega_x$ cross-section, $x$={xcut:.2f}')
    plt.colorbar(sc, ax=ax)

    # (c) rms profile vs y+
    yk = np.round(yplus, 8).ravel()
    order = np.argsort(yk); yy = yk[order]
    flat = ox.reshape(-1, nz)[order]
    uu = u.reshape(-1, nz)[order]
    edges = np.unique(yy)
    prof = np.array([flat[yy == e].std() for e in edges])
    ax = axes[2]
    ax.plot(edges, prof, 'k-', lw=1.8)
    ax.axvline(15, color='r', ls=':', lw=1, label='$y^+=15$')
    ax.set(xlabel='$y^+$', ylabel="$\\omega_x'$ rms", xscale='log',
           title='(c) streamwise-vorticity rms')
    ax.grid(alpha=.3); ax.legend()
    pk = edges[np.argmax(prof)]
    print(f'  omega_x rms peaks at y+ = {pk:.1f} (KMM/literature: y+ ~ 20, '
          f'with a wall value ~0.35-0.40 in u_tau^2/nu units)')
    print(f'  rms at wall {prof[0]:.3f}, peak {prof.max():.3f}')

    fig.suptitle(f'run01 FOSLS-3D minimal channel, $Re_\\tau$=180, t={t:.3f}', y=1.02)
    fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
