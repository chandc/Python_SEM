"""The three measurements that carry the FOSLS-vs-fractional-step comparison:
pointwise divergence (FOSLS's advantage), small-scale content (its cost), and
how the small scales evolved over the run.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT = os.path.join(_R, 'figs_fosls_vs_fs', 'evidence.png')
RT = 180.0

def legvander(x, N):
    V = np.zeros((len(x), N+1)); V[:, 0] = 1.0
    if N >= 1: V[:, 1] = x
    for n in range(1, N):
        V[:, n+1] = ((2*n+1)*x*V[:, n] - n*V[:, n-1])/(n+1)
    return V

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, deriv as DV, fourier as FR
    from lssem2d.lgl import lgl_nodes
    import minchan as MC
    s = MC.setup(); m = s['m']; nz = s['nz']; N = s['N']
    Y = np.empty((m.nelem, N+1, N+1))
    for e in range(m.nelem): Y[e] = m.ynod[e][None, :]
    yp = np.round(np.minimum(Y, 2.0-Y)*RT, 8); ypu = np.unique(yp)
    xg = np.asarray(lgl_nodes(N)[0] if isinstance(lgl_nodes(N), tuple) else lgl_nodes(N),
                    float).ravel()[:N+1]
    Vi = np.linalg.inv(legvander(xg, N))
    cks = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))
    seed = os.path.join(_R, 'scratch/fs_seed/seed_ckpt.npz')

    def fields(f):
        d = np.load(f); C = OP.to_complex(d['U'])
        t = float(d['t']) if 't' in d.files else 0.0
        return C, t

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) pointwise |div u| profile
    for f, lab, col in ((seed, 'fractional step (projected)', 'C1'),
                        (cks[-1], None, 'k')):
        C, t = fields(f)
        du = (DV.ddx(C[..., OP.U_:OP.U_+1, :], s['D'], m.facx)
              + DV.ddy(C[..., OP.V_:OP.V_+1, :], s['D'], m.facy)
              + 1j*s['kz']*C[..., OP.W_:OP.W_+1, :])
        u2 = np.sqrt((np.abs(C[..., :3, :])**2).mean())
        dm = np.sqrt((np.abs(du[..., 0, :])**2).mean(axis=-1)).ravel()
        prof = np.array([dm[yp.ravel() == v].mean() for v in ypu])/u2
        ax[0].semilogy(ypu, prof, '-', color=col, lw=2.2,
                       label=lab or f'FOSLS t={t:.2f}')
    ax[0].set(xlabel='$y^+$', ylabel=r'$|\nabla\!\cdot\!u|\,/\,\mathrm{rms}|u|$',
              xlim=(0, 180), title='(a) POINTWISE divergence')
    ax[0].grid(alpha=.3, which='both'); ax[0].legend(fontsize=9)

    # (b) Legendre modal spectrum
    for f, lab, col in ((seed, 'fractional step (skew form)', 'C1'),
                        (cks[-1], None, 'k')):
        C, t = fields(f)
        u = FR.to_physical(np.ascontiguousarray(C[..., OP.U_:OP.U_+1, :]), nz)[..., 0, :]
        a = np.einsum('qj,epjz->epqz', Vi, np.einsum('pi,eijz->epjz', Vi, u))
        e = (a**2).mean(axis=(0, 3))
        deg = np.maximum.outer(np.arange(N+1), np.arange(N+1))
        sp = np.array([np.sqrt(e[deg == d].mean()) for d in range(N+1)])
        ax[1].semilogy(np.arange(N+1), sp, 'o-', color=col, lw=2, ms=5,
                       label=lab or f'FOSLS t={t:.2f}')
    ax[1].set(xlabel='Legendre degree', ylabel='rms modal amplitude of $u$',
              title='(b) intra-element spectrum')
    ax[1].grid(alpha=.3, which='both'); ax[1].legend(fontsize=9)

    # (c) omega_x rms and streak spacing vs time
    ts, oxr, sk = [], [], []
    for f in cks:
        C, t = fields(f)
        ox = FR.to_physical(np.ascontiguousarray(C[..., OP.OX_:OP.OX_+1, :]), nz)[..., 0, :]
        u = FR.to_physical(np.ascontiguousarray(C[..., OP.U_:OP.U_+1, :]), nz)[..., 0, :]
        sel = np.abs(yp - 20.0) < 4.0
        ts.append(t); oxr.append(ox[sel].std())
        sub = u[np.abs(yp-12.0) < 3.0]; sub = sub - sub.mean()
        c = np.array([np.mean(sub*np.roll(sub, k, axis=1)) for k in range(nz//2+1)])
        c /= c[0]
        sk.append(2*(s['lz']/nz)*int(np.argmin(c))*RT)
    ax[2].plot(ts, oxr, 'ko-', lw=2, ms=4, label=r"$\omega_x$ rms at $y^+\!=\!20$")
    ax[2].set(xlabel='$t$', ylabel=r"$\omega_x$ rms")
    a2 = ax[2].twinx()
    a2.plot(ts, sk, 's--', color='C3', lw=1.6, ms=4, label='streak spacing $\\Delta z^+$')
    a2.axhline(100, color='g', ls=':', lw=1.5)
    a2.set_ylabel('streak spacing $\\Delta z^+$', color='C3')
    ax[2].set_title('(c) small scales over the run')
    ax[2].grid(alpha=.3)
    h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[2].legend(h1+h2, l1+l2+['canonical 100'], fontsize=8, loc='center right')
    print(f'  omega_x rms {oxr[0]:.2f} -> {oxr[-1]:.2f}  ({100*(oxr[-1]/oxr[0]-1):+.0f}%)')
    print(f'  streak dz+  {sk[0]:.0f} -> {sk[-1]:.0f}')
    fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
