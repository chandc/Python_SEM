"""Plot the Poiseuille start-up comparison from scratch/pois2d_vs_*_dt*.npz."""
import os, sys, glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pois2d_vs import u_exact

DT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
SUF = sys.argv[2] if len(sys.argv) > 2 else ''
OUT = f'scratch/pois2d_vs_dt{DT:g}{SUF}.png'
COL = {'jacobi': 'k', 'pmg2': 'b', 'vschwarz': 'r'}
LAB = {'jacobi': 'Jacobi', 'pmg2': 'PMG2 ladder (4,2)+direct', 'vschwarz': 'vertex-patch Schwarz + p=2 coarse'}


def main():
    runs = {}
    for f in sorted(glob.glob(f'scratch/pois2d_vs_*_dt{DT:g}{SUF}.npz')):
        d = np.load(f); runs[str(d['kind'])] = d
    T = float(list(runs.values())[0]['T_end'])
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # (a) centreline history
    a = ax[0, 0]; tt = np.linspace(T/400, T, 400)
    a.plot(tt, u_exact(0.5, tt), 'k-', lw=1, label='exact transient')
    for k, d in runs.items():
        a.plot(d['ts'], d['uc'], 'o', color=COL[k], ms=5, mfc='none', label=LAB[k])
    a.set(xlabel='t', ylabel='$u_c(t)$', title='centreline velocity, start-up from rest'); a.legend(fontsize=8); a.grid(alpha=.3)
    # (b) final profile
    a = ax[0, 1]; yy = np.linspace(0, 1, 200)
    a.plot(6*yy*(1-yy), yy, 'k--', lw=1, label='parabola $6y(1-y)$')
    for k, d in runs.items():
        y = d['y'][:, :]; U = d['U'][..., 0]; e0 = 0
        ys = y.ravel(); us = U[:, 0, :].ravel(); o = np.argsort(ys)
        a.plot(us[o], ys[o], '.', color=COL[k], ms=3, label=f'{LAB[k]}  (err vs exact(T) {float(d["err_final_transient"]):.1e})')
    a.set(xlabel='u', ylabel='y', title=f'profile at T={T:g}'); a.legend(fontsize=7); a.grid(alpha=.3)
    # (c) iterations per step
    a = ax[1, 0]
    for k, d in runs.items():
        a.semilogy(np.arange(1, len(d['cg_it'])+1), d['cg_it'], color=COL[k], lw=1, label=f'{LAB[k]}: mean {np.mean(d["cg_it"]):.0f}, max {np.max(d["cg_it"])}')
    a.set(xlabel='time step', ylabel='CG iterations', title='CG iterations per step (tol 1e-10 abs, 1e-8 rel)'); a.legend(fontsize=8); a.grid(alpha=.3)
    # (d) cumulative wall
    a = ax[1, 1]
    for k, d in runs.items():
        cg = np.cumsum(d['cg_t']); bts = np.zeros_like(cg)
        if d['build_t'].size:                      # builds happen every `refresh` steps
            kk = max(1, len(cg)//len(d["build_t"])); bts[::kk][:len(d["build_t"])] = d["build_t"][:len(bts[::kk])]
        bt = np.cumsum(bts)
        a.plot(np.arange(1, len(cg)+1), cg + bt, color=COL[k], lw=1.5, label=f'{LAB[k]}: CG {np.sum(d["cg_t"]):.0f}s + build {np.sum(d["build_t"]):.0f}s')
        if d['build_t'].size: a.plot(np.arange(1, len(cg)+1), cg, color=COL[k], lw=0.8, ls=':')
    a.set(xlabel='time step', ylabel='cumulative wall [s]', title='wall time (numpy backend, Mac; dotted = CG only)'); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(f'2D periodic channel, body-force Poiseuille start-up, Re=100, 4x4 elements N=8, dt={DT:g} (c={1/DT:.0f}), BDF2, production step_bdf', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT, dpi=130); print('wrote', OUT)

if __name__ == '__main__':
    main()
