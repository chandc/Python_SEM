"""Centreline profiles of the balanced-weighting cavity at dt = 0.1, 1e-2, 1e-3, 1e-4 vs Ghia."""
import os, sys, glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('CAV_RE', '1000')
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import centreline_u, centreline_v, GHIA_X, GHIA_V
from dt_study_balanced import latest

if __name__ == '__main__':
    gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']; o = np.argsort(GHIA_X)
    N = 15; m = build_channel(1.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    runs = [('dt=0.1 (t=151, steady)', 'scratch/ghia_n15_wsqrt_dt1e-1', 'C3'),
            ('dt=1e-2 (t=90)', 'scratch/ghia_n15_w0.1_steady', 'C0'),
            ('dt=1e-3 (restart, t=91.3)', 'scratch/ghia_n15_wsqrt_dt1e-3_from_steady', 'C2'),
            ('dt=1e-4 (restart, t=90.1)', 'scratch/ghia_n15_wsqrt_dt1e-4_from_steady', 'C1'),
            ('legacy dt=1e-2 plateau', 'scratch/ghia_n15_dt1e-2', 'k')]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    for lab, d, c in runs:
        z = np.load(latest(d)); U = z['U0']
        y, u = centreline_u(U, m, N); x, v = centreline_v(U, m, N)
        ru = np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)); rv = np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2))
        ls = ':' if 'legacy' in lab else '-'
        ax[0].plot(u, y, ls, color=c, lw=1.3, label=f'{lab}  rms {ru:.4f}'); ax[1].plot(x, v, ls, color=c, lw=1.3, label=f'{lab}  rms {rv:.4f}')
        ax[2].plot(u, y, ls, color=c, lw=1.3, marker='.', ms=4, label=lab)
    for a in ax[::2]: a.plot(gu, gy, 'o', mfc='none', ms=7, color='r', label='Ghia et al. 1982')
    ax[1].plot(GHIA_X, GHIA_V, 'o', mfc='none', ms=7, color='r', label='Ghia et al. 1982')
    ax[0].set(xlabel='u', ylabel='y', title='u on x = 0.5'); ax[1].set(xlabel='x', ylabel='v', title='v on y = 0.5')
    ax[2].set(xlabel='u', ylabel='y', title='u near the lid (zoom)', xlim=(-0.45, 1.02), ylim=(0.74, 1.0))
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle('Re=1000 cavity, 4x4 N=15, balanced weighting w=√dt: dt = 0.1 / 1e-2 / 1e-3 / 1e-4 vs Ghia')
    fig.tight_layout(); out = 'figs_fosls_vs_fs/ghia_balanced_dt_study.png'; fig.savefig(out, dpi=130); print('wrote', out)
