"""Re=1000 cavity, N=15: balanced-weighting steady march vs legacy vs dt=1 vs Ghia."""
import os, sys, glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('CAV_RE', '1000')
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import centreline_u, centreline_v, GHIA_X, GHIA_V

if __name__ == '__main__':
    gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']; o = np.argsort(GHIA_X)
    m15 = build_channel(1.0, 1.0, 4, 4, 15, bcs=(1, 1, 1, 2)); m15.compute_global_indices()
    m30 = build_channel(1.0, 1.0, 4, 4, 30, bcs=(1, 1, 1, 2)); m30.compute_global_indices()
    bal = sorted(glob.glob('scratch/ghia_n15_w0.1_steady/checkpoint_*.npz'))[-1]
    runs = [('balanced w=√dt, dt=0.01', bal, m15, 15, 'C0', '-', 2.0),
            ('legacy, dt=0.01 (t=90 plateau)', 'scratch/ghia_n15_dt1e-2/final.npz', m15, 15, 'C3', '-', 1.2),
            ('legacy, dt=1 (t=223)', 'scratch/ghia_n15_dt1/final.npz', m15, 15, 'C2', '--', 1.2),
            ('N=30 steady reference', 'scratch/pmg_ghia_cavity_lad_N30_r1.npz', m30, 30, 'k', ':', 1.2)]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    rows = []
    for lab, f, m, N, c, ls, lw in runs:
        z = np.load(f, allow_pickle=True); U = z['U0'] if 'U0' in z else z['U']
        t = float(z['t']) if 't' in z else np.nan
        y, u = centreline_u(U, m, N); x, v = centreline_v(U, m, N)
        ru = np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)); rv = np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2))
        mu = np.max(np.abs(np.interp(gy, y, u) - gu)); mv = np.max(np.abs(np.interp(GHIA_X[o], x, v) - GHIA_V[o]))
        if 'balanced' in lab: lab += f' (t={t:.0f})'
        rows.append((lab, t, ru, rv, mu, mv, u.min(), v.min(), v.max()))
        ax[0].plot(u, y, ls, color=c, lw=lw, label=f'{lab}  rms {ru:.4f}')
        ax[1].plot(x, v, ls, color=c, lw=lw, label=f'{lab}  rms {rv:.4f}')
        ax[2].plot(u, y, ls, color=c, lw=lw, marker='.' if N == 15 else None, ms=4, label=lab)
    ax[0].plot(gu, gy, 'o', mfc='none', ms=7, color='r', label='Ghia et al. 1982')
    ax[1].plot(GHIA_X, GHIA_V, 'o', mfc='none', ms=7, color='r', label='Ghia et al. 1982')
    ax[2].plot(gu, gy, 'o', mfc='none', ms=7, color='r', label='Ghia et al. 1982')
    ax[0].set(xlabel='u', ylabel='y', title='u on the vertical centreline x = 0.5')
    ax[1].set(xlabel='x', ylabel='v', title='v on the horizontal centreline y = 0.5')
    ax[2].set(xlabel='u', ylabel='y', title='u near the lid (zoom, GLL nodes marked)', xlim=(-0.45, 1.02), ylim=(0.74, 1.0))
    for a in ax: a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle('Re=1000 lid-driven cavity, 4x4 elements N=15: balanced weighting vs legacy vs Ghia')
    fig.tight_layout(); out = 'figs_fosls_vs_fs/ghia_balanced_steady.png'; fig.savefig(out, dpi=130)
    print(f'{"run":40s} {"t":>5} {"rms u":>7} {"rms v":>7} {"max|du|":>8} {"max|dv|":>8} {"u_min":>8} {"v_min":>8} {"v_max":>7}')
    for lab, t, ru, rv, mu, mv, umin, vmin, vmax in rows:
        print(f'{lab:40s} {t:5.0f} {ru:7.4f} {rv:7.4f} {mu:8.4f} {mv:8.4f} {umin:8.4f} {vmin:8.4f} {vmax:7.4f}')
    print(f'{"Ghia extrema":40s} {"":>5} {"":>7} {"":>7} {"":>8} {"":>8} {gu.min():8.4f} {GHIA_V.min():8.4f} {GHIA_V.max():7.4f}')
    print('wrote', out)
