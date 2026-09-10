"""rms distance to Ghia (u on x=0.5, v on y=0.5) of the N=15 cavity steady/plateau
state as a function of dt: legacy weighting vs balanced w=sqrt(dt)."""
import os, sys
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
    def rms(d):
        z = np.load(latest(d)); U = z['U0']
        y, u = centreline_u(U, m, N); x, v = centreline_v(U, m, N)
        return (np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)),
                np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2)), float(z['t']))
    legacy = [(1.0, 'scratch/ghia_n15_dt1'), (0.1, 'scratch/ghia_n15_dt0.1'), (0.03, 'scratch/ghia_n15_dt0.03'), (0.01, 'scratch/ghia_n15_dt1e-2')]
    balanced = [(0.1, 'scratch/ghia_n15_wsqrt_dt1e-1'), (0.01, 'scratch/ghia_n15_w0.1_steady'),
                (0.001, 'scratch/ghia_n15_wsqrt_dt1e-3_from_steady'), (0.0001, 'scratch/ghia_n15_wsqrt_dt1e-4_from_steady')]
    L = [(dt,) + rms(d) for dt, d in legacy]; B = [(dt,) + rms(d) for dt, d in balanced]
    ref_u, ref_v = 3.21e-3, 6.66e-3          # converged N=30 (dt=1) reference
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for a, col, nm in ((ax[0], 1, 'rms u vs Ghia (x = 0.5)'), (ax[1], 2, 'rms v vs Ghia (y = 0.5)')):
        a.loglog([r[0] for r in L], [r[col] for r in L], 'ks--', ms=7, label='legacy (momentum row × dt)')
        a.loglog([r[0] for r in B], [r[col] for r in B], 'o-', color='C0', ms=7, label='balanced (w_mom = w_mass = √dt)')
        a.axhline(ref_u if col == 1 else ref_v, color='C3', ls=':', label='N=30 reference (resolution floor)')
        a.set(xlabel='dt', ylabel='rms', title=nm); a.grid(alpha=.3, which='both'); a.legend(fontsize=8)
        a.set_ylim(2e-3, 5e-2)
    fig.suptitle('Re=1000 cavity, 4x4 N=15: steady-state distance to Ghia as a function of dt')
    fig.tight_layout(); out = 'figs_fosls_vs_fs/ghia_rms_vs_dt.png'; fig.savefig(out, dpi=130)
    print(f'{"weighting":10s} {"dt":>7} {"t":>6} {"rms u":>7} {"rms v":>7}')
    for nm, rows in (('legacy', L), ('balanced', B)):
        for dt, ru, rv, t in rows: print(f'{nm:10s} {dt:7g} {t:6.1f} {ru:7.4f} {rv:7.4f}')
    print('wrote', out)
