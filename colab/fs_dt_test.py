"""Is the projection path stable and accurate at the least-squares time step?

    python colab/fs_dt_test.py [--backend torch] [--graph]

The two codes run at different steps -- 3.5e-4 for the projection, 8e-4 for
least squares -- and cost per eddy turnover scales with 1/dt, so this is worth
2.3x to the comparison.  Both use RKW3 with the same sqrt(3) CFL limit, so there
is no obvious reason the projection needs the smaller step; it may simply never
have been tried.

WHAT IS COMPARED, and it is not the timing.  Each dt is marched to the SAME
physical time from the same restart, and the run reports u_tau, the relative
pointwise divergence and the maximum velocity.  A larger step that is unstable
shows as a growing divergence or a blow-up; one that is merely less accurate
shows as a small shift in u_tau.  The reference is the small step.
"""
import argparse, os, runpy, sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = os.path.join(_R, 'fractional_step')
sys.path.insert(0, FS); sys.path.insert(0, os.path.join(FS, 'scratch'))
os.chdir(FS)
import numpy as np


def run_to(tend, dt, backend, graph, restart):
    argv = ['fs_minchan_stats.py', '--restart', restart, '--backend', backend,
            '--dt', repr(dt), '--tend', repr(tend), '--outdir', f'/tmp/fs_dt_{dt:g}',
            '--consistent']
    if graph:
        argv.append('--graph')
    sys.argv = argv
    G = runpy.run_path(os.path.join('scratch', 'fs_minchan_stats.py'), run_name='__main__')
    import lssem3d
    from lssem3d import project as PJ, device as DEV
    s, Uc = G['s'], G['Uc']
    Uh = DEV.to_host(Uc) if hasattr(DEV, 'to_host') else Uc
    dd = PJ.divergence(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'])
    rdiv = float(np.sqrt((abs(DEV.to_host(dd))**2).sum()/(abs(Uh)**2).sum()))
    return dict(utau=float(G['utau_of'](Uh)), div=rdiv,
                umax=float(np.abs(Uh).max()), t=float(G['t']),
                steps=int(round((tend - 15.9527)/dt)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='torch')
    ap.add_argument('--graph', action='store_true')
    ap.add_argument('--span', type=float, default=0.04, help='physical time to march')
    ap.add_argument('--restart', default=os.path.join(
        _R, 'results/minchan_re180_E/state_t15.95.npz'))
    a = ap.parse_args()
    t0 = float(np.load(a.restart)['t'])
    tend = t0 + a.span
    print(f'marching {a.span} time units from t={t0:.4f} at two step sizes\n')
    out = {}
    for dt in (3.5e-4, 8e-4):
        out[dt] = run_to(tend, dt, a.backend, a.graph, a.restart)
        r = out[dt]
        print(f"  dt={dt:7.1e}  {r['steps']:4d} steps  u_tau={r['utau']:.5f}  "
              f"div={r['div']:.3e}  max|u|={r['umax']:.4f}", flush=True)
    a_, b_ = out[3.5e-4], out[8e-4]
    du = abs(b_['utau'] - a_['utau'])/max(abs(a_['utau']), 1e-30)*100
    print(f"\nu_tau differs by {du:.2f} %; divergence "
          f"{a_['div']:.2e} -> {b_['div']:.2e}")
    if not np.isfinite(b_['umax']) or b_['umax'] > 5*a_['umax']:
        print('VERDICT: the larger step is UNSTABLE -- keep 3.5e-4')
    elif du < 1.0 and b_['div'] < 3*a_['div']:
        print('VERDICT: the larger step holds -- worth 2.3x per eddy turnover')
    else:
        print('VERDICT: stable but less accurate; judge against the 1-3 % '
              'database spread before adopting')
