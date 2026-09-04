"""What is the warm start worth?

`channel3d.stage` implements it fully -- a dict keyed by stage index, cached
solution passed to pcg as x0, new one stored -- and `minchan.advance`'s docstring
even advertises it.  `minchan.run` has never passed it.  At dt=8e-4 successive
stage solves are highly correlated, so x0 from the previous step should start CG
most of the way there.

Production settings: tol=1e-6 (the measured policy, sec 7F), 17 modes, 6x18 N=8.
Counts are max-over-stages, as channel3d.step reports them.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

NSTEP, TOL = 6, 1e-6

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP
    import minchan as MC
    s = MC.setup(); dt = 8e-4
    U0 = np.load(os.path.join(_R, 'scratch/fs_seed/seed_ckpt.npz'))['U']
    Minv = MC._precond(s, dt)
    print(f'channel 6x18 N=8, dt={dt:g}, tol={TOL:g}, {NSTEP} steps\n', flush=True)
    out = {}
    for tag, use_warm in (('cold (production)', False), ('WARM START', True)):
        U = U0.copy()
        Np = np.zeros(OP.to_complex(U0).shape[:-2] + (3, s['nk']), dtype=complex)
        warm = {} if use_warm else None
        its, t0 = [], time.perf_counter()
        for i in range(NSTEP):
            kw = dict(warm=warm) if use_warm else {}
            U, Np, it = MC.advance(s, U, Np, dt, Minv, tol=TOL, max_iter=30000, **kw)
            its.append(int(it))
        tw = time.perf_counter()-t0
        out[tag] = (its, tw)
        print(f'{tag:18s} its {its}  total {sum(its):6d}  wall {tw:7.1f}s', flush=True)
    (ic, tc), (iw, tww) = out['cold (production)'], out['WARM START']
    print(f'\n  steady-state (steps 2-{NSTEP}):  cold {sum(ic[1:])/len(ic[1:]):7.0f}   '
          f'warm {sum(iw[1:])/len(iw[1:]):7.0f}   '
          f'-> {sum(ic[1:])/max(sum(iw[1:]),1):.2f}x fewer iterations')
    print(f'  wall over all {NSTEP} steps: {tc:.1f}s -> {tww:.1f}s   = {tc/tww:.2f}x')
    print('\n  NOTE step 1 is cold in both cases -- the warm cache is empty.')

if __name__ == '__main__':
    main()
