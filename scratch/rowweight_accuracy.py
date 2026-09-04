"""What does w_mom=100 COST in accuracy?

scratch/rowweight_opt.py: w_mom=100 gives 4.28x fewer CG iterations at every c
from 525 to 15000, with the same c^-2 scaling as the default -- the inherited
constant is simply 100x too small.  But the row weights ARE the least-squares
functional and the discrete system is overdetermined (8 rows, 7 unknowns), so a
different weighting picks a different discrete solution.

That opt script's |dU| column was NOT an accuracy measure: its RHS was built as
b = A x_true with A depending on rw, so the answer is x_true by construction and
the drift it showed was CG wandering in the null space.

Here: one real RKW3 step from the channel seed, identical initial state, both
weightings solved TIGHT (1e-11), and compare.  Same physics, same dt, same
everything -- the difference is what the weighting costs.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

SEED = os.path.join(_R, 'scratch', 'fs_seed', 'seed_ckpt.npz')

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, deriv as DV
    import minchan as MC
    _orig = OP.momentum_row_weights
    def force(wm):
        OP.momentum_row_weights = (lambda c, w7=None, w_mom=None, w_vort=None, _w=wm:
                                   _orig(c, w7=w7 if w7 is not None else 1e-4,
                                         w_mom=_w, w_vort=w_vort))
    s = MC.setup(); U0 = np.load(SEED)['U']; dt = 8e-4
    res = {}
    print(f'channel 6x18 N=8, one RKW3 step from the seed, dt={dt:g}, tol=1e-11\n', flush=True)
    print(f'{"w_mom":>7} {"CG":>7} {"wall":>8}   {"rms div u":>11} {"rms div om":>11}')
    for wm in (1.0, 30.0, 100.0):
        force(wm)
        Minv = MC._precond(s, dt)
        Np = np.zeros(OP.to_complex(U0).shape[:-2] + (3, s['nk']), dtype=complex)
        t0 = time.perf_counter()
        U1, _, it = MC.advance(s, U0.copy(), Np, dt, Minv, tol=1e-11, max_iter=40000)
        tw = time.perf_counter()-t0
        C = OP.to_complex(U1)
        du = (DV.ddx(C[..., OP.U_:OP.U_+1, :], s['D'], s['m'].facx)
              + DV.ddy(C[..., OP.V_:OP.V_+1, :], s['D'], s['m'].facy)
              + 1j*s['kz']*C[..., OP.W_:OP.W_+1, :])
        do = (DV.ddx(C[..., OP.OX_:OP.OX_+1, :], s['D'], s['m'].facx)
              + DV.ddy(C[..., OP.OY_:OP.OY_+1, :], s['D'], s['m'].facy)
              + 1j*s['kz']*C[..., OP.OZ_:OP.OZ_+1, :])
        nrm = lambda a: float(np.sqrt((np.abs(a)**2).mean()))
        res[wm] = (U1, it, nrm(du), nrm(do))
        print(f'{wm:7.0f} {it:7d} {tw:7.1f}s   {nrm(du):11.4e} {nrm(do):11.4e}', flush=True)
    U1 = res[1.0][0]
    sc = np.abs(U1).max()
    print()
    for wm in (30.0, 100.0):
        d = np.abs(res[wm][0]-U1).max()/sc
        print(f'  w_mom={wm:.0f} vs default: max|dU|/max|U| = {d:.3e}   '
              f'CG {res[1.0][1]} -> {res[wm][1]} ({res[1.0][1]/res[wm][1]:.2f}x)')
    print('\n  dt=8e-4 with u~15 means one step moves the field ~1e-2 in relative terms,')
    print('  so a difference well below that is inside the timestep error.')

if __name__ == '__main__':
    main()
