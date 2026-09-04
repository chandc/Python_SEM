"""Can a CHEAPER V-cycle turn 8.5x fewer iterations into less wall clock?

sec 7T.2: PMG needs 402 CG against Jacobi's 3436, but a deg=6 cycle costs ~16
operator-applies, so it loses 2x on wall.  Cost per cycle is ~2*deg at the fine
level plus ~0.31*2*deg at p=4, so deg is the lever.  Break-even is ~250
iterations at deg=6; at deg=2 a cycle is ~6 applies and break-even moves to ~900.

Reports EFFECTIVE WORK = its * (applies per cycle + 1), which is the quantity a
bandwidth-bound machine pays, alongside measured wall.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 20000

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, precond as P3, solver3d as S3, timestep as T
    import channel3d as C, minchan as MC
    s = MC.setup(); dt = 8e-4
    cc = T.implicit_coeff(dt, 0); rw = OP.momentum_row_weights(cc)
    shape = (s['m'].nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    A0 = lambda x: S3.normal_op(x, s['D'], s['m'].facx, s['m'].facy, s['kz'],
                                s['nu'], cc, s['m'], s['mask'], s['m'].wq, 0.0, rw)
    rng = np.random.default_rng(0)
    b = A0(rng.standard_normal(shape)*s['mask']); b /= np.linalg.norm(b)
    print(f'channel p=8, c={cc:.0f}, tol={TOL:g}.  work = its*(applies/cycle + 1)\n', flush=True)
    print(f'{"precond":>22} {"CG":>6} {"wall":>8} {"appl/cyc":>9} {"work":>8} {"vs Jac":>7}')

    t0 = time.perf_counter()
    MJ = C.make_precond(s, dt, 0.0, rowweight=True)[0]
    _, itj, _ = S3.pcg(b, s['D'], s['m'].facx, s['m'].facy, s['kz'], s['nu'], cc,
                       mesh=s['m'], mask=s['mask'], M_inv=MJ, tol=TOL,
                       max_iter=CAP, wq=s['m'].wq, kap=0.0, rw=rw)
    tj = time.perf_counter()-t0
    wj = itj*2
    print(f'{"Jacobi":>22} {itj:6d} {tj:7.1f}s {1:9d} {wj:8d} {1.0:6.2f}x', flush=True)

    for deg in (1, 2, 3, 4, 6):
        M = P3.PMG(s['m'], s['nk'], s['nz'], s['nu'], cc, s['kz'], kap=0.0, rw=rw,
                   orders=(8,4,2), deg=deg, pin_p=True, direct_coarse=False,
                   mask=s['mask'])
        t0 = time.perf_counter()
        _, it, _ = S3.pcg(b, s['D'], s['m'].facx, s['m'].facy, s['kz'], s['nu'], cc,
                          mesh=s['m'], mask=s['mask'], M_inv=M, tol=TOL,
                          max_iter=CAP, wq=s['m'].wq, kap=0.0, rw=rw)
        tw = time.perf_counter()-t0
        ap = 2*deg + 0.31*2*deg          # fine level + p=4 level
        work = int(it*(ap + 1))
        print(f'{"PMG (8,4,2) deg=%d" % deg:>22} {it:6d} {tw:7.1f}s {ap:9.1f} '
              f'{work:8d} {wj/max(work,1):6.2f}x', flush=True)

if __name__ == '__main__':
    main()
