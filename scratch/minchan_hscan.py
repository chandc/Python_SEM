"""Why 4786 CG iterations?  Isolate h-refinement from dt and resolution.

sec 8.6 priced the minimal channel from a [flat ... 2x] bracket on iteration count
under h-refinement, and flagged it as ASSUMED.  The first timed step came in at
4786 CG/step against Stage 5's 650 -- 7x, far outside that bracket.  Everything
else about the two cases differs too (dt, Re_tau, box), so this varies ONE thing:
the number of elements, at fixed N, nz, dt and nu.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v,'1')
SC=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.dirname(SC)); sys.path.insert(0,SC)
import numpy as np
from lssem3d import backend, operator as OP, solver3d as S3, timestep as T
import minchan as M


def one_solve(ex, ey, nz=32, dt=2e-3, N=8):
    s = M.setup(N=N, ex=ex, ey=ey, nz=nz)
    U = M.initial_state(s)
    c = T.implicit_coeff(dt, 0)
    rw = OP.momentum_row_weights(c)
    m = s['m']
    d = S3.jacobi_diagonal_analytic((m.nelem, N+1, N+1, OP.NVAR_R, s['nk']), s['D'],
                                    m.facx, m.facy, s['kz'], s['nu'], c, m,
                                    s['mask'], m.wq, rw=rw)
    Minv = S3.jacobi_inverse(d, s['mask'])
    b = S3.normal_op(U, s['D'], m.facx, m.facy, s['kz'], s['nu'], c, m,
                     s['mask'], m.wq, 0.0, rw)
    t0 = time.perf_counter()
    _, it, _ = S3.pcg(b, s['D'], m.facx, m.facy, s['kz'], s['nu'], c, mesh=m,
                      mask=s['mask'], M_inv=Minv, tol=1e-6, max_iter=40000,
                      wq=m.wq, rw=rw)
    return it, time.perf_counter()-t0, m.nelem


def main():
    backend.set_backend('numba')
    print(f'{"elems":>10} {"nelem":>6} {"h_x":>7} {"CG":>7} {"wall":>8}  (N=8, nz=32, dt=2e-3)')
    prev = None
    for ex, ey in ((2, 6), (3, 9), (4, 12), (6, 18)):
        it, w, ne = one_solve(ex, ey)
        r = f'{it/prev:5.2f}x' if prev else '  --  '
        print(f'{f"{ex}x{ey}":>10} {ne:6d} {M.LX/ex:7.3f} {it:7d} {w:7.1f}s  {r}')
        prev = it


if __name__ == '__main__':
    main()
