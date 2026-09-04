"""PMG vs Jacobi INSIDE CG on the channel operator -- the fair comparison.

An earlier attempt ran each preconditioner as a stationary iteration
x <- x + M(b - Ax) and reported both as "stalled".  That test was wrong: an
undamped Richardson sweep needs rho(I - MA) < 1, which neither satisfies on a
normal-equations operator, and Jacobi "diverged" at 1.98 in it while working
perfectly well inside CG at 6026 iterations.  Preconditioner quality for CG is
about the SPECTRUM of M^-1 A, not about M being a convergent splitting.

Same A, same b, same tolerance; count iterations.  That is the only number that
decides whether run01 should be on point Jacobi.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

TOL, CAP = 1e-8, 8000

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, precond as P3, solver3d as S3, timestep as T
    import channel3d as C, minchan as MC
    s = MC.setup(); dt = 8.0e-4
    cc = T.implicit_coeff(dt, 0); rw = OP.momentum_row_weights(cc)
    print(f'channel N=8, 108 elems, nk={s["nk"]}, c={cc:.1f}, tol={TOL:g}, cap={CAP}\n',
          flush=True)
    shape = (s['m'].nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    # b MUST lie in range(A).  A random masked vector does not: A is singular on
    # its null directions, CG then has nothing to converge to, and it diverges --
    # measured, Jacobi hit 2.3e+22 on a random b while converging in 6026 on the
    # real one.  Manufacture b = A x_true instead, which is consistent by
    # construction and is what the stage assembly produces.
    rng = np.random.default_rng(0)
    x_true = rng.standard_normal(shape)*s['mask']
    A0 = lambda x: S3.normal_op(x, s['D'], s['m'].facx, s['m'].facy, s['kz'],
                                s['nu'], cc, s['m'], s['mask'], s['m'].wq, 0.0, rw)
    b = A0(x_true); b /= np.linalg.norm(b)

    def go(tag, M, tb):
        t0 = time.perf_counter()
        x, it, rt = S3.pcg(b, s['D'], s['m'].facx, s['m'].facy, s['kz'], s['nu'], cc,
                           mesh=s['m'], mask=s['mask'], M_inv=M, tol=TOL,
                           max_iter=CAP, wq=s['m'].wq, kap=0.0, rw=rw)
        tw = time.perf_counter()-t0
        cap = '  CAPPED' if it >= CAP else ''
        print(f'{tag:26s} build {tb:5.1f}s   CG {it:6d}   solve {tw:7.1f}s'
              f'   final rel res {float(np.max(np.abs(rt))):.2e}{cap}', flush=True)

    t0 = time.perf_counter(); MJ = C.make_precond(s, dt, 0.0, rowweight=True)[0]
    go('Jacobi', MJ, time.perf_counter()-t0)

    for orders, dc in (((8,4,2), 'element'), ((8,4), 'element'),
                       ((8,6,4,2), 'element')):
        t0 = time.perf_counter()
        try:
            M = P3.PMG(s['m'], s['nk'], s['nz'], s['nu'], cc, s['kz'], kap=0.0,
                       rw=rw, orders=orders, deg=6, pin_p=True,
                       direct_coarse=dc, mask=s['mask'])
        except Exception as e:
            print(f'PMG {orders} {dc}: BUILD FAILED {type(e).__name__}: {str(e)[:70]}',
                  flush=True); continue
        go(f'PMG {orders} {"DIRECT" if dc else "cheb"}', M, time.perf_counter()-t0)

if __name__ == '__main__':
    main()
