"""Controlled A/B: cheb-coarse vs the REFERENCE DirectCoarse vs my DirectCoarseE,
all inside PMG, on a mesh small enough that the reference (O(global dof) probing)
is affordable.

An exact coarse solve must not need MORE CG iterations than a polynomial
smoother.  On the channel it did (441 DIRECT vs 402 cheb).  Either DirectCoarseE
is wrong, or the effect is real and belongs to the coarse OPERATOR rather than
the coarse SOLVER -- PMG rediscretises at order pc instead of forming the
Galerkin product, and solving a rediscretised coarse operator exactly can
over-correct.  Running the reference alongside separates those two.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3

    N, ex, ey, nz = 8, 3, 3, 4
    nk = nz//2 + 1
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = 2.0*np.pi; m.compute_global_indices()
    kz = np.arange(nk)*2.0
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0)
    nu = 1/180.
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)

    for cc in (525.0, 5405.4):
        rw = OP.momentum_row_weights(cc)
        A = lambda x: S3.normal_op(x, diff_matrix(N), m.facx, m.facy, kz, nu, cc,
                                   m, mask, m.wq, 0.0, rw)
        rng = np.random.default_rng(0)
        b = A(rng.standard_normal(shape)*mask); b /= np.linalg.norm(b)
        print(f'\n--- c = {cc:g}  ({ex}x{ey} N={N} nz={nz}, {int(mask.sum())} free dof) ---')
        for tag, dc in (('cheb coarse', False), ('DirectCoarse  (ref)', True),
                        ('DirectCoarseE (new)', 'element')):
            t0 = time.perf_counter()
            M = P3.PMG(m, nk, nz, nu, cc, kz, kap=0.0, rw=rw, orders=(8, 4, 2),
                       deg=6, coarse_deg=10, pin_p=True, direct_coarse=dc, mask=mask)
            tb = time.perf_counter()-t0
            x, it, rt = S3.pcg(b, diff_matrix(N), m.facx, m.facy, kz, nu, cc,
                               mesh=m, mask=mask, M_inv=M, tol=1e-10,
                               max_iter=4000, wq=m.wq, kap=0.0, rw=rw)
            print(f'  {tag:22s} build {tb:6.2f}s   CG {it:5d}   '
                  f'res {float(np.max(np.abs(rt))):.2e}')

if __name__ == '__main__':
    main()
