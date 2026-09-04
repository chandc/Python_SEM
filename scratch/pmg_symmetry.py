"""Is the V-cycle SYMMETRIC in the inner product pcg uses?

CG assumes an SPD preconditioner.  If M is not self-adjoint, CG's short
recurrence loses its optimality and the iteration count blows up -- while each
individual eigenvector still looks perfectly well reduced, because a
non-symmetric M can reduce every eigenvector and still destroy conjugacy.  That
is exactly the contradiction in the data: 0/118 eigenvectors survive a V-cycle,
yet CG needs 402 iterations.

_restrict does  x*mw -> P^T (x) P^T -> gs -> *mask_coarse
_prolong  does  P (x) P -> *mask_fine
No mw and no gs on the prolong side, so these need not be adjoint.

Tests <M r1, r2>_mw == <r1, M r2>_mw directly.  Also checks the transfer pair
<R x, y>_c == <x, P y>_f, which is where any asymmetry must originate.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np

def main(N=8, ex=2, ey=2, cc=5405.4):
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3

    nz, nk = 4, 1
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = 2.0*np.pi; m.compute_global_indices()
    kz = np.array([5.882])
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0)
    nu, D = 1/180., diff_matrix(N)
    rw = OP.momentum_row_weights(cc)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    mw = S3.multiplicity_weight(m, shape)          # the weight pcg's _dot uses
    dot = lambda a, b: float((a*b*mw).sum())

    for orders in ((8, 4, 2), (8, 4)):
        M = P3.PMG(m, nk, nz, nu, cc, kz, kap=0.0, rw=rw, orders=orders, deg=6,
                   pin_p=True, direct_coarse='element', mask=mask)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, m, mask,
                                   m.wq, 0.0, rw)
        rng = np.random.default_rng(0)
        # residuals must be legitimate: in range(A) and consistent across copies
        r1 = A(rng.standard_normal(shape)*mask)
        r2 = A(rng.standard_normal(shape)*mask)
        a, b = dot(M(r1), r2), dot(r1, M(r2))
        print(f'PMG{orders}:  <M r1,r2> = {a: .8e}')
        print(f'{"":14s}<r1,M r2> = {b: .8e}')
        print(f'{"":14s}RELATIVE ASYMMETRY = {abs(a-b)/max(abs(a),abs(b)):.3e}'
              f'{"   <-- NOT SYMMETRIC" if abs(a-b)/max(abs(a),abs(b)) > 1e-8 else "   symmetric"}')
        # positive definite?
        q = dot(M(r1), r1)
        print(f'{"":14s}<M r,r> = {q:.4e}{"  (>0 ok)" if q > 0 else "  <-- NOT POSITIVE"}')

        # where does it come from: is R the adjoint of P?
        lv0, lv1 = M.levels[0], M.levels[1]
        mw1 = S3.multiplicity_weight(lv1.m, lv1.shape)
        x = A(rng.standard_normal(shape)*mask)
        y = rng.standard_normal(lv1.shape)*lv1.mask
        lhs = float((M._restrict(x, 0)*y*mw1).sum())
        rhs = float((x*M._prolong(y, 0)*mw).sum())
        print(f'{"":14s}<R x,y>_c = {lhs: .6e}   <x,P y>_f = {rhs: .6e}   '
              f'rel diff {abs(lhs-rhs)/max(abs(lhs),abs(rhs)):.3e}\n')

if __name__ == '__main__':
    main()
