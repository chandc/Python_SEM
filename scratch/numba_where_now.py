"""Under numba, where does a CG iteration actually spend its time NOW?

The fused kernels cut apply_L + apply_LT from ~30 passes over the state to ~2.
But a CG iteration also does gather-scatter, two masks, and ~7 full-state vector
operations (axpy, dot, the diagonal preconditioner) -- all still NumPy, all pure
bandwidth.  If those were ~19% of an iteration before, they are ~78% after, and
that is Amdahl INSIDE the iteration.

That would explain the unresolved spread in 3D_STATUS.md sec 7M -- numba worth
6.2x on a bare matvec but only 2.4x end to end -- and it would say exactly where
the next factor is.  Or it would not, and the spread is something else.  Measure.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem3d import backend, bc as BC, operator as OP, solver3d as S3


def timeit(f, reps=20):
    f(); t = time.perf_counter()
    for _ in range(reps):
        r = f()
    return (time.perf_counter() - t)/reps


def main(N=6, ex=3, ey=3, nk=9):
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = 2.0*np.pi; m.compute_global_indices()
    D = diff_matrix(N)
    kz = np.arange(nk, dtype=float)
    c, nu = 600.0, 1.0/180.0
    mask = BC.build_mask(m, nk, pin_p=True)
    rw = OP.momentum_row_weights(c)
    U = S3.make_continuous(m, np.random.default_rng(0).standard_normal(mask.shape))*mask
    V = U.copy(); Minv = np.ones_like(U)
    mw = S3.multiplicity_weight(m, U.shape)

    print(f'N={N} {ex}x{ey}e nk={nk}  (the Stage 5 channel shape)')
    print(f'{"piece":>34} {"numpy":>10} {"numba":>10}')
    rows = {}
    for name in ('numpy', 'numba'):
        backend.set_backend(name)
        rows[name] = {
            'apply_L + apply_LT (the kernels)': timeit(
                lambda: OP.apply_LT(OP.apply_L(U, D, m.facx, m.facy, kz, nu, c,
                                               m.wq, 0.0, rw),
                                    D, m.facx, m.facy, kz, nu, c, 0.0)),
            'gather-scatter gs()': timeit(lambda: S3.gs(m, U)),
            'full normal_op': timeit(
                lambda: S3.normal_op(U, D, m.facx, m.facy, kz, nu, c, m, mask,
                                     m.wq, 0.0, rw)),
            'CG vector ops (x+a*p, r-a*Ap, z=r*Minv)': timeit(
                lambda: (U + 0.5*V, U - 0.5*V, U*Minv)),
            'CG inner products (2x _dot)': timeit(
                lambda: (S3._dot(U, V, mw), S3._dot(V, V, mw))),
        }
    backend.set_backend('numpy')
    for k in rows['numpy']:
        print(f'{k:>34} {rows["numpy"][k]*1e3:9.2f}ms {rows["numba"][k]*1e3:9.2f}ms')

    for name in ('numpy', 'numba'):
        r = rows[name]
        it = r['full normal_op'] + r['CG vector ops (x+a*p, r-a*Ap, z=r*Minv)'] \
             + r['CG inner products (2x _dot)']
        out = it - r['full normal_op']
        print(f'\n  {name}: one CG iteration ~ {it*1e3:.2f}ms, of which '
              f'{out*1e3:.2f}ms ({100*out/it:.0f}%) is OUTSIDE the fused kernel')
    a = rows['numpy']; b = rows['numba']
    ita = a['full normal_op'] + a['CG vector ops (x+a*p, r-a*Ap, z=r*Minv)'] + a['CG inner products (2x _dot)']
    itb = b['full normal_op'] + b['CG vector ops (x+a*p, r-a*Ap, z=r*Minv)'] + b['CG inner products (2x _dot)']
    print(f'  predicted end-to-end gain from these pieces: {ita/itb:.2f}x '
          f'(measured on the channel: 2.36-2.49x)')


if __name__ == '__main__':
    main()
