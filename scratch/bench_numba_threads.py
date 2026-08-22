"""Does the numba backend still scale across threads?

`parallel.pcg` splits the z-modes over a ThreadPoolExecutor.  That only works if
the kernel releases the GIL -- an njit function without nogil=True would show a
flat line here, which is the exact failure this script exists to catch.
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem3d import backend, bc as BC, operator as OP, solver3d as S3, parallel as PAR


def main(N=8, ex=4, ey=4, nk=16):
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    D = diff_matrix(N)
    kz = np.arange(nk, dtype=float)
    c, nu = 525.0, 1.0/180.0
    mask = BC.build_mask(m, nk, pin_p=True)
    rw = OP.momentum_row_weights(c)
    x = S3.make_continuous(m, np.random.default_rng(0).standard_normal(mask.shape))*mask
    print(f'{"backend":>8} {"workers":>8} {"wall":>9} {"scaling":>9}')
    for name in ('numpy', 'numba'):
        if not backend.available(name):
            continue
        backend.set_backend(name)
        b = S3.normal_op(x, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw)
        diag = S3.jacobi_diagonal_analytic(mask.shape, D, m.facx, m.facy, kz, nu,
                                           c, m, mask, m.wq, rw=rw)
        Minv = S3.jacobi_inverse(diag, mask)
        base = None
        for w in (1, 2, 4, 6):
            kw = dict(mesh=m, mask=mask, M_inv=Minv, tol=1e-4, max_iter=4000,
                      wq=m.wq, rw=rw, workers=w)
            PAR.pcg(b, D, m.facx, m.facy, kz, nu, c, **dict(kw, max_iter=3))
            t = time.perf_counter()
            _, it, _ = PAR.pcg(b, D, m.facx, m.facy, kz, nu, c, **kw)
            el = time.perf_counter() - t
            base = base or el
            print(f'{name:>8} {w:>8} {el:>8.2f}s {base/el:>8.2f}x   ({it} it)')
    backend.set_backend('numpy')


if __name__ == '__main__':
    main()
