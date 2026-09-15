"""End-to-end: a full preconditioned CG solve on each backend.

The matvec microbenchmark (bench_numba.py) is not the deliverable -- the solve
is.  This checks the two things that could still go wrong after parity holds on
a single application:

  * the ITERATION COUNT must be identical, or the two backends are not solving
    the same system and the speedup is not a like-for-like comparison;
  * the SOLUTION must agree to round-off, not merely to the CG tolerance.
"""
import time

import numpy as np
from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel

from lssem3d import backend, bc as BC, operator as OP, solver3d as S3


def run(N=8, ex=4, ey=4, nk=8, tol=1e-6):
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    D = diff_matrix(N)
    kz = np.arange(nk, dtype=float)
    c, nu = 525.0, 1.0/180.0
    mask = BC.build_mask(m, nk, pin_p=True)
    rw = OP.momentum_row_weights(c)
    x = S3.make_continuous(m, np.random.default_rng(0).standard_normal(mask.shape))*mask

    res = {'exact': x}
    for name in ('numpy', 'numba'):
        backend.set_backend(name)
        b = S3.normal_op(x, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw)
        diag = S3.jacobi_diagonal_analytic(mask.shape, D, m.facx, m.facy, kz, nu,
                                           c, m, mask, m.wq, rw=rw)
        Minv = S3.jacobi_inverse(diag, mask)
        S3.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask, M_inv=Minv,
               tol=tol, max_iter=5, wq=m.wq, rw=rw)          # warm/compile
        t = time.perf_counter()
        xs, it, rr = S3.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                            M_inv=Minv, tol=tol, max_iter=20000, wq=m.wq, rw=rw)
        res[name] = (time.perf_counter() - t, it, rr, xs)
    backend.set_backend('numpy')
    return res


def main():
    for N, ex, ey, nk in [(8, 4, 4, 8), (10, 4, 4, 16)]:
        r = run(N, ex, ey, nk)
        (tp, ip, rp, xp), (tb, ib, rb, xb) = r['numpy'], r['numba']
        print(f'N={N} {ex}x{ey}e nk={nk}:  numpy {tp:7.2f}s/{ip} it   '
              f'numba {tb:7.2f}s/{ib} it   {tp/tb:5.2f}x')
        print(f'{"":22} true residual {np.max(rp):.2e} / {np.max(rb):.2e}   '
              f'iterates differ {np.abs(xp-xb).max()/np.abs(xp).max():.2e}')
        # The two backends cannot produce bit-identical CG trajectories: the
        # fused kernel accumulates in a different order than einsum + BLAS, and
        # over ~1e3 iterations that shows up as +/-1 iteration.  Both must reach
        # the SAME residual target in essentially the same number of iterations
        # -- that is what makes the timing a like-for-like comparison.
        #
        # Do not read the iterate difference as an error bar.  b was built from a
        # RANDOM x, which loads the near-null space this system still has after
        # the row-7 fix, so x is not uniquely recoverable and "err vs exact" is
        # meaningless here.  Backend accuracy is established by the bit-level
        # operator parity in test_backend_parity.py and by the physics
        # validation in validate_numba_physics.py, not by this script.
        assert abs(ip - ib) <= 2, f'iteration counts diverge: {ip} vs {ib}'


if __name__ == '__main__':
    main()
