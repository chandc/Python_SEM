"""numpy vs numba backend on the normal operator -- the 99.4% of a step.

Reports per-matvec time, so the number is comparable across problem sizes and
directly proportional to solver wall clock (CG cost is iterations x matvec).
Compilation is warmed up first, or it lands inside the first timing.
"""
import time

import numpy as np
from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel

from lssem3d import backend, bc as BC, operator as OP, solver3d as S3


def bench(N=8, ex=4, ey=4, nk=8, reps=5):
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    D = diff_matrix(N)
    kz = np.arange(nk, dtype=float)
    c, nu = 525.0, 1.0/180.0
    mask = BC.build_mask(m, nk, pin_p=True)
    rw = OP.momentum_row_weights(c)
    U = S3.make_continuous(m, np.random.default_rng(0).standard_normal(mask.shape))*mask
    args = (D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw)

    out = {}
    for name in ('numpy', 'numba'):
        if not backend.available(name):
            continue
        backend.set_backend(name)
        r = S3.normal_op(U, *args)                      # warm up / compile
        t = time.perf_counter()
        for _ in range(reps):
            r = S3.normal_op(U, *args)
        out[name] = (time.perf_counter() - t)/reps
        out[name + '_chk'] = float(np.abs(r).max())
    backend.set_backend('numpy')
    return m, out


def main():
    print(f"{'case':>22} {'numpy':>10} {'numba':>10} {'speedup':>8}  agreement")
    for N, ex, ey, nk in [(6, 2, 2, 1), (8, 4, 4, 8), (10, 4, 4, 16),
                          (12, 6, 6, 16), (8, 8, 8, 32)]:
        m, o = bench(N, ex, ey, nk)
        dof = m.nelem*(N+1)**2*OP.NVAR_R*nk
        s = o['numpy']/o['numba']
        agree = abs(o['numpy_chk'] - o['numba_chk'])/max(o['numpy_chk'], 1e-30)
        print(f"N={N:2d} {ex}x{ey}e nk={nk:2d} {dof/1e6:5.2f}M "
              f"{o['numpy']*1e3:8.1f}ms {o['numba']*1e3:8.1f}ms {s:7.2f}x  {agree:.1e}")


if __name__ == '__main__':
    main()
