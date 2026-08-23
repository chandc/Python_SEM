"""Where does the per-matvec HOST time go?

At a tiny problem size the GPU work is negligible, so the wall clock IS the
Python/CUDA dispatch cost -- which is what pins the Colab A100 at a flat
11.45 ms regardless of size.  Measuring it on any GPU therefore tells us what
to fuse; the number is a property of the code, not the device.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, cupy as cp
import lssem3d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d.kernels_cupy import _L0, _LT, _to_complex, _to_real

L = 2*np.pi
g = cp.asarray


def bits(N, ex, nz):
    m = build_channel(L, L, ex, ex, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)
    D = diff_matrix(N); kz = FR.wavenumbers(nz, L)
    c, nu = 525.0, 1/180.
    rw = OP.momentum_row_weights(c)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    U = S3.make_continuous(m, np.random.default_rng(0).standard_normal(shape))*mask
    return (m, g(D), g(m.facx), g(m.facy), g(kz), nu, c, g(m.wq), g(mask),
            g(rw), g(U))


def t(f, n=30):
    f(); cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter()-t0)/n*1e3


if __name__ == '__main__':
    lssem3d.set_backend('cupy')
    for lab, (N, ex, nz) in (('tiny  (dispatch-dominated)', (4, 2, 4)),
                             ('production 6.17 M dof', (8, 11, 88))):
        m, D, fx, fy, kz, nu, c, wq, mask, rw, U = bits(N, ex, nz)
        Uc = _to_complex(U)
        R = OP.apply_L(U, D, fx, fy, kz, nu, c, wq, 0.0, rw)
        print(f'\n{lab}   ({U.size/1e6:.3f} M dof)')
        print(f'   normal_op total   {t(lambda: S3.normal_op(U, D, fx, fy, kz, nu, c, m, mask, wq, 0.0, rw)):8.3f} ms')
        print(f'     apply_L         {t(lambda: OP.apply_L(U, D, fx, fy, kz, nu, c, wq, 0.0, rw)):8.3f} ms')
        print(f'       _L0 only      {t(lambda: _L0(Uc, D, fx, fy, kz, nu, c, 0.0)):8.3f} ms')
        print(f'     apply_LT        {t(lambda: OP.apply_LT(R, D, fx, fy, kz, nu, c, 0.0)):8.3f} ms')
        print(f'     gather-scatter  {t(lambda: S3.gs(m, U)):8.3f} ms')
