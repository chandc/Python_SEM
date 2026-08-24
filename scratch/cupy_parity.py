"""CuPy backend parity + speed against the NumPy reference.

    docker run --rm --gpus all -v "$PWD":/work -w /work lssem-cupy:latest \
           python scratch/cupy_parity.py
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, cupy as cp
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
import lssem3d
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR

L = 2*np.pi


def case(N=8, ex=6, nz=24, nu=1/180., c=525.0):
    m = build_channel(L, L, ex, ex, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)
    D = diff_matrix(N); kz = FR.wavenumbers(nz, L)
    rw = OP.momentum_row_weights(c)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    U = S3.make_continuous(m, np.random.default_rng(0).standard_normal(shape))*mask
    return m, D, kz, mask, rw, U, nu, c


def main():
    m, D, kz, mask, rw, U, nu, c = case()
    args_np = (D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw)
    lssem3d.set_backend('numpy')
    ref = S3.normal_op(U, *args_np)
    lssem3d.set_backend('cupy')
    g = cp.asarray
    args_cp = (g(D), g(m.facx), g(m.facy), g(kz), nu, c, m, g(mask), g(m.wq),
               0.0, g(rw))
    out = S3.normal_op(g(U), *args_cp)
    assert isinstance(out, cp.ndarray), 'result left the device!'
    rel = float(cp.abs(out - g(ref)).max())/float(np.abs(ref).max())
    print(f'normal_op parity  max rel err = {rel:.3e}   ({"PASS" if rel < 1e-12 else "FAIL"})')

    def timeit(f, n=10):
        f(); cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        cp.cuda.Stream.null.synchronize()
        return (time.perf_counter()-t0)/n
    lssem3d.set_backend('numpy')
    t_np = timeit(lambda: S3.normal_op(U, *args_np), 3)
    lssem3d.set_backend('cupy')
    Ug = g(U)
    t_cp = timeit(lambda: S3.normal_op(Ug, *args_cp), 10)
    dof = U.size
    print(f'matvec  numpy(host) {t_np*1e3:8.1f} ms   cupy(GB10) {t_cp*1e3:8.1f} ms'
          f'   speedup {t_np/t_cp:5.2f}x   ({dof/1e6:.2f} M dof)')


if __name__ == '__main__':
    main()
