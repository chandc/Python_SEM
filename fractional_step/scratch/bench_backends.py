"""Head-to-head: NumPy vs CuPy vs PyTorch on the same operator, same machine.

    python scratch/bench_backends.py <backend> [<backend> ...]

Run once per container -- the CuPy image is torch-free by design and the NGC
torch image has no CuPy -- and compare the printed numbers.  NumPy is measured
in BOTH so the two runs share a reference and land on one scale.

Fairness rules, each of which changes the answer if skipped:
  * float64 everywhere (this project validates sigma to seven digits);
  * a warm-up call before timing -- both libraries compile on first use;
  * an explicit device synchronise before stopping the clock, or the GPU
    timings measure queue submission rather than work;
  * identical case, identical dof, identical row weights.

SCOPE.  'torch' here is the committed `kernels_torch.py`.  The fused `cuda`
backend was still uncommitted when this branch was taken, so it is NOT in this
comparison and may well be faster than both.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
import lssem3d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR

L = 2*np.pi


def build(N, ex, nz, nu=1/180., c=525.0):
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


def move(be, arrs):
    if be == 'numpy':
        return arrs
    if be == 'cupy':
        import cupy as cp
        return [cp.asarray(a) for a in arrs]
    import torch
    dev = torch.device('cuda')
    return [torch.as_tensor(np.ascontiguousarray(a), device=dev,
                            dtype=torch.float64) for a in arrs]


def sync(be):
    if be == 'cupy':
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    elif be in ('torch', 'cuda'):
        import torch
        torch.cuda.synchronize()


def bench(be, N, ex, nz, reps=10):
    m, D, kz, mask, rw, U, nu, c = build(N, ex, nz)
    lssem3d.set_backend(be)
    Ug, Dg, fxg, fyg, kzg, wqg, mg, rwg = move(
        be, [U, D, m.facx, m.facy, kz, m.wq, mask, rw])
    kw = dict(mesh=m, mask=mg, wq=wqg, kap=0.0, rw=rwg)
    f = lambda: S3.normal_op(Ug, Dg, fxg, fyg, kzg, nu, c, **kw)
    f(); sync(be)
    n = reps if be != 'numpy' else max(3, reps//3)
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    sync(be)
    return (time.perf_counter()-t0)/n, U.size


if __name__ == '__main__':
    cases = [(8, 6, 24), (8, 8, 32), (8, 11, 48), (8, 11, 88)]
    bes = sys.argv[1:] or ['numpy']
    print(f"{'case':>18}{'Mdof':>8}" + ''.join(f'{b:>13}' for b in bes))
    for (N, ex, nz) in cases:
        row, dof = [], 0
        for be in bes:
            try:
                t, dof = bench(be, N, ex, nz)
                row.append(f'{t*1e3:.2f} ms')
            except Exception as e:
                row.append(type(e).__name__)
        print(f'  N={N} {ex}x{ex} nz={nz:<3}{dof/1e6:8.2f}'
              + ''.join(f'{r:>13}' for r in row), flush=True)
