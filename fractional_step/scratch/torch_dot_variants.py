"""Does the torch path pay the same starved-reduction cost as CuPy?

CuPy computed the per-mode inner product 17x off bandwidth because ~19 M
inputs reduce to 65 outputs -- roughly one block per output, most of a
108-SM card idle -- and a GEMM against a row of ones fixed it (9.41 ->
0.81 ms).  The shape belongs to the problem, not to CuPy.  But torch's
reduction kernels are tuned differently and may already handle it, so this
MEASURES rather than assumes.  Run on a CUDA box:

    python scratch/torch_dot_variants.py [ex ey nz N]
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, torch

ex, ey, nz, N = (int(v) for v in (sys.argv[1:5] or [16, 16, 128, 8]))
nk, nelem, n, nvar = nz//2 + 1, ex*ey, N + 1, 14
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cpu':
    print('NO CUDA -- correctness only, timings meaningless\n')
else:
    print(f'{torch.cuda.get_device_name(0)}, torch {torch.__version__}\n')
shape = (nelem, n, n, nvar, nk)
g = torch.Generator(device='cpu').manual_seed(0)
mk = lambda: torch.randn(shape, generator=g, dtype=torch.float64).to(dev)
a, b, w = mk(), mk(), mk().abs() + 0.5
M = int(np.prod(shape[:-1]))
print(f'{a.numel()/1e6:.2f} M elements -> {nk} outputs '
      f'(reduction ratio {M:,}:1)\n')

def cur(a, b, w):
    return torch.sum(a*b*w, dim=(0, 1, 2, 3)).reshape(1, 1, 1, 1, nk)

ONES = torch.ones((1, M), dtype=torch.float64, device=dev)
def gemv(a, b, w):
    return (ONES @ (a*b*w).reshape(M, nk)).reshape(1, 1, 1, 1, nk)

def flat(a, b, w):
    return (a*b*w).reshape(M, nk).sum(dim=0).reshape(1, 1, 1, 1, nk)

ref = cur(a, b, w)
for name, fn in (('current  sum(dim=0..3)', cur), ('reshape + sum(dim=0)', flat),
                 ('ones @ X   (cuBLAS)', gemv)):
    got = fn(a, b, w)
    assert got.shape == ref.shape, (name, got.shape, ref.shape)
    err = float((got - ref).abs().max()/ref.abs().max())
    for _ in range(3):
        fn(a, b, w)
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn(a, b, w)
    if dev == 'cuda':
        torch.cuda.synchronize()
    print(f'  {name:<26} {(time.perf_counter()-t0)*50:7.2f} ms   '
          f'rel err {err:.2e}')
print(f'\n  bandwidth bound (3 reads): '
      f'{3*a.numel()*8/1356e9*1e3:.2f} ms')
print('\n  If "current" is already near the bound, torch handles this shape\n'
      '  and the CuPy result does NOT carry over -- report that, do not\n'
      '  assume the 17x.')
