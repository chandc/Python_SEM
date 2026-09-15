"""CuPy vs PyTorch on the same GPU, same arrays, same operator.

    python scratch/backend_shootout.py cupy  [ex ey nz N]
    python scratch/backend_shootout.py torch [ex ey nz N]

Run BOTH and compare.  Separate processes deliberately: two GPU allocators in
one process interfere, and library init would be charged to whichever went
second.

READ THE TWO HALVES DIFFERENTLY.  The first half times the real solver, so it
answers "which backend should I run today" -- and it is a comparison of
IMPLEMENTATIONS, not libraries.  The CuPy path has hand-written fused
ElementwiseKernels and a GEMM inner product; the torch path is deliberately
unfused (kernels_torch.py: torch has no comparable fusion mechanism) and uses
a 4-D contraction where CuPy uses 5-D.  The second half times primitives, so
it answers "why", and only there is it library-vs-library.

ONE HYPOTHESIS THIS TESTS.  kernels_torch.py records that torch is faster
UNFUSED than batched (5.3 vs 10.1 ms at 88^3) because the 5-D contraction
'pi,eijvk->epjvk' falls off a fast path the 4-D 'pi,eijm->epjm' stays on --
so it merges the field and mode axes.  The CuPy path uses the 5-D form and
its derivatives measure 5x off bandwidth.  If the same merge helps CuPy, that
is ~3 ms of a 12.5 ms iteration.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np

which = (sys.argv[1] if len(sys.argv) > 1 else 'cupy').lower()
ex, ey, nz, N = (int(v) for v in (sys.argv[2:6] or [16, 16, 128, 8]))
import lssem3d; lssem3d.set_backend(which)
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR, deriv as DV
from lssem3d import device as DEV

if which == 'cupy':
    import cupy as bk
    to = lambda a: bk.asarray(np.ascontiguousarray(a))
    sync = lambda: bk.cuda.Stream.null.synchronize()
    dev = bk.cuda.runtime.getDeviceProperties(0)['name'].decode()
    ver = f'cupy {bk.__version__}'
else:
    import torch as bk
    to = lambda a: torch.as_tensor(np.ascontiguousarray(a)).cuda()
    torch = bk
    sync = torch.cuda.synchronize
    dev = torch.cuda.get_device_name(0)
    ver = f'torch {torch.__version__}'

L = 2*np.pi
m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
nk = nz//2 + 1
mask = to(BC.build_mask(m, nk, pin_p=False, nz=nz))
D, fx, fy = to(diff_matrix(N)), to(m.facx), to(m.facy)
kz, wq = to(FR.wavenumbers(nz, L)), to(m.wq)
U = to(np.random.default_rng(0).standard_normal(
    (m.nelem, N+1, N+1, OP.NVAR_R, nk)))
nu, c = 6.25e-4, 1.0/0.0039
rw = to(OP.momentum_row_weights(c))
nbytes = U.nbytes if which == 'cupy' else U.element_size()*U.numel()
size = U.size if which == 'cupy' else U.numel()
BW = 1356e9
print(f'{dev} | {ver}')
print(f'{ex}x{ey} N={N} Nz={nz}: {size/1e6:.2f} M dof, '
      f'field array {nbytes/2**20:.0f} MiB\n')

def t(label, fn, reps=10, bound=None):
    for _ in range(3):
        fn()
    sync(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(); ms = (time.perf_counter()-t0)/reps*1e3
    ex_ = '' if bound is None else f'   (bound {bound:5.2f}, {ms/bound:4.1f}x)'
    print(f'  {label:<30} {ms:7.2f} ms{ex_}')
    return ms

print('THE REAL SOLVER  (implementation vs implementation)')
mv = t('normal_op', lambda: S3.normal_op(U, D, fx, fy, kz, nu, c, m, mask,
                                         wq, 0.0, rw), 10, 20*nbytes/BW*1e3)
b_ = S3.gs(m, U)*mask
def solve(n):
    sync(); t0 = time.perf_counter()
    S3.pcg(b_, D, fx, fy, kz, nu, c, mesh=m, mask=mask, M_inv=None,
           tol=1e-30, max_iter=n, wq=wq, rw=rw, check_every=10)
    sync(); return time.perf_counter()-t0
solve(5)
per = (solve(60)-solve(20))/40*1e3
print(f'  {"CG iteration (differenced)":<30} {per:7.2f} ms')

print('\nPRIMITIVES  (library vs library -- same call, same data)')
Uc = OP.to_complex(U)
t('to_complex', lambda: OP.to_complex(U), 10, 2*nbytes/BW*1e3)
t('gather-scatter', lambda: S3.gs(m, U), 10, 3*nbytes/BW*1e3)
t('elementwise  U*mask', lambda: U*mask, 20, 3*nbytes/BW*1e3)
# DEV.to_device, not a hasattr('device') probe: numpy 2.x arrays HAVE a
# .device attribute, so that probe silently skips the conversion.
mw = DEV.to_device(S3.multiplicity_weight(m, tuple(U.shape)), U)
t('_dot  (as the solver calls it)', lambda: S3._dot(U, U, mw), 20,
  3*nbytes/BW*1e3)
if which == 'cupy':
    t('  raw sum(axis=0..3)', lambda: (U*U*mw).sum(axis=(0, 1, 2, 3)), 20,
      3*nbytes/BW*1e3)
else:
    t('  raw sum(dim=0..3)', lambda: torch.sum(U*U*mw, dim=(0, 1, 2, 3)), 20,
      3*nbytes/BW*1e3)

print('\nDERIVATIVE CONTRACTION  (5-D vs 4-D -- the kernels_torch hypothesis)')
xp = bk if which == 'cupy' else torch
V, K = Uc.shape[-2], Uc.shape[-1]
U4 = Uc.reshape(Uc.shape[0], N+1, N+1, V*K)
d5 = t('5-D  pi,eijvk->epjvk', lambda: xp.einsum('pi,eijvk->epjvk', D.astype(
    Uc.dtype) if which == 'cupy' else D.to(Uc.dtype), Uc), 10)
d4 = t('4-D  pi,eijm->epjm  (merged)', lambda: xp.einsum('pi,eijm->epjm', D.astype(
    Uc.dtype) if which == 'cupy' else D.to(Uc.dtype), U4), 10)
print(f'\n  4-D is {d5/d4:.2f}x the 5-D form on this backend')
print(f'  (kernels_torch.py measured torch preferring 4-D by ~1.9x at 88^3)')
