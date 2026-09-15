"""Does torch.compile find the Phase 6 wins that were hand-written for CuPy?

    python scratch/torch_compile_experiment.py [ex ey nz N]

CUPY_BACKEND.md Phase 6 took the CuPy CG iteration 12.04 -> 9.62 ms with three
hand-written changes: folding the metric scaling into the fused kernel,
fusing L^T W L so the real<->complex round trip happens twice instead of four
times, and moving the row/quadrature weights inside the kernel.  All three are
elementwise-fusion wins, which is exactly what inductor exists to do.

If torch.compile finds them automatically, torch matches CuPy with none of the
hand-written kernels -- and that is a real argument for torch, since
kernels_torch.py deliberately has no fusion mechanism.

TWO WAYS THIS CAN LEGITIMATELY FAIL, both reported rather than hidden:
  - inductor's complex128 support is limited, and these kernels are complex
    throughout.  A graph break or a silent eager fallback yields "no speedup"
    for a reason that has nothing to do with the idea.
  - compile time is minutes and is NOT included in the per-call timings, so it
    is reported separately.  For a run of thousands of steps it amortises; for
    a short job it does not.

Correctness is checked against the uncompiled result before any timing.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, torch

ex, ey, nz, N = (int(v) for v in (sys.argv[1:5] or [16, 16, 128, 8]))
import lssem3d; lssem3d.set_backend('torch')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d import kernels_torch as KT

if not torch.cuda.is_available():
    sys.exit('needs CUDA')
print(f'{torch.cuda.get_device_name(0)} | torch {torch.__version__}')
L = 2*np.pi
m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
nk = nz//2 + 1
to = lambda a: torch.as_tensor(np.ascontiguousarray(a)).cuda()
mask = to(BC.build_mask(m, nk, pin_p=False, nz=nz))
D, fx, fy = to(diff_matrix(N)), to(m.facx), to(m.facy)
kz, wq = to(FR.wavenumbers(nz, L)), to(m.wq)
U = to(np.random.default_rng(0).standard_normal(
    (m.nelem, N+1, N+1, OP.NVAR_R, nk)))
nu, c = 6.25e-4, 1.0/0.0039
rw = to(OP.momentum_row_weights(c))
print(f'{U.numel()/1e6:.2f} M dof\n')

def t(label, fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter()-t0)/reps*1e3
    print(f'  {label:<34} {ms:7.2f} ms')
    return ms

nop = lambda: S3.normal_op(U, D, fx, fy, kz, nu, c, m, mask, wq, 0.0, rw)
ref = nop().clone()
base_mv = t('normal_op  (eager)', nop)
b_ = S3.gs(m, U)*mask
def solve(n):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    S3.pcg(b_, D, fx, fy, kz, nu, c, mesh=m, mask=mask, M_inv=None,
           tol=1e-30, max_iter=n, wq=wq, rw=rw, check_every=10)
    torch.cuda.synchronize(); return time.perf_counter()-t0
solve(5)
base_it = (solve(60)-solve(20))/40*1e3
print(f'  {"CG iteration (eager, differenced)":<34} {base_it:7.2f} ms')

# --- compile the inner, pure-tensor kernels -----------------------------
# The PUBLIC apply_L/apply_LT branch on isinstance(...) for the numpy parity
# path, which graph-breaks.  _apply_L/_apply_LT take tensors only.
print('\ncompiling _apply_L / _apply_LT ...')
raw_L, raw_LT = KT._apply_L, KT._apply_LT
try:
    t0 = time.perf_counter()
    cL, cLT = torch.compile(raw_L), torch.compile(raw_LT)
    KT._apply_L, KT._apply_LT = cL, cLT
    got = nop()
    torch.cuda.synchronize()
    ctime = time.perf_counter()-t0
    err = float((got - ref).abs().max()/ref.abs().max())
    print(f'  compiled in {ctime:.0f} s, rel err vs eager {err:.2e}')
    if err > 1e-12:
        print('  *** RESULT DIFFERS -- not a usable speedup ***')
    t('normal_op  (compiled)', nop)
    solve(5)
    ci = (solve(60)-solve(20))/40*1e3
    print(f'  {"CG iteration (compiled)":<34} {ci:7.2f} ms')
    print(f'\n  speedup: matvec {base_mv/t("normal_op  (compiled, re-timed)", nop):.2f}x, '
          f'iteration {base_it/ci:.2f}x')
    print(f'  CuPy after hand-written Phase 6: 6.40 ms matvec, 9.62 ms iteration')
except Exception as e:
    KT._apply_L, KT._apply_LT = raw_L, raw_LT
    print(f'  COMPILE FAILED: {type(e).__name__}: {str(e)[:400]}')
    print('  Reported as-is -- inductor has limited complex128 support and these\n'
          '  kernels are complex throughout.  That is a real answer, not a\n'
          '  failed experiment: it means the Phase 6 wins are NOT available to\n'
          '  the torch path for free.')
