"""Does fixing PMG's mask change its performance?

    python scratch/pmg_bench.py [Nlist] [ne] [nz]

3D_STATUS.md 7K measured PMG's iteration ratio PINNED at 7.3-7.4x across
N = 8/12/16 -- growing with N in lockstep with Jacobi rather than staying flat
as multigrid should -- and closed the chapter on wall-clock grounds.

That measurement was taken with PMG building its own level masks via
build_mask(pin_p=True), which pins pressure at EVERY Fourier mode.  The driver
pins it at k = 0 only, which is what the physics needs: for k != 0 the ik*p
term in the z-momentum row already determines pressure uniquely.  On a
3x3 N=8 Nz=16 mesh that left 60 dofs FREE in the solve but PINNED in the
preconditioner -- 32 pressure real parts and their imaginary partners -- so the
V-cycle returned exactly zero on 60 directions CG was searching.  M was
singular on the space CG operates in.

The symmetry gate cannot see this: the old V-cycle is symmetric to 2.95e-17
AND singular.  Symmetry is not correctness.

Whether 60 dofs out of ~82000 explains a pinned 7.4x is not obvious either way
-- 0.07% of the space, but the WORST-conditioned 0.07%, pressure carrying the
1/c^2 weight at exactly the modes where it is least controlled.  Measure it.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

backend = 'numpy'
if '--backend' in sys.argv:
    backend = sys.argv[sys.argv.index('--backend') + 1]
argv = [a for a in sys.argv[1:] if not a.startswith('-')
        and a not in ('cupy', 'numpy', 'torch')]
Ns = [int(v) for v in (argv[0].split(',') if argv else ['8'])]
ne = int(argv[1]) if len(argv) > 1 else 3
nz = int(argv[2]) if len(argv) > 2 else 16

if backend != 'numpy':
    sys.exit(
        f"PMG has no GPU path -- run this without --backend (numpy is the\n"
        f"default).  _Level passes HOST mesh.facx / mesh.wq / D / kz / mask\n"
        f"straight into normal_op, so a cupy kernel receives numpy arrays and\n"
        f"raises 'Unsupported type numpy.ndarray'.  DirectCoarse is host-only\n"
        f"too: numpy loops, a Cholesky, and a dense basis construction.\n\n"
        f"This does not weaken the comparison -- CG ITERATION COUNTS do not\n"
        f"depend on the backend.  It is, though, a practical argument against\n"
        f"PMG independent of its convergence: the production solver is\n"
        f"GPU-resident, and a host round trip inside the CG loop was measured\n"
        f"at 21.9x the matvec (TORCH_VERIFY_PLAN.md V3).  A V-cycle per\n"
        f"iteration would pay that every iteration.")

import lssem3d; lssem3d.set_backend(backend)
import tgv_gpu_run as TG
from lssem3d import operator as OP, solver3d as S3, timestep as T, precond as PC

ops = TG.Ops(backend)
dt, CAP = 0.0039, 20000
print(f'backend={backend}, {ne}x{ne} elements, Nz={nz}, dt={dt}\n')
print(f'{"N":>3} {"Jacobi":>8} {"PMG old":>9} {"PMG fixed":>10} '
      f'{"old ratio":>10} {"fixed ratio":>12}')
for N in Ns:
    cfg = dict(nu=6.25e-4, N=N, ex=ne, ey=ne, nz=nz, tend=1.0, snap=1.0,
               cfl=1.0, tol=1e-6)
    s = TG.setup(cfg, ops.t, ops)
    U0 = ops.to_dev(TG.ic_tgv(s))
    Np0 = ops.zeros_c(tuple(OP.to_complex(U0).shape[:-2]) + (3, s['nk']),
                      OP.to_complex(U0))
    c = T.implicit_coeff(dt, 0)
    rw = OP.momentum_row_weights(c)
    shape = (s['m'].nelem, N+1, N+1, OP.NVAR_R, s['nk'])
    dj = S3.jacobi_diagonal_analytic(shape, s['D'], s['m'].facx, s['m'].facy,
                                     s['kz'], cfg['nu'], c, s['m'], s['mask'],
                                     s['m'].wq, 0.0, rw=rw)
    Mj = ops.to_dev(S3.jacobi_inverse(dj, s['mask']))
    orders = (N, N//2, 2) if N >= 8 and (N//2) > 2 else (N, 2)
    common = dict(rw=rw, orders=orders, deg=6)
    old = PC.PMG(s['m'], s['nk'], nz, cfg['nu'], c, s['kz'], **common)
    new = PC.PMG(s['m'], s['nk'], nz, cfg['nu'], c, s['kz'],
                 mask=s['mask'], **common)
    rwd = ops.to_dev(rw)
    r = []
    for M in (Mj, old, new):
        _, _, it = TG.stage(s, U0, Np0, 0, dt, {0: M}, rwd, cfg['tol'],
                            max_iter=CAP, check_every=10)
        r.append(it)
    print(f'{N:>3} {r[0]:>8} {r[1]:>9} {r[2]:>10} '
          f'{r[0]/max(r[1],1):>10.2f} {r[0]/max(r[2],1):>12.2f}', flush=True)
print('\n  ratios are Jacobi/PMG -- above 1 means PMG needs fewer iterations.')
print('  7K measured 7.4x with the OLD (mask-inconsistent) preconditioner.')
