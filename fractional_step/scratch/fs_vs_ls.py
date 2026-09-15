"""PRELIMINARY timing: fractional step vs VVP least-squares, same case.

    python scratch/fs_vs_ls.py [N ne nz nsteps]

NOT Phase 3.  Phase 3 is TGV Re = 800 at 88^3 on the A100 against the run
finishing on the Mac, and needs a GPU backend for the projection path, which is
numpy-only today.  This is the honest preliminary: identical mesh, identical
dt, identical tolerance, same machine, single-threaded numpy.

What it can and cannot say.  The ITERATION ratio should carry over to any
backend -- it is a property of the operators, not the hardware.  The WALL
ratio is CPU-numpy and will shift on a GPU: the least-squares path has hand-
fused kernels and a GEMM inner product from this session's work, while the
projection path has none of that, so this understates it.  Read the iteration
counts as the signal and the wall time as a rough bound.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                     operator as OP, solver3d as S3, timestep as T)
import tgv_gpu_run as TG
import fs_phase2 as F2

a = [int(v) for v in sys.argv[1:5]] or [8, 4, 16, 5]
N, ne, nz, nsteps = (a + [8, 4, 16, 5][len(a):])[:4]
NU, DT, TOL = 0.01, 0.01, 1e-8
print(f'{ne}x{ne} elements, N={N}, Nz={nz}, dt={DT}, tol={TOL:.0e}, '
      f'{nsteps} steps, numpy\n')

# ---------------------------------------------------------------- VVP LSSEM
ops = TG.Ops('numpy')
cfg = dict(nu=NU, N=N, ex=ne, ey=ne, nz=nz, tend=1.0, snap=1.0, cfl=1.0,
           tol=TOL)
s = TG.setup(cfg, np, ops)
U = TG.ic_tgv(s)
Np_ = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
Minv, rws = TG.precond(s, DT)
t0 = time.perf_counter(); ls_it = 0
for i in range(nsteps):
    for k in range(T.NSTAGE):
        U, Np_, it = TG.stage(s, U, Np_, k, DT, Minv, rws[k], TOL,
                              check_every=1)
        ls_it += it
ls_t = time.perf_counter() - t0
print(f'  VVP LSSEM        {ls_it/nsteps:8.0f} CG iters/step   '
      f'{ls_t/nsteps:8.3f} s/step')

# -------------------------------------------------------- fractional step
sf = F2.build(N=N, ne=ne, nz=nz, nu=NU, tol=TOL)
Uc = F2.ic_tgv(sf)
pc = np.zeros((sf['m'].nelem, N+1, N+1, 1, sf['nk']), dtype=complex)
Nprev = np.zeros((sf['m'].nelem, N+1, N+1, 3, sf['nk']), dtype=complex)
pre = [HH.fdm_preconditioner(sf['m'], N,
                             T.implicit_coeff(DT, k) + NU*(sf['kz']**2),
                             NU, sf['mask_u'], 6, sf['nk'])
       for k in range(T.NSTAGE)]
t0 = time.perf_counter(); fs_u = fs_p = 0
for i in range(nsteps):
    for k in range(T.NSTAGE):
        sf['Mu'] = pre[k]
        Nk = -CV.convective(Uc, sf['D'], sf['m'].facx, sf['m'].facy,
                            sf['kz'], sf['nz'])
        Uc, pc, inf = PJ.substage(sf, Uc, pc, Nk, Nprev, k, DT)
        Nprev = Nk
        fs_u += inf[0]; fs_p += inf[2]
fs_t = time.perf_counter() - t0
print(f'  fractional step  {(fs_u+fs_p)/nsteps:8.0f} CG iters/step   '
      f'{fs_t/nsteps:8.3f} s/step'
      f'   (velocity {fs_u/nsteps:.0f}, pressure {fs_p/nsteps:.0f})')
print(f'\n  iterations: {ls_it/max(fs_u+fs_p,1):.1f}x fewer for fractional step')
print(f'  wall clock: {ls_t/max(fs_t,1e-9):.2f}x faster for fractional step')
print('\n  The iteration ratio is a property of the operators and should carry\n'
      '  to any backend.  The wall ratio is numpy-CPU and UNDERSTATES the\n'
      '  least-squares path, which has fused kernels and a GEMM inner product\n'
      '  this session added; the projection path has neither yet.')
