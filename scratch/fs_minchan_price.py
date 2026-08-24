"""Price the MINIMAL CHANNEL (Re_tau = 180) on the fractional-step path.

    python scratch/fs_minchan_price.py [--backend cupy]

Configuration copied from scratch/minchan.py so the two paths are comparable:
Jimenez-Moin minimal flow unit, Lx = pi, Ly = 2, Lz = 0.34*pi, 6x18 elements at
N = 8 with Nz = 32, delta = u_tau = 1 and nu = 1/180, driven by a constant body
force f_x = u_tau^2/delta = 1.

WHY THIS CASE AND NOT AN EXTRAPOLATION.  GPU_PORT_PLAN.md sec 0 records the
least-squares path at ~35 days on the Mac for this configuration -- MEASURED,
and replacing an earlier estimate of 14 hours that was wrong by 8x because an
assumed CG-iteration bracket under h-refinement did not hold.  Chaining
20.6x (fractional step vs LSSEM, measured on periodic TGV) onto that would
cross two different flow types, and the channel is exactly where the projection
path is weakest: it has WALLS, where its temporal order is ~1.6 against the
least-squares path's 2.00, and where its pressure Poisson is harder than in a
periodic box.

Pair this with:   LSSEM3D_BACKEND=cupy python scratch/minchan.py price
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

backend = ('cupy' if '--backend' not in sys.argv
           else sys.argv[sys.argv.index('--backend') + 1])
import lssem3d; lssem3d.set_backend(backend)
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                     fourier as FR, solver3d as S3, timestep as T)

RE_TAU, DELTA = 180.0, 1.0
LX, LZ, FX = np.pi, 0.34*np.pi, 1.0
N, EX, EY, NZ = 8, 6, 18, 32
NU, TOL = 1.0/RE_TAU, 1e-6

m = build_channel(LX, 2.0*DELTA, EX, EY, N, bcs=(0, 0, 1, 1))
m.periodic_x = LX
m.compute_global_indices()
nk, n = NZ//2 + 1, N + 1
kz = FR.wavenumbers(NZ, LZ)
mask_u = PJ.build_masks(m, nk, NZ, 3, wall=True)
mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False)
ind = np.zeros(mask_p.shape); ind[0, 0, 0, 0, 0] = 1.0
mask_p[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
if backend == 'cupy':
    import cupy as xp
    g = lambda a: xp.asarray(np.ascontiguousarray(a))
    print(f'GPU  {xp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
else:
    xp = np; g = lambda a: a
v = np.ones(mask_p[..., 0:1, 0:1].shape)*mask_p[..., 0:1, 0:1]
mw1 = S3.multiplicity_weight(m, mask_p.shape)[..., 0:1, 0:1]
D = diff_matrix(N)
s = dict(m=m, D=D, N=N, nz=NZ, nk=nk, nu=NU, kz=kz, lz=LZ, tol=TOL,
         incremental=False, mask_u=g(mask_u), mask_p=g(mask_p),
         Dg=g(D), fxg=g(m.facx), fyg=g(m.facy), wqg=g(m.wq), kzg=g(kz),
         wq3=g(m.wq[..., None, None]), wq1=g(m.wq[..., None, None]),
         mw1=g(mw1), null_kz0=g(v), null_norm=float((v*v*mw1).sum()),
         wall_u=g(PJ.wall_indicator(m, nk, NZ, 3)), ubc=None, backend=backend,
         check_every=None)
# PRESSURE PRECONDITIONER: p-multigrid, not one-level FDM.  On this operator
# FDM is no better than a plain diagonal and both grow like sqrt(elements)
# (254 -> 734 as elements go 8 -> 72), because the slow modes are global.  A
# V-cycle is FLAT at 9 iterations across that whole range.
use_pmg = '--fdm' not in sys.argv
if use_pmg:
    from lssem3d import hpmg
    t0 = time.perf_counter()
    s['Mp'] = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, nk, NZ, wall=False,
                                pin_kz0=True, deg=6, like=s['mask_p'])
    print(f'pressure preconditioner: p-multigrid  '
          f'(setup {time.perf_counter()-t0:.1f}s, amortised over the run)')
else:
    s['Mp'] = HH.fdm_preconditioner(m, N, kz**2, 1.0, s['mask_p'], 2, nk,
                                    like=s['mask_p'])
    print('pressure preconditioner: one-level FDM')

# laminar Poiseuille start, u = u_tau^2/(2 nu) * y(2-y) scaled to Re_tau
yy = np.empty((m.nelem, n, n))
for e in range(m.nelem):
    yy[e] = m.ynod[e][None, :]
up = np.zeros((m.nelem, n, n, 3, NZ))
up[..., 0, :] = (RE_TAU/2.0*(yy*(2.0 - yy)))[..., None]
Uc = g(FR.to_modes(up))
fp = np.zeros((m.nelem, n, n, 3, NZ)); fp[..., 0, :] = FX
Fm = g(FR.to_modes(fp))
pc = xp.zeros((m.nelem, n, n, 1, nk), dtype=complex)
Nprev = xp.zeros((m.nelem, n, n, 3, nk), dtype=complex)

DT = 1.0e-3
pre = [HH.fdm_preconditioner(m, N, T.implicit_coeff(DT, k) + NU*(kz**2), NU,
                             s['mask_u'], 6, nk, like=s['mask_u'])
       for k in range(T.NSTAGE)]
dof = mask_u.size
print(f'Re_tau={RE_TAU:.0f}  {EX}x{EY} elem N={N} Nz={NZ}  '
      f'Lx+={LX*RE_TAU:.0f} Lz+={LZ*RE_TAU:.0f}')
print(f'velocity dof {dof/1e6:.2f} M   dt={DT:g}\n')


def step():
    global Uc, pc, Nprev
    tot = 0
    for k in range(T.NSTAGE):
        s['Mu'] = pre[k]
        Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], NZ) + Fm
        Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
        Nprev = Nk
        tot += inf[0] + inf[2]
    return tot


sync = (lambda: xp.cuda.Stream.null.synchronize()) if backend == 'cupy' \
    else (lambda: None)
step(); sync()
t0 = time.perf_counter(); its = 0
for _ in range(2):
    its += step()
sync()
per = (time.perf_counter() - t0)/2
print(f'PRICE  {per:.2f} s/step   {its/2:.0f} CG iters/step')
for T_ in (10000, 50000, 200000):
    print(f'         {T_:>7d} steps -> {T_*per/3600:>7.1f} h '
          f'({T_*per/86400:.1f} days)')
print('\n  least-squares on the Mac for this configuration: ~35 days MEASURED'
      '\n  (GPU_PORT_PLAN.md sec 0), 4786 CG per STAGE, 114-150 s/step.'
      '\n  Pair with: LSSEM3D_BACKEND=cupy python scratch/minchan.py price')
