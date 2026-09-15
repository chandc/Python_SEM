"""Is the fractional-step path less accurate in the divergence condition?

    python scratch/fs_divergence.py

A concrete mechanism says it should be.  The projection gives
u^{n+1} = uhat - dt*grad(phi) with grad^2 phi = div(uhat)/dt, so

    div u^{n+1} = div(uhat) - dt * div(grad(phi))

which vanishes ONLY if the discrete div(grad) equals the Laplacian the Poisson
solve inverts.  Here the divergence is STRONG form (ddx + ddy + i*kz) and the
Poisson operator is the WEAK stiffness matrix K.  Different operators, so the
cancellation is inexact by construction.

LSSEM has no projection at all: div u is a weighted ROW of the least-squares
functional, penalised rather than enforced.  3D_STATUS.md sec 7P.1 records it
holding div u / |grad u| at 5e-09 on the Stokes case -- and 1.1e-01 on a run
whose INITIAL condition carried divergence, since a penalty does not project.

So the two fail differently, and which is better is a measurement.  Same mesh,
same dt, same initial condition, TGV with convection.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem3d import (project as PJ, helmholtz as HH, convect as CV, deriv as DV,
                     operator as OP, solver3d as S3, timestep as T)
import fs_phase2 as F2
import tgv_gpu_run as TG

N, NE, NZ, NU, DT, TOL = 8, 3, 8, 0.01, 0.01, 1e-10
NSTEP = 20


def divnorm(Uc, D, fx, fy, kz, wq):
    """||div u|| / ||grad u||_F -- the ratio 7P.1 uses, so the numbers compare."""
    u, v, w = (Uc[..., i:i+1, :] for i in range(3))
    d = DV.ddx(u, D, fx) + DV.ddy(v, D, fy) + 1j*kz*w
    g2 = 0.0
    for q in (u, v, w):
        for t in (DV.ddx(q, D, fx), DV.ddy(q, D, fy), 1j*kz*q):
            g2 += float(np.sum(np.abs(t)**2 * wq[..., None, None]))
    dn = float(np.sum(np.abs(d)**2 * wq[..., None, None]))
    return np.sqrt(dn/max(g2, 1e-300))


# ---- is the D.G operator symmetric?  CG needs it, and LGL summation-by-parts
# only gives it on a periodic seam.
_s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=TOL)
_m, _D = _s['m'], _s['D']
_mk, _mw = _s['mask_p'], S3.multiplicity_weight(_s['m'], _s['mask_p'].shape)
_rng = np.random.default_rng(0)
_x = S3.gs(_m, _rng.standard_normal(_mk.shape))*_mk
_y = S3.gs(_m, _rng.standard_normal(_mk.shape))*_mk
_A = lambda v: HH.apply_dg(v, _D, _m.facx, _m.facy, _m.wq, _s['kz'], _m, _mk)
_num = abs(np.sum(_x*_A(_y)*_mw) - np.sum(_y*_A(_x)*_mw))
print(f'D.G symmetry (continuous probes): '
      f'{_num/abs(np.sum(_x*_A(_x)*_mw)):.2e}')
print(f'D.G positive:  x^T A x = {np.sum(_x*_A(_x)*_mw):.4e}\n')

# ---- fractional step, both pressure operators
for dg in (False, True):
  s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=TOL)
  s['dg_pressure'] = dg
  Uc = F2.ic_tgv(s)
  pc = np.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
  Np_ = np.zeros((s['m'].nelem, N+1, N+1, 3, s['nk']), dtype=complex)
  pre = [HH.fdm_preconditioner(s['m'], N, T.implicit_coeff(DT, k)+NU*(s['kz']**2),
                               NU, s['mask_u'], 6, s['nk'])
         for k in range(T.NSTAGE)]
  D, fx, fy, kz, wq = s['D'], s['m'].facx, s['m'].facy, s['kz'], s['m'].wq
  fs = [divnorm(Uc, D, fx, fy, kz, wq)]
  for i in range(NSTEP):
    for k in range(T.NSTAGE):
        s['Mu'] = pre[k]
        Nk = -CV.convective(Uc, D, fx, fy, kz, NZ)
        Uc, pc, _ = PJ.substage(s, Uc, pc, Nk, Np_, k, DT)
        Np_ = Nk
    fs.append(divnorm(Uc, D, fx, fy, kz, wq))
  globals()['fs_dg' if dg else 'fs_wk'] = fs

# ---- VVP least squares
ops = TG.Ops('numpy')
cfg = dict(nu=NU, N=N, ex=NE, ey=NE, nz=NZ, tend=1.0, snap=1.0, cfl=1.0, tol=TOL)
sl = TG.setup(cfg, np, ops)
U = TG.ic_tgv(sl)
Npl = np.zeros(OP.to_complex(U).shape[:-2] + (3, sl['nk']), dtype=complex)
Minv, rws = TG.precond(sl, DT)
vel = lambda U: np.ascontiguousarray(
    OP.to_complex(U)[..., [OP.U_, OP.V_, OP.W_], :])
ls = [divnorm(vel(U), D, fx, fy, kz, wq)]
for i in range(NSTEP):
    for k in range(T.NSTAGE):
        U, Npl, _ = TG.stage(sl, U, Npl, k, DT, Minv, rws[k], TOL, check_every=1)
    ls.append(divnorm(vel(U), D, fx, fy, kz, wq))

print(f'TGV, {NE}x{NE} N={N} Nz={NZ}, dt={DT}, tol={TOL:.0e}\n')
print(f'{"step":>5} {"FS weak K":>12} {"FS D.G":>12} {"VVP LSSEM":>12}')
for i in (0, 1, 5, 10, NSTEP):
    print(f'{i:>5} {fs_wk[i]:>12.3e} {fs_dg[i]:>12.3e} {ls[i]:>12.3e}')
print(f'\n  ||div u|| / ||grad u||_F, the ratio 3D_STATUS.md sec 7P.1 uses.')
