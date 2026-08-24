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


# ---- fractional step
s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=TOL)
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
print(f'{"step":>5} {"fractional step":>17} {"VVP LSSEM":>13}   ratio')
for i in (0, 1, 5, 10, NSTEP):
    print(f'{i:>5} {fs[i]:>17.3e} {ls[i]:>13.3e}   {fs[i]/max(ls[i],1e-300):>7.1f}x')
print(f'\n  ||div u|| / ||grad u||_F, the ratio 3D_STATUS.md sec 7P.1 uses.')
