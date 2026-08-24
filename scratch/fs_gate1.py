"""PHASE 1b / GATE 1: Stokes decay in a CHANNEL, against sigma = 9.3137399.

    python scratch/fs_gate1.py

The project's own gate, and the first test of the fractional-step path that has
BOTH a non-zero pressure and NO-SLIP WALLS.  Phase 1a established the splitting
is second order on a periodic, zero-pressure case (order 1.99), so a failure
here points at the wall treatment.

The least-squares path passes this at order 2.00:

    dt = 0.01     sigma = 9.3153041   rel err 1.679e-04
    dt = 0.005    sigma = 9.3141300   rel err 4.188e-05
    dt = 0.0025   sigma = 9.3138373   rel err 1.045e-05

Mesh, initial state and energy come from `stokes3d` unchanged, so the two paths
see identical geometry and identical initial data -- only the solve differs.

WALL TREATMENT.  Dirichlet on the intermediate velocity, uhat = 0.  NOTHING on
the pressure correction: homogeneous Neumann is the NATURAL boundary condition
of the weak form, so imposing nothing IS dphi/dn = 0.  The rotational term
-nu*div(uhat) in the pressure update is what should hold the tangential slip to
O(dt^{3/2}) instead of O(dt); if the order degrades, that is the first thing to
distrust, and sec 3.2's consistent condition
dp/dn = -nu * n . (curl omega) is the next handle.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import stokes3d as SD
from lssem3d import (project as PJ, helmholtz as HH, operator as OP,
                     solver3d as S3, timestep as T, fourier as FR)


def build(N=8, nz=8, tol=1e-12):
    sd = SD.setup(N=N, nz=nz)
    m, nk = sd['m'], sd['nk']
    mask_u = PJ.build_masks(m, nk, nz, 3, wall=True)
    mask_p = PJ.build_masks(m, nk, nz, 1, wall=False)
    ind = np.zeros(mask_p.shape); ind[0, 0, 0, 0, 0] = 1.0
    mask_p[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
    s = dict(m=m, D=sd['D'], N=N, nz=nz, nk=nk, nu=SD.NU, kz=sd['kz'],
             lz=sd['lz'], mask_u=mask_u, mask_p=mask_p, tol=tol,
             wq3=m.wq[..., None, None], wq1=m.wq[..., None, None])
    s['Mp'] = HH.fdm_preconditioner(m, N, sd['kz']**2, 1.0, mask_p, 2, nk)
    return s, sd


def precond_u(s, dt, k):
    lam = T.implicit_coeff(dt, k) + s['nu']*(s['kz']**2)
    s['Mu'] = HH.fdm_preconditioner(s['m'], s['N'], lam, s['nu'],
                                    s['mask_u'], 6, s['nk'])


def energy(s, Uc):
    """Matches stokes3d.energy: 0.5 int (u^2+v^2+w^2) dV."""
    Up = FR.to_physical(Uc, s['nz'])
    e = sum(Up[..., f, :]**2 for f in range(3))
    return 0.5*float(np.sum(e*s['m'].wq[..., None]))*(s['lz']/s['nz'])


print('GATE 1  Stokes decay in a channel, fractional-step path')
print(f'   analytic sigma = {SD.SIGMA_2D}\n')
rows = []
for dt in (0.01, 0.005, 0.0025):
    s, sd = build()
    U0, _ = SD.initial_state(sd, mode='kz0')
    Uc0 = OP.to_complex(U0)
    Uc = np.ascontiguousarray(Uc0[..., [OP.U_, OP.V_, OP.W_], :])
    pc = np.ascontiguousarray(Uc0[..., OP.P_:OP.P_+1, :])
    Uc = Uc*s['mask_u'][..., :3, :]          # enforce no-slip on the IC
    Z = np.zeros_like(Uc)
    nstep = int(round(0.05/dt))
    ts, Es = [0.0], [energy(s, Uc)]
    dv, pm = [], []
    for i in range(nstep):
        for k in range(T.NSTAGE):
            precond_u(s, dt, k)
            Uc, pc, _ = PJ.substage(s, Uc, pc, Z, Z, k, dt)
        ts.append((i+1)*dt); Es.append(energy(s, Uc))
        dv.append(float(np.abs(PJ.divergence(Uc, s['D'], s['m'].facx,
                                             s['m'].facy, s['kz'])).max()))
        pm.append(float(np.abs(pc).max()))
    print(f'      |div u| {dv[0]:.2e} -> {dv[-1]:.2e}    '
          f'max|p| {pm[0]:.2e} -> {pm[-1]:.2e}    '
          f'E {Es[0]:.4e} -> {Es[-1]:.4e}')
    ts, Es = np.array(ts), np.array(Es)
    k0 = len(ts)//2
    sig = -0.5*np.polyfit(ts[k0:], np.log(Es[k0:]/Es[0]), 1)[0]
    rel = abs(sig - SD.SIGMA_2D)/SD.SIGMA_2D
    rows.append((dt, sig, rel))
    print(f'   dt = {dt:<8g} sigma = {sig:.7f}   rel err {rel:.3e}', flush=True)
order = np.log2(rows[0][2]/rows[-1][2])/2
ok = rows[-1][2] < 2e-5 and order > 1.7
print(f'\n   convergence order in dt = {order:.2f} (expect ~2)   '
      f'{"PASS" if ok else "FAIL"}')
print(f'   least-squares path for comparison: 9.3153041 / 9.3141300 / '
      f'9.3138373, order 2.00')
