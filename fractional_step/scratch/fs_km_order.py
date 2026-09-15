"""Does ONE projection per step recover second order at walls?

    python scratch/fs_km_order.py

Per-SUBSTAGE projection with the Kim-Moin correction measures ~1.6 order on
Gate 1 (channel, no-slip).  Kim & Moin report 2 and the reference figure shows
it plainly: their no-slip curve has slope ~1, their corrected curve ~2.

The suspected cause is structural rather than a coding error.  The correction
    uhat|wall = u^{n+1}|wall + dt grad(phi^{n-1})|wall
extrapolates over a UNIFORM step, and the SMR weights beta = (0.2315, 0.2083,
0.1667) sum to 0.606 rather than 1 -- so applied per substage it is scaled to
the wrong interval.  Kim & Moin apply it once per step.

This runs the reference's own sequence -- four-stage RK for convection only,
one Crank-Nicolson viscous solve, one projection -- against the same analytic
sigma = 9.3137399.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import stokes3d as SD
from lssem3d import project as PJ, helmholtz as HH, operator as OP
import fs_gate1 as G

DTS = (0.02, 0.01, 0.005, 0.0025)


def run(dt, N=8):
    s, sd = G.build(N=N)
    s['incremental'] = False
    U0, _ = SD.initial_state(sd, mode='kz0')
    Uc0 = OP.to_complex(U0)
    Uc = np.ascontiguousarray(Uc0[..., [OP.U_, OP.V_, OP.W_], :])
    Uc = Uc*s['mask_u'][..., :3, :]
    # the CN viscous operator is FIXED across the run: 2/dt + nu*kz^2
    s['Mu'] = HH.fdm_preconditioner(s['m'], N, 2.0/dt + s['nu']*(s['kz']**2),
                                    s['nu'], s['mask_u'], 6, s['nk'])
    phi = None
    nstep = int(round(0.05/dt))
    ts, Es = [0.0], [G.energy(s, Uc)]
    for i in range(nstep):
        Uc, phi, _, _ = PJ.step_kim_moin(s, Uc, phi, dt)
        ts.append((i+1)*dt); Es.append(G.energy(s, Uc))
    ts, Es = np.array(ts), np.array(Es)
    k0 = len(ts)//2
    sig = -0.5*np.polyfit(ts[k0:], np.log(Es[k0:]/Es[0]), 1)[0]
    return abs(sig - SD.SIGMA_2D)/SD.SIGMA_2D, sig


print(f'GATE 1, ONE projection per step (Kim-Moin structure)')
print(f'   analytic sigma = {SD.SIGMA_2D}\n')
errs = []
for dt in DTS:
    e, sg = run(dt)
    errs.append(e)
    print(f'   dt = {dt:<9g} sigma = {sg:.7f}   rel err {e:.3e}', flush=True)
o = [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]
print(f'\n   pairwise orders: ' + ', '.join(f'{x:.2f}' for x in o))
print(f'   per-SUBSTAGE projection gave: 1.64, 1.48, 1.77 (~1.6)')
print(f'   least-squares path:           2.00')
