"""GATE 1: PRESSURE-FREE projection + the Kim-Moin wall correction.

The two error sources were separated and each addressed alone:

  incremental + uhat = 0        UNSTABLE (sigma 9.316 -> 9.944 over 40 steps)
  pressure-free + uhat = 0      stable, FIRST order (0.95, 0.97, 0.97)
  incremental + Kim-Moin        still unstable, but less so (order -0.80)

The pressure-free form is stable and its first-order error is the tangential
slip; the Kim-Moin correction removes exactly that slip.  Together they are the
scheme spectral channel codes in the Kim-Moin-Moser lineage actually use, and
the combination is what this measures.


The incremental form is unstable here (sigma runs to 9.94 over 40 steps), and
the likely reason is not only the wall condition: an INCREMENTAL scheme carries
a pressure increment scaled by beta_k*dt per substage, but the SMR weights sum
to beta_0+beta_1+beta_2 = 0.606, not 1, so those increments do not reconstruct
a consistent pressure over the step.  RKW3 projection codes in the Kim-Moin-
Moser lineage apply a PRESSURE-FREE projection per substage instead.

If that form is second order, Gate 1 passes and the consistent Neumann
condition of sec 3.2 is unnecessary -- so this is measured before it is built.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import stokes3d as SD
from lssem3d import project as PJ, operator as OP, timestep as T
from fs_gate1 import build, precond_u, energy

print(f'GATE 1, pressure-free projection.  analytic sigma = {SD.SIGMA_2D}\n')
rows = []
for dt in (0.01, 0.005, 0.0025, 0.00125):
    s, sd = build()
    s['incremental'] = False
    U0, _ = SD.initial_state(sd, mode='kz0')
    Uc0 = OP.to_complex(U0)
    Uc = np.ascontiguousarray(Uc0[..., [OP.U_, OP.V_, OP.W_], :])
    pc = np.ascontiguousarray(Uc0[..., OP.P_:OP.P_+1, :])
    Uc = Uc*s['mask_u'][..., :3, :]
    Z = np.zeros_like(Uc)
    nstep = int(round(0.05/dt))
    ts, Es = [0.0], [energy(s, Uc)]
    for i in range(nstep):
        for k in range(T.NSTAGE):
            precond_u(s, dt, k)
            Uc, pc, _ = PJ.substage(s, Uc, pc, Z, Z, k, dt)
        ts.append((i+1)*dt); Es.append(energy(s, Uc))
    ts, Es = np.array(ts), np.array(Es)
    k0 = len(ts)//2
    sig = -0.5*np.polyfit(ts[k0:], np.log(Es[k0:]/Es[0]), 1)[0]
    rel = abs(sig - SD.SIGMA_2D)/SD.SIGMA_2D
    rows.append((dt, sig, rel))
    inst = -0.5*np.diff(np.log(Es))/dt
    print(f'   dt = {dt:<9g} sigma = {sig:.7f}   rel err {rel:.3e}   '
          f'sigma(t) {inst[0]:.4f} -> {inst[-1]:.4f}')
o1 = np.log2(rows[0][2]/rows[1][2])
o2 = np.log2(rows[1][2]/rows[2][2])
o3 = np.log2(rows[2][2]/rows[3][2])
print(f'\n   pairwise orders in dt: {o1:.2f}, {o2:.2f}, {o3:.2f}')
ok = rows[-1][2] < 2e-5 and min(o1, o2, o3) > 1.7
print(f'   {"PASS" if ok else "FAIL"}   '
      f'(least-squares path: 9.3153041 / 9.3141300 / 9.3138373, order 2.00)')
