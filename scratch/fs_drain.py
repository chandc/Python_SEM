"""Does the Gate 1 energy drain track c_k, or track step count?

    python scratch/fs_drain.py

Gate 1 fails with the instantaneous decay rate drifting UPWARD and
ACCELERATING at dt = 0.0025 (sigma(t): 9.3162 -> 9.4448 against an analytic
9.3137399), while dt = 0.005 sits almost exactly on the analytic rate.  Two
things change together as dt shrinks and they have to be separated:

  c_k = 1/(beta_k*dt)   432, 864, 1728 at the three tested dt.  The project
                        already documents a stability window in this quantity
                        for the least-squares path (timestep.py: "budget
                        max_k 1/(beta_k*dt) against the measured stability
                        window, NOT 1.5/dt"; AMASS_RESOLVED.md).  An analogous
                        limit on the projection path would be a finding, not a
                        bug.

  step count            0.05/dt gives 5, 10, 20 steps.  An error committed once
                        per substage accumulates with step count regardless of
                        c_k.

TEST A fixes the STEP COUNT and varies dt, so c_k varies and accumulation does
not.  TEST B fixes dt (hence c_k) and varies the step count.  One run each
settles which.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import stokes3d as SD
from lssem3d import project as PJ, operator as OP, timestep as T
from fs_gate1 import build, precond_u, energy


def run(dt, nstep):
    s, sd = build()
    U0, _ = SD.initial_state(sd, mode='kz0')
    Uc0 = OP.to_complex(U0)
    Uc = np.ascontiguousarray(Uc0[..., [OP.U_, OP.V_, OP.W_], :])
    pc = np.ascontiguousarray(Uc0[..., OP.P_:OP.P_+1, :])
    Uc = Uc*s['mask_u'][..., :3, :]
    Z = np.zeros_like(Uc)
    Es = [energy(s, Uc)]
    for i in range(nstep):
        for k in range(T.NSTAGE):
            precond_u(s, dt, k)
            Uc, pc, _ = PJ.substage(s, Uc, pc, Z, Z, k, dt)
        Es.append(energy(s, Uc))
    inst = -0.5*np.diff(np.log(np.array(Es)))/dt
    return inst


ck = lambda dt: max(1.0/(T.BETA[k]*dt) for k in range(3))
EX = SD.SIGMA_2D
print(f'analytic sigma = {EX}\n')

print('TEST A -- FIXED STEP COUNT (20), varying dt so only c_k changes')
print(f'   {"dt":>9} {"max c_k":>9} {"sigma[1]":>10} {"sigma[20]":>10} '
      f'{"drift":>10} {"per step":>10}')
for dt in (0.01, 0.005, 0.0025, 0.00125):
    ins = run(dt, 20)
    d = ins[-1] - ins[0]
    print(f'   {dt:>9g} {ck(dt):>9.0f} {ins[0]:>10.4f} {ins[-1]:>10.4f} '
          f'{d:>+10.4f} {d/20:>+10.5f}')

print('\nTEST B -- FIXED dt = 0.0025 (c_k fixed), varying step count')
print(f'   {"steps":>7} {"sigma[1]":>10} {"sigma[last]":>12} {"drift":>10} '
      f'{"per step":>10}')
for ns in (5, 10, 20, 40):
    ins = run(0.0025, ns)
    d = ins[-1] - ins[0]
    print(f'   {ns:>7} {ins[0]:>10.4f} {ins[-1]:>12.4f} {d:>+10.4f} '
          f'{d/ns:>+10.5f}')
print('\n  If drift-per-step grows with c_k in A but is constant in B, the\n'
      '  mechanism is c_k.  If it is constant in A and the TOTAL drift grows\n'
      '  linearly with steps in B, it is a fixed per-substage error.')
