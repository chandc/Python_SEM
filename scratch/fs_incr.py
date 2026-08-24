"""Is the growing mode the INCREMENTAL pressure feeding back?

Test B showed the Gate 1 drift growing super-linearly with step count at fixed
c_k (0.012, 0.038, 0.129, 0.628 over 5/10/20/40 steps) -- an exponentially
growing numerical mode, not a fixed per-substage error.

In an incremental projection the pressure accumulates, p^k = p^{k-1} + phi -
nu*div(uhat), and re-enters the next substage through -beta_k*dt*grad(p).  That
is a feedback loop, and an inconsistent wall condition on phi is exactly the
sort of thing it can amplify.  The pressure-free form breaks the loop: no
grad(p) in the right-hand side, no accumulation.  It costs an order in pressure
-- which is why it is a DIAGNOSTIC here, not a fix.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import stokes3d as SD
from lssem3d import project as PJ, operator as OP, timestep as T
from fs_gate1 import build, precond_u, energy


def run(dt, nstep, incremental):
    s, sd = build()
    s['incremental'] = incremental
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
    return -0.5*np.diff(np.log(np.array(Es)))/dt


print(f'analytic sigma = {SD.SIGMA_2D}\n')
print('drift in sigma(t) at dt = 0.0025, growing with steps if a mode grows')
print(f'   {"steps":>7} {"INCREMENTAL":>24} {"PRESSURE-FREE":>24}')
print(f'   {"":>7} {"sig[1] -> sig[last]":>24} {"sig[1] -> sig[last]":>24}')
for ns in (5, 10, 20, 40):
    a = run(0.0025, ns, True)
    b = run(0.0025, ns, False)
    print(f'   {ns:>7} {a[0]:>10.4f} ->{a[-1]:>10.4f}  '
          f'{b[0]:>10.4f} ->{b[-1]:>10.4f}')
print('\n   If the right column stays flat while the left runs away, the')
print('   incremental pressure feedback is the mechanism, and the fix is the')
print('   consistent wall condition of FRACTIONAL_STEP_PLAN.md sec 3.2 --')
print('   NOT dropping the increment, which would cost an order in pressure.')
