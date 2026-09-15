"""TEMPORAL ORDER of the fractional-step path with the wall condition.

    python scratch/fs_order.py

Gate 1's criterion was wrong, not the scheme.  At N = 8 the sigma error reached
3.5e-6 at dt = 0.00125 and the pairwise orders read 1.48, 1.77, 4.15 -- the
last inflated because the error had hit the SPATIAL floor and stopped
responding to dt.  A ratio between two numbers that are both at the floor
measures nothing.

The fix is to lower the floor until the temporal term is what is being
measured, and to say where the floor is rather than fitting through it.  Each N
is run to a fixed final time over the same dt ladder; the order is fitted only
over pairs whose error is well clear of that N's floor.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import stokes3d as SD
from lssem3d import project as PJ, helmholtz as HH, operator as OP, timestep as T
from lssem3d import solver3d as S3, fourier as FR
import fs_gate1 as G


def run(N, dt, incremental=False):
    s, sd = G.build(N=N)
    s['incremental'] = incremental
    U0, _ = SD.initial_state(sd, mode='kz0')
    Uc0 = OP.to_complex(U0)
    Uc = np.ascontiguousarray(Uc0[..., [OP.U_, OP.V_, OP.W_], :])
    pc = np.ascontiguousarray(Uc0[..., OP.P_:OP.P_+1, :])
    Uc = Uc*s['mask_u'][..., :3, :]
    Z = np.zeros_like(Uc)
    nstep = int(round(0.05/dt))
    ts, Es = [0.0], [G.energy(s, Uc)]
    for i in range(nstep):
        for k in range(T.NSTAGE):
            G.precond_u(s, dt, k)
            Uc, pc, _ = PJ.substage(s, Uc, pc, Z, Z, k, dt)
        ts.append((i+1)*dt); Es.append(G.energy(s, Uc))
    ts, Es = np.array(ts), np.array(Es)
    k0 = len(ts)//2
    sig = -0.5*np.polyfit(ts[k0:], np.log(Es[k0:]/Es[0]), 1)[0]
    return abs(sig - SD.SIGMA_2D)/SD.SIGMA_2D, sig


DTS = (0.02, 0.01, 0.005, 0.0025)
print(f'analytic sigma = {SD.SIGMA_2D}\n')
print(f'{"N":>3} ' + ' '.join(f'{d:>11g}' for d in DTS) + '   pairwise orders')
for N in (8, 12):
    errs, sigs = [], []
    for dt in DTS:
        e, sg = run(N, dt)
        errs.append(e); sigs.append(sg)
    o = [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]
    print(f'{N:>3} ' + ' '.join(f'{e:>11.3e}' for e in errs) + '   '
          + ', '.join(f'{x:.2f}' for x in o), flush=True)
    # asymptotic order: only pairs where BOTH errors are >10x the finest error
    fl = errs[-1]
    keep = [i for i in range(len(errs)-1) if errs[i+1] > 10*fl or i == 0]
    if keep:
        print(f'{"":>3} floor ~{fl:.1e}; order over pairs clear of it: '
              f'{np.mean([o[i] for i in keep]):.2f}')
