"""Does the TGV balance error scale with dt?

    python scratch/fs_dt_test.py

The Re = 800 run reported -dE/dt / 2nuOmega at 1.2038 by t = 0.687, when
Omega is only 97 and the flow is still smooth -- so this is not gradient
sharpening, it is ~20% extra dissipation present from the start.  E and Omega
otherwise TRACK the least-squares reference (Omega 714.7 against ~710 at
t = 5.45), and the run then went unstable at t = 5.51.

The scaling says what it is:
   error halving with dt      -> O(dt), the pressure-free projection, which
                                 omits grad(p) from the momentum step
   error quartering with dt   -> O(dt^2), a step-size limit
   error INDEPENDENT of dt    -> structural, and dt will not fix it
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('cupy')
import cupy as cp
from lssem3d import project as PJ, helmholtz as HH, hpmg, timestep as T
import fs_phase2 as F2

N, NE, NZ, NU = 8, 6, 32, 1.0/800.0
TEND = 0.5
DT0 = 0.00567493
print(f'TGV Re=800-like, {NE}x{NE} N={N} Nz={NZ}, to t={TEND}\n')
print(f'{"scheme":>12} {"dt":>10} {"mean balance":>13} {"|bal-1|":>9}')
import itertools
for scheme, div in itertools.product(('kim_moin', 'substage'), (1, 2, 4)):
    dt = DT0/div
    s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-8, backend='cupy')
    s['force'] = None
    s['Mp'] = hpmg.HelmholtzPMG(s['m'], N, s['kz']**2, 1.0, 1, s['nk'], NZ,
                                wall=False, pin_kz0=True, deg=6,
                                like=s['mask_p'])
    s['Mu'] = HH.fdm_preconditioner(s['m'], N, 2.0/dt + NU*(s['kz']**2), NU,
                                    s['mask_u'], 6, s['nk'], like=s['mask_u'])
    if scheme == 'substage':
        # per-substage RKW3/CN projection needs its own preconditioner per
        # stage, at c_k rather than 2/dt
        pre = [HH.fdm_preconditioner(s['m'], N,
                                     T.implicit_coeff(dt, k) + NU*(s['kz']**2),
                                     NU, s['mask_u'], 6, s['nk'],
                                     like=s['mask_u'])
               for k in range(T.NSTAGE)]
    Uc = F2.ic_tgv(s)
    phi = None
    pc = cp.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
    Np_ = cp.zeros((s['m'].nelem, N+1, N+1, 3, s['nk']), dtype=complex)
    E, Om = F2.diagnostics(s, Uc)
    bals = []
    for i in range(int(round(TEND/dt))):
        pE, pO = E, Om
        if scheme == 'kim_moin':
            Uc, phi, _, pc = PJ.step_kim_moin(s, Uc, phi, dt, pc)
        else:
            from lssem3d import convect as CV
            for k in range(T.NSTAGE):
                s['Mu'] = pre[k]
                Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'],
                                    s['kzg'], NZ)
                Uc, pc, _ = PJ.substage(s, Uc, pc, Nk, Np_, k, dt)
                Np_ = Nk
        E, Om = F2.diagnostics(s, Uc)
        d = 2*NU*0.5*(Om + pO)
        if d > 0 and i:
            bals.append((-(E - pE)/dt)/d)
    mb = float(np.mean(bals))
    print(f'{scheme:>12} {dt:>10.6f} {mb:>13.4f} {abs(mb-1):>9.4f}',
          flush=True)
print('\n  least-squares reference holds this at ~0.999 throughout.')
