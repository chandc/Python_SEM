"""Kim & Moin fractional step (RK4 convection + one CN + one projection),
TGV Re = 800 at 88^3, skew-symmetric convection.

    python scratch/fs_km_tgv_run.py [--price] [--pc-u fdm|jacobi] [--pc-p pmg|fdm|jacobi]

Same grid, nu, dt and diagnostic columns as fs_tgv_run.py so the three
trajectories (least-squares VVP, RKW3 substage projection, Kim-Moin) compare
line for line.

TWO DIFFERENCES FROM THE EARLIER kim_moin MEASUREMENT that made it look O(dt):
  1. pc is CARRIED (incremental pressure).  Pressure-free splitting is first
     order; the 1.2296 balance in the driver note was measured with pc=None.
  2. skew=True.  x-y is not dealiased, and Kim & Moin use skew for exactly
     that reason.

Log is opened 'w', NOT 'a' -- an appended log holding several runs is how a
previous analysis mixed run 1's head with run 3's tail.
"""
import sys, os, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

backend = 'cupy' if '--backend' not in sys.argv else \
    sys.argv[sys.argv.index('--backend') + 1]
price = '--price' in sys.argv
arg = lambda f, d: (sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d)
outdir = arg('--outdir', '/work/_km800')
chk_min = float(arg('--chk-minutes', 20))
PC_U, PC_P = arg('--pc-u', 'fdm'), arg('--pc-p', 'pmg')

import lssem3d; lssem3d.set_backend(backend)
from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                     fourier as FR, hpmg)
import fs_phase2 as F2

L = 2*np.pi
NU, NE, N, NZ, TEND = 1.0/800.0, 11, 8, 88, 4.0*np.pi
TOL = 1e-6

s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=TOL, backend=backend)
Uc = F2.ic_tgv(s)
xp = np
if backend == 'cupy':
    import cupy as xp
    print(f'GPU  {xp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')

Up = FR.to_physical(Uc if backend == 'numpy' else xp.asnumpy(Uc), NZ)
dt = float(CV.max_dt_for_cfl(np.concatenate(
    [Up, np.zeros(Up.shape[:-2] + (4, NZ))], axis=-2),
    s['D'], s['m'].facx, s['m'].facy, L, NZ, 1.0))
dt = min(dt, 0.02)
nstep = int(np.ceil(TEND/dt))
print(f'{NE}x{NE} N={N} Nz={NZ}, nu={NU:.5g}, dt={dt:.6g}, {nstep} steps to '
      f't={TEND:.4f}   [Kim-Moin RK4/CN, skew=True, incremental p]')

dev = (lambda a: xp.asarray(a)) if backend == 'cupy' else (lambda a: a)


def make_pc(kind, lam, mu, mask, nfield, tag):
    t0 = time.perf_counter()
    if kind == 'fdm':
        M = HH.fdm_preconditioner(s['m'], N, lam, mu, mask, nfield, s['nk'],
                                  like=mask)
    elif kind == 'jacobi':
        d = HH.jacobi_diagonal_analytic(s['m'], N, s['wq'], lam, mu, nfield,
                                        s['nk'], mask=None)
        inv = dev(HH.jacobi_inverse(d))
        mk = mask
        M = lambda r, _i=inv, _m=mk: r*_i*_m
    elif kind == 'pmg':
        M = hpmg.HelmholtzPMG(s['m'], N, lam, mu, nfield, s['nk'], NZ,
                              wall=False, pin_kz0=True, deg=6, like=mask)
    else:
        raise SystemExit(f'unknown preconditioner {kind}')
    print(f'{tag}: {kind}  (setup {time.perf_counter()-t0:.1f}s)')
    return M


# ONE viscous solve per step at lam = 2/dt + nu kz^2, so ONE velocity
# preconditioner -- the RKW3 substage path needs three, one per beta_k.
s['Mu'] = make_pc(PC_U, 2.0/dt + NU*(s['kz']**2), NU, s['mask_u'], 6, 'velocity')
s['Mp'] = make_pc(PC_P, s['kz']**2, 1.0, s['mask_p'], 1, 'pressure')

pc = xp.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
phi = xp.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
sync = (lambda: xp.cuda.Stream.null.synchronize()) if backend == 'cupy' \
    else (lambda: None)


def step():
    global Uc, pc, phi
    Uc, phi, inf, pc = PJ.step_kim_moin(s, Uc, phi, dt, pc=pc, skew=True)
    return inf[0] + inf[2]


if price:
    step(); sync()
    t0 = time.perf_counter(); its = 0
    for _ in range(2):
        its += step()
    sync()
    per = (time.perf_counter() - t0)/2
    print(f'PRICE  {per:.2f} s/step   {its/2:.0f} CG iters/step   '
          f'-> {nstep*per/3600:.1f} h for {nstep} steps', flush=True)
    E, Om = F2.diagnostics(s, Uc)          # exercise the run path, not just step
    print(f'       diagnostics ok: E={E:.6f} Om={Om:.5f}')
    sys.exit()

os.makedirs(outdir, exist_ok=True)
log = open(f'{outdir}/km_re800.log', 'w')
log.write(f'# Kim-Moin RK4/CN skew=True incremental-p, {NE}x{NE} N={N} '
          f'Nz={NZ} nu={NU:.5g} dt={dt:.6g} pc_u={PC_U} pc_p={PC_P}\n')
E, Om = F2.diagnostics(s, Uc)
prevE, prevOm = E, Om
t = 0.0
w0 = last = time.perf_counter()
for i in range(nstep):
    tot = step()
    t += dt
    E, Om = F2.diagnostics(s, Uc)
    den = 2*NU*0.5*(Om + prevOm)
    if den <= 0 or not np.isfinite(E):
        print(f'BLEW UP at t={t:.4f}: E={E}, Omega={Om}', flush=True)
        log.write(f'BLEW UP at t={t:.4f}: E={E}, Omega={Om}\n'); log.flush()
        break
    bal = (-(E - prevE)/dt)/den if i else 1.0
    prevE, prevOm = E, Om
    if i % 10 == 0 or i == nstep-1:
        line = (f't={t:8.4f}  E={E:.6f}  Om={Om:.5f}  '
                f'-dE/dt / 2nuOm={bal:.4f}  CG={tot}  '
                f'[{time.perf_counter()-w0:.0f}s]')
        print(line, flush=True); log.write(line + '\n'); log.flush()
    if time.perf_counter() - last > chk_min*60:
        cvt = (lambda a: a) if backend == 'numpy' else xp.asnumpy
        np.savez(f'{outdir}/chk_tmp.npz', U=cvt(Uc), p=cvt(pc),
                 phi=cvt(phi), t=t, step=i+1, dt=dt)
        os.replace(f'{outdir}/chk_tmp.npz', f'{outdir}/chk_latest.npz')
        last = time.perf_counter()
print(f'DONE t={t:.4f} in {(time.perf_counter()-w0)/3600:.2f} h')
