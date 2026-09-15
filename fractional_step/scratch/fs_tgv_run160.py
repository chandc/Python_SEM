"""PHASE 3: TGV Re = 800 at 88^3 on the fractional-step path.

    python scratch/fs_tgv_run.py [--price] [--backend cupy] [--outdir DIR]

The A/B that replaces the 30-100x estimate with a measurement.  Configuration
is copied from tgv3d.py's re800_88 so it matches the least-squares run
finishing on the Mac: 11x11 elements at N = 8, Nz = 88, nu = 1/800, to
t = 4*pi, dt from the same CFL rule.  Diagnostics are logged in the same
columns so the two trajectories can be compared line for line.

Checkpoints every --chk-minutes, atomically (temp name then rename), with the
same reasoning as the least-squares driver: RKW3's ZETA[0] = 0, so the
convective history carries no information across a step boundary and a
checkpoint needs only the state.
"""
import sys, os, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

backend = 'cupy' if '--backend' not in sys.argv else \
    sys.argv[sys.argv.index('--backend') + 1]
price = '--price' in sys.argv
outdir = (sys.argv[sys.argv.index('--outdir') + 1]
          if '--outdir' in sys.argv else '/work/_fs800_160')
chk_min = float(sys.argv[sys.argv.index('--chk-minutes') + 1]
                if '--chk-minutes' in sys.argv else 20)

import lssem3d; lssem3d.set_backend(backend)
from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                     timestep as T, fourier as FR)
import fs_phase2 as F2

L = 2*np.pi
NU, NE, N, NZ, TEND = 1.0/800.0, 20, 8, 160, 4.0*np.pi
TOL = 1e-6

s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=TOL, backend=backend)
Uc = F2.ic_tgv(s)
xp = np
if backend == 'cupy':
    import cupy as xp
    print(f'GPU  {xp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
pc = xp.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
Nprev = xp.zeros((s['m'].nelem, N+1, N+1, 3, s['nk']), dtype=complex)

# dt by the same CFL rule the least-squares driver uses, so the two match
Up = FR.to_physical(Uc if backend == 'numpy' else xp.asnumpy(Uc), NZ)
dt = float(CV.max_dt_for_cfl(np.concatenate(
    [Up, np.zeros(Up.shape[:-2] + (4, NZ))], axis=-2),
    s['D'], s['m'].facx, s['m'].facy, L, NZ, 1.0))
dt = min(dt, 0.02)
nstep = int(np.ceil(TEND/dt))
dof = Uc.size if backend == 'cupy' else Uc.size
print(f'{NE}x{NE} N={N} Nz={NZ}, nu={NU:.5g}, dt={dt:.6g}, {nstep} steps to '
      f't={TEND:.4f}')
print(f'velocity dof {dof*2/1e6:.2f} M (3 complex fields)\n')

# PRESSURE: p-multigrid, device-resident.  One-level FDM is no better than a
# plain diagonal on this operator and both grow with element count.
from lssem3d import hpmg
t0 = time.perf_counter()
# cache_path: the coarse factorisation is host-bound (~90 min at 20x20 N=8)
# and pure setup, so a checkpoint restart reloads it instead of repaying it
# (keyed on the parameters; a config change rebuilds and overwrites)
s['Mp'] = hpmg.HelmholtzPMG(s['m'], N, s['kz']**2, 1.0, 1, s['nk'], NZ,
                            wall=False, pin_kz0=True, deg=6,
                            like=s['mask_p'],
                            cache_path=f'{outdir}/hpmg_cache.npz')
print(f'pressure: device p-multigrid (setup {time.perf_counter()-t0:.1f}s)')
# VELOCITY: one CN solve per step at lam = 2/dt, so ONE preconditioner, not
# three -- step_kim_moin does a single projection per step rather than one per
# RKW3 substage.
# VELOCITY: per-substage RKW3/CN, so one preconditioner per stage at c_k.
#
# SCHEME CHOICE IS NOT FREE, and the two options have OPPOSITE strengths,
# measured on this case:
#     periodic, energy balance   substage 1.0003 flat | kim_moin 1.2296, O(dt)
#     walls, temporal order      substage ~1.6        | kim_moin ~2.2
# RKW3/CN interleaves viscous damping into every substage, which the balance
# rewards; Kim-Moin's single CN solve per step is a coarser convection/
# diffusion split.  But only Kim-Moin's correction is consistent at a wall,
# because it extrapolates over a whole step.  TGV is periodic, so: substage.
pre = [HH.fdm_preconditioner(s['m'], N,
                             T.implicit_coeff(dt, k) + NU*(s['kz']**2),
                             NU, s['mask_u'], 6, s['nk'], like=s['mask_u'])
       for k in range(T.NSTAGE)]
pc = xp.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
Nprev = xp.zeros((s['m'].nelem, N+1, N+1, 3, s['nk']), dtype=complex)
sync = (lambda: xp.cuda.Stream.null.synchronize()) if backend == 'cupy' \
    else (lambda: None)


def step():
    global Uc, pc, Nprev
    tot = 0
    for k in range(T.NSTAGE):
        s['Mu'] = pre[k]
        # SKEW-SYMMETRIC form.  The advective form conserves energy only when
        # div u = 0, and this path carries 3.6e-04 where the least-squares path
        # holds 4.1e-07.  Kim & Moin use skew for exactly this reason
        # ("following Horiuti's recommendation ... to control aliasing
        # errors"); it did not carry over, and the previous run blew up at
        # t = 9.32 with enstrophy 155% above the reference.
        Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], NZ,
                            skew=True)
        Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, dt)
        Nprev = Nk
        tot += inf[0] + inf[2]
    return tot


if price:
    # profile where an iteration actually goes, before quoting a speedup
    import lssem3d.helmholtz as _HH
    step(); sync()                       # warm: JIT, pool growth
    t0 = time.perf_counter(); its = 0
    for _ in range(2):
        its += step()
    sync()
    per = (time.perf_counter() - t0)/2
    print(f'PRICE  {per:.2f} s/step   {its/2:.0f} CG iters/step   '
          f'-> {nstep*per/3600:.1f} h for {nstep} steps', flush=True)
    print(f'       least-squares reference on the Mac: ~87 s/step, ~3600 '
          f'iters/step, 53.6 h')
    sys.exit()

os.makedirs(outdir, exist_ok=True)
log = open(f'{outdir}/fs_re800.log', 'a')
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
        break
    bal = (-(E - prevE)/dt)/den if i else 1.0
    prevE, prevOm = E, Om
    if i % 10 == 0 or i == nstep-1:
        line = (f't={t:8.4f}  E={E:.6f}  Om={Om:.5f}  '
                f'-dE/dt / 2nuOm={bal:.4f}  CG={tot}  '
                f'[{time.perf_counter()-w0:.0f}s]')
        print(line, flush=True); log.write(line + '\n'); log.flush()
    if time.perf_counter() - last > chk_min*60:
        u = Uc if backend == 'numpy' else xp.asnumpy(Uc)
        p_ = pc if backend == 'numpy' else xp.asnumpy(pc)
        np.savez(f'{outdir}/chk_tmp.npz', U=u, p=p_, t=t, step=i+1, dt=dt)
        os.replace(f'{outdir}/chk_tmp.npz', f'{outdir}/chk_latest.npz')
        last = time.perf_counter()
print(f'DONE t={t:.4f} in {(time.perf_counter()-w0)/3600:.2f} h')
