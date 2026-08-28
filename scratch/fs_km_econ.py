"""Kim-Moin RK4-CN with the CONSISTENT E projection: TGV Re=800 at 88^3.

    python scratch/fs_km_econ.py [--smoke] [--nsteps N]

The completion of the Kim-Moin arc: thesis form = stable but O(dt) energy
defect; strong-gradient stage pressure = defect fixed but destabilised at
t=5.1 by the K-vs-DG mismatch; THIS = stage force and projection both built
from E = G^T M^{-1} G, so the sweep's weak divergence production cancels
identically.  Pass criteria: survives t=5.1, balance ~1.00 throughout,
completes t = 4*pi.
"""
import os, sys, time
# parent thread count: with --modepar the workers own most cores (4x3);
# a 12-thread parent oversubscribes to 24 threads on 12 P-cores and thrashed
# production to 30 s/step against the 3.2 benchmark.
NT = '4' if ('--smoke' in sys.argv or '--modepar' in sys.argv) else '12'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = NT
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np


def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import (project as PJ, helmholtz as HH, epmg, fourier as FR,
                         timestep as T)
    import fs_phase2 as F2
    NU, NE, N, NZ = 1.0/800.0, 11, 8, 88
    TEND = 4.0*np.pi
    arg = lambda f, d: (sys.argv[sys.argv.index(f)+1] if f in sys.argv else d)
    arg0 = lambda f, d: (sys.argv[sys.argv.index(f)+1] if f in sys.argv else d)
    s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-6, backend='numpy')
    # UNPINNED pressure mask for E (fs_phase2 pins; undo)
    mask_p = PJ.build_masks(s['m'], s['nk'], NZ, 1, wall=False)
    s['mask_p'] = mask_p
    v = np.ones(mask_p[..., 0:1, 0:1].shape)*mask_p[..., 0:1, 0:1]
    s['null_kz0'] = v
    s['null_norm'] = float((v*v*s['mw1']).sum())
    s['consistent_p'] = True
    s['tol_p'] = 1e-5
    t0 = time.perf_counter()
    s['Mp'] = epmg.ConsistentPMG(s['m'], N, s['kz'], s['nk'], NZ, deg=6,
                                 like=mask_p, wall=False)
    print(f'E-multigrid setup {time.perf_counter()-t0:.1f}s', flush=True)
    t0 = time.perf_counter()
    s['null_basis_kz0'] = epmg.kz0_null_basis(s['m'], N, s['kz'], s['nk'],
                                              NZ, mask_p, s['mask_u'])
    s['purge_lanes'] = (0, s['nk'] - 1)      # both real modes (nz even)
    print(f'kz0 kernel basis in {time.perf_counter()-t0:.1f}s', flush=True)
    NW = int(arg0('--modepar', '0'))
    if NW > 0:
        from lssem3d.modepar import ModePool
        L = 2*np.pi
        cfg = dict(N=N, ex=NE, ey=NE, nz=NZ, nu=NU, dt=0.00567493, tol=1e-6,
                   tol_p=1e-5, lx=float(L), lz=float(L), geom='periodic',
                   with_e=True, km=True)
        s['modepool'] = ModePool(cfg, nworkers=NW, blas_threads=3)
        print(f'mode pool: {NW} workers (E + velocity)', flush=True)
    dt = 0.00567493
    rst = arg0('--restart', '')
    if rst:
        d0 = np.load(rst)
        Uc = d0['U']; t_start = float(d0['t'])
        print(f'restarting from {rst} at t={t_start:.4f}', flush=True)
    else:
        Uc = F2.ic_tgv(s); t_start = 0.0
    s['Mu'] = HH.fdm_preconditioner(s['m'], N, 2.0/dt + NU*(s['kz']**2), NU,
                                    s['mask_u'], 6, s['nk'], like=s['mask_u'])
    pc = np.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
    phi = np.zeros_like(pc)
    nstep = int(arg('--nsteps', int(np.ceil((TEND - t_start)/dt))))
    out = 'scratch/_km_econ'
    os.makedirs(out, exist_ok=True)
    log = open(f'{out}/km_econ.log', 'a')
    E, Om = F2.diagnostics(s, Uc)
    prevE, prevOm = E, Om
    t = t_start
    w0 = last = time.perf_counter()
    for i in range(nstep):
        Uc, phi, inf, pc = PJ.step_kim_moin(s, Uc, phi, dt, pc=pc, skew=True)
        t += dt
        E, Om = F2.diagnostics(s, Uc)
        den = 2*NU*0.5*(Om + prevOm)
        if den <= 0 or not np.isfinite(E):
            line = f'BLEW UP at t={t:.4f}'
            print(line, flush=True); log.write(line + '\n'); log.flush()
            break
        bal = (-(E - prevE)/dt)/den if i else 1.0
        prevE, prevOm = E, Om
        if i % 10 == 0 or i == nstep - 1:
            line = (f't={t:8.4f}  E={E:.6f}  Om={Om:.5f}  '
                    f'-dE/dt / 2nuOm={bal:.4f}  CG={inf[0]+inf[2]}  '
                    f'[{time.perf_counter()-w0:.0f}s]')
            print(line, flush=True); log.write(line + '\n'); log.flush()
        if time.perf_counter() - last > 20*60:
            np.savez(f'{out}/chk_tmp.npz', U=Uc, p=pc, phi=phi, t=t, dt=dt)
            os.replace(f'{out}/chk_tmp.npz', f'{out}/chk_latest.npz')
            last = time.perf_counter()
    np.savez(f'{out}/final.npz', U=Uc, p=pc, phi=phi, t=t, dt=dt)
    print(f'DONE t={t:.4f} in {(time.perf_counter()-w0)/3600:.2f} h',
          flush=True)


if __name__ == '__main__':
    main()
