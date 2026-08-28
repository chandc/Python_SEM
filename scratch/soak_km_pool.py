"""KM+E with mode pool: correctness vs serial + s/step, tiny grid then 88^3."""
import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='4'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import project as PJ, helmholtz as HH, epmg
    from lssem3d.modepar import ModePool
    import fs_phase2 as F2
    big = '--big' in sys.argv
    NU = 1/800.
    NE, N, NZ = (11, 8, 88) if big else (4, 6, 16)
    dt = 0.00567493 if big else 0.005
    s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-6, backend='numpy')
    mask_p = PJ.build_masks(s['m'], s['nk'], NZ, 1, wall=False)
    s['mask_p'] = mask_p
    v = np.ones(mask_p[..., 0:1, 0:1].shape)*mask_p[..., 0:1, 0:1]
    s['null_kz0'] = v; s['null_norm'] = float((v*v*s['mw1']).sum())
    s['consistent_p'] = True; s['tol_p'] = 1e-5
    L = 2*np.pi
    cfg = dict(N=N, ex=NE, ey=NE, nz=NZ, nu=NU, dt=dt, tol=1e-6,
               tol_p=1e-5, lx=float(L), lz=float(L), geom='periodic',
               with_e=True, km=True)
    t0 = time.time()
    pool = None
    if '--serial' not in sys.argv:
        pool = ModePool(cfg, nworkers=4, blas_threads=3)
        s['modepool'] = pool
    else:
        s['Mp'] = epmg.ConsistentPMG(s['m'], N, s['kz'], s['nk'], NZ, deg=6,
                                     wall=False, like=mask_p)
        s['null_basis_kz0'] = epmg.kz0_null_basis(s['m'], N, s['kz'],
                                                  s['nk'], NZ, mask_p,
                                                  s['mask_u'])
        s['purge_lanes'] = (0, s['nk'] - 1)
    s['Mu'] = HH.fdm_preconditioner(s['m'], N, 2.0/dt + NU*(s['kz']**2), NU,
                                    s['mask_u'], 6, s['nk'], like=s['mask_u'])
    # parent still needs Mp/null basis for the non-pool fallback path: skip
    if pool is not None:
        s['Mp'] = lambda r: r
    Uc = F2.ic_tgv(s)
    pc = np.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
    phi = np.zeros_like(pc)
    nrm = lambda a: float(np.sqrt((np.abs(a)**2).sum()))
    # first step includes worker setup
    t0 = time.time()
    Uc, phi, inf, pc = PJ.step_kim_moin(s, Uc, phi, dt, pc=pc, skew=True)
    print(f'first step (worker setup incl): {time.time()-t0:.0f}s  '
          f'|u| {nrm(Uc):.6e}', flush=True)
    t0 = time.time(); nst = 200
    for i in range(nst):
        Uc, phi, inf, pc = PJ.step_kim_moin(s, Uc, phi, dt, pc=pc, skew=True)
        if i % 40 == 39:
            print(f'step {i+2}: |u| {nrm(Uc):.6e}  CG {inf[0]+inf[2]}', flush=True)
    print(f's/step = {(time.time()-t0)/nst:.2f}', flush=True)
    if pool is not None:
        pool.close()

if __name__ == '__main__':
    main()
