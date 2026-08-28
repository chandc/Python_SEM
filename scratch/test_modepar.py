import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
sys.argv = ['x']
import numpy as np, importlib
sw = importlib.import_module('fs_minchan_sweep')

def main():
    s = sw.build()
    from lssem3d import project as PJ, convect as CV, timestep as T, helmholtz as HH
    from lssem3d.modepar import ModePool
    d = np.load('scratch/_minchan_fs/final_s20.npz')
    DT = 3.5e-4
    cfg = dict(N=sw.N, ex=sw.EX, ey=sw.EY, nz=sw.NZ, nu=sw.NU, dt=DT,
               tol=1e-6, lx=float(np.pi), lz=float(0.34*np.pi))
    t0 = time.time()
    pool = ModePool(cfg, nworkers=4, blas_threads=3)
    # warm: first solve triggers worker setup completion
    print(f'pool spawned in {time.time()-t0:.1f}s (setup completes on first solve)',
          flush=True)
    s['modepool'] = pool
    pre = [HH.fdm_preconditioner(s['m'], sw.N,
           T.implicit_coeff(DT,k)+sw.NU*(s['kz']**2), sw.NU, s['mask_u'], 6,
           s['nk'], like=s['mask_u']) for k in range(T.NSTAGE)]
    Uc, pc = d['U'].copy(), d['p'].copy()
    s['ubc'] = None; Nprev = np.zeros_like(Uc)
    def step():
        nonlocal Uc, pc, Nprev
        tot = 0
        for k in range(T.NSTAGE):
            s['Mu'] = pre[k]
            Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'],
                                sw.NZ, skew=True) + s['Fm']
            Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
            Nprev = Nk; tot += inf[0] + inf[2]
        return tot
    t0 = time.time(); step(); print(f'first step (incl worker setup): '
                                    f'{time.time()-t0:.1f}s', flush=True)
    t0 = time.time(); its = 0
    for _ in range(5): its += step()
    per = (time.time()-t0)/5
    print(f'modepar: {per:.2f} s/step  CG(max/blk) {its/5:.0f}/step  '
          f'|U| {float(np.sqrt((np.abs(Uc)**2).sum())):.8e}', flush=True)
    pool.close()

if __name__ == '__main__':
    main()
