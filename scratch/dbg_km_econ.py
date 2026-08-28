"""Localize the KM+E blowup: tiny grid, one step, phase-by-phase norms."""
import os, sys
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='4'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import (project as PJ, helmholtz as HH, epmg, convect as CV,
                         solver3d as S3, deriv as DV)
    import fs_phase2 as F2
    NU, NE, N, NZ = 1/800., 4, 6, 16
    s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-8, backend='numpy')
    mask_p = PJ.build_masks(s['m'], s['nk'], NZ, 1, wall=False)
    s['mask_p'] = mask_p
    v = np.ones(mask_p[..., 0:1, 0:1].shape)*mask_p[..., 0:1, 0:1]
    s['null_kz0'] = v; s['null_norm'] = float((v*v*s['mw1']).sum())
    s['consistent_p'] = True; s['tol_p'] = 1e-7
    s['Mp'] = epmg.ConsistentPMG(s['m'], N, s['kz'], s['nk'], NZ, deg=6,
                                 like=mask_p, wall=False)
    s['null_basis_kz0'] = epmg.kz0_null_basis(s['m'], N, s['kz'], s['nk'],
                                              NZ, mask_p, s['mask_u'])
    Uc = F2.ic_tgv(s)
    dt = 0.005
    s['Mu'] = HH.fdm_preconditioner(s['m'], N, 2.0/dt + NU*(s['kz']**2), NU,
                                    s['mask_u'], 6, s['nk'], like=s['mask_u'])
    nrm = lambda a: float(np.sqrt((np.abs(a)**2).sum()))
    # full step
    pc = np.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
    Uc2, phi, inf, pc2 = PJ.step_kim_moin(s, Uc, np.zeros_like(pc), dt,
                                          pc=np.zeros_like(pc), skew=True)
    print(f'stage solve: it {s["_dbg_stage_p"][0]}  res {s["_dbg_stage_p"][1]:.1e}')
    print(f'after full step: |u| {nrm(Uc2):.3e}  it_u {inf[0]} it_p {inf[2]} '
          f'res_p {inf[3]:.1e}')
    # 3 more steps
    phi = np.zeros_like(pc); pcl = pc2
    for i in range(9):
        Uc2, phi, inf, pcl = PJ.step_kim_moin(s, Uc2, phi, 0.005, pc=pcl,
                                              skew=True)
        print(f'step {i+2}: |u| {nrm(Uc2):.3e}  stage {s["_dbg_stage_p"]}')

if __name__ == '__main__':
    main()
