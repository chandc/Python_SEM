"""Is ConsistentPMG.subset a valid preconditioner?  Solve on modes {3,4,5}
with (a) full-PMG-parent path as reference, (b) subset-PMG, (c) identity."""
import os, sys
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='8'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import project as PJ, epmg, solver3d as S3
    from lssem2d.lgl import diff_matrix
    import fs_phase2 as F2
    NU, NE, N, NZ = 1/800., 4, 6, 16
    s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-6, backend='numpy')
    m = s['m']
    mask_p = PJ.build_masks(m, s['nk'], NZ, 1, wall=False)
    mask_u = s['mask_u']
    wq3 = m.wq[..., None, None]
    Mginv = 1.0/S3.gs(m, wq3 + np.zeros_like(wq3))
    kz = s['kz']; D = diff_matrix(N)
    pmg = epmg.ConsistentPMG(m, N, kz, s['nk'], NZ, deg=6, wall=False)
    idx = np.array([3, 4, 5])
    kz_a, mp_a, mu_a = kz[idx], mask_p[..., idx], mask_u[..., idx]
    A = lambda p_: PJ.apply_E(p_, D, m.facx, m.facy, wq3, kz_a, m, mp_a,
                              mu_a, Mginv)
    # a C0 rhs on the block, from a velocity field (consistent by construction)
    rng = np.random.default_rng(3)
    uv = S3.gs(m, rng.standard_normal(mask_u[..., idx].shape))*mu_a
    from lssem3d import deriv as DV
    uc = PJ._join(uv)
    b = (DV.ddxT(wq3*uc[...,0:1,:], D, m.facx)
         + DV.ddyT(wq3*uc[...,1:2,:], D, m.facy)
         - 1j*kz_a*(wq3*uc[...,2:3,:]))
    b = S3.gs(m, PJ._split(b))*mp_a
    sub = pmg.subset(idx)
    # symmetry check of the subset V-cycle in the mw metric
    mw = S3.multiplicity_weight(m, mp_a.shape)
    r1 = S3.gs(m, rng.standard_normal(mp_a.shape))*mp_a
    r2 = S3.gs(m, rng.standard_normal(mp_a.shape))*mp_a
    d1 = float((r1*sub(r2)*mw).sum()); d2 = float((sub(r1)*r2*mw).sum())
    print(f'subset V-cycle symmetry: {d1:.6e} vs {d2:.6e}  '
          f'rel {abs(d1-d2)/max(abs(d1),1e-30):.2e}')
    for nm, M in (('subset-PMG', sub), ('identity', lambda r: r)):
        x, it, res = PJ._pcg(A, b.copy(), M, m, 1e-7, 1)
        print(f'{nm:>11}: it {it}  res {res:.2e}')

if __name__ == '__main__':
    main()
