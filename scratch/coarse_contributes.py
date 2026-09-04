"""Is the coarse correction contributing ANYTHING?

Every symptom points one way: PMG grows in lockstep with Jacobi (4.76x vs 4.29x
at k_z=0), an EXACT coarse solve is no better than a degree-10 polynomial
(441 vs 402), and 2-, 3- and 4-level ladders are indistinguishable
(434 / 441 / 405).  A working V-cycle cannot be insensitive to all three.

So compare, on the same problem:
  smoother ALONE  -- Chebyshev4 on the fine level, no coarse grid at all
  full V-cycle    -- same smoother plus restriction/coarse-solve/prolongation

If they agree, the coarse correction is inert and the defect is in the transfer
operators or in how the correction is applied -- not in FOSLS, not in p.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 40000

def main(ex=3, ey=3, nz=32, cc=5405.4, kcol=1):
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3, fourier as FR
    nk = nz//2 + 1
    kz_all = FR.wavenumbers(nz, 0.34*np.pi)
    kz = np.array([float(kz_all[kcol])])
    print(f'{ex}x{ey}, k_z={kz[0]:.2f}, c={cc:g}, tol={TOL:g}\n')
    print(f'{"p":>3} {"jacobi":>8} {"cheb only":>10} {"V-cycle":>9} {"coarse buys":>12}')
    for N, orders in ((8,(8,4,2)), (12,(12,6,3)), (16,(16,8,4,2)), (20,(20,10,5))):
        m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
        m.periodic_x = np.pi; m.compute_global_indices()
        mf = BC.build_mask(m, nk, pin_p=False, nz=nz)
        BC.pin_dof(m, mf, OP.P_, 0); BC.pin_dof(m, mf, OP.NVAR+OP.P_, 0)
        mask = np.ascontiguousarray(mf[..., kcol:kcol+1])
        nu, D = 1/180., diff_matrix(N)
        rw = OP.momentum_row_weights(cc)
        sh = (m.nelem, N+1, N+1, OP.NVAR_R, 1)
        kwn = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kwn)
        b = A(np.random.default_rng(0).standard_normal(sh)*mask); b /= np.linalg.norm(b)
        run = lambda M: int(np.max(S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m,
                                          mask, M, TOL, CAP, None, m.wq, 0.0, rw)[1]))
        lv = P3._Level(m, 1, nz, nu, cc, kz, 0.0, rw, False, mask=mask)
        itj = run(lv.M_inv)
        cheb = P3.Chebyshev4(lv.A, lv.M_inv, lv.shape, deg=6)
        itc = run(cheb)
        M = P3.PMG(m, 1, nz, nu, cc, kz, kap=0.0, rw=rw, orders=orders, deg=6,
                   pin_p=True, direct_coarse='element', mask=mask)
        itv = run(M)
        print(f'{N:3d} {itj:8d} {itc:10d} {itv:9d} {itc/max(itv,1):11.2f}x', flush=True)
    print('\nlast column ~1.0 => the coarse grid is INERT: the V-cycle is just its smoother.')

if __name__ == '__main__':
    main()
