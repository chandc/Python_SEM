"""DOES THE ITERATION COUNT GROW WITH p?  Per mode, with a CORRECT mask.

p-independence is the whole point of the p-ladder and 2D delivers it (sec 6.9:
1.05x over N=8..24).  Earlier 3D sweeps said 5x, but both were contaminated:

  * the p-sweep (110->556) used a correct mask but INCLUDED k_z=0, and k_z=0
    needs 3442 Jacobi iterations against 50 for the top mode, so it measured
    k_z=0 and nothing else;
  * the h-sweep (11->41) called build_mask(m, 1, ...) at k_z=5.88, which zeroes
    the whole imaginary half because it believes column 0 is k=0 -- a halved,
    wrong problem.

So: build the mask ONCE for the full nk and SLICE it, which is the only way to
get a single k_z!=0 mode masked correctly.  Then sweep p at fixed mesh, for
k_z = 0 and two k_z != 0, Jacobi and PMG.

If k_z != 0 is flat in p and k_z = 0 is not, the p-growth is a property of the
k_z = 0 block -- whose softest mode is 100% PRESSURE at cond 9.0e4 -- and not of
p-multigrid or of the FOSLS formulation.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 40000
LADDER = {8: (8,4,2), 12: (12,6,3), 16: (16,8,4,2), 20: (20,10,5)}

def main(ex=3, ey=3, nz=32, cc=5405.4):
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3, fourier as FR
    nk = nz//2 + 1
    kz_all = FR.wavenumbers(nz, 0.34*np.pi)
    picks = [(0, 'k_z=0'), (1, f'k_z={kz_all[1]:.1f}'), (4, f'k_z={kz_all[4]:.1f}')]
    print(f'{ex}x{ey} elements, c={cc:g}, nu=1/180, tol={TOL:g}; mask sliced from full nk={nk}\n',
          flush=True)
    for kcol, klab in picks:
        print(f'--- {klab} ---', flush=True)
        print(f'{"p":>3} {"gDOF":>8} {"jacobi":>8} {"pmg":>6} {"ratio":>7}', flush=True)
        res = []
        for N in (8, 12, 16, 20):
            m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
            m.periodic_x = np.pi; m.compute_global_indices()
            mfull = BC.build_mask(m, nk, pin_p=False, nz=nz)
            BC.pin_dof(m, mfull, OP.P_, 0); BC.pin_dof(m, mfull, OP.NVAR+OP.P_, 0)
            mask = np.ascontiguousarray(mfull[..., kcol:kcol+1])
            kz = np.array([float(kz_all[kcol])])
            nu, D = 1/180., diff_matrix(N)
            rw = OP.momentum_row_weights(cc)
            sh = (m.nelem, N+1, N+1, OP.NVAR_R, 1)
            kwn = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
            A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kwn)
            b = A(np.random.default_rng(0).standard_normal(sh)*mask)
            b /= np.linalg.norm(b)
            Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(sh, D, m.facx,
                                   m.facy, kz, nu, cc, **kwn), mask)
            _, itj, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m, mask, Mi,
                               TOL, CAP, None, m.wq, 0.0, rw)
            M = P3.PMG(m, 1, nz, nu, cc, kz, kap=0.0, rw=rw, orders=LADDER[N],
                       deg=6, pin_p=True, direct_coarse='element', mask=mask)
            _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m, mask, M,
                               TOL, CAP, None, m.wq, 0.0, rw)
            itj, itp = int(np.max(itj)), int(np.max(itp))
            print(f'{N:3d} {int(mask.sum()):8d} {itj:8d} {itp:6d} {itj/max(itp,1):6.1f}x',
                  flush=True)
            res.append((itj, itp))
        r = np.array(res, float)
        print(f'    growth p=8->20:  jacobi {r[-1,0]/r[0,0]:.2f}x   '
              f'PMG {r[-1,1]/r[0,1]:.2f}x'
              f'{"   <-- p-INDEPENDENT" if r[-1,1]/r[0,1] < 1.5 else "   <-- GROWS"}\n',
              flush=True)
    print('2D contrast (PMG_ALGORITHM 6.9): jacobi 4.00x, ladder 1.05x')

if __name__ == '__main__':
    main()
