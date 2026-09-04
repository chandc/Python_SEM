"""Locate the momentum row-weight optimum, and find how it scales with c.

scratch/rowweight_sweep.py: w_mom=1e2 gives 131 CG against the default's 561 --
4.28x -- at c=5405.  rw[4:7] = w_mom/c^2, so the default is 1/c^2 = 3.4e-8 and
the optimum is near 3.4e-6.  1/c^1.5 = 2.5e-6 is suspiciously close, so sweep c
too and fit the exponent: rw_opt ~ c^-alpha.

ACCURACY IS NOT FREE.  The row weights ARE the least-squares functional, and the
discrete system is overdetermined (8 rows, 7 unknowns) hence inconsistent, so a
different weighting gives a different discrete solution -- exactly the trade sec 7J
measured for w7.  So also report how far the solution moves.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 60000

def main(N=8, ex=3, ey=3, kcol=1):
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, solver3d as S3, fourier as FR
    nz, nk = 32, 17
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = np.pi; m.compute_global_indices()
    mf = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mf, OP.P_, 0); BC.pin_dof(m, mf, OP.NVAR+OP.P_, 0)
    mask = np.ascontiguousarray(mf[..., kcol:kcol+1])
    kz = np.array([float(FR.wavenumbers(nz, 0.34*np.pi)[kcol])])
    nu, D = 1/180., diff_matrix(N)
    sh = (m.nelem, N+1, N+1, OP.NVAR_R, 1)
    rng = np.random.default_rng(0)
    xt = rng.standard_normal(sh)*mask

    opt = []
    for cc in (525.0, 1500.0, 5405.4, 15000.0):
        print(f'\n--- c = {cc:g}  (default rw_mom = 1/c^2 = {1/cc**2:.2e}) ---', flush=True)
        print(f'{"w_mom":>8} {"rw[4:7]":>10} | {"CG":>7} {"gain":>7} {"|dU|/|U| vs default":>20}')
        best, bit, ref = None, None, None
        for wm in (1.0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4):
            rw = OP.momentum_row_weights(cc, w_mom=wm)
            kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
            A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kw)
            b = A(xt); b /= np.linalg.norm(b)
            Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(sh, D, m.facx,
                                   m.facy, kz, nu, cc, **kw), mask)
            U, it, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m, mask, Mi,
                              TOL, CAP, None, m.wq, 0.0, rw)
            it = int(np.max(it))
            if ref is None:
                ref, base = U.copy(), it
                dsol = 0.0
            else:
                dsol = float(np.abs(U-ref).max()/max(np.abs(ref).max(), 1e-300))
            if best is None or it < bit: best, bit = wm, it
            print(f'{wm:8.0e} {wm/cc**2:10.2e} | {it:7d} {base/max(it,1):6.2f}x '
                  f'{dsol:20.3e}', flush=True)
        opt.append((cc, best, bit))
        print(f'  best w_mom = {best:.0e}  ->  rw = {best/cc**2:.2e}')
    a = np.array(opt, float)
    if len(a) > 1:
        alpha = -np.polyfit(np.log(a[:, 0]), np.log(a[:, 1]/a[:, 0]**2), 1)[0]
        print(f'\n  optimal rw_mom ~ c^-{alpha:.2f}   (default assumes c^-2.00)')
    print('\n  |dU|/|U| is how far the SOLUTION moves -- the weights are the functional.')

if __name__ == '__main__':
    main()
