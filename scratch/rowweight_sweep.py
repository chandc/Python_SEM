"""Is the momentum row weight 1/c^2 the right SCALING for conditioning?

The Jacobi diagonal is exact (scratch/jacobi_check.py: 3.1e-16), so ~2000 CG
iterations is genuinely what the operator costs.  cond(A) = cond(L)^2 is
intrinsic to normal equations, but the SCALING of L is free, and
momentum_row_weights picks 1/c^2 on the authority of "lssem2d's legacy scaling:
momentum rows divided by c so their mass coefficient is 1" -- a convention, not
an optimum.  At c=5405 that is 3.4e-8, which is exactly why the softest mode is
100% pressure (pressure enters the functional only through the momentum rows).

Sweep w_mom and w_vort and measure cond(D^-1 A) and CG.  A better scaling would
be a free win at the channel's own c -- no new preconditioner, no new theory.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 60000

def main(N=8, ex=3, ey=3, cc=5405.4, kcol=1):
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
    print(f'{ex}x{ey} N={N} k_z={kz[0]:.2f} c={cc:g}; default w_mom=1 (i.e. 1/c^2)\n')
    print(f'{"w_mom":>9} {"w_vort":>8} | {"CG":>7} {"vs default":>11}')
    base = None
    for wm, wv in ((1.0,1.0), (1e2,1.0), (1e4,1.0), (1e6,1.0), (1e8,1.0),
                   (1.0,1e-2), (1.0,1e2), (1e4,1e-2), (1e4,1e2)):
        rw = OP.momentum_row_weights(cc, w_mom=wm, w_vort=wv)
        kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kw)
        b = A(np.random.default_rng(0).standard_normal(sh)*mask)
        b /= np.linalg.norm(b)
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(sh, D, m.facx, m.facy,
                               kz, nu, cc, **kw), mask)
        it = int(np.max(S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m, mask, Mi,
                               TOL, CAP, None, m.wq, 0.0, rw)[1]))
        if base is None: base = it
        print(f'{wm:9.0e} {wv:8.0e} | {it:7d} {base/max(it,1):10.2f}x', flush=True)
    print('\nw_mom scales rw[4:7] = w_mom/c^2; w_vort scales rw[1:4].')
    print('A CG drop here is a free win: same operator family, same solve, better scaling.')

if __name__ == '__main__':
    main()
