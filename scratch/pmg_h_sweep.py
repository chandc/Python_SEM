"""Does the 3D V-cycle degrade under h-refinement?

cond(M A) = 2.45 at 2x2 (predicted CG ~15) but the SAME p=8, nk=1 configuration
needs 204 on a 4x4 cavity and 402 on the 6x18 channel.  The formulation, the
smoother, the coarse solve and symmetry are all cleared.  The variable left is
ELEMENT COUNT -- and p-multigrid coarsens p only, never h.

2D sec 6.9's h-sweep IMPROVED (0.70x over 2x2..8x8), which PMG_ALGORITHM credits to
DirectCoarse absorbing the h-scale as the coarsest level grows with the mesh.
This measures whether that carries to 3D, with the coarse level solved EXACTLY
(DirectCoarseE) so the coarse solver cannot be blamed.

Fixed N=8, nk=1, so p and mode count are constant and only h moves.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 30000

def main(N=8, cc=5405.4):
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3
    nz, nk = 4, 1
    print(f'N={N} fixed, k_z=5.88, c={cc:g}, nk=1, coarse solved EXACTLY\n', flush=True)
    print(f'{"mesh":>7} {"gDOF":>8} | {"jacobi":>7} | {"pmg":>6} {"cond(MA)":>10} | {"ratio":>6}')
    rows = []
    for ex, ey in ((2,2), (3,3), (4,4), (6,6), (8,8)):
        m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
        m.periodic_x = 2.0*np.pi; m.compute_global_indices()
        kz = np.array([5.882])
        mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0)
        nu, D = 1/180., diff_matrix(N)
        rw = OP.momentum_row_weights(cc)
        shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
        kwn = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kwn)
        rng = np.random.default_rng(0)
        b = A(rng.standard_normal(shape)*mask); b /= np.linalg.norm(b)
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(shape, D, m.facx,
                               m.facy, kz, nu, cc, **kwn), mask)
        _, itj, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m, mask, Mi, TOL,
                           CAP, None, m.wq, 0.0, rw)
        itj = int(np.max(itj))
        M = P3.PMG(m, nk, nz, nu, cc, kz, kap=0.0, rw=rw, orders=(8,4,2), deg=6,
                   pin_p=True, direct_coarse='element', mask=mask)
        _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m, mask, M, TOL,
                           CAP, None, m.wq, 0.0, rw)
        itp = int(np.max(itp))
        cs = ''
        if ex <= 3:   # dense cond only where it fits
            mwa = S3.multiplicity_weight(m, shape)
            cols, seen = [], set()
            for e in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        for f in range(OP.NVAR_R):
                            if mask[e,i,j,f,0] == 0.0: continue
                            ed = np.zeros(shape); ed[e,i,j,f,0] = 1.0
                            g = S3.gs(m, ed)
                            k2 = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                            if not k2 or k2 in seen: continue
                            seen.add(k2); cols.append(g)
            Bm = np.stack([c.ravel() for c in cols], axis=1)
            Bw = (Bm*mwa.ravel()[:, None]).T
            T = np.empty((Bm.shape[1],)*2)
            for a in range(Bm.shape[1]):
                T[:, a] = Bw @ M(A(Bm[:, a].reshape(shape))).ravel()
            ev = np.linalg.eigvals(T).real; ev = ev[ev > 1e-14*ev.max()]
            cs = f'{ev.max()/ev.min():.3e}'
        print(f'{ex}x{ey:<5} {int(mask.sum()):8d} | {itj:7d} | {itp:6d} {cs:>10} | '
              f'{itj/max(itp,1):5.1f}x', flush=True)
        rows.append((ex, itj, itp))
    r = np.array(rows, float)
    print(f'\n  growth {int(r[0,0])}x{int(r[0,0])} -> {int(r[-1,0])}x{int(r[-1,0])}:  '
          f'jacobi {r[-1,1]/r[0,1]:.2f}x   PMG {r[-1,2]/r[0,2]:.2f}x')
    print('  2D for contrast (PMG_ALGORITHM 6.9 h-sweep): jacobi 3.76x, ladder 0.70x')

if __name__ == '__main__':
    main()
