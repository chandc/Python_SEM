"""Part 2 of the ADN block test.

(a) Is the (u, omega) pair coercive in H(div) x L2 at large c?  Precondition
    with the RIESZ MAP of that norm rather than the diagonal blocks of A:
        B_u = A_uu - K_curl,u   (= mass + div-div + O(nu^2/c^2))   [H(div)]
        B_w = M_w               (the curl-row mass matrix)          [L2]
        B_p = A_pp                                                  [Poisson]
    where K_curl,u and M_w come from assembling the operator with ONLY the
    three vorticity-definition rows switched on.  If the cond is flat and
    small, H(div) x L2 x H1 is the norm the large-c functional is equivalent
    to, and the (u,omega) solve is an H(div) problem.
(b) h-sweep of the exact 2-block (u,w | p) preconditioner at fixed p.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sla
from adn_block_precond import assemble, blockdiag_solver, pcg_ritz, VEL, VOR, PRE, NVAR


def build(ex, ey, N, cc, kcol, nz=16, rw_override=None):
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_channel
    from lssem3d import bc as BC, operator as OP, fourier as FR
    from lssem3d.precond import _Level
    nk = nz//2+1; kz_all = FR.wavenumbers(nz, 0.34*np.pi)
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0,0,1,1))
    m.periodic_x = np.pi; m.compute_global_indices()
    mfull = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mfull, OP.P_, 0); BC.pin_dof(m, mfull, OP.NVAR+OP.P_, 0)
    mask = np.ascontiguousarray(mfull[..., kcol:kcol+1])
    kz = np.array([float(kz_all[kcol])])
    rw = OP.momentum_row_weights(cc) if rw_override is None else rw_override
    lev = _Level(m, 1, nz, 1/180., cc, kz, 0.0, rw, False, mask=mask)
    A, free, field = assemble(lev, 0)
    return A, free, field, kz[0]


def main():
    print('(a) Riesz-map blocks  B_u = H(div) [A_uu - K_curl], B_w = M_w, B_p = A_pp   (2x2 elements)')
    print(f'{"c":>7} {"kz":>6} {"p":>3} | {"blk3 diag(A)":>14} | {"blk3 Hdiv/L2/H1":>16} | {"blk2 uw|p":>14}')
    for cc in (1.0, 5405.4):
        for kcol in (0, 1):
            for N in (4, 6, 8, 12):
                A, free, field, kz = build(2, 2, N, cc, kcol)
                rwc = np.zeros(8); rwc[1:4] = 1.0            # curl rows only
                Ac, _, _, _ = build(2, 2, N, cc, kcol, rw_override=rwc)
                A = A[free][:, free].tocsr(); Ac = Ac[free][:, free].tocsr(); field = field[free]
                nd = A.shape[0]
                b = A@np.random.default_rng(0).standard_normal(nd); b /= np.linalg.norm(b)
                iu, iw, ip = (np.flatnonzero(np.isin(field, g)) for g in (VEL, VOR, PRE))
                Bu = (A[iu][:, iu] - Ac[iu][:, iu]).tocsc(); Bw = Ac[iw][:, iw].tocsc(); Bp = A[ip][:, ip].tocsc()
                lus = [(iu, spla.splu(Bu)), (iw, spla.splu(Bw)), (ip, spla.splu(Bp))]
                def riesz(r):
                    z = np.zeros_like(r)
                    for g, lu in lus: z[g] = lu.solve(r[g])
                    return z
                r3 = pcg_ritz(A, blockdiag_solver(A, [iu, iw, ip]), b)
                rr = pcg_ritz(A, riesz, b)
                r2 = pcg_ritz(A, blockdiag_solver(A, [np.concatenate([iu, iw]), ip]), b)
                print(f'{cc:7.0f} {kz:6.2f} {N:3d} | {r3[0]:5d} {r3[1]:8.1e} | {rr[0]:7d} {rr[1]:8.1e} | {r2[0]:5d} {r2[1]:8.1e}', flush=True)
            print()
    print('(b) h-sweep, p=8, exact 2-block (u,w | p), c=5405')
    for kcol in (0, 1):
        for ex in (2, 3, 4):
            A, free, field, kz = build(ex, ex, 8, 5405.4, kcol)
            A = A[free][:, free].tocsr(); field = field[free]; nd = A.shape[0]
            b = A@np.random.default_rng(0).standard_normal(nd); b /= np.linalg.norm(b)
            iu, iw, ip = (np.flatnonzero(np.isin(field, g)) for g in (VEL, VOR, PRE))
            rj = pcg_ritz(A, lambda r: r/A.diagonal(), b)
            r2 = pcg_ritz(A, blockdiag_solver(A, [np.concatenate([iu, iw]), ip]), b)
            print(f'  kz={kz:5.2f} {ex}x{ex} p=8 ndof={nd:6d}: jacobi {rj[0]:5d} {rj[1]:8.1e} | blk2 {r2[0]:4d} {r2[1]:8.1e}', flush=True)

if __name__ == '__main__':
    main()
