"""Part 3: the norm is on the PAIR, not on u and omega separately.

J(u,w) = |u|^2 + |div u|^2 + |curl u - w|^2 is a norm on (u, w - curl u), so the
Riesz map is block-diagonal only after the triangular change of variables.
Test the block-LDL^T preconditioner of the (u,w) block with the Schur
complement S_u = A_uu - A_uw A_ww^-1 A_wu REPLACED by the H(div) operator
M + K_div (= A_uu - K_curl), i.e. dropping the discrete curl-representation
defect, and A_ww by the curl-row mass M_w:

    z_w0 = M_w^-1 r_w
    (M + K_div) z_u = r_u - A_uw z_w0
    z_w  = z_w0 - M_w^-1 A_wu z_u
    z_p  = A_pp^-1 r_p

Everything but the H(div) solve is matrix-free.  If kappa is flat and small,
the whole large-c preconditioning problem IS an H(div) solve.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np, scipy.sparse.linalg as spla
from adn_block_precond import blockdiag_solver, pcg_ritz, VEL, VOR, PRE
from adn_block_precond2 import build


def main():
    print(f'{"c":>7} {"kz":>6} {"p":>3} | {"LDL Hdiv+Mw+App":>16} | {"LDL exact Su":>14} | {"blk2 uw|p":>14}')
    for cc in (5405.4, 1.0):
        for kcol in (0, 1):
            for N in (4, 6, 8, 12):
                A, free, field, kz = build(2, 2, N, cc, kcol)
                rwc = np.zeros(8); rwc[1:4] = 1.0
                Ac, _, _, _ = build(2, 2, N, cc, kcol, rw_override=rwc)
                A = A[free][:, free].tocsr(); Ac = Ac[free][:, free].tocsr(); field = field[free]
                nd = A.shape[0]
                b = A@np.random.default_rng(0).standard_normal(nd); b /= np.linalg.norm(b)
                iu, iw, ip = (np.flatnonzero(np.isin(field, g)) for g in (VEL, VOR, PRE))
                Auw = A[iu][:, iw].tocsr(); Awu = A[iw][:, iu].tocsr()
                Mw = spla.splu(Ac[iw][:, iw].tocsc()); Hdiv = spla.splu((A[iu][:, iu] - Ac[iu][:, iu]).tocsc())
                App = spla.splu(A[ip][:, ip].tocsc())
                Aww_lu = spla.splu(A[iw][:, iw].tocsc())
                # exact Schur complement (dense) for the control column
                Aww_inv_Awu = np.column_stack([Aww_lu.solve(Awu[:, j].toarray().ravel()) for j in range(len(iu))])
                Su = A[iu][:, iu].toarray() - Auw@Aww_inv_Awu
                Su_lu = spla.splu(__import__('scipy.sparse').sparse.csc_matrix(Su))
                def ldl(r, S, Wsolve):
                    z = np.zeros_like(r)
                    zw0 = Wsolve(r[iw])
                    zu = S(r[iu] - Auw@zw0)
                    z[iu] = zu; z[iw] = zw0 - Wsolve(Awu@zu); z[ip] = App.solve(r[ip])
                    return z
                r_h = pcg_ritz(A, lambda r: ldl(r, Hdiv.solve, Mw.solve), b)
                r_s = pcg_ritz(A, lambda r: ldl(r, Su_lu.solve, Aww_lu.solve), b)
                r_2 = pcg_ritz(A, blockdiag_solver(A, [np.concatenate([iu, iw]), ip]), b)
                print(f'{cc:7.0f} {kz:6.2f} {N:3d} | {r_h[0]:7d} {r_h[1]:8.1e} | {r_s[0]:5d} {r_s[1]:8.1e} | {r_2[0]:5d} {r_2[1]:8.1e}', flush=True)
            print()

if __name__ == '__main__':
    main()
