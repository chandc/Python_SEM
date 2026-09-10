"""Option F: low-order-refined (LOR) preconditioning for the 2D FOSLS operator.

A_LOR = the SAME least-squares functional (rows R0..R3 of lssem._apply_L_numpy,
fu = fv = 0, no pseudo-time) discretised with BILINEAR Q1 elements on the GLL
sub-cells of every spectral element, 2x2 Gauss quadrature per cell, same global
dof numbering (gidx*4 + field), same mask.  It is sparse (<= 9 nodes x 4 fields
per row) and shares its dof set with the SEM operator.

Tests, at c = 5405 and c = 1, N = 8, 12, 16:
  (1) spectral equivalence: kappa(A_LOR^-1 A_SEM) via CG Ritz values with an
      exact sparse-LU LOR solve as preconditioner (this is the ceiling for any
      LOR-based method);
  (2) smoothed-aggregation AMG on A_LOR (per-field near-null space, energy
      prolongation -- the F2 configuration) as preconditioner for A_SEM;
  (3) the same AMG on its own matrix A_LOR (how good is AMG on the LOR graph).
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, pyamg
from lssem2d.lssem import ls_coeffs
from adn_block_precond import pcg_ritz
from adn_hiptmair_2d import setup

U_, V_, P_, W_ = 0, 1, 2, 3


def build_lor(m, st):
    """Sparse Q1 FOSLS operator on the GLL sub-mesh, global numbering gidx*4+f."""
    N = m.N; n = N + 1; nu = st.nu
    a_mass, a_flux, _ = ls_coeffs(st)
    gid = m.gidx; ndof = (int(gid.max()) + 1)*4
    # Gauss 2x2 on the unit square
    gq = np.array([0.5 - 0.5/np.sqrt(3), 0.5 + 0.5/np.sqrt(3)])
    rows, cols, vals = [], [], []
    for e in range(m.nelem):
        x = m.xnod[e]; y = m.ynod[e]
        for i in range(N):
            for j in range(N):
                dx = x[i+1] - x[i]; dy = y[j+1] - y[j]
                nodes = [gid[e, i, j], gid[e, i+1, j], gid[e, i, j+1], gid[e, i+1, j+1]]   # (00,10,01,11)
                Ac = np.zeros((16, 16))
                for xi in gq:
                    for eta in gq:
                        Nk = np.array([(1-xi)*(1-eta), xi*(1-eta), (1-xi)*eta, xi*eta])
                        Nx = np.array([-(1-eta), (1-eta), -eta, eta])/dx
                        Ny = np.array([-(1-xi), -xi, (1-xi), xi])/dy
                        L = np.zeros((4, 16))                       # rows R0..R3, cols node k*4+f
                        for k in range(4):
                            L[0, k*4+U_] += a_mass*Nk[k]; L[0, k*4+P_] += a_flux*Nx[k]; L[0, k*4+W_] += a_flux*nu*Ny[k]
                            L[1, k*4+V_] += a_mass*Nk[k]; L[1, k*4+P_] += a_flux*Ny[k]; L[1, k*4+W_] -= a_flux*nu*Nx[k]
                            L[2, k*4+U_] += Nx[k];        L[2, k*4+V_] += Ny[k]
                            L[3, k*4+W_] += Nk[k];        L[3, k*4+U_] += Ny[k];        L[3, k*4+V_] -= Nx[k]
                        Ac += (dx*dy/4.0)*(L.T @ L)
                gd = np.array([nd*4 + f for nd in nodes for f in range(4)])
                rows.append(np.repeat(gd, 16)); cols.append(np.tile(gd, 16)); vals.append(Ac.ravel())
    A = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)).tocsr()
    return 0.5*(A + A.T)


def main():
    print(f'{"c":>6} {"N":>3} {"ndof":>6} {"nnz/row":>7} | {"jacobi":>14} | {"exact LOR solve":>16} | {"AMG(LOR) -> SEM":>16} {"lev":>3} {"opcx":>5} | {"AMG(LOR) -> LOR":>16} | {"xAx SEM/LOR rand,smooth":>24}')
    for DT in (1.85e-4, 1.0):
        for N in (8, 12, 16):
            m, st, fu, fv, A, free, g = setup(N, DT)
            t0 = time.time(); AL = build_lor(m, st); tb = time.time() - t0
            Af = A[free][:, free].tocsr(); ALf = AL[free][:, free].tocsr(); nd = Af.shape[0]
            fld = (np.arange(free.size) % 4)[free]
            rng = np.random.default_rng(0); b = Af@rng.standard_normal(nd); b /= np.linalg.norm(b)
            xr = rng.standard_normal(nd)
            # smooth test vector: low-frequency field on the mesh
            gx = np.zeros(free.size); gy = np.zeros(free.size)
            for e in range(m.nelem):
                for f in range(4):
                    gx[g[e, :, :, f]] = m.xnod[e][:, None]; gy[g[e, :, :, f]] = m.ynod[e][None, :]
            xs = (np.sin(np.pi*gx)*np.sin(np.pi*gy))[free]*(1 + 0.3*(fld == 2))
            ratio = lambda x: (x@(Af@x))/(x@(ALf@x))
            rJ = pcg_ritz(Af, lambda r: r/Af.diagonal(), b, maxit=8000)
            lu = spla.splu(ALf.tocsc()); rE = pcg_ritz(Af, lambda r: lu.solve(r), b, maxit=8000)
            B = np.zeros((nd, 4))
            for v in range(4): B[fld == v, v] = 1.0
            ml = pyamg.smoothed_aggregation_solver(ALf, B=B, max_coarse=20, smooth='energy'); M = ml.aspreconditioner(cycle='V')
            rA = pcg_ritz(Af, lambda r: M.matvec(r), b, maxit=8000)
            bL = ALf@rng.standard_normal(nd); bL /= np.linalg.norm(bL)
            rL = pcg_ritz(ALf, lambda r: M.matvec(r), bL, maxit=8000)
            print(f'{1/DT:6.0f} {N:3d} {nd:6d} {ALf.nnz/nd:7.1f} | {rJ[0]:5d} {rJ[1]:8.1e} | {rE[0]:7d} {rE[1]:8.1e} | {rA[0]:7d} {rA[1]:8.1e} {len(ml.levels):3d} {ml.operator_complexity():5.2f} | {rL[0]:7d} {rL[1]:8.1e} | {ratio(xr):11.2f} {ratio(xs):11.2f}   [LOR build {tb:.0f}s]', flush=True)
        print()

if __name__ == '__main__':
    main()
