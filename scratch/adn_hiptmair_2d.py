"""2D time-dependent FOSLS: (A) does the 3D ADN diagnosis hold, (B) does a
Hiptmair (nodal auxiliary-space) sweep reach the divergence-free kernel?

2D VVP, unknowns (u, v, p, w), cavity mesh as p_indep_2d.py, fu=fv=0,
legacy weights a_mass=1, a_flux=dt  ->  dt = 1.85e-4 is the channel's c=5405.

(A) assembled A: Jacobi | (u,v,w)|p exact | (u,v)|w|p exact ; softest Jacobi
    mode of the Schur complement S_u and its divergence.
(B) on S_u (w eliminated, A_ww ~ diagonal GLL mass):
      Jacobi           D_S^-1
      Hiptmair (add)   D_S^-1 + C D_psi^-1 C^T,   C: psi -> (psi_y, -psi_x)
      Hiptmair (sym)   J -> H -> J multiplicative
    where D_psi = diag(C^T S_u C).  A stand-alone smoother cannot be flat in p;
    what we look for is kappa dropping by orders and the p-growth softening,
    which is what would make it a usable PMG smoother.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sla
import lssem2d
from lssem2d import solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from lssem2d.operators import dUdx, dUdy
import fosls_assemble as FA
from adn_block_precond import pcg_ritz, blockdiag_solver

RE, EX = 1000.0, 4
U_, V_, P_, W_ = 0, 1, 2, 3


def setup(N, DT):
    lssem2d.set_backend('numpy')
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)
    A, free, g = FA.assemble(st, fu, fv, pin_p=True)
    return m, st, fu, fv, A, free, g


def curl_map(m, st, g, free):
    """C: global psi (scalar node) -> global (u,v) dofs, nodal-averaged grad-perp,
    rows of fixed u/v dofs and columns of boundary psi nodes zeroed."""
    n = m.N + 1; D = st.D if hasattr(st, 'D') else diff_matrix(m.N)
    gid = m.gidx; ng = int(gid.max()) + 1
    mult = np.zeros(ng); np.add.at(mult, gid.ravel(), 1.0); mw = 1.0/mult
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            Psi = np.zeros((m.nelem, n, n)); Psi[:, i, j] = 1.0
            U = dUdy(Psi, D, m.facy); V = -dUdx(Psi, D, m.facx)
            col = gid[:, i, j]                                   # (nelem,)
            for e in range(m.nelem):
                r = gid[e].ravel()
                rows.append(g[e, :, :, U_].ravel()); cols.append(np.full(r.size, col[e])); vals.append(U[e].ravel()*mw[r])
                rows.append(g[e, :, :, V_].ravel()); cols.append(np.full(r.size, col[e])); vals.append(V[e].ravel()*mw[r])
    ndof = free.size
    C = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ng)).tocsr()
    # psi fixed (=0) on nodes where v or u is prescribed (walls, lid)
    fixed_node = np.zeros(ng, bool)
    for f in (U_, V_):
        gf = g[..., f].ravel(); fixed_node[gid.ravel()[~free[gf]]] = True
    C = sp.diags(free.astype(float)) @ C @ sp.diags((~fixed_node).astype(float))
    return C.tocsr(), fixed_node


def main():
    for DT in (1.85e-4, 1.0):
        print(f'\n===== dt={DT:g}  (c = 1/dt = {1/DT:.0f}), Re={RE:.0f}, {EX}x{EX} elements =====')
        print(f'{"N":>3} {"ndof":>6} | {"jacobi A":>14} | {"(u,v,w)|p":>14} | {"(u,v)|w|p":>14} | {"jacobi S_u":>14} | {"Hiptmair add":>14} | {"Hiptmair sym":>14} | softest-Jacobi S_u mode: |div|,|curl| / |u|')
        for N in (8, 12, 16, 20):
            t0 = time.time()
            m, st, fu, fv, A, free, g = setup(N, DT)
            field = np.arange(free.size) % 4
            C_full, _ = curl_map(m, st, g, free)
            Af = A[free][:, free].tocsr(); fld = field[free]; nd = Af.shape[0]
            iu = np.flatnonzero(np.isin(fld, [U_, V_])); iw = np.flatnonzero(fld == W_); ip = np.flatnonzero(fld == P_)
            rng = np.random.default_rng(0)
            b = Af@rng.standard_normal(nd); b /= np.linalg.norm(b)
            rJ = pcg_ritz(Af, lambda r: r/Af.diagonal(), b)
            r2 = pcg_ritz(Af, blockdiag_solver(Af, [np.concatenate([iu, iw]), ip]), b)
            r3 = pcg_ritz(Af, blockdiag_solver(Af, [iu, iw, ip]), b)
            # --- Schur complement in u (A_ww ~ diagonal mass) ---
            Auu = Af[iu][:, iu]; Auw = Af[iu][:, iw]; Awu = Af[iw][:, iu]; Aww = Af[iw][:, iw]
            dw = Aww.diagonal(); offw = abs(Aww - sp.diags(dw)).max()/dw.max()
            Su = (Auu - Auw @ sp.diags(1.0/dw) @ Awu).tocsr(); Su = 0.5*(Su + Su.T)
            DS = Su.diagonal()
            bS = Su@rng.standard_normal(len(iu)); bS /= np.linalg.norm(bS)
            rSJ = pcg_ritz(Su, lambda r: r/DS, bS)
            # --- Hiptmair ---
            C = C_full[free][:, :].tocsr()[iu]              # rows: u,v dofs of the free system
            keep = np.flatnonzero(np.asarray(abs(C).sum(0)).ravel() > 0); C = C[:, keep].tocsr()
            Ppsi = (C.T @ Su @ C).tocsr(); Dpsi = Ppsi.diagonal(); Dpsi[Dpsi <= 0] = np.inf
            # damping: an undamped multiplicative sweep is indefinite when
            # lambda_max(D^-1 S) > 2 (it is), which turns CG into NaN.
            def lmax(op, n, it=20):
                v = rng.standard_normal(n); v /= np.linalg.norm(v)
                for _ in range(it):
                    w = op(v); lam = np.linalg.norm(w); v = w/lam
                return lam
            wJ = 1.0/lmax(lambda v: (Su@v)/DS, len(iu))
            wH = 1.0/lmax(lambda v: (Ppsi@v)/Dpsi, len(keep))
            Hadd = lambda r: r/DS + C@((C.T@r)/Dpsi)
            def Hsym(r):
                z = wJ*r/DS
                z = z + wH*(C@((C.T@(r - Su@z))/Dpsi))
                return z + wJ*(r - Su@z)/DS
            rHa = pcg_ritz(Su, Hadd, bS)
            rHs = pcg_ritz(Su, Hsym, bS)
            # --- softest Jacobi mode of S_u (dense, N<=12) ---
            note = ''
            if N <= 12:
                Sd = Su.toarray(); d = DS
                w, V = sla.eigh(Sd/np.sqrt(np.outer(d, d))); v = V[:, 0]/np.sqrt(d)
                # divergence / curl of that field via the element operators
                loc = np.zeros((m.nelem, m.N+1, m.N+1, 4)); vf = np.zeros(free.size); vf[np.flatnonzero(free)[iu]] = v
                loc = vf[g]                                  # (nelem,n,n,4)
                D = diff_matrix(m.N)
                div = dUdx(loc[..., U_], D, m.facx) + dUdy(loc[..., V_], D, m.facy)
                crl = dUdx(loc[..., V_], D, m.facx) - dUdy(loc[..., U_], D, m.facy)
                wq = m.wq; nrm = lambda f: np.sqrt((wq*f*f).sum())
                un = np.sqrt((wq*(loc[..., U_]**2 + loc[..., V_]**2)).sum())
                note = f'div {nrm(div)/un:8.2e}  curl {nrm(crl)/un:8.2e}'
            print(f'{N:3d} {nd:6d} | {rJ[0]:5d} {rJ[1]:8.1e} | {r2[0]:5d} {r2[1]:8.1e} | {r3[0]:5d} {r3[1]:8.1e} | {rSJ[0]:5d} {rSJ[1]:8.1e} | {rHa[0]:5d} {rHa[1]:8.1e} | {rHs[0]:5d} {rHs[1]:8.1e} | {note}   [A_ww offdiag/diag {offw:.1e}, wJ {wJ:.2f} wH {wH:.2f}, {time.time()-t0:.0f}s]', flush=True)

if __name__ == '__main__':
    main()
