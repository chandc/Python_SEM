"""Vertex-patch Schwarz on the FULL 2D operator A (u,v,p,w in the patch, no
vorticity elimination), at c=5405 and c=1, p-sweep and h-sweep with/without
the Galerkin p=2 coarse correction on the full system."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from copy import copy
import lssem2d
from lssem2d.precond import _p_interp
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
import fosls_assemble as FA
from adn_block_precond import pcg_ritz
from adn_schwarz_2d import schwarz


def setup(N, DT, EX):
    lssem2d.set_backend('numpy')
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/1000, dt=DT, fac1=1.0)
    n = N + 1; fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)
    A, free, g = FA.assemble(st, fu, fv, pin_p=True)
    return m, A, free, g


def coarse_full(Af, m, g, free, pc=2):
    mc = copy(m); mc.N = pc; mc.nterm = pc + 1
    mc.gidx = -np.ones((m.nelem, pc+1, pc+1), dtype=int)
    mc.xnod = np.zeros((m.nelem, pc+1)); mc.ynod = np.zeros((m.nelem, pc+1)); mc.wq = np.zeros((m.nelem, pc+1, pc+1))
    mc.setup_derived(); mc.compute_global_indices()
    L = _p_interp(pc, m.N); gid = m.gidx; ng = int(gid.max())+1
    mult = np.zeros(ng); np.add.at(mult, gid.ravel(), 1.0); mw = 1.0/mult
    ngc = int(mc.gidx.max())+1
    rows, cols, vals = [], [], []
    Le = np.kron(L, L)
    for e in range(m.nelem):
        for f in range(4):
            rf = g[e, :, :, f].ravel(); cc = mc.gidx[e].ravel()*4 + f
            rows.append(np.repeat(rf, cc.size)); cols.append(np.tile(cc, rf.size))
            vals.append((Le*mw[gid[e].ravel()][:, None]).ravel())
    P = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(free.size, 4*ngc)).tocsr()
    P = P[np.flatnonzero(free)]
    keep = np.flatnonzero(np.asarray(abs(P).sum(0)).ravel() > 0); P = P[:, keep].tocsr()
    Ac = (P.T @ Af @ P).toarray(); Ac = 0.5*(Ac + Ac.T); Ac += 1e-12*np.linalg.eigvalsh(Ac).max()*np.eye(Ac.shape[0])
    cf = sla.cho_factor(Ac)
    return lambda r: P@sla.cho_solve(cf, P.T@r)


def run(N, DT, EX):
    m, A, free, g = setup(N, DT, EX)
    Af = A[free][:, free].tocsr(); nd = Af.shape[0]
    idx = -np.ones(free.size, dtype=int); idx[np.flatnonzero(free)] = np.arange(nd)
    corners = {}
    for e in range(m.nelem):
        for (a, b) in ((0, 0), (0, N), (N, 0), (N, N)):
            corners.setdefault(m.gidx[e, a, b], []).append(e)
    def dofs_of(es):
        d = idx[np.concatenate([g[e].ravel() for e in es])]; return np.unique(d[d >= 0])
    vp = [dofs_of(es) for es in corners.values()]
    rng = np.random.default_rng(0); b = Af@rng.standard_normal(nd); b /= np.linalg.norm(b)
    V = schwarz(Af, vp); Cc = coarse_full(Af, m, g, free)
    rJ = pcg_ritz(Af, lambda r: r/Af.diagonal(), b)
    rV = pcg_ritz(Af, V, b)
    rVC = pcg_ritz(Af, lambda r: V(r) + Cc(r), b)
    return nd, rJ, rV, rVC, max(p.size for p in vp)


def main():
    print(f'{"c":>6} {"mesh":>5} {"N":>3} {"ndof":>6} | {"jacobi":>14} | {"vertex patch":>14} | {"vpatch+coarse":>14} | patch dofs')
    for DT in (1.85e-4, 1.0):
        for N in (8, 12, 16):
            t0 = time.time(); nd, rJ, rV, rVC, ps = run(N, DT, 4)
            print(f'{1/DT:6.0f} {"4x4":>5} {N:3d} {nd:6d} | {rJ[0]:5d} {rJ[1]:8.1e} | {rV[0]:5d} {rV[1]:8.1e} | {rVC[0]:5d} {rVC[1]:8.1e} | {ps}   [{time.time()-t0:.0f}s]', flush=True)
        print()
    print('h-sweep at N=8, c=5405')
    for EX in (4, 6, 8):
        t0 = time.time(); nd, rJ, rV, rVC, ps = run(8, 1.85e-4, EX)
        print(f'{5405:6.0f} {f"{EX}x{EX}":>5} {8:3d} {nd:6d} | {rJ[0]:5d} {rJ[1]:8.1e} | {rV[0]:5d} {rV[1]:8.1e} | {rVC[0]:5d} {rVC[1]:8.1e} | {ps}   [{time.time()-t0:.0f}s]', flush=True)

if __name__ == '__main__':
    main()
