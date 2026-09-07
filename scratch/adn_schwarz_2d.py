"""2D: the AFW / Pavarino prescription on the vorticity-eliminated operator S_u.

The nodal Hiptmair sweep failed (adn_hiptmair_2d.log): on C0 GLL elements
the discretely divergence-free kernel has no potential representation, so the
auxiliary-space sweep never reaches it.  The other H(div) prescription is the
OVERLAPPING VERTEX-PATCH Schwarz smoother (Arnold-Falk-Winther 2000), which
for spectral elements is Pavarino's p-independent additive Schwarz.

Test on S_u (c = 5405):  Jacobi | element blocks | vertex patches (overlap =
all elements sharing the vertex) | each + Galerkin p=2 coarse correction.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sla
from copy import copy
from lssem2d.precond import _p_interp
from adn_block_precond import pcg_ritz
from adn_hiptmair_2d import setup, U_, V_, P_, W_, EX

PC = 2


def schwarz(Su, patches):
    lus = [(idx, sla.cho_factor(Su[idx][:, idx].toarray())) for idx in patches if idx.size]
    def apply(r):
        z = np.zeros_like(r)
        for idx, cf in lus: z[idx] += sla.cho_solve(cf, r[idx])
        return z
    return apply


def coarse(Su, m, g, free, iu, pc=PC):
    """Galerkin p=pc coarse correction P (P^T S_u P)^-1 P^T on the u,v dofs."""
    mc = copy(m); mc.N = pc; mc.nterm = pc + 1
    mc.gidx = -np.ones((m.nelem, pc+1, pc+1), dtype=int)
    mc.xnod = np.zeros((m.nelem, pc+1)); mc.ynod = np.zeros((m.nelem, pc+1)); mc.wq = np.zeros((m.nelem, pc+1, pc+1))
    mc.setup_derived(); mc.compute_global_indices()
    L = _p_interp(pc, m.N)                                   # (n_f, n_c)
    gid = m.gidx; ng = int(gid.max())+1
    mult = np.zeros(ng); np.add.at(mult, gid.ravel(), 1.0); mw = 1.0/mult
    ngc = int(mc.gidx.max())+1
    rows, cols, vals = [], [], []
    for e in range(m.nelem):
        Le = np.kron(L, L)                                    # (n_f^2, n_c^2): fine (a,b) <- coarse (i,j)
        for f in (U_, V_):
            rf = g[e, :, :, f].ravel(); cc = mc.gidx[e].ravel()*2 + f
            rows.append(np.repeat(rf, cc.size)); cols.append(np.tile(cc, rf.size))
            vals.append((Le*mw[gid[e].ravel()][:, None]).ravel())
    P = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(free.size, 2*ngc)).tocsr()
    P = P[np.flatnonzero(free)[iu]]                           # rows: free u,v dofs in S_u numbering
    keep = np.flatnonzero(np.asarray(abs(P).sum(0)).ravel() > 0); P = P[:, keep].tocsr()
    Ac = (P.T @ Su @ P).toarray(); Ac = 0.5*(Ac + Ac.T)
    w = np.linalg.eigvalsh(Ac); Ac += 1e-12*w.max()*np.eye(Ac.shape[0])
    cf = sla.cho_factor(Ac)
    return lambda r: P@sla.cho_solve(cf, P.T@r), P.shape[1]


def main():
    DT = 1.85e-4
    print(f'S_u (omega eliminated), dt={DT:g} (c=5405), Re=1000, {EX}x{EX} elements, coarse p={PC}')
    print(f'{"N":>3} {"ndof_u":>7} | {"jacobi":>14} | {"elem blocks":>14} | {"vertex patch":>14} | {"jac+coarse":>14} | {"elem+coarse":>14} | {"vpatch+coarse":>16} | {"ncoarse":>7}')
    for N in (8, 12, 16):
        t0 = time.time()
        m, st, fu, fv, A, free, g = setup(N, DT)
        field = np.arange(free.size) % 4
        Af = A[free][:, free].tocsr(); fld = field[free]
        iu = np.flatnonzero(np.isin(fld, [U_, V_])); iw = np.flatnonzero(fld == W_)
        Auu = Af[iu][:, iu]; Auw = Af[iu][:, iw]; Awu = Af[iw][:, iu]; dw = Af[iw][:, iw].diagonal()
        Su = (Auu - Auw @ sp.diags(1.0/dw) @ Awu).tocsr(); Su = 0.5*(Su + Su.T); DS = Su.diagonal()
        # dof bookkeeping: free-u index of (e, a, b, f)
        loc2su = -np.ones(free.size, dtype=int); loc2su[np.flatnonzero(free)[iu]] = np.arange(len(iu))
        def dofs_of(elems):
            d = np.concatenate([g[e, :, :, [U_, V_]].ravel() for e in elems])
            d = loc2su[d]; return np.unique(d[d >= 0])
        elem_patches = [dofs_of([e]) for e in range(m.nelem)]
        # vertex patches: elements sharing a corner node
        corners = {}
        for e in range(m.nelem):
            for (a, b) in ((0, 0), (0, N), (N, 0), (N, N)):
                corners.setdefault(m.gidx[e, a, b], []).append(e)
        vert_patches = [dofs_of(es) for es in corners.values()]
        rng = np.random.default_rng(0); b = Su@rng.standard_normal(len(iu)); b /= np.linalg.norm(b)
        Cc, nc = coarse(Su, m, g, free, iu)
        J = lambda r: r/DS; E = schwarz(Su, elem_patches); V = schwarz(Su, vert_patches)
        res = [pcg_ritz(Su, pre, b) for pre in (J, E, V,
               lambda r: J(r) + Cc(r), lambda r: E(r) + Cc(r), lambda r: V(r) + Cc(r))]
        print(f'{N:3d} {len(iu):7d} | ' + ' | '.join(f'{it:5d} {k:8.1e}' for it, k in res) + f' | {nc:7d}   [{time.time()-t0:.0f}s, {len(vert_patches)} vertex patches, max {max(p.size for p in vert_patches)} dofs]', flush=True)

if __name__ == '__main__':
    main()
