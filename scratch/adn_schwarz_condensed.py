"""EXACT memory reduction of the vertex-patch Schwarz solve by static
condensation of element interiors (substructuring).

Interior nodes of an element (local i,j in 1..N-1) couple only to that
element's own dofs, so within a patch the interior block K_II is block
diagonal over the patch's elements and is SHARED by every patch containing
the element.  Factor once per element:   K_II,e  (n_I = (N-1)^2 F)
and once per patch the dense Schur complement on the patch's edge dofs:
    S_P = K_BB - K_BI K_II^-1 K_IB          (n_B ~ 12 N F  for 4 elements)
Apply: z_B = S_P^-1 (r_B - K_BI K_II^-1 r_I),  z_I = K_II^-1 (r_I - K_IB z_B).
Identical to the dense patch solve to round-off -- same CG iterations --
with memory  sum_e n_I^2/2 + sum_P n_B^2/2  instead of  sum_P n_P^2/2.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from adn_block_precond import pcg_ritz
from adn_schwarz_2d import schwarz
from adn_schwarz_2d_full import setup, coarse_full


def condensed_schwarz(Af, m, g, free, N, patches_elems):
    """patches_elems: list of element lists.  Returns (apply, stored_entries)."""
    nd = Af.shape[0]; idx = -np.ones(free.size, dtype=int); idx[np.flatnonzero(free)] = np.arange(nd)
    Ad = Af.toarray() if nd <= 30000 else None
    def sub(r, c):
        return (Ad[np.ix_(r, c)] if Ad is not None else Af[r][:, c].toarray())
    # element interior dofs (free) and their factors, shared across patches
    int_dofs, int_fac = {}, {}
    entries = 0
    for e in range(m.nelem):
        d = idx[g[e, 1:N, 1:N, :].ravel()]; d = np.unique(d[d >= 0]); int_dofs[e] = d
        K = sub(d, d); sc = 1/np.sqrt(np.diag(K)); int_fac[e] = (sc, sla.cho_factor(K*sc[:, None]*sc[None, :], lower=True))
        entries += d.size*(d.size+1)//2
    def isolve(e, r):
        sc, cf = int_fac[e]; return sc*sla.cho_solve(cf, sc*r)
    pats = []
    for es in patches_elems:
        alld = np.unique(np.concatenate([idx[g[e].ravel()] for e in es])); alld = alld[alld >= 0]
        I = [int_dofs[e] for e in es]; Iall = np.concatenate(I)
        B = np.setdiff1d(alld, Iall)
        KBB = sub(B, B); S = KBB.copy()
        KIB = [sub(d, B) for d in I]
        for e, d, kib in zip(es, I, KIB):
            X = np.column_stack([isolve(e, kib[:, j]) for j in range(kib.shape[1])])   # K_II^-1 K_IB
            S -= kib.T @ X
        S = 0.5*(S + S.T); sc = 1/np.sqrt(np.diag(S)); Sf = (sc, sla.cho_factor(S*sc[:, None]*sc[None, :], lower=True))
        entries += B.size*(B.size+1)//2
        pats.append((es, I, KIB, B, Sf))
    def apply(r):
        z = np.zeros_like(r)
        for es, I, KIB, B, (sc, cf) in pats:
            rB = r[B].copy(); yI = []
            for e, d, kib in zip(es, I, KIB):
                y = isolve(e, r[d]); yI.append(y); rB -= kib.T @ y
            zB = sc*sla.cho_solve(cf, sc*rB); z[B] += zB
            for e, d, kib, y in zip(es, I, KIB, yI):
                z[d] += y - isolve(e, kib @ zB)
        return z
    return apply, entries


def main():
    DT = 1.85e-4
    print(f'2D, c=5405, 4x4 elements: dense vertex patch vs statically condensed (exact), both + p=2 coarse')
    print(f'{"N":>3} {"ndof":>6} | {"dense: it":>10} {"kappa":>8} {"entries":>10} {"ms/it":>7} | {"condensed: it":>13} {"kappa":>8} {"entries":>10} {"ms/it":>7} | {"mem ratio":>9}')
    for N in (8, 12, 16, 24):
        m, A, free, g = setup(N, DT, 4)
        Af = A[free][:, free].tocsr(); nd = Af.shape[0]
        idx = -np.ones(free.size, dtype=int); idx[np.flatnonzero(free)] = np.arange(nd)
        corners = {}
        for e in range(m.nelem):
            for (a, b) in ((0, 0), (0, N), (N, 0), (N, N)):
                corners.setdefault(m.gidx[e, a, b], []).append(e)
        pe = list(corners.values())
        def dofs_of(es):
            d = idx[np.concatenate([g[e].ravel() for e in es])]; return np.unique(d[d >= 0])
        vp = [dofs_of(es) for es in pe]
        rng = np.random.default_rng(0); b = Af@rng.standard_normal(nd); b /= np.linalg.norm(b)
        Cc = coarse_full(Af, m, g, free)
        V = schwarz(Af, vp); dense_entries = sum(p.size*(p.size+1)//2 for p in vp)
        t0 = time.perf_counter(); rd = pcg_ritz(Af, lambda r: V(r) + Cc(r), b); td = (time.perf_counter()-t0)/rd[0]
        Cnd, cond_entries = condensed_schwarz(Af, m, g, free, N, pe)
        t0 = time.perf_counter(); rc = pcg_ritz(Af, lambda r: Cnd(r) + Cc(r), b); tc = (time.perf_counter()-t0)/rc[0]
        print(f'{N:3d} {nd:6d} | {rd[0]:10d} {rd[1]:8.1e} {dense_entries:10d} {1e3*td:7.1f} | {rc[0]:13d} {rc[1]:8.1e} {cond_entries:10d} {1e3*tc:7.1f} | {dense_entries/cond_entries:8.1f}x', flush=True)

if __name__ == '__main__':
    main()
