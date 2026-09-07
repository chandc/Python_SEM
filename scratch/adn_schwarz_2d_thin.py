"""Minimal-overlap Schwarz: element + ONE node layer of each edge neighbour,
vs the full vertex patch.  Decides the 3D memory question: a vertex patch is
4 elements (16x an element block, dense); element+1 layer is ~1.5x."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
from adn_block_precond import pcg_ritz
from adn_schwarz_2d import schwarz
from adn_schwarz_2d_full import setup, coarse_full


def thin_patches(m, g, idx, N, layers=1):
    gid = m.gidx; nel = m.nelem
    node_elems = {}
    for e in range(nel):
        for gn in np.unique(gid[e]): node_elems.setdefault(gn, set()).add(e)
    patches = []
    for e in range(nel):
        nodes = set(np.unique(gid[e]).tolist())
        # edge neighbours: elements sharing >= 2 nodes with e
        nb = {}
        for gn in np.unique(gid[e]):
            for e2 in node_elems[gn]:
                if e2 != e: nb[e2] = nb.get(e2, 0) + 1
        for e2, cnt in nb.items():
            if cnt < 2: continue
            shared = np.isin(gid[e2], gid[e])           # (n,n) bool on e2's local grid
            rows = np.flatnonzero(shared.all(axis=1)); cols = np.flatnonzero(shared.all(axis=0))
            for r in rows:                              # shared row r of e2 -> add rows r±layers
                for L in range(1, layers+1):
                    for rr in (r-L, r+L):
                        if 0 <= rr <= N: nodes.update(gid[e2, rr, :].tolist())
            for c in cols:
                for L in range(1, layers+1):
                    for cc in (c-L, c+L):
                        if 0 <= cc <= N: nodes.update(gid[e2, :, cc].tolist())
        d = idx[(np.array(sorted(nodes))[:, None]*4 + np.arange(4)[None, :]).ravel()]
        patches.append(np.unique(d[d >= 0]))
    return patches


def main():
    DT = 1.85e-4
    print(f'full 2D operator, c=5405, 4x4 elements: element block | element+1 layer | element+2 layers | vertex patch, each + p=2 coarse')
    print(f'{"N":>3} {"ndof":>6} | {"elem+coarse":>14} | {"elem+1L+coarse":>16} | {"elem+2L+coarse":>16} | {"vpatch+coarse":>14} | patch dofs (elem, 1L, 2L, vp)')
    for N in (8, 12, 16):
        t0 = time.time()
        m, A, free, g = setup(N, DT, 4)
        Af = A[free][:, free].tocsr(); nd = Af.shape[0]
        idx = -np.ones(free.size, dtype=int); idx[np.flatnonzero(free)] = np.arange(nd)
        def dofs_of(es):
            d = idx[np.concatenate([g[e].ravel() for e in es])]; return np.unique(d[d >= 0])
        elem = [dofs_of([e]) for e in range(m.nelem)]
        l1 = thin_patches(m, g, idx, N, 1); l2 = thin_patches(m, g, idx, N, 2)
        corners = {}
        for e in range(m.nelem):
            for (a, b) in ((0, 0), (0, N), (N, 0), (N, N)):
                corners.setdefault(m.gidx[e, a, b], []).append(e)
        vp = [dofs_of(es) for es in corners.values()]
        rng = np.random.default_rng(0); b = Af@rng.standard_normal(nd); b /= np.linalg.norm(b)
        Cc = coarse_full(Af, m, g, free)
        res = []
        for P in (elem, l1, l2, vp):
            Sw = schwarz(Af, P); res.append(pcg_ritz(Af, lambda r, Sw=Sw: Sw(r) + Cc(r), b))
        sizes = tuple(max(p.size for p in P) for P in (elem, l1, l2, vp))
        print(f'{N:3d} {nd:6d} | ' + ' | '.join(f'{it:5d} {k:8.1e}' for it, k in res) + f' | {sizes}   [{time.time()-t0:.0f}s]', flush=True)

if __name__ == '__main__':
    main()
