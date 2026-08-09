"""Load a Fortran LSSEM grid file (bfs_grid.dat format) into a lssem2d Mesh.

Format:
  nelem nterm
  wht(1..nelem)          element heights
  wid(1..nelem)          element widths
  per element:
    xp(nterm)            GLL nodes mapped to the element x-range
    yp(nterm)            GLL nodes mapped to the element y-range
    iwest ieast isouth inorth      (1-based; 0 = boundary)
    ibcw  ibce  ibcs   ibcn        (0 interior, 1 wall, 2 lid, 3 inlet, 4 outlet)
"""
import sys
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import Mesh


def load(path):
    toks = open(path).read().split()
    k = 0
    def take(m):
        nonlocal k
        v = toks[k:k+m]; k += m; return v
    nelem, nterm = int(take(1)[0]), int(take(1)[0])
    N = nterm - 1
    wht = np.array([float(t) for t in take(nelem)])
    wid = np.array([float(t) for t in take(nelem)])

    m = Mesh(nelem, N)
    XP = np.zeros((nelem, nterm)); YP = np.zeros((nelem, nterm))
    for e in range(nelem):
        XP[e] = [float(t) for t in take(nterm)]
        YP[e] = [float(t) for t in take(nterm)]
        w, ee, s, nn = [int(t) for t in take(4)]
        m.neighbour[e] = [w-1, ee-1, s-1, nn-1]        # 1-based -> 0-based, 0 -> -1
        m.bc[e] = [int(t) for t in take(4)]
    assert k == len(toks), f"trailing tokens: {len(toks)-k}"

    m.x0 = XP[:, 0].copy(); m.y0 = YP[:, 0].copy()
    m.hx = wid.copy();      m.hy = wht.copy()
    m.setup_derived()
    # setup_derived rebuilds xnod/ynod from x0+hx*(xi+1)/2; check it reproduces the file
    dx = np.max(np.abs(m.xnod - XP)); dy = np.max(np.abs(m.ynod - YP))
    m.compute_global_indices()
    return m, dx, dy


if __name__ == '__main__':
    from lssem2d.assembly import gather_scatter
    import collections
    p = sys.argv[1] if len(sys.argv) > 1 else \
        '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/bfs_grid.dat'
    m, dx, dy = load(p)
    n = m.N + 1
    print(f"loaded {p}")
    print(f"  nelem={m.nelem}  order N={m.N} (nterm={n})  dofs={m.nelem*n*n*4}")
    print(f"  node-coordinate reconstruction error: dx={dx:.2e}  dy={dy:.2e}")
    print(f"  x range [{m.x0.min():.3f}, {(m.x0+m.hx).max():.3f}]"
          f"   y range [{m.y0.min():.3f}, {(m.y0+m.hy).max():.3f}]")
    print(f"  total area = {np.sum(m.wq):.6f}")
    NAME = {0:'-',1:'wall',2:'lid',3:'inlet',4:'outlet'}
    cnt = collections.Counter()
    for e in range(m.nelem):
        for d in range(4):
            if m.bc[e, d]: cnt[NAME[m.bc[e, d]]] += 1
    print(f"  boundary edges: {dict(cnt)}")
    # connectivity invariants
    OPP = {0:1, 1:0, 2:3, 3:2}; bad = 0
    for i in range(m.nelem):
        for d in range(4):
            j = m.neighbour[i, d]
            if j >= 0 and m.neighbour[j, OPP[d]] != i: bad += 1
            if j >= 0 and m.bc[i, d] != 0: bad += 1
            if j < 0 and m.bc[i, d] == 0: bad += 1
    print(f"  connectivity invariant violations: {bad}")
    mult = gather_scatter(m, np.ones((m.nelem, n, n)))
    print(f"  multiplicity histogram: {dict(sorted(collections.Counter(mult.ravel().astype(int)).items()))}")
    # where is the inlet, and what y does it span?
    inl = [e for e in range(m.nelem) if m.bc[e, 0] == 3]
    print(f"  inlet elements {inl}: y in "
          f"[{min(m.y0[e] for e in inl):.3f}, {max(m.y0[e]+m.hy[e] for e in inl):.3f}]"
          f" at x={m.x0[inl[0]]:.3f}")
