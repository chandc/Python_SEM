"""Meshes for the low-order FOSLS code.

Structured quadrilateral meshes only, for now.  gmsh/meshio arrive with the
triangle stage; the point of starting structured is that the SPECTRAL code can
run the identical geometry, so the first validation has ground truth.

The invariants in `check` are the gate-G1 criteria.  The Euler characteristic
is there because it catches what an area check cannot: a duplicated or dangling
node leaves the total area correct while breaking the connectivity.  That
failure is not hypothetical in this project -- an earlier curvilinear cylinder
mesh had 224 unmerged nodes, was hydrodynamically a SLIT, and had an area
correct to 1e-15.
"""
import numpy as np

NF = 4                                  # u, v, p, omega


class QuadMesh:
    """Nodes, quad connectivity (counter-clockwise), and tagged boundary edges."""

    def __init__(self, xy, quads, edge_tags):
        self.xy = np.asarray(xy, float)          # (nnode, 2)
        self.quads = np.asarray(quads, int)      # (nelem, 4)
        self.edge_tags = edge_tags               # {name: (nedge, 2) node pairs}

    nnode = property(lambda s: s.xy.shape[0])
    nelem = property(lambda s: s.quads.shape[0])
    ndof = property(lambda s: s.xy.shape[0]*NF)

    def areas(self):
        """Signed area of each quad by the shoelace formula."""
        x = self.xy[self.quads, 0]
        y = self.xy[self.quads, 1]
        return 0.5*sum(x[:, i]*y[:, (i+1) % 4] - x[:, (i+1) % 4]*y[:, i]
                       for i in range(4))

    def edges(self):
        """All element edges as sorted node pairs, with their incidence count."""
        from collections import Counter
        c = Counter()
        for q in self.quads:
            for i in range(4):
                c[tuple(sorted((int(q[i]), int(q[(i+1) % 4]))))] += 1
        return c

    def check(self, exact_area=None, tol=1e-12):
        """Gate G1.  Returns (ok, list of (name, detail, passed))."""
        out = []
        a = self.areas()
        out.append(('all element areas strictly positive',
                    f'min {a.min():.6e}', bool(a.min() > 0)))
        if exact_area is not None:
            rel = abs(a.sum() - exact_area)/abs(exact_area)
            out.append(('total area vs analytic',
                        f'{a.sum():.12f} vs {exact_area:.12f}, rel {rel:.2e}',
                        bool(rel < tol)))
        c = self.edges()
        inc = np.array(list(c.values()))
        out.append(('every edge shared by 1 or 2 elements',
                    f'counts {sorted(set(inc.tolist()))}',
                    bool(set(inc.tolist()) <= {1, 2})))
        nb = int((inc == 1).sum())
        tagged = sum(len(v) for v in self.edge_tags.values())
        out.append(('tagged boundary edges == unshared edges',
                    f'{tagged} tagged, {nb} unshared', tagged == nb))
        V, E, F = self.nnode, len(c), self.nelem
        out.append(('Euler characteristic V - E + F',
                    f'{V} - {E} + {F} = {V-E+F}', (V - E + F) == 1))
        # a duplicated node would leave area and Euler intact only if it were
        # also disconnected, so check coordinates are distinct as well
        uniq = len({tuple(np.round(p, 12)) for p in self.xy})
        out.append(('no duplicated node coordinates',
                    f'{uniq} unique of {self.nnode}', uniq == self.nnode))
        return all(p for _, _, p in out), out


def build_rect(x0, x1, y0, y1, nx, ny, distort=0.0, seed=0):
    """Structured quad mesh on a rectangle.

    `distort` perturbs interior nodes by that fraction of the cell size, which
    makes the elements non-affine -- the case where no quadrature rule is exact
    and the Q1 map contributes a rational integrand.  Keep it 0 for the affine
    tests that have ground truth.
    """
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    if distort > 0.0:
        rng = np.random.default_rng(seed)
        hx, hy = (x1 - x0)/nx, (y1 - y0)/ny
        X[1:-1, 1:-1] += distort*hx*rng.uniform(-1, 1, (nx-1, ny-1))
        Y[1:-1, 1:-1] += distort*hy*rng.uniform(-1, 1, (nx-1, ny-1))
    xy = np.column_stack([X.ravel(), Y.ravel()])
    nid = lambda i, j: i*(ny + 1) + j
    quads = np.array([[nid(i, j), nid(i+1, j), nid(i+1, j+1), nid(i, j+1)]
                      for i in range(nx) for j in range(ny)], dtype=int)
    tags = {
        'left':   [(nid(0, j),  nid(0, j+1))  for j in range(ny)],
        'right':  [(nid(nx, j), nid(nx, j+1)) for j in range(ny)],
        'bottom': [(nid(i, 0),  nid(i+1, 0))  for i in range(nx)],
        'top':    [(nid(i, ny), nid(i+1, ny)) for i in range(nx)],
    }
    return QuadMesh(xy, quads, {k: np.array(v, int) for k, v in tags.items()})
