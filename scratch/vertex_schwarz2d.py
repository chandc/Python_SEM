"""Overlapping vertex-patch additive Schwarz preconditioner for lssem2d, built
the production way: the GLOBAL operator stays matrix-free; the only dense
objects are the patch matrices, assembled from element-local blocks that are
obtained by probing the element operator (n*n*4 probes for the whole mesh,
exactly as DirectCoarseE does in 3D), then Cholesky-factored once.

    patch(v) = every dof of every element that has mesh vertex v as a corner
    z = sum_patches R_v^T K_v^-1 R_v r   (+ optional two-level coarse term)

Interface: same as PMG2 -- construct once per linearisation, call z = M(r)
with r in redundant local storage (nelem, n, n, 4), returns z likewise.
"""
import time
import numpy as np
import scipy.linalg as sla

from lssem2d.assembly import gather_scatter
from lssem2d.lssem import apply_L, apply_LT

NV = 4


def element_blocks(state, fu, fv):
    """A_e = L0_e^T W L0_e for every element at once, (nelem, nde, nde),
    local index ((i*n)+j)*NV + var."""
    m = state.mesh; n = m.N + 1; nde = n*n*NV
    blocks = np.empty((m.nelem, nde, nde))
    U = np.zeros((m.nelem, n, n, NV))
    for col in range(nde):
        i, j, v = np.unravel_index(col, (n, n, NV))
        U[:] = 0.0; U[:, i, j, v] = 1.0
        blocks[:, :, col] = apply_LT(state, apply_L(state, U, fu, fv), fu, fv).reshape(m.nelem, nde)
    return blocks


class VertexSchwarz2D:
    name = 'vschwarz'

    def __init__(self, state, fu, fv, pin_p=False, coarse=None, verbose=False):
        t0 = time.time()
        m = state.mesh; n = m.N + 1; N = m.N
        self.state, self.fu, self.fv, self.pin_p = state, fu, fv, pin_p
        self.mask = state.get_global_mask(pin_p=pin_p)                      # 0 = fixed
        mult = gather_scatter(m, np.ones((m.nelem, n, n, NV)))
        self.mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
        gid = m.gidx; ng = int(gid.max()) + 1
        self.g = (gid[..., None]*NV + np.arange(NV)).astype(np.int64)         # (nelem,n,n,NV) global dof
        self.ndof = ng*NV
        self.coarse = coarse                                                 # callable r -> z (local), or None

        blocks = element_blocks(state, fu, fv)
        free = np.zeros(self.ndof, bool)
        np.logical_or.at(free, self.g.ravel(), self.mask.ravel() > 0.5)
        self.free = free

        # vertex patches: elements sharing a corner node
        corners = {}
        for e in range(m.nelem):
            for (a, b) in ((0, 0), (0, N), (N, 0), (N, N)):
                corners.setdefault(int(gid[e, a, b]), []).append(e)
        self.patches = []
        gflat = self.g.reshape(m.nelem, -1)
        # elements touching each global node (for the ring of neighbours)
        node_elems = {}
        for e in range(m.nelem):
            for gn in np.unique(gid[e]): node_elems.setdefault(int(gn), []).append(e)
        for es in corners.values():
            dofs = np.unique(np.concatenate([gflat[e] for e in es]))
            dofs = dofs[free[dofs]]
            loc = -np.ones(self.ndof, dtype=np.int64); loc[dofs] = np.arange(dofs.size)
            # The Schwarz block is R A R^T of the ASSEMBLED operator: every
            # element that touches a patch dof contributes, including the ring
            # of neighbours outside the patch (they carry part of the stiffness
            # of the patch-boundary nodes).  Summing only the patch's own
            # elements leaves those nodes under-stiffened and the block singular.
            ring = sorted({e2 for e in es for gn in np.unique(gid[e]) for e2 in node_elems[int(gn)]})
            K = np.zeros((dofs.size, dofs.size))
            for e in ring:
                ge = gflat[e]; li = loc[ge]; keep = li >= 0
                K[np.ix_(li[keep], li[keep])] += blocks[e][np.ix_(keep, keep)]
            K = 0.5*(K + K.T)
            # symmetric Jacobi equilibration first: the pressure rows carry
            # a_flux^2 ~ dt^2 ~ 1e-8 against O(1) mass rows, and unscaled
            # Cholesky loses those pivots to round-off (potrf info>0).
            sc = 1.0/np.sqrt(np.diag(K))
            Ks = K*sc[:, None]*sc[None, :]
            try:
                fac = ('chol', sla.cho_factor(Ks, lower=True, check_finite=False))
            except sla.LinAlgError:
                fac = ('lu', sla.lu_factor(Ks, check_finite=False)); self.n_lu_fallback = getattr(self, 'n_lu_fallback', 0) + 1
            self.patches.append((dofs, sc, fac))
        self.setup_time = time.time() - t0
        self.npatch = len(self.patches); self.maxdofs = max(d.size for d, _, _ in self.patches)
        if verbose:
            print(f'  VertexSchwarz2D: {self.npatch} patches, max {self.maxdofs} dofs, '
                  f'setup {self.setup_time:.1f}s ({blocks.shape[1]} probes)')

    def __call__(self, r):
        r = r*self.mask
        rg = np.bincount(self.g.ravel(), weights=(r*self.mw).ravel(), minlength=self.ndof)
        zg = np.zeros(self.ndof)
        for dofs, sc, (kind, fac) in self.patches:
            rs = rg[dofs]*sc
            zg[dofs] += sc*(sla.cho_solve(fac, rs, check_finite=False) if kind == 'chol'
                            else sla.lu_solve(fac, rs, check_finite=False))
        z = zg[self.g]*self.mask
        if self.coarse is not None:
            z = z + self.coarse(r)
        return z


def make_coarse(state, fu, fv, M_inv, pin_p, pc=2):
    """Two-level term P A_c^-1 P^T using PMG2's own transfers and direct coarse solve."""
    from lssem2d.precond import PMG2
    pmg = PMG2(state, fu, fv, M_inv, pin_p, pc=pc, coarse_solver='direct')
    return lambda r: pmg._prolong(pmg._coarse_solve(pmg._restrict(r)))
