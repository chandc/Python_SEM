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
from lssem2d import obc

NV = 4


def element_blocks(state, fu, fv):
    """A_e for every element at once, (nelem, nde, nde), local index
    ((i*n)+j)*NV + var.

    A_e = L0_e^T W L0_e  PLUS the Dong outflow term B^T B on any bc == 6 edge.
    That second piece is what `newton_step` adds to the matvec (solver.py step
    3.5), and it is element-local -- apply_B reads only the i = N edge of its own
    element -- so it belongs in the element block.  Omitting it would build the
    preconditioner from a DIFFERENT operator than the one being solved, precisely
    on the boundary the condition exists to treat.  No-op when the mesh has no
    bc == 6 edge, which is the case for the cavity, the channel and the Gartling
    pressure outlet; the Armaly/BFS runs with the Dong condition need it."""
    m = state.mesh; n = m.N + 1; nde = n*n*NV
    blocks = np.empty((m.nelem, nde, nde))
    U = np.zeros((m.nelem, n, n, NV))
    has_obc = obc.obc_active(state)
    for col in range(nde):
        i, j, v = np.unravel_index(col, (n, n, NV))
        U[:] = 0.0; U[:, i, j, v] = 1.0
        out = apply_LT(state, apply_L(state, U, fu, fv), fu, fv)
        if has_obc:
            obc.apply_BT(state, obc.apply_B(state, U), out)
        blocks[:, :, col] = out.reshape(m.nelem, nde)
    return blocks


class VertexSchwarz2D:
    name = 'vschwarz'

    def __init__(self, state, fu, fv, pin_p=False, coarse=None, verbose=False, pou=False):
        t0 = time.time()
        m = state.mesh; n = m.N + 1; N = m.N
        self.state, self.fu, self.fv, self.pin_p = state, fu, fv, pin_p
        self.pou = pou
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
        self._build_pou([d for d, _, _ in self.patches])
        self.setup_time = time.time() - t0
        self.npatch = len(self.patches); self.maxdofs = max(d.size for d, _, _ in self.patches)
        if verbose:
            print(f'  VertexSchwarz2D: {self.npatch} patches, max {self.maxdofs} dofs, '
                  f'setup {self.setup_time:.1f}s ({blocks.shape[1]} probes)')

    def _build_pou(self, patch_dof_lists):
        """Partition-of-unity weight D = 1/(patch count), applied as D^1/2 on both
        sides so the preconditioner stays SPD:

            M^-1 = D^1/2 ( sum_v R_v^T A_v^-1 R_v ) D^1/2  (+ coarse, unweighted).

        Without it an interior dof is corrected once per patch that contains it --
        four times in 2D -- and the additive sum over-corrects by that factor.
        Weighting by the inverse multiplicity is standard for overlapping Schwarz
        on spectral elements (Fischer 1997; Lottes & Fischer 2005; Stiller 2016
        measures 1.5-3x fewer iterations from it).  Default off: dw = 1 reproduces
        the unweighted method exactly."""
        self.dw = np.ones(self.ndof)
        if not getattr(self, 'pou', False):
            return
        cnt = np.zeros(self.ndof)
        for dofs in patch_dof_lists:
            cnt[dofs] += 1.0
        self.dw = 1.0/np.sqrt(np.maximum(cnt, 1.0))

    def __call__(self, r):
        r = r*self.mask
        rg = np.bincount(self.g.ravel(), weights=(r*self.mw).ravel(), minlength=self.ndof)*self.dw
        zg = np.zeros(self.ndof)
        for dofs, sc, (kind, fac) in self.patches:
            rs = rg[dofs]*sc
            zg[dofs] += sc*(sla.cho_solve(fac, rs, check_finite=False) if kind == 'chol'
                            else sla.lu_solve(fac, rs, check_finite=False))
        zg *= self.dw
        z = zg[self.g]*self.mask
        if self.coarse is not None:
            z = z + self.coarse(r)
        return z


def make_coarse(state, fu, fv, M_inv, pin_p, pc=2):
    """Two-level term P A_c^-1 P^T using PMG2's own transfers and direct coarse solve."""
    from lssem2d.precond import PMG2
    pmg = PMG2(state, fu, fv, M_inv, pin_p, pc=pc, coarse_solver='direct')
    return lambda r: pmg._prolong(pmg._coarse_solve(pmg._restrict(r)))


class VertexSchwarzCondensed2D(VertexSchwarz2D):
    """Same preconditioner, statically condensed (exact):
        per element : factor of the interior block K_II,e and the coupling K_IB,e
                      (interior nodes couple only to their own element -> shared
                      by every patch containing e)
        per patch   : dense Schur complement on the patch's EDGE dofs
                      S = K_BB - sum_e K_BI,e K_II,e^-1 K_IB,e
    Apply: z_B = S^-1 (r_B - sum_e K_BI,e K_II,e^-1 r_I,e),  z_I,e = K_II,e^-1 (r_I,e - K_IB,e z_B).
    """
    name = 'vschwarz-condensed'

    def __init__(self, state, fu, fv, pin_p=False, coarse=None, verbose=False, pou=False):
        t0 = time.time()
        m = state.mesh; n = m.N + 1; N = m.N
        self.state, self.fu, self.fv, self.pin_p = state, fu, fv, pin_p
        self.pou = pou
        self.mask = state.get_global_mask(pin_p=pin_p)
        mult = gather_scatter(m, np.ones((m.nelem, n, n, NV)))
        self.mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
        gid = m.gidx; ng = int(gid.max()) + 1
        self.g = (gid[..., None]*NV + np.arange(NV)).astype(np.int64); self.ndof = ng*NV
        self.coarse = coarse
        blocks = element_blocks(state, fu, fv)
        free = np.zeros(self.ndof, bool); np.logical_or.at(free, self.g.ravel(), self.mask.ravel() > 0.5); self.free = free
        gflat = self.g.reshape(m.nelem, -1)
        loc_int = np.zeros((n, n, NV), bool); loc_int[1:N, 1:N, :] = True; loc_int = loc_int.ravel()
        # ---- per element: interior factor and interior->edge coupling ----
        self.eint, self.eedge, self.efac, self.eKIB = {}, {}, {}, {}
        self.bytes = 0
        for e in range(m.nelem):
            ge = gflat[e]; fr = free[ge]
            I = np.flatnonzero(loc_int & fr); B = np.flatnonzero(~loc_int & fr)
            K = blocks[e]
            KII = K[np.ix_(I, I)]; sc = 1.0/np.sqrt(np.diag(KII))
            self.efac[e] = (sc, sla.cho_factor(KII*sc[:, None]*sc[None, :], lower=True, check_finite=False))
            self.eKIB[e] = K[np.ix_(I, B)].copy()
            self.eint[e], self.eedge[e] = ge[I], ge[B]              # global dofs
            self.bytes += (I.size*(I.size+1)//2 + I.size*B.size)*8
        def isolve(e, r):
            sc, cf = self.efac[e]; return sc*sla.cho_solve(cf, sc*r, check_finite=False)
        self._isolve = isolve
        # ---- per patch: edge Schur complement ----
        corners, node_elems = {}, {}
        for e in range(m.nelem):
            for (a, b) in ((0, 0), (0, N), (N, 0), (N, N)):
                corners.setdefault(int(gid[e, a, b]), []).append(e)
            for gn in np.unique(gid[e]): node_elems.setdefault(int(gn), []).append(e)
        self.patches = []
        for es in corners.values():
            Bp = np.unique(np.concatenate([self.eedge[e] for e in es]))          # edge dofs of patch elements
            ring = sorted({e2 for e in es for gn in np.unique(gid[e]) for e2 in node_elems[int(gn)]})
            locB = -np.ones(self.ndof, dtype=np.int64); locB[Bp] = np.arange(Bp.size)
            S = np.zeros((Bp.size, Bp.size))
            for e in ring:                                    # K_BB from every element touching the patch edge dofs
                ge = gflat[e]; li = locB[ge]; k = li >= 0
                S[np.ix_(li[k], li[k])] += blocks[e][np.ix_(k, k)]
            emaps = []
            for e in es:                                      # subtract K_BI K_II^-1 K_IB of the patch's own elements
                cols = locB[self.eedge[e]]; KIB = self.eKIB[e]
                X = np.column_stack([isolve(e, KIB[:, j]) for j in range(KIB.shape[1])])
                S[np.ix_(cols, cols)] -= KIB.T @ X
                emaps.append((e, cols))
            S = 0.5*(S + S.T); sc = 1.0/np.sqrt(np.diag(S))
            self.patches.append((Bp, emaps, sc, sla.cho_factor(S*sc[:, None]*sc[None, :], lower=True, check_finite=False)))
            self.bytes += Bp.size*(Bp.size+1)//2*8
        # a patch corrects its edge dofs AND the interiors of its own elements
        self._build_pou([np.concatenate([Bp] + [self.eint[e] for e, _ in emaps])
                         for Bp, emaps, _, _ in self.patches])
        self.setup_time = time.time() - t0
        self.npatch = len(self.patches); self.maxdofs = max(p[0].size for p in self.patches)
        if verbose:
            print(f'  VertexSchwarzCondensed2D: {self.npatch} patches, max edge dofs {self.maxdofs}, '
                  f'interior dofs/elem {max(v.size for v in self.eint.values())}, stored {self.bytes/1e6:.1f} MB, setup {self.setup_time:.1f}s')

    def __call__(self, r):
        r = r*self.mask
        rg = np.bincount(self.g.ravel(), weights=(r*self.mw).ravel(), minlength=self.ndof)*self.dw
        zg = np.zeros(self.ndof)
        for Bp, emaps, sc, cf in self.patches:
            rB = rg[Bp].copy(); ys = []
            for e, cols in emaps:
                y = self._isolve(e, rg[self.eint[e]]); ys.append(y); rB[cols] -= self.eKIB[e].T @ y
            zB = sc*sla.cho_solve(cf, sc*rB, check_finite=False); zg[Bp] += zB
            for (e, cols), y in zip(emaps, ys):
                zg[self.eint[e]] += y - self._isolve(e, self.eKIB[e] @ zB[cols])
        zg *= self.dw
        z = zg[self.g]*self.mask
        if self.coarse is not None:
            z = z + self.coarse(r)
        return z
