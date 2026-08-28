"""p-multigrid for the CONSISTENT pressure operator E = G^T M^{-1} G.

The K-based V-cycle preconditions E poorly (465 CG iterations on a turbulent
channel state, vs ~15 for K on its own system): the mass-weighted adjoint
pairing shifts the spectrum enough that K's coarse corrections miss.  This
V-cycle is built ON E at every level:

  * levels at descending polynomial order, each with its OWN velocity mask
    (E's essential BCs live in the velocity space) and assembled inverse mass;
  * Chebyshev smoothing against E, lambda_max from power iteration
    (the K analytic diagonal is reused as the smoothing diagonal -- E's exact
    diagonal is expensive and Chebyshev only needs the right scaling);
  * dense coarse E per Fourier mode, assembled by global-dof probing, inverted
    by eigendecomposition with NULL DEFLATION: eigenvalues below 1e-10*max are
    zeroed rather than pseudo-inverted, so the constant (and anything
    numerically null) is annihilated instead of amplified -- the pinv fallback
    on the K-coarse is what poisoned CG at 4000 iterations.

Interface matches HelmholtzPMG: M(r) with r split-real (nelem,n,n,2,nk).
"""
import numpy as np

from . import device as DEV
from . import helmholtz as HH
from . import solver3d as S3
from .precond import p_interp, coarsen_mesh, Chebyshev4
from lssem2d.lgl import diff_matrix


class _ELvl:
    def __init__(self, mesh, p, kz, nk, nz, like=None, wall=True):
        from . import project as PJ
        self.m, self.p = mesh, p
        Dh = diff_matrix(p)
        mask_p = PJ.build_masks(mesh, nk, nz, 1, wall=False)   # UNPINNED
        mask_u = PJ.build_masks(mesh, nk, nz, 3, wall=wall)
        wq3 = mesh.wq[..., None, None]
        Mginv = 1.0/S3.gs(mesh, wq3 + np.zeros_like(wq3))
        self.shape = mask_p.shape
        # smoothing diagonal: K's analytic diagonal, right scaling for E
        dh = HH.jacobi_diagonal_analytic(mesh, p, mesh.wq, kz**2, 1.0, 2, nk,
                                         mask_p)
        Minv_h = HH.jacobi_inverse(dh, mask_p)
        mwh = S3.multiplicity_weight(mesh, self.shape)
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        self.D, self.mask = to(Dh), to(mask_p)
        self.mask_u, self.wq3, self.Mginv = to(mask_u), to(wq3), to(Mginv)
        self.fx, self.fy = to(mesh.facx), to(mesh.facy)
        self.kz = to(np.asarray(kz, dtype=float))
        self.M_inv, self.mw = to(Minv_h), to(mwh)
        self.mask_h = mask_p
        mref = self.mask
        self.A = lambda v: PJ.apply_E(v, self.D, self.fx, self.fy, self.wq3,
                                      self.kz, mesh, mref, self.mask_u,
                                      self.Mginv)


class _EDirect:
    """Dense coarse E per mode; eigendecomposition with null deflation."""

    def __init__(self, lvl, like=None):
        shape, m, mask = lvl.shape, lvl.m, lvl.mask_h
        nk = shape[-1]
        nsp = int(np.prod(shape[:-1]))
        self.lvl, self.shape, self.nk, self.nsp = lvl, shape, nk, nsp
        # host-side probing: build the E-apply with HOST arrays
        from . import project as PJ
        Dh = diff_matrix(lvl.p)
        wq3 = m.wq[..., None, None]
        mask_u = PJ.build_masks(m, nk, int(2*(nk - 1)), 3, wall=True)
        # nz only enters build_masks via real_mode_columns; reuse level's mask
        mask_u = DEV.to_host(lvl.mask_u) if hasattr(DEV, 'to_host') else (
            lvl.mask_u if isinstance(lvl.mask_u, np.ndarray) else
            np.asarray(lvl.mask_u.get()))
        Mginv = 1.0/S3.gs(m, wq3 + np.zeros_like(wq3))
        kzh = (np.asarray(lvl.kz.get()) if not isinstance(lvl.kz, np.ndarray)
               else lvl.kz)
        Ah = lambda v: PJ.apply_E(v, Dh, m.facx, m.facy, wq3, kzh, m,
                                  mask, mask_u, Mginv)
        mwh = np.asarray(S3.multiplicity_weight(m, shape))
        Bs, Ais = [], []
        for k in range(nk):
            cols, seen = [], set()
            for e in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        for f in range(shape[3]):
                            if mask[e, i, j, f, k] == 0.0:
                                continue
                            oh = np.zeros(shape)
                            oh[e, i, j, f, k] = 1.0
                            g = S3.gs(m, oh)
                            key = tuple(np.flatnonzero(
                                np.abs(g[..., k].ravel()) > .5))
                            if not key or key in seen:
                                continue
                            seen.add(key)
                            cols.append((g*mask)[..., k].ravel())
            if not cols:
                Bs.append(None); Ais.append(None); continue
            B = np.stack(cols, axis=1)
            A = np.empty((B.shape[1],)*2)
            for a in range(B.shape[1]):
                full = np.zeros(shape)
                full[..., k] = B[:, a].reshape(shape[:-1])
                A[:, a] = B.T @ (Ah(full)[..., k].ravel()*mwh[..., k].ravel())
            A = 0.5*(A + A.T)
            w, V = np.linalg.eigh(A)
            winv = np.where(w > 1e-10*max(w[-1], 1e-300), 1.0/np.maximum(w, 1e-300), 0.0)
            Ais.append((V*winv) @ V.T)
            Bs.append(B)
        wmax = max((B.shape[1] for B in Bs if B is not None), default=0)
        Bb = np.zeros((nk, nsp, wmax)); Ab = np.zeros((nk, wmax, wmax))
        for k, (B, Ai) in enumerate(zip(Bs, Ais)):
            if B is None:
                continue
            Bb[k, :, :B.shape[1]] = B
            Ab[k, :Ai.shape[0], :Ai.shape[1]] = Ai
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        self.Bb, self.Ab = to(Bb), to(Ab)

    def __call__(self, r):
        x = (r*self.lvl.mw).reshape(self.nsp, self.nk).T[:, :, None]
        y = self.Bb @ (self.Ab @ (self.Bb.transpose(0, 2, 1) @ x))
        return y[:, :, 0].T.reshape(self.shape)*self.lvl.mask


class ConsistentPMG:
    def __init__(self, mesh, N, kz, nk, nz, orders=None, deg=6,
                 like=None, wall=True):
        orders = orders or tuple(o for o in (N, max(2, N//2), 2)
                                 if o <= N and o >= 2)
        orders = tuple(sorted(set(orders), reverse=True))
        meshes = [mesh] + [coarsen_mesh(mesh, p) for p in orders[1:]]
        self.lv = [_ELvl(mm, p, kz, nk, nz, like, wall=wall)
                   for mm, p in zip(meshes, orders)]
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        Ph = [p_interp(c, f) for f, c in zip(orders[:-1], orders[1:])]
        self.P = [to(p) for p in Ph]
        self.R = [to(np.ascontiguousarray(p.T)) for p in Ph]
        self.sm = []
        for l in self.lv[:-1]:
            v = DEV.to_device(
                np.random.default_rng(0).standard_normal(l.shape),
                l.mask)*l.mask
            lm = 0.0
            for _ in range(25):
                w = l.M_inv*l.A(v)
                nw = float(DEV.sqrt((w*w).sum()))
                if nw <= 1e-300:
                    break
                lm, v = nw, w/nw
            self.sm.append(Chebyshev4(l.A, l.M_inv, l.shape, deg=deg,
                                      lam_max=lm))
        self.coarse = _EDirect(self.lv[-1], like)

    def _restrict(self, x, i):
        t = DEV.einsum('bj,eijvk->eibvk', self.R[i], x*self.lv[i].mw)
        c = DEV.einsum('ai,eibvk->eabvk', self.R[i], t)
        return S3.gs(self.lv[i+1].m, c)*self.lv[i+1].mask

    def _prolong(self, xc, i):
        t = DEV.einsum('bj,eijvk->eibvk', self.P[i], xc)
        return DEV.einsum('ai,eibvk->eabvk', self.P[i], t)*self.lv[i].mask

    def _v(self, r, i):
        if i == len(self.lv) - 1:
            return self.coarse(r)
        z = self.sm[i](r)
        z = z + self._prolong(self._v(self._restrict(r - self.lv[i].A(z), i),
                                      i + 1), i)
        return z + self.sm[i](r - self.lv[i].A(z))

    def __call__(self, r):
        return self._v(r, 0)

    def subset(self, idx):
        """V-cycle restricted to a mode subset -- for _pcg's freezing."""
        import copy
        from .precond import Chebyshev4
        idx = np.asarray(idx)
        sub = object.__new__(ConsistentPMG)
        sub.P, sub.R = self.P, self.R
        sub.lv = []
        for l in self.lv:
            c = copy.copy(l)
            c.mask = l.mask[..., idx]
            c.mask_u = l.mask_u[..., idx]
            c.M_inv = l.M_inv[..., idx]
            c.mw = l.mw[..., idx]
            c.kz = l.kz[idx]
            c.shape = tuple(l.shape[:-1]) + (len(idx),)
            from . import project as PJ
            c.A = (lambda v, c=c: PJ.apply_E(v, c.D, c.fx, c.fy, c.wq3,
                                             c.kz, c.m, c.mask, c.mask_u,
                                             c.Mginv))
            sub.lv.append(c)
        sub.sm = [Chebyshev4(c.A, c.M_inv, None, deg=s0.deg,
                             lam_max=s0.rho, safety=1.0)
                  for c, s0 in zip(sub.lv[:-1], self.sm)]
        co = object.__new__(_EDirect)
        co.lvl = sub.lv[-1]
        co.shape = sub.lv[-1].shape
        co.nk = len(idx)
        co.nsp = self.coarse.nsp
        co.Bb, co.Ab = self.coarse.Bb[idx], self.coarse.Ab[idx]
        sub.coarse = co
        return sub


def kz0_null_basis(mesh, N, kz, nk, nz, mask_p, mask_u, tol=1e-9):
    """Exact null basis of E's k_z = 0 block, by dense assembly + eigh.

    On WALLED meshes the kernel is the constant alone (measured: dim 1), and a
    single-vector purge suffices.  On PERIODIC meshes the collocated P_N-P_N
    pairing has additional spurious pressure modes (checkerboard family); a
    preconditioner that does not respect them amplifies roundoff into 1e29
    (measured), so every kernel vector must be purged.  One-off dense cost.

    Returns (nelem, n, n) basis vectors, mw-orthonormalised, as a list.
    """
    import numpy as np
    from . import project as PJ
    from . import solver3d as S3
    wq3 = mesh.wq[..., None, None]
    Mginv = 1.0/S3.gs(mesh, wq3 + np.zeros_like(wq3))
    kzh = np.asarray(kz, dtype=float)
    Ah = lambda v: PJ.apply_E(v, __import__('lssem2d.lgl', fromlist=['diff_matrix']).diff_matrix(N),
                              mesh.facx, mesh.facy, wq3, kzh, mesh,
                              mask_p, mask_u, Mginv)
    shape = mask_p.shape
    mwh = np.asarray(S3.multiplicity_weight(mesh, shape))
    cols, seen = [], set()
    n1 = N + 1
    for e in range(shape[0]):
        for i in range(n1):
            for j in range(n1):
                if mask_p[e, i, j, 0, 0] == 0.0:
                    continue
                oh = np.zeros(shape); oh[e, i, j, 0, 0] = 1.0
                g = S3.gs(mesh, oh)
                key = tuple(np.flatnonzero(np.abs(g[..., 0, 0].ravel()) > .5))
                if not key or key in seen:
                    continue
                seen.add(key)
                cols.append((g*mask_p)[..., 0, 0].ravel())
    B = np.stack(cols, axis=1)
    A = np.empty((B.shape[1],)*2)
    for a in range(B.shape[1]):
        full = np.zeros(shape); full[..., 0, 0] = B[:, a].reshape(shape[:3])
        A[:, a] = B.T @ (Ah(full)[..., 0, 0].ravel()*mwh[..., 0, 0].ravel())
    A = 0.5*(A + A.T)
    w, V = np.linalg.eigh(A)
    null = V[:, w < tol*max(w[-1], 1e-300)]
    print(f'kz0 kernel dimension: {null.shape[1]} '
          f'(smallest eigs: {np.array2string(w[:4], precision=2)})')
    out = []
    mw0 = mwh[..., 0, 0].ravel()
    for c in range(null.shape[1]):
        vec = (B @ null[:, c]).reshape(shape[:3])
        # mw-orthonormalise against previous
        fv = vec.ravel()
        for prev in out:
            fv = fv - (fv*prev.ravel()*mw0).sum()*prev.ravel()
        nn = np.sqrt((fv*fv*mw0).sum())
        out.append((fv/nn).reshape(shape[:3]))
    return out
