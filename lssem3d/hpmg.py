"""p-multigrid for the scalar Helmholtz/Poisson — the coarse grid FDM lacks.

WHY THIS EXISTS.  `helmholtz.fdm_preconditioner` is one-level additive Schwarz:
an EXACT element-local inverse plus gather-scatter, with no coarse grid.  On
the Re_tau = 180 channel Poisson that buys nothing over a plain diagonal, and
both grow with element count:

    elements   Jacobi   FDM
    2x4           254   231
    6x12          734   826      ~sqrt(elements)

which is the signature.  For Poisson the slow modes are GLOBAL and smooth, so
exactness inside an element cannot touch them.

AND THIS IS THE CASE WHERE MULTIGRID WORKS.  3D_STATUS.md sec 7K closed p-MG
for the VVP LEAST-SQUARES operator after measuring 7.4x fewer iterations for
~27x the cost per iteration, and sec 7K.2 explained why: that operator's slow
modes are ROUGH, which a coarse grid cannot reach.  Poisson's are smooth.  The
closure does not transfer, and the machinery it left behind -- p_interp,
coarsen_mesh, Chebyshev4 -- is operator-agnostic and reused here unchanged.

The smoother needs a diagonal at EVERY level, per dt, so it uses
helmholtz.jacobi_diagonal_analytic: closed form, and exact to 2.1e-16 against a
true diagonal probed one global dof at a time.
"""
import hashlib
import os

import numpy as np

from . import device as DEV
from . import helmholtz as HH
from . import solver3d as S3
from .precond import p_interp, coarsen_mesh, Chebyshev4
from lssem2d.lgl import diff_matrix


class _Lvl:
    """One polynomial order.  Everything the V-cycle touches lives on `like`.

    The masks, the analytic diagonal and the multiplicity weight are all built
    in NumPy -- they are closed-form setup, evaluated once -- and moved once.
    What must NOT stay on the host is anything the operator reads per
    application: D, facx, facy, wq and mask.  Leaving those behind is what made
    the first version cost 275x a matvec.
    """

    def __init__(self, mesh, p, lam, mu, nfield_c, nk, nz, wall, pin_kz0,
                 like=None, mask_h=None, Minv_h=None):
        from . import project as PJ
        self.m, self.p = mesh, p
        Dh = diff_matrix(p)
        if mask_h is None:
            mask = PJ.build_masks(mesh, nk, nz, nfield_c, wall=wall)
            if pin_kz0:
                ind = np.zeros(mask.shape)
                ind[0, 0, 0, 0, 0] = 1.0
                mask[..., 0, 0] *= (S3.gs(mesh, ind)[..., 0, 0] < 0.5)
        else:
            mask = mask_h
        self.shape = mask.shape
        if Minv_h is None:
            dh = HH.jacobi_diagonal_analytic(mesh, p, mesh.wq, lam, mu,
                                             2*nfield_c, nk, mask)
            Minv_h = HH.jacobi_inverse(dh, mask)
        mwh = S3.multiplicity_weight(mesh, self.shape)
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        self.D, self.mask = to(Dh), to(mask)
        self.fx, self.fy, self.wq = to(mesh.facx), to(mesh.facy), to(mesh.wq)
        self.M_inv = to(Minv_h)
        self.mw = to(mwh)
        self.mask_h, self.Minv_h = mask, Minv_h
        # lam is per-mode (kz^2) and is MULTIPLIED into the operator, so it is
        # read every application and has to move as well.  mu is a scalar.
        lam_d = to(np.asarray(lam, dtype=float)) if np.ndim(lam) else lam
        self.lam_d, self.mu_v = lam_d, mu
        self.A = lambda v: HH.apply(v, self.D, self.fx, self.fy, self.wq,
                                    lam_d, mu, mesh, self.mask)


class _Direct:
    """Exact coarsest solve, in the GLOBAL (continuous) basis, per mode.

    Per mode because the operator is block diagonal in k_z -- one small dense
    factorisation each rather than one huge one.  Cholesky, not pinv: the
    matrix is a Galerkin projection of an SPD operator, and pinv's rcond would
    silently truncate exactly the near-null directions a coarse solve exists to
    resolve.
    """

    def __init__(self, lvl, like=None, Bb=None, Ab=None):
        from .precond import _factor_spd
        shape, m, mask = lvl.shape, lvl.m, lvl.mask_h
        nk = shape[-1]
        nsp = int(np.prod(shape[:-1]))
        self.lvl, self.shape, self.nk, self.nsp = lvl, shape, nk, nsp
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        if Bb is not None:
            # CACHED: the probe/factorise loop below is the ~cubic host-side
            # cost this skips -- the products depend only on the setup key.
            self.Bb_h, self.Ab_h = Bb, Ab
            self.Bb, self.Ab = to(Bb), to(Ab)
            return
        Dh, wqh = diff_matrix(lvl.p), m.wq
        mwh = np.asarray(S3.multiplicity_weight(m, shape))
        Ah = lambda v: HH.apply(v, Dh, m.facx, m.facy, wqh, lvl.lam, lvl.mu,
                                m, mask)
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
                            # RESTRICT TO THIS MODE'S SLICE.  The basis vector
                            # is a full-array one-hot, but only mode k is
                            # non-zero -- carrying the whole array made every
                            # matmul touch nk times more data than it needed.
                            cols.append(g[..., k].ravel())
            if not cols:
                Bs.append(None); Ais.append(None); continue
            B = np.stack(cols, axis=1)                     # (nsp, ncol)
            A = np.empty((B.shape[1], B.shape[1]))
            for a in range(B.shape[1]):
                full = np.zeros(shape)
                full[..., k] = B[:, a].reshape(shape[:-1])
                A[:, a] = B.T @ (Ah(full)[..., k].ravel()
                                 * mwh[..., k].ravel())
            fac = _factor_spd(0.5*(A + A.T))
            Ai = np.stack([fac(e) for e in np.eye(B.shape[1])], axis=1)
            Bs.append(B); Ais.append(0.5*(Ai + Ai.T))
        # BATCH ACROSS MODES: pad to a common width with ZERO columns and a
        # ZERO inverse block, which contribute nothing, so all nk solves become
        # three batched matmuls instead of 3*nk small ones.
        w = max((B.shape[1] for B in Bs if B is not None), default=0)
        Bb = np.zeros((nk, nsp, w))
        Ab = np.zeros((nk, w, w))
        for k, (B, Ai) in enumerate(zip(Bs, Ais)):
            if B is None:
                continue
            Bb[k, :, :B.shape[1]] = B
            Ab[k, :Ai.shape[0], :Ai.shape[1]] = Ai
        self.Bb_h, self.Ab_h = Bb, Ab
        self.Bb, self.Ab = to(Bb), to(Ab)

    def __call__(self, r):
        # (nsp, nk) -> (nk, nsp, 1), three batched matmuls, back again
        x = (r*self.lvl.mw).reshape(self.nsp, self.nk).T[:, :, None]
        y = self.Bb @ (self.Ab @ (self.Bb.transpose(0, 2, 1) @ x))
        return y[:, :, 0].T.reshape(self.shape)*self.lvl.mask


def _setup_key(mesh, N, lam, mu, nfield_c, nk, nz, orders, deg, wall,
               pin_kz0):
    """Hash of everything the setup products depend on.

    Geometry and BC codes rather than object identity, so a cache written by
    one process validates in another.  lam is the per-mode kz^2 table and mu a
    scalar; both go in as raw float64 bytes -- bit-exact, no repr rounding.
    """
    h = hashlib.sha256()
    for a in (mesh.x0, mesh.y0, mesh.hx, mesh.hy, mesh.bc,
              np.asarray(lam, dtype=float), np.asarray(mu, dtype=float)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((mesh.nelem, N, nfield_c, nk, nz, orders, deg, wall,
                   pin_kz0,
                   getattr(mesh, 'periodic_x', None),
                   getattr(mesh, 'periodic_y', None))).encode())
    return h.hexdigest()


_CACHE_VERSION = 1


class HelmholtzPMG:
    """V-cycle preconditioner for lam*M + mu*K.  Callable: z = M^-1 r.

    `cache_path`: npz holding the host-side setup products -- per-level masks,
    Jacobi inverse diagonals, smoother lam_max, the interpolation matrices and
    the coarse factors Bb/Ab.  The coarse factorisation is the ~cubic-in-
    elements host cost (85 s at 11x11 N=8, ~90 min at 20x20), and it depends
    only on (mesh, N, lam, mu, BC flags), not on the solution state, so a
    restart can reload it.  The file is keyed by a hash of those parameters:
    on mismatch it is rebuilt and overwritten, never trusted.
    """

    def __init__(self, mesh, N, lam, mu, nfield_c, nk, nz, orders=None,
                 deg=6, wall=False, pin_kz0=True, like=None, cache_path=None):
        orders = orders or tuple(o for o in (N, max(2, N//2), 2)
                                 if o <= N and o >= 2)
        orders = tuple(sorted(set(orders), reverse=True))
        key = _setup_key(mesh, N, lam, mu, nfield_c, nk, nz, orders, deg,
                         wall, pin_kz0)
        cache = None
        if cache_path is not None and os.path.exists(cache_path):
            z = np.load(cache_path)
            if (z['version'].item() == _CACHE_VERSION
                    and z['key'].item() == key):
                cache = z
            else:
                print(f'hpmg: cache {cache_path} does not match this setup, '
                      f'rebuilding')
        meshes = [mesh] + [coarsen_mesh(mesh, p) for p in orders[1:]]
        self.lv = []
        for i, (mm, p) in enumerate(zip(meshes, orders)):
            l = _Lvl(mm, p, lam, mu, nfield_c, nk, nz, wall, pin_kz0, like,
                     mask_h=cache[f'mask{i}'] if cache is not None else None,
                     Minv_h=cache[f'minv{i}'] if cache is not None else None)
            l.lam, l.mu = lam, mu
            self.lv.append(l)
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        Ph = ([cache[f'P{i}'] for i in range(len(orders) - 1)]
              if cache is not None else
              [p_interp(c, f) for f, c in zip(orders[:-1], orders[1:])])
        self.P = [to(p) for p in Ph]
        self.R = [to(np.ascontiguousarray(p.T)) for p in Ph]
        self.sm, lmaxs = [], []
        for i, l in enumerate(self.lv[:-1]):
            if cache is not None:
                lm = float(cache[f'lmax{i}'])
            else:
                # estimate_lambda_max starts from a HOST random vector; give it
                # a device one so the power iteration never leaves the device
                v = DEV.to_device(
                    np.random.default_rng(0).standard_normal(l.shape),
                    l.mask)*l.mask
                lm = 0.0
                for _ in range(20):
                    w = l.M_inv*l.A(v)
                    nw = float(DEV.sqrt((w*w).sum()))
                    if nw <= 1e-300:
                        break
                    lm, v = nw, w/nw
            lmaxs.append(lm)
            self.sm.append(Chebyshev4(l.A, l.M_inv, l.shape, deg=deg,
                                      lam_max=lm))
        self.coarse = _Direct(self.lv[-1], like,
                              Bb=cache['Bb'] if cache is not None else None,
                              Ab=cache['Ab'] if cache is not None else None)
        self.like = like
        if cache_path is not None and cache is None:
            out = {'version': np.int64(_CACHE_VERSION), 'key': key,
                   'Bb': self.coarse.Bb_h, 'Ab': self.coarse.Ab_h}
            for i, l in enumerate(self.lv):
                out[f'mask{i}'] = l.mask_h
                out[f'minv{i}'] = l.Minv_h
            for i, (p, lm) in enumerate(zip(Ph, lmaxs)):
                out[f'P{i}'] = p
                out[f'lmax{i}'] = np.float64(lm)
            # atomic: a restart killed mid-write must not leave a torn file
            # that the next start half-trusts
            d = os.path.dirname(os.path.abspath(cache_path))
            os.makedirs(d, exist_ok=True)
            tmp = os.path.join(d, '.' + os.path.basename(cache_path) + '.tmp')
            with open(tmp, 'wb') as f:
                np.savez(f, **out)
            os.replace(tmp, cache_path)

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
        """Device-resident when built with `like`: no transfer, no NumPy."""
        return self._v(r, 0)

    def subset(self, idx):
        """V-cycle restricted to a subset of Fourier modes (slice, no rebuild).

        Every per-mode tensor lives on the last axis; the interpolation
        matrices are mode-independent; each smoother's spectral bound over a
        subset is bounded by the full-set bound, so lam_max carries over.
        Used by helmholtz.solve's mode-adaptive freezing.
        """
        import copy
        from .precond import Chebyshev4
        sub = object.__new__(HelmholtzPMG)
        sub.P, sub.R, sub.like = self.P, self.R, self.like
        sub.lv = []
        for l in self.lv:
            c = copy.copy(l)
            c.mask = l.mask[..., idx]
            c.M_inv = l.M_inv[..., idx]
            c.mw = l.mw[..., idx]
            c.shape = tuple(l.shape[:-1]) + (len(idx),)
            lam_c = (l.lam_d[idx] if hasattr(l.lam_d, 'ndim')
                     and getattr(l.lam_d, 'ndim', 0) else l.lam_d)
            c.A = (lambda v, c=c, lam_c=lam_c:
                   HH.apply(v, c.D, c.fx, c.fy, c.wq, lam_c, c.mu_v, c.m,
                            c.mask))
            sub.lv.append(c)
        sub.sm = [Chebyshev4(c.A, c.M_inv, None, deg=s0.deg,
                             lam_max=s0.rho, safety=1.0)
                  for c, s0 in zip(sub.lv[:-1], self.sm)]
        co = object.__new__(_Direct)
        co.lvl = sub.lv[-1]
        co.shape = sub.lv[-1].shape
        co.nk = len(idx)
        co.nsp = self.coarse.nsp
        co.Bb, co.Ab = self.coarse.Bb[idx], self.coarse.Ab[idx]
        sub.coarse = co
        return sub
