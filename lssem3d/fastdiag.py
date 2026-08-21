"""Fast-diagonalization block preconditioner for the per-mode LS normal solve.

WHY.  Jacobi cannot see operator structure: at production settings the stage
solve costs O(10^3-10^4) CG iterations, ~99% of runtime, and the matvec is
bandwidth-bound so cores do not help.  But with legacy row weights the
field-by-field diagonal blocks of the normal operator A = Qt Q L^T W rw L are
SEPARABLE 2D Helmholtz operators on the tensor-product mesh:

    block(f) = a_f * Ax (x) My  +  b_f * Mx (x) Ay  +  sig_f(k) * Mx (x) My

with per-field coefficients (e = nu^2/c^2; k = k_z; legacy weight 1/c on the
momentum rows):

    field   rows it appears in                     a       b       sigma
    u       cont(dx), oz-def(dy), oy-def(ik), mom  1       1       k^2 + 1
    v       oz-def(dx), cont(dy), ox-def(ik), mom  1       1       k^2 + 1
    w       oy-def(dx), ox-def(dy), cont(ik), mom  1       1       k^2 + 1
    ox      odiv(dx), mom-z(nu dy/c), def, mom-y   1       e       1 + e k^2
    oy      mom-z(nu dx/c), odiv(dy), def, mom-x   e       1       1 + e k^2
    oz      mom-y(nu dx/c), mom-x(nu dy/c), def,
            odiv(ik)                               e       e       1 + k^2
    p       mom-x(dx/c), mom-y(dy/c), mom-z(ik/c)  1/c^2   1/c^2   k^2/c^2

The couplings the block form DROPS (u-p, u-omega) carry weights 1/c and nu/c
-- small at production c, which is why this is a good preconditioner exactly
where the solver is expensive.  Kronecker sums invert by fast diagonalization
(Lynch-Rice-Thomas): two generalized 1D eigenproblems A1 Z = M1 Z Lam,
Z^T M1 Z = I, solved ONCE (the GLL mass is diagonal, so it is one symmetric
eigh per direction), give

    block(f)^-1 = (Zx (x) Zy) diag(1/(a lam_x + b lam_y + sig)) (Zx (x) Zy)^T

applied as four small dense matmuls -- cheaper than one CG matvec.  The
eigenvectors are mode- and stage-independent; only the scalar denominators
change with (k_z, c).

SCOPE.  Doubly-periodic tensor meshes from build_channel (the TGV rigs).
Walls need per-BC 1D eigenbases -- a later extension.  The preconditioner is
symmetric in the multiplicity-weighted PCG inner product: gather averages the
(equal) copies of each assembled-residual node, the global solve is symmetric,
and the scatter replicates -- on the continuous subspace that composition is
congruent to the symmetric global inverse.

USE.  fd = FastDiagPeriodic(mesh, D, N, nz, kz, nu, c, mask);  S3.pcg(...,
M_inv=fd, ...) -- solver3d.pcg already accepts a callable.  Null directions
(the k = 0 pressure constant) get a zeroed inverse entry, and the output is
re-masked so CG stays in the free subspace.
"""
import numpy as np
from lssem2d.lgl import lgl_weights
from . import operator as OP


def _assemble_1d(E, N, h, D, w):
    """Assembled periodic 1D mass (diagonal) and stiffness on E elements."""
    ng = E*N
    M1 = np.zeros(ng)
    A1 = np.zeros((ng, ng))
    Ael = (2.0/h)*(D.T @ (w[:, None]*D))          # sum_q w_q D[q,i] D[q,j]
    for e in range(E):
        g = [(e*N + i) % ng for i in range(N + 1)]
        M1[g] += 0.5*h*w
        A1[np.ix_(g, g)] += Ael
    return M1, A1


def _eig(M1, A1):
    s = np.sqrt(M1)
    lam, Q = np.linalg.eigh(A1/np.outer(s, s))
    lam[np.abs(lam) < 1e-10] = 0.0                # periodic null mode, exactly
    return lam, Q/s[:, None]                      # Z with Z^T M Z = I


class FastDiagPeriodic:
    def __init__(self, mesh, D, N, nz, kz, nu, c, mask):
        n = N + 1
        hx, hy = float(mesh.hx[0]), float(mesh.hy[0])
        assert np.allclose(mesh.hx, hx) and np.allclose(mesh.hy, hy), \
            'tensor mesh with uniform elements required'
        Ltot_x = hx*round((mesh.xnod.max() - mesh.xnod.min())/hx)
        Ex = int(round((mesh.xnod.max() - mesh.xnod.min())/hx))
        Ey = int(round((mesh.ynod.max() - mesh.ynod.min())/hy))
        assert Ex*Ey == mesh.nelem, 'expected full tensor element grid'
        w = lgl_weights(N)
        Mx, Ax = _assemble_1d(Ex, N, hx, D, w)
        My, Ay = _assemble_1d(Ey, N, hy, D, w)
        self.lamx, self.Zx = _eig(Mx, Ax)
        self.lamy, self.Zy = _eig(My, Ay)
        ngx, ngy = Ex*N, Ey*N

        # local (e, i, j) -> global (gx, gy); build_channel rasters e = ex*Ey+ey
        self.GX = np.empty((mesh.nelem, n), dtype=int)
        self.GY = np.empty((mesh.nelem, n), dtype=int)
        for e in range(mesh.nelem):
            ex, ey = divmod(e, Ey)
            self.GX[e] = [(ex*N + i) % ngx for i in range(n)]
            self.GY[e] = [(ey*N + j) % ngy for j in range(n)]
        mult = np.zeros((ngx, ngy))
        np.add.at(mult, (self.GX[:, :, None], self.GY[:, None, :]), 1.0)
        self.mult = mult
        self.mask = mask

        # denominators: (ngx, ngy, 14, nk) from the per-field (a, b, sigma)
        kz = np.atleast_1d(np.asarray(kz, dtype=float))
        e2 = (nu/c)**2
        ic2 = 1.0/c**2
        k2 = kz**2
        one = np.ones_like(k2)
        a7 = np.array([1, 1, 1, 1, e2, e2, ic2])
        b7 = np.array([1, 1, 1, e2, 1, e2, ic2])
        sig7 = np.stack([k2 + 1, k2 + 1, k2 + 1,
                         1 + e2*k2, 1 + e2*k2, 1 + k2, ic2*k2])  # (7, nk)
        a14 = np.concatenate([a7, a7]); b14 = np.concatenate([b7, b7])
        sig14 = np.concatenate([sig7, sig7], axis=0)              # (14, nk)
        den = (a14[None, None, :, None]*self.lamx[:, None, None, None]
               + b14[None, None, :, None]*self.lamy[None, :, None, None]
               + sig14[None, None, :, :])
        self.dinv = np.where(den > 1e-10, 1.0/np.where(den == 0, 1, den), 0.0)

    def __call__(self, r):
        G = np.zeros(self.mult.shape + r.shape[3:])
        np.add.at(G, (self.GX[:, :, None], self.GY[:, None, :]), r)
        G /= self.mult[:, :, None, None]
        T = np.einsum('ap,abfk->pbfk', self.Zx, G)
        T = np.einsum('bq,pbfk->pqfk', self.Zy, T)
        T *= self.dinv
        T = np.einsum('ap,pqfk->aqfk', self.Zx, T)
        G = np.einsum('bq,aqfk->abfk', self.Zy, T)
        z = G[self.GX[:, :, None], self.GY[:, None, :]]
        return z*self.mask if self.mask is not None else z
