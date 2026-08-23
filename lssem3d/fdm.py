"""Fast-diagonalisation preconditioner for the VVP least-squares operator.

WHY THIS AND NOT A DIAGONAL.  Point-Jacobi removes metric and element-size
variation but does nothing about the N-dependence of the spectral-element
condition number.  Sweeping N at FIXED dof (CUPY_BACKEND.md) puts CG iterations
at N^1.01 -- 3.9x from N = 4 to N = 16 -- while Jacobi's relative benefit erodes
(86.8x -> 47.6x).  So the payoff from an N-independent preconditioner is a
difference of SLOPES, and it compounds with every order added.  Costed at ~1
extra matvec per application the crossover is near N ~ 10-12; below that this
is a loss and the diagonal should be used instead.

WHY IT APPLIES.  Classical FDM needs a sum of tensor products, and the full
operator couples all seven fields at every node.  But a field-block-diagonal
preconditioner needs only each DIAGONAL block separable, and every one of them
is, to machine precision (scratch/fdm_structure.py verifies this by building
the blocks numerically from the real operator):

    u, v, w   a*K(x)M + b*M(x)K + s*M(x)M          full 2-D Helmholtz
    ox        a*K(x)M + s*M(x)M                    x-stiffness only
    oy        b*M(x)K + s*M(x)M                    y-stiffness only
    oz        s*M(x)M                              pure mass -- already diagonal
    p         (1/c^2)(a*K(x)M + b*M(x)K + s*M(x)M)

ox/oy/oz differ because row 7 is oxx + oyy + ik*oz: each vorticity component
sees one direction.  The legacy 1/c^2 momentum weighting is load-bearing here
-- it cancels the c^2 from c*u and leaves a clean unit mass term, which is what
makes the momentum blocks separable at all.

THE MECHANICS.  Solve the 1-D generalised eigenproblem K1 s = lam M1 s once and
M1-orthonormalise, so S^T M1 S = I and S^T K1 S = Lam.  Then the SAME S
diagonalises the 2-D block in both directions and the inverse is
transform -> divide -> transform back, at O(N^4) per element instead of O(N^6).
K1 and M1 are REFERENCE-element quantities, so one eigendecomposition serves
every element, field and mode; only the diagonal shift changes.

WHAT IT DROPS.  The field coupling, 37% of the operator by Frobenius norm.  A
preconditioner only has to cluster the spectrum, not invert the operator, so
whether that is acceptable is a measured question -- see scratch/fdm_bench.py.
"""
import numpy as np

from . import device as DEV
from . import operator as OP
from lssem2d.lgl import diff_matrix, lgl_weights


def reference_factors(N):
    """S, Lam for the reference element: S^T M1 S = I, S^T K1 S = diag(Lam).

    Depends only on N, so it is computed once and reused for every element,
    field and mode.
    """
    w = lgl_weights(N)
    D = diff_matrix(N)
    K1 = D.T @ np.diag(w) @ D
    M1 = np.diag(w)
    # M1 is diagonal and positive, so the generalised problem reduces to a
    # symmetric one -- more accurate than inverting M1 and calling eig.
    r = np.sqrt(w)
    C = (K1/r[:, None])/r[None, :]
    lam, Q = np.linalg.eigh(0.5*(C + C.T))
    S = Q/r[:, None]                       # undo the scaling: S^T M1 S = I
    return S, lam


def _shifts(field, lam, a, b, s):
    """Diagonal of the block in the eigenbasis, shape (n1, n1)."""
    li, lj = lam[:, None], lam[None, :]
    if field in (OP.U_, OP.V_, OP.W_, OP.P_):
        return a*li + b*lj + s
    if field == OP.OX_:
        return a*li + s
    if field == OP.OY_:
        return b*lj + s
    return np.full((lam.size, lam.size), s)      # oz: pure mass


def build(mesh, N, kz, nu, c, rw, mask, kap=0.0, floor=1e-30, like=None):
    """Return a callable preconditioner z = M^-1 r.

    `floor` guards the pressure block, whose k = 0 mode is singular on an
    element (the constant-pressure null space).  The assembled system is
    regularised by the pin in bc.py, but the element-local block is not.
    """
    S, lam = reference_factors(N)
    n1 = N + 1
    nk = len(kz)
    fx = np.asarray(mesh.facx, dtype=float)
    fy = np.asarray(mesh.facy, dtype=float)
    w7 = rw[7] if rw is not None else 1.0
    wm = rw[4] if rw is not None else 1.0     # the 1/c^2 momentum weight

    d = np.empty((mesh.nelem, n1, n1, OP.NVAR_R, nk))
    for e in range(mesh.nelem):
        ax, by, m0 = fx[e]/fy[e], fy[e]/fx[e], 1.0/(fx[e]*fy[e])
        for k in range(nk):
            kk = float(kz[k])**2
            for f, (a, b, s) in {
                    OP.U_:  (ax, by, (1.0 + kk)*m0),
                    OP.V_:  (ax, by, (1.0 + kk)*m0),
                    OP.W_:  (ax, by, (1.0 + kk)*m0),
                    OP.OX_: (w7*ax, 0.0, (1.0 + nu*nu*kk*wm)*m0),
                    OP.OY_: (0.0, w7*by, (1.0 + nu*nu*kk*wm)*m0),
                    OP.OZ_: (0.0, 0.0, (1.0 + w7*kk + nu*nu*kk*wm)*m0),
                    OP.P_:  (wm*ax, wm*by, wm*kk*m0)}.items():
                blk = _shifts(f, lam, a, b, s)
                d[e, :, :, f, k] = blk
                d[e, :, :, f + OP.NVAR, k] = blk      # imaginary part, same block
    dinv = np.where(np.abs(d) < floor, 0.0, 1.0/np.where(d == 0, 1.0, d))

    Sd = DEV.to_device(S, like) if like is not None else S
    dinv = DEV.to_device(dinv, like) if like is not None else dinv
    maskd = mask

    def apply(r):
        # transform -> divide -> transform back, one direction at a time
        g = DEV.einsum('pi,eijvk->epjvk', Sd, r)
        g = DEV.einsum('qj,epjvk->epqvk', Sd, g)
        g = g*dinv
        g = DEV.einsum('ip,epqvk->eiqvk', Sd, g)
        g = DEV.einsum('jq,eiqvk->eijvk', Sd, g)
        return g*maskd if maskd is not None else g

    return apply
