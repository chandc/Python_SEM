"""(x,y) derivatives that carry arbitrary trailing axes (fields, modes).

WHY THIS EXISTS RATHER THAN REUSING lssem2d.operators.  Those routines are
shape-locked to exactly (nelem, n, n): `dUdx` does `np.matmul(D, U)` followed by
`out *= facx[:, None, None]`, both of which assume three dimensions.  The 3D
layout puts fields and z-modes on trailing axes -- (nelem, n, n, var, mode) --
precisely so that the whole mode set rides along as one batched contraction
(3D_DEVELOPMENT_PLAN.md sec 1.1).  Passing that through the 2D routines raises a
broadcast error, which is how this was caught.

lssem2d is NOT modified.  These are new functions with the same mathematics and
the same fac convention, and `lssem3d/tests/test_deriv.py` asserts they agree
with the 2D versions to the last bit on 3-dimensional input, so the reuse claim
is checked rather than assumed.
"""
import numpy as np

from . import device as DEV


def _fac(fac, U):
    """Reshape (nelem,) to broadcast against (nelem, n, n, *trailing)."""
    return fac.reshape((-1,) + (1,)*(U.ndim - 1))


def _xp(a):
    return DEV.xp(a)


def ddx(U, D, facx):
    """d/dx.  U is (nelem, n, n, ...); contracts D over the i index.

    tensordot, NOT einsum: np.einsum's C kernel is single-threaded, and the
    profile put 70% of a channel step inside it (7.9 of 11.3 s over 3 steps).
    tensordot reshapes to one large GEMM, which Accelerate/cuBLAS thread.
    out[e,p,j,...] = sum_i D[p,i] U[e,i,j,...]  ->  tensordot over (i).
    """
    xp = _xp(U)
    t = xp.tensordot(D, U, axes=([1], [1]))       # (p, e, j, ...)
    return xp.moveaxis(t, 0, 1)*_fac(facx, U)


def ddy(U, D, facy):
    """d/dy.  Contracts D over the j index."""
    xp = _xp(U)
    t = xp.tensordot(D, U, axes=([1], [2]))       # (q, e, i, ...)
    return xp.moveaxis(t, 0, 2)*_fac(facy, U)


def ddxT(S, D, facx):
    """Adjoint of ddx (transpose of the differentiation matrix)."""
    xp = _xp(S)
    t = xp.tensordot(D, S, axes=([0], [1]))       # (i, e, j, ...)
    return xp.moveaxis(t, 0, 1)*_fac(facx, S)


def ddyT(S, D, facy):
    """Adjoint of ddy."""
    xp = _xp(S)
    t = xp.tensordot(D, S, axes=([0], [2]))       # (j, e, i, ...)
    return xp.moveaxis(t, 0, 2)*_fac(facy, S)
