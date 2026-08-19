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


def _fac(fac, U):
    """Reshape (nelem,) to broadcast against (nelem, n, n, *trailing)."""
    return fac.reshape((-1,) + (1,)*(U.ndim - 1))


def ddx(U, D, facx):
    """d/dx.  U is (nelem, n, n, ...); contracts D over the i index."""
    return np.einsum('pi,eij...->epj...', D, U)*_fac(facx, U)


def ddy(U, D, facy):
    """d/dy.  Contracts D over the j index."""
    return np.einsum('qj,eij...->eiq...', D, U)*_fac(facy, U)


def ddxT(S, D, facx):
    """Adjoint of ddx (transpose of the differentiation matrix)."""
    return np.einsum('pi,epj...->eij...', D, S)*_fac(facx, S)


def ddyT(S, D, facy):
    """Adjoint of ddy."""
    return np.einsum('qj,eiq...->eij...', D, S)*_fac(facy, S)
