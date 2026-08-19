"""The batched derivatives must agree with lssem2d exactly on 3-D input.

This is the check that licenses lssem3d/deriv.py existing at all: it is a
reimplementation, so it has to be pinned to the original rather than merely
believed.  Any divergence here would silently change every operator downstream.
"""
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.operators import dUdx, dUdy, DxT, DyT
from lssem3d import deriv as DV

N, EX = 5, 3


def _geom():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    return m, diff_matrix(N)


def test_matches_lssem2d_bitwise_on_3d_input():
    m, D = _geom()
    U = np.random.default_rng(0).standard_normal((m.nelem, N+1, N+1))
    for mine, theirs, fac in ((DV.ddx, dUdx, m.facx), (DV.ddy, dUdy, m.facy),
                              (DV.ddxT, DxT, m.facx), (DV.ddyT, DyT, m.facy)):
        a = mine(U.copy(), D, fac)
        b = theirs(U.copy(), D, fac)
        assert np.abs(a - b).max() < 1e-13, mine.__name__


def test_trailing_axes_are_independent():
    """A (var, mode) batch must give the same answer as looping the slices.

    This is the property the whole batched design rests on.
    """
    m, D = _geom()
    U = np.random.default_rng(1).standard_normal((m.nelem, N+1, N+1, 4, 6))
    got = DV.ddx(U, D, m.facx)
    for v in range(4):
        for k in range(6):
            one = DV.ddx(U[..., v, k], D, m.facx)
            assert np.abs(got[..., v, k] - one).max() < 1e-14


def test_adjointness():
    m, D = _geom()
    rng = np.random.default_rng(2)
    a = rng.standard_normal((m.nelem, N+1, N+1, 2))
    b = rng.standard_normal((m.nelem, N+1, N+1, 2))
    for d, dT, fac in ((DV.ddx, DV.ddxT, m.facx), (DV.ddy, DV.ddyT, m.facy)):
        lhs = float(np.sum(d(a, D, fac)*b))
        rhs = float(np.sum(a*dT(b, D, fac)))
        assert abs(lhs - rhs)/max(abs(lhs), 1e-300) < 1e-12
