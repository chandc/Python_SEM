"""STAGE 2: a single non-zero k_z against a hand-derived analytic solution.

    uv run --quiet python -m pytest lssem3d/tests/test_stage2_mms.py -q

Stage 1 pinned the operator at k_z = 0 against lssem2d.  That comparison is blind
to every `i*k_z` term, since they all vanish there -- so the transverse fields
(w, ox, oy) and the whole z-coupling are still untested against anything except
each other.  This closes that gap.

CONSTRUCTION.  Take two vector potentials, each giving an exactly
divergence-free velocity for any smooth phi, chi:

    A = (0, phi e^{ikz}, 0)   ->   u = (-ik phi,      0,   phi_x) e^{ikz}
    A = (chi e^{ikz}, 0, 0)   ->   u = (      0, ik chi,  -chi_y) e^{ikz}

Superposed (the operator is linear -- convection is explicit and not in L):

    u = -ik phi        v = ik chi        w = phi_x - chi_y

with the curl worked out by hand,

    ox = phi_xy - chi_yy + k^2 chi
    oy = k^2 phi - phi_xx + chi_xy
    oz = ik (chi_x + phi_y)

and p = q, free.  By construction ALL FIVE constraint rows vanish identically --
continuity, the three vorticity definitions, and the vorticity divergence -- so
those are machine-zero tests with no tolerance to tune.  The three momentum rows
must equal a forcing that is also derived by hand below.

PHI, CHI AND Q ARE POLYNOMIALS of degree <= N, so the GLL differentiation matrix
is EXACT on them.  That is what lets this demand ~1e-12 rather than a
discretisation-limited tolerance: any failure is an operator bug, not truncation.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP

N, EX = 6, 2
NU, C = 0.037, 2.5
KZS = [0.0, 1.0, 2.5, -3.25]


# ---- polynomial potentials and every derivative used below, by hand ----------
def phi(x, y):      return x**2*y**3 + x*y
def phi_x(x, y):    return 2*x*y**3 + y
def phi_y(x, y):    return 3*x**2*y**2 + x
def phi_xx(x, y):   return 2*y**3
def phi_xy(x, y):   return 6*x*y**2 + 1.0
def phi_yy(x, y):   return 6*x**2*y
def phi_xxx(x, y):  return np.zeros_like(x)
def phi_xyy(x, y):  return 12*x*y


def chi(x, y):      return x**3*y + y**2
def chi_x(x, y):    return 3*x**2*y
def chi_y(x, y):    return x**3 + 2*y
def chi_xx(x, y):   return 6*x*y
def chi_xy(x, y):   return 3*x**2
def chi_yy(x, y):   return 2.0*np.ones_like(x)
def chi_xxy(x, y):  return 6*x
def chi_yyy(x, y):  return np.zeros_like(x)


def q(x, y):        return x**3 - y**2
def q_x(x, y):      return 3*x**2
def q_y(x, y):      return -2*y


@pytest.fixture(scope='module')
def geom():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    n = N+1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]
        Y[e] = m.ynod[e][None, :]
    return m, diff_matrix(N), X, Y


def exact_state(X, Y, k):
    """(nelem, n, n, 7, 1) complex, satisfying all five constraint rows."""
    U = np.zeros(X.shape + (OP.NVAR, 1), dtype=complex)
    ik = 1j*k
    U[..., OP.U_, 0] = -ik*phi(X, Y)
    U[..., OP.V_, 0] = ik*chi(X, Y)
    U[..., OP.W_, 0] = phi_x(X, Y) - chi_y(X, Y)
    U[..., OP.OX_, 0] = phi_xy(X, Y) - chi_yy(X, Y) + k*k*chi(X, Y)
    U[..., OP.OY_, 0] = k*k*phi(X, Y) - phi_xx(X, Y) + chi_xy(X, Y)
    U[..., OP.OZ_, 0] = ik*(chi_x(X, Y) + phi_y(X, Y))
    U[..., OP.P_, 0] = q(X, Y)
    return U


def exact_momentum(X, Y, k, nu, c):
    """The three momentum-row values the operator must return, by hand."""
    ik = 1j*k
    f = np.zeros(X.shape + (3,), dtype=complex)
    # row 4:  c*u + p_x + nu*(oz_y - ik*oy)
    f[..., 0] = (-ik*c*phi(X, Y) + q_x(X, Y)
                 + nu*ik*(phi_yy(X, Y) + phi_xx(X, Y) - k*k*phi(X, Y)))
    # row 5:  c*v + p_y + nu*(ik*ox - oz_x)
    f[..., 1] = (ik*c*chi(X, Y) + q_y(X, Y)
                 + nu*ik*(k*k*chi(X, Y) - chi_yy(X, Y) - chi_xx(X, Y)))
    # row 6:  c*w + ik*p + nu*(oy_x - ox_y)
    f[..., 2] = (c*(phi_x(X, Y) - chi_y(X, Y)) + ik*q(X, Y)
                 + nu*(k*k*phi_x(X, Y) - phi_xxx(X, Y) + chi_xxy(X, Y)
                       - phi_xyy(X, Y) + chi_yyy(X, Y) - k*k*chi_y(X, Y)))
    return f


@pytest.mark.parametrize('k', KZS)
def test_constraint_rows_vanish(geom, k):
    """Rows 0,1,2,3,7 are identically zero for a divergence-free curl pair.

    No tolerance to tune: the potentials are polynomials of degree <= N, so the
    GLL derivative is exact and this is a machine-precision statement.
    """
    m, D, X, Y = geom
    R = OP.apply_L0_complex(exact_state(X, Y, k), D, m.facx, m.facy, k, NU, C)
    for row, name in ((0, 'continuity'), (1, 'vorticity-x'), (2, 'vorticity-y'),
                      (3, 'vorticity-z'), (7, 'vorticity-divergence')):
        err = np.abs(R[..., row, :]).max()
        assert err < 1e-11, f'k={k}, {name} row: {err:.3e}'


@pytest.mark.parametrize('k', KZS)
def test_momentum_rows_match_the_analytic_forcing(geom, k):
    """Rows 4,5,6 equal a forcing derived by hand, not by another code path."""
    m, D, X, Y = geom
    R = OP.apply_L0_complex(exact_state(X, Y, k), D, m.facx, m.facy, k, NU, C)
    f = exact_momentum(X, Y, k, NU, C)
    for row in (4, 5, 6):
        got = R[..., row, 0]
        want = f[..., row-4]
        scale = max(np.abs(want).max(), 1e-30)
        err = np.abs(got - want).max()/scale
        assert err < 1e-11, f'k={k}, momentum row {row}: rel err {err:.3e}'


def test_transverse_fields_are_actually_exercised(geom):
    """Negative control: at k != 0 the state must have non-trivial w, ox, oy.

    Stage 1 could not see these -- everything with an i*k vanishes at k_z = 0 --
    so without this the test above could pass on a degenerate state and prove
    nothing about the z-coupling.
    """
    m, D, X, Y = geom
    U = exact_state(X, Y, 2.5)
    for f, name in ((OP.W_, 'w'), (OP.OX_, 'ox'), (OP.OY_, 'oy')):
        assert np.abs(U[..., f, :]).max() > 0.1, f'{name} is trivial'
    assert np.abs(U[..., OP.U_, :].imag).max() > 0.1, 'u has no imaginary part'


def test_kz0_case_degenerates_as_expected(geom):
    """At k = 0 the same construction must collapse to a 2D state: u, oz and the
    whole imaginary half vanish, leaving only v=0, w = phi_x - chi_y in plane.

    Documents WHY Stage 1 could not have caught a z-coupling bug."""
    m, D, X, Y = geom
    U = exact_state(X, Y, 0.0)
    assert np.abs(U[..., OP.U_, :]).max() == 0.0
    assert np.abs(U[..., OP.V_, :]).max() == 0.0
    assert np.abs(U[..., OP.OZ_, :]).max() == 0.0
    assert np.abs(U.imag).max() == 0.0
