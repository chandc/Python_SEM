"""STAGE 1, criterion 1: the 3D operator at k_z = 0 must BE the 2D operator.

    uv run --quiet python -m pytest lssem3d/tests/test_stage1_vs_2d.py -q

3D_DEVELOPMENT_PLAN.md Stage 1 calls this the single most valuable test in the
project, and the quadrature-weight bug (fixed in 67130d0) is why: it survived
every symmetry and adjointness test, because those are satisfied by the WRONG
operator too.  A bit-level comparison against code that is already validated
against Gartling, Ghia and Kovasznay does not have that blind spot.

THE CORRESPONDENCE.  lssem2d's rows (lssem.py _apply_L_numpy) are

    su0 = (a_mass*u + a_flux*(conv + p_x + nu*om_y)) * wq      momentum x
    su1 = (a_mass*v + a_flux*(conv + p_y - nu*om_x)) * wq      momentum y
    su2 = (a_p*p + u_x + v_y) * wq                             continuity
    su3 = (om + u_y - v_x) * wq                                vorticity

and lssem3d's in-plane rows at k_z = 0 are

    row 4 = c*u + p_x + nu*oz_y                                momentum x
    row 5 = c*v + p_y - nu*oz_x                                momentum y
    row 0 = u_x + v_y                                          continuity
    row 3 = v_x - u_y - oz                                     vorticity

so with c = a_mass, a_flux = 1 (w_mom = 1), a_p = 0 (no AC) and the linearisation
zeroed (conv = 0), three rows must agree exactly and the fourth agrees up to an
overall SIGN: lssem2d writes the vorticity residual as om - (v_x - u_y), lssem3d
as (v_x - u_y) - oz.  Both define the same oz and both square to the same
functional; only the row's sign differs, and that is a convention, not an error.
The test asserts the sign flip explicitly rather than taking an absolute value,
so a genuine sign error elsewhere would still fail.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L as apply_L2D
from lssem3d import operator as OP

N, EX = 5, 3
NU = 0.017
A_MASS = 2.75            # c in 3D, a_mass in 2D

# lssem2d field order: u, v, p, om
I2_U, I2_V, I2_P, I2_OM = 0, 1, 2, 3


@pytest.fixture(scope='module')
def setup():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    # a_mass = w_mass*fac1/dt -> pick fac1 = 1, dt = 1/A_MASS, w_mass = 1
    st = SolverState(m, D, nu=NU, dt=1.0/A_MASS, fac1=1.0, w_mom=1.0, w_mass=1.0)
    z = np.zeros((m.nelem, N+1, N+1))
    st.update_linearisation(z, z)          # conv == 0: the Stokes-like operator
    return m, D, st, z


def _random_2d(m, seed):
    return np.random.default_rng(seed).standard_normal((m.nelem, N+1, N+1, 4))


def _embed(U2, m):
    """2D state (u,v,p,om) -> 3D split-real state at a single k_z = 0 mode."""
    U3 = np.zeros((m.nelem, N+1, N+1, OP.NVAR_R, 1))
    U3[..., OP.U_, 0] = U2[..., I2_U]
    U3[..., OP.V_, 0] = U2[..., I2_V]
    U3[..., OP.P_, 0] = U2[..., I2_P]
    U3[..., OP.OZ_, 0] = U2[..., I2_OM]
    return U3                               # imaginary half left at zero


def test_ls_coeffs_are_what_this_test_assumes(setup):
    """Guard the premise: a_mass == A_MASS and a_flux == 1, or the comparison
    below is not comparing what its docstring claims."""
    from lssem2d.lssem import ls_coeffs
    _, _, st, _ = setup
    a_mass, a_flux, _ = ls_coeffs(st)
    assert a_mass == pytest.approx(A_MASS)
    assert a_flux == pytest.approx(1.0)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_3d_at_kz0_reproduces_the_2d_operator(setup, seed):
    """The four in-plane rows, row by row, against lssem2d."""
    m, D, st, z = setup
    U2 = _random_2d(m, seed)
    U3 = _embed(U2, m)

    su = np.asarray(apply_L2D(st, U2, z, z)).copy()
    R3 = OP.apply_L(U3, D, m.facx, m.facy, 0.0, NU, A_MASS, m.wq)

    pairs = ((0, 4, +1.0),        # 2D su0 (mom x)      <-> 3D row 4
             (1, 5, +1.0),        # 2D su1 (mom y)      <-> 3D row 5
             (2, 0, +1.0),        # 2D su2 (continuity) <-> 3D row 0
             (3, 3, -1.0))        # 2D su3 (vorticity)  <-> -(3D row 3)
    for r2, r3, sgn in pairs:
        got = R3[..., r3, 0]
        want = sgn*su[..., r2]
        scale = max(np.abs(want).max(), 1e-30)
        err = np.abs(got - want).max()/scale
        assert err < 1e-13, (f'2D row {r2} vs 3D row {r3} (sign {sgn:+.0f}): '
                             f'rel err {err:.3e}')


def test_transverse_rows_vanish_for_an_inplane_state(setup):
    """Embedding a purely 2D state must leave w, ox, oy rows at exactly zero."""
    m, D, st, z = setup
    R3 = OP.apply_L(_embed(_random_2d(m, 7), m), D, m.facx, m.facy, 0.0,
                    NU, A_MASS, m.wq)
    for r in (1, 2, 6, 7):
        assert np.abs(R3[..., r, 0]).max() == 0.0, f'row {r} should be identically zero'


def test_imaginary_half_stays_zero(setup):
    """A real k_z = 0 state must produce no imaginary residual at all."""
    m, D, st, z = setup
    R3 = OP.apply_L(_embed(_random_2d(m, 8), m), D, m.facx, m.facy, 0.0,
                    NU, A_MASS, m.wq)
    assert np.abs(R3[..., OP.NROW:, 0]).max() == 0.0


def test_the_weights_are_required_for_this_to_hold(setup):
    """Negative control: drop wq and the comparison must FAIL.

    This is what pins the fix in 67130d0 -- without it, an unweighted operator
    would sail through every other test in the suite.
    """
    m, D, st, z = setup
    U2 = _random_2d(m, 9)
    su = np.asarray(apply_L2D(st, U2, z, z)).copy()
    R3 = OP.apply_L(_embed(U2, m), D, m.facx, m.facy, 0.0, NU, A_MASS)  # no wq
    err = np.abs(R3[..., 0, 0] - su[..., 2]).max()/np.abs(su[..., 2]).max()
    assert err > 1e-3, 'unweighted operator matched 2D; the weights are inert'
