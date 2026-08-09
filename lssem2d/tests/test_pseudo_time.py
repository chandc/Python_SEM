"""The pseudo-time term (dtau) must change the OPERATOR and not the RESIDUAL.

See PSEUDO_TIME_DESIGN.md.  The term comes from the 1996 F77 source
(reference/tj_channel_1996.f), where the momentum row carries a bare `u` inside
the dt bracket and the transpose coefficient is (dt+fac1).  Because `u` and `fu`
are assigned from the same array before the residual is formed, its contribution
to the residual is identically zero -- it modifies the Jacobian only.

The failure mode these tests exist to catch: apply_L carries kappa*u
unconditionally (it is also the operator applied to the increment), so if
_drop_pseudo is not applied where a RESIDUAL is formed, dtau silently changes the
equations being solved and moves the converged answer at leading order.
"""
import numpy as np
import pytest

from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT, ls_coeffs, ls_pseudo
from lssem2d.mesh import build_channel
from lssem2d import solver as S

N = 5


def _case(dtau=None, **kw):
    mesh = build_channel(1.0, 1.0, 2, 2, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/100.0, dt=0.5, fac1=1.0,
                     dtau=dtau, **kw)
    rng = np.random.default_rng(7)
    shp = (mesh.nelem, N+1, N+1)
    U = rng.standard_normal(shp + (4,))
    fu = np.ascontiguousarray(U[..., 0])
    fv = np.ascontiguousarray(U[..., 1])
    st.update_linearisation(fu, fv)
    return st, U, fu, fv


@pytest.mark.parametrize("kw", [{}, dict(w_mom=0.1, w_mass=0.0),
                                dict(w_mom=2.0, w_mass=1.0)])
def test_default_is_a_no_op(kw):
    """dtau=None must give kappa=0 for every weighting, so nothing changes."""
    st, _, _, _ = _case(None, **kw)
    assert ls_pseudo(st) == 0.0


def test_kappa_is_a_flux_over_dtau():
    for dtau in (10.0, 1.0, 0.1):
        st, _, _, _ = _case(dtau)
        _, a_flux, _ = ls_coeffs(st)
        assert ls_pseudo(st) == pytest.approx(a_flux / dtau)


@pytest.mark.parametrize("dtau", [10.0, 1.0, 0.1])
def test_operator_gains_exactly_kappa(dtau):
    """L gains kappa*u*wq on the momentum rows; L^T gains kappa*su on the same."""
    sa, U, fu, fv = _case(dtau)
    sb, _, _, _ = _case(None)
    sb.update_linearisation(fu, fv)
    kap = ls_pseudo(sa)

    dL = apply_L(sa, U, fu, fv).copy() - apply_L(sb, U, fu, fv).copy()
    expect = np.zeros_like(dL)
    expect[..., 0] = kap * U[..., 0] * sa.mesh.wq
    expect[..., 1] = kap * U[..., 1] * sa.mesh.wq
    assert np.abs(dL - expect).max() < 1e-13

    su = np.ascontiguousarray(np.random.default_rng(3).standard_normal(dL.shape))
    dT = apply_LT(sa, su.copy(), fu, fv).copy() - apply_LT(sb, su.copy(), fu, fv).copy()
    expectT = np.zeros_like(dT)
    expectT[..., 0] = kap * su[..., 0]
    expectT[..., 1] = kap * su[..., 1]
    assert np.abs(dT - expectT).max() < 1e-12


@pytest.mark.parametrize("dtau", [10.0, 1.0, 0.1])
def test_residual_is_unchanged(dtau):
    """The whole contract: dtau must not move the equations being solved."""
    def residual(dt_):
        st, U, fu, fv = _case(dt_)
        f = np.ascontiguousarray(U[..., 0] / 2.0)
        g = np.ascontiguousarray(U[..., 1] / 2.0)
        st.update_linearisation(f, g)
        r = apply_L(st, U, f, g) - np.zeros_like(U)
        S._drop_pseudo(st, r, U)
        return r

    assert np.abs(residual(dtau) - residual(None)).max() < 1e-14


def test_residual_check_is_live():
    """Without the cancellation the residual DOES move -- so the test above bites."""
    st, U, fu, fv = _case(1.0)
    f = np.ascontiguousarray(U[..., 0] / 2.0)
    g = np.ascontiguousarray(U[..., 1] / 2.0)
    st.update_linearisation(f, g)
    r_uncancelled = apply_L(st, U, f, g).copy()

    st0, _, _, _ = _case(None)
    st0.update_linearisation(f, g)
    r_ref = apply_L(st0, U, f, g).copy()

    assert np.abs(r_uncancelled - r_ref).max() > 1e-3


@pytest.mark.parametrize("dtau", [None, 1.0, 0.1])
def test_jacobi_still_matches_the_true_diagonal(dtau):
    """compute_jacobi must track apply_L, pseudo-time included.

    Element-INTERIOR nodes only: apply_A gather-scatters, so a single local node
    set to 1 is not a global unit vector where elements meet.
    """
    st, U, fu, fv = _case(dtau)
    M_inv = S.compute_jacobi(st, fu, fv)
    mask = st.get_global_mask()
    shape = (st.mesh.nelem, N+1, N+1, 4)

    rng = np.random.default_rng(11)
    checked = 0
    for ij in rng.integers([0, 1, 1, 0], [st.mesh.nelem, N, N, 4], size=(12, 4)):
        ij = tuple(ij)
        if mask[ij] == 0.0 or M_inv[ij] == 0.0:
            continue
        e = np.zeros(shape)
        e[ij] = 1.0
        d_true = S.apply_A(st, e, fu, fv)[ij]
        assert abs(d_true - 1.0/M_inv[ij]) <= 1e-10 * abs(d_true)
        checked += 1
    assert checked > 0, "no interior dofs were actually checked"
