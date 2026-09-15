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


# --- non-monotone line-search window scoping -------------------------------
# Separate concern from dtau, but the same file already builds the fixtures.

def test_line_search_window_is_per_time_step():
    """state._ls_hist must not carry merits across time levels.

    The GLL reference is max(J) over the window, and J is measured against
    su_history, which step_bdf rebuilds at every time level.  A merit retained
    from an earlier step compares a different functional and would license a
    step the current level should reject.
    """
    mesh = build_channel(1.0, 1.0, 2, 2, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/100.0, dt=0.5, fac1=1.0)
    rng = np.random.default_rng(5)
    U = rng.standard_normal((mesh.nelem, N+1, N+1, 4)) * 0.05
    hist = [U]

    # poison the window with a merit no real iterate at the next level produces
    st._ls_hist = [1.0e9]
    S.step_bdf(st, hist, max_newton=2, line_search=True, cgsfac=1e-3,
               cg_max_iter=2000)

    assert 1.0e9 not in st._ls_hist, "stale merit survived into the next time step"
    assert len(st._ls_hist) <= 2, "window grew beyond this step's sub-iterations"


def test_reset_line_search_clears():
    st = SolverState(build_channel(1.0, 1.0, 2, 2, N), diff_matrix(N),
                     nu=0.01, dt=0.5, fac1=1.0)
    st._ls_hist = [3.0, 4.0]
    S.reset_line_search(st)
    assert st._ls_hist == []


@pytest.mark.parametrize("max_newton,expected", [(1, 10), (2, 1), (5, 1)])
def test_ls_memory_default_depends_on_regime(max_newton, expected, monkeypatch):
    """step_bdf picks Armijo for sub-iterated solves, GLL for a single step.

    max_newton > 1 is a bounded solve at a fixed time level and must be
    monotone: sub-iteration 0 drops J by ~3 decades from the previous time
    level, and a window retaining that value lets GLL accept steps that grow J.
    max_newton == 1 means successive calls form one continuous iteration, which
    is legitimately non-monotone.
    """
    seen = {}
    real = S.newton_step

    def spy(*a, **kw):
        seen['ls_memory'] = kw.get('ls_memory')
        return real(*a, **kw)

    monkeypatch.setattr(S, 'newton_step', spy)

    mesh = build_channel(1.0, 1.0, 2, 2, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/100.0, dt=0.5, fac1=1.0)
    U = np.random.default_rng(5).standard_normal((mesh.nelem, N+1, N+1, 4))*0.05
    S.step_bdf(st, [U], max_newton=max_newton, line_search=True,
               cgsfac=1e-3, cg_max_iter=2000)
    assert seen['ls_memory'] == expected


def test_ls_memory_explicit_value_is_respected(monkeypatch):
    """An explicit ls_memory must override the regime default."""
    seen = {}
    real = S.newton_step

    def spy(*a, **kw):
        seen['ls_memory'] = kw.get('ls_memory')
        return real(*a, **kw)

    monkeypatch.setattr(S, 'newton_step', spy)

    mesh = build_channel(1.0, 1.0, 2, 2, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/100.0, dt=0.5, fac1=1.0)
    U = np.random.default_rng(5).standard_normal((mesh.nelem, N+1, N+1, 4))*0.05
    S.step_bdf(st, [U], max_newton=5, line_search=True, ls_memory=7,
               cgsfac=1e-3, cg_max_iter=2000)
    assert seen['ls_memory'] == 7


# --- p-MG coarse operator must match the fine one --------------------------

@pytest.mark.parametrize("kw", [
    dict(w_mom=1.0, w_mass=1.0),
    dict(w_mom=0.1, w_mass=0.0),
    dict(w_mom=1.0, w_mass=1.0, dtau=0.3),
])
def test_pmg_coarse_carries_the_same_weighting(kw):
    """The p-MG coarse SolverState must inherit w_mom/w_mass/dtau.

    Built with only (nu, dt, fac1), ls_coeffs takes its LEGACY branch on the
    coarse grid -- (fac1, dt) instead of (fac1*w_mass/dt, w_mom) -- so the
    V-cycle preconditions a differently-weighted operator.  Measured cost of
    that mismatch at w_mom = w_mass = 1, dt = 0.1: 5955 CG iterations against
    597, a factor of 10.
    """
    from lssem2d import precond as P
    from lssem2d.lssem import ls_coeffs, ls_pseudo

    mesh = build_channel(1.0, 1.0, 2, 2, 8, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(8), nu=1.0/389.0, dt=0.1, fac1=1.0, **kw)
    fu = np.zeros((mesh.nelem, 9, 9))
    fv = np.zeros((mesh.nelem, 9, 9))
    st.update_linearisation(fu, fv)
    M_inv = S.compute_jacobi(st, fu, fv)

    pmg = P.make('pmg2', st, fu, fv, M_inv, False, pc=4, deg=4, coarse_deg=10)

    assert pmg.sc.w_mom == st.w_mom
    assert pmg.sc.w_mass == st.w_mass
    assert getattr(pmg.sc, 'dtau', None) == getattr(st, 'dtau', None)
    # the coefficients themselves, which is what actually matters
    assert ls_coeffs(pmg.sc)[:2] == pytest.approx(ls_coeffs(st)[:2])
    assert ls_pseudo(pmg.sc) == pytest.approx(ls_pseudo(st))
