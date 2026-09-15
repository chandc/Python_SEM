"""STAGE 4: spectral convergence in (x,y) and in z, on a full 3D MMS.

    uv run --quiet python -m pytest lssem3d/tests/test_stage4_mms.py -q

WHY THIS CANNOT REUSE STAGE 2'S SOLUTION.  `test_stage2_mms.py` deliberately
used potentials that are POLYNOMIALS of degree <= N, so the GLL differentiation
matrix is EXACT on them and the operator can be pinned to ~1e-12.  That is the
right choice for finding operator bugs, and exactly the wrong one here: with no
truncation error there is no convergence rate to measure.  Stage 4 keeps the
same construction and swaps in TRIGONOMETRIC potentials, which no polynomial
basis represents exactly, so the error is pure truncation.

The construction (identical algebra to Stage 2, generic in phi, chi, q):

    u = -ik phi        v = ik chi        w = phi_x - chi_y
    ox = phi_xy - chi_yy + k^2 chi
    oy = k^2 phi - phi_xx + chi_xy
    oz = ik (chi_x + phi_y)
    p  = q

which makes ALL FIVE constraint rows vanish identically in exact arithmetic --
continuity, three vorticity definitions, vorticity divergence.  So for those
rows the computed residual IS the discretisation error, with no analytic forcing
to derive and no cancellation to get wrong.  That is what makes them the
cleanest possible probe of spatial accuracy.

GATE (plan sec 4, Stage 4): error must fall FASTER THAN ANY ALGEBRAIC RATE in N,
and exponentially in Nz.  Both are checked as such -- not against a fixed
tolerance, which would pass for a merely-convergent scheme.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, convect as CV, fourier as FR

EX = 2
NU, C = 0.037, 2.5
LZ = 2.0*np.pi

# incommensurate-ish wavenumbers, so no accidental symmetry hides a term
A, B = 1.7, 2.3
E, F = 1.3, 0.9


# --------- trigonometric potentials, every derivative the operator needs -----
def phi(x, y):      return np.sin(A*x)*np.cos(B*y)
def phi_x(x, y):    return A*np.cos(A*x)*np.cos(B*y)
def phi_y(x, y):   return -B*np.sin(A*x)*np.sin(B*y)
def phi_xx(x, y):  return -A*A*np.sin(A*x)*np.cos(B*y)
def phi_xy(x, y):  return -A*B*np.cos(A*x)*np.sin(B*y)
def phi_yy(x, y):  return -B*B*np.sin(A*x)*np.cos(B*y)
def phi_xxx(x, y): return -A**3*np.cos(A*x)*np.cos(B*y)
def phi_xyy(x, y): return -A*B*B*np.cos(A*x)*np.cos(B*y)


def chi(x, y):      return np.cos(E*x)*np.sin(F*y)
def chi_x(x, y):   return -E*np.sin(E*x)*np.sin(F*y)
def chi_y(x, y):    return F*np.cos(E*x)*np.cos(F*y)
def chi_xx(x, y):  return -E*E*np.cos(E*x)*np.sin(F*y)
def chi_xy(x, y):  return -E*F*np.sin(E*x)*np.cos(F*y)
def chi_yy(x, y):  return -F*F*np.cos(E*x)*np.sin(F*y)
def chi_xxy(x, y): return -E*E*F*np.cos(E*x)*np.cos(F*y)
def chi_yyy(x, y): return -F**3*np.cos(E*x)*np.cos(F*y)


def q(x, y):        return np.sin(1.1*x)*np.sin(0.7*y)
def q_x(x, y):      return 1.1*np.cos(1.1*x)*np.sin(0.7*y)
def q_y(x, y):      return 0.7*np.sin(1.1*x)*np.cos(0.7*y)


def geom(N):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    n = N+1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]
        Y[e] = m.ynod[e][None, :]
    return m, diff_matrix(N), X, Y


def exact_state(X, Y, k):
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
    ik = 1j*k
    f = np.zeros(X.shape + (3,), dtype=complex)
    f[..., 0] = (-ik*c*phi(X, Y) + q_x(X, Y)
                 + nu*ik*(phi_yy(X, Y) + phi_xx(X, Y) - k*k*phi(X, Y)))
    f[..., 1] = (ik*c*chi(X, Y) + q_y(X, Y)
                 + nu*ik*(k*k*chi(X, Y) - chi_yy(X, Y) - chi_xx(X, Y)))
    f[..., 2] = (c*(phi_x(X, Y) - chi_y(X, Y)) + ik*q(X, Y)
                 + nu*(k*k*phi_x(X, Y) - phi_xxx(X, Y) + chi_xxy(X, Y)
                       - phi_xyy(X, Y) + chi_yyy(X, Y) - k*k*chi_y(X, Y)))
    return f


def errors_at(N, k):
    """(constraint-row error, momentum-row error) at polynomial order N."""
    m, D, X, Y = geom(N)
    R = OP.apply_L0_complex(exact_state(X, Y, k), D, m.facx, m.facy, k, NU, C)
    con = max(np.abs(R[..., r, :]).max() for r in (0, 1, 2, 3, 7))
    fm = exact_momentum(X, Y, k, NU, C)
    mom = max(np.abs(R[..., r, 0] - fm[..., r-4]).max() for r in (4, 5, 6))
    return con, mom


# ------------------------------------------------- spatial: spectral in N

ORDERS = [4, 6, 8, 10, 12]
KS = [0.0, 2.0]


@pytest.mark.parametrize('k', KS)
def test_spatial_error_falls_faster_than_any_algebraic_rate(k):
    """Spectral convergence, stated as a rate rather than a tolerance.

    An h^p scheme has error ~ C N^-p, so log e vs log N is a STRAIGHT line and
    the local slope is constant.  Spectral convergence steepens without bound.
    Requiring the final slope to beat the initial one by a wide margin is a test
    a merely high-order scheme fails; a fixed tolerance is not.
    """
    Ns = np.array(ORDERS, dtype=float)
    con = np.array([errors_at(int(n), k)[0] for n in ORDERS])
    mom = np.array([errors_at(int(n), k)[1] for n in ORDERS])
    for name, err in (('constraint', con), ('momentum', mom)):
        if err.max() < 1e-13:
            continue                       # k=0 degenerates; covered elsewhere
        assert err[-1] < err[0], f'{name}: no convergence at all ({err})'
        lo = np.log(err[1]/err[0])/np.log(Ns[1]/Ns[0])       # early slope
        hi = np.log(err[-1]/err[-2])/np.log(Ns[-1]/Ns[-2])   # late slope
        assert hi < lo - 2.0, (
            f'{name} k={k}: slope {hi:.1f} vs {lo:.1f} -- algebraic, '
            f'not spectral.  errors {err}')


@pytest.mark.parametrize('k', KS)
def test_spatial_error_reaches_round_off_by_N12(k):
    """Absolute check to accompany the rate: a spectral method on an analytic
    solution should be at round-off by N = 12 on this domain."""
    con, mom = errors_at(12, k)
    assert con < 1e-9, f'k={k}: constraint rows {con:.3e}'
    assert mom < 1e-9, f'k={k}: momentum rows {mom:.3e}'


def test_the_mms_is_not_secretly_polynomial():
    """Negative control.  If the potentials happened to be exactly representable
    the error would be ~1e-15 at EVERY N, the convergence test above would have
    nothing to measure, and it would still pass its `err[-1] < err[0]` clause on
    noise.  Demand a genuinely resolvable-but-unresolved error at low N.
    """
    con, mom = errors_at(4, 2.0)
    assert max(con, mom) > 1e-6, (
        f'error at N=4 is only {max(con, mom):.3e} -- the manufactured solution '
        f'is being represented exactly, so there is no truncation to converge')


# ------------------------------------------------ z: spectral in Nz

def gz(z):    return np.exp(np.sin(z))          # analytic, all modes non-zero
def gz_p(z):  return np.cos(z)*np.exp(np.sin(z))


def z_error(nz):
    """Error in the convective term for u = (0, 0, g(z)), whose exact
    z-momentum contribution is g g' -- computed through the full dealiased
    pipeline, not through the transform alone."""
    N = 4
    m, D, _, _ = geom(N)
    kz = FR.wavenumbers(nz, LZ)
    z = np.arange(nz)*LZ/nz
    Uh = np.zeros((m.nelem, N+1, N+1, OP.NVAR, len(kz)), dtype=complex)
    Uh[..., OP.W_, :] = FR.to_modes(gz(z))[None, None, None, :]
    Nh = CV.convective(Uh, D, m.facx, m.facy, kz, nz)
    got = FR.to_physical(Nh[0, 0, 0, 2, :], nz)
    return np.abs(got - gz(z)*gz_p(z)).max()


def _r2(x, y):
    """Coefficient of determination of a straight-line fit y = a x + b."""
    a, b = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (a*x + b))**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1.0 - ss_res/ss_tot, a


def test_z_convergence_is_exponential_not_algebraic():
    """The distinguishing test, not just "the error is small".

    Exponential convergence is a straight line in log(err) vs **Nz**.
    Algebraic convergence is a straight line in log(err) vs **log Nz**.
    Both curves fall steeply, so smallness alone cannot tell them apart -- fit
    BOTH and require the exponential model to win.

    Restricted to Nz <= 24: by Nz = 32 the error is 5.0e-14, i.e. round-off,
    and fitting a rate through a round-off floor measures noise.
    """
    nzs = np.array([8, 12, 16, 24], dtype=float)
    errs = np.array([z_error(int(nz)) for nz in nzs])
    assert errs[0] > 1e-8, f'Nz=8 already converged ({errs[0]:.3e}); no rate'
    assert np.all(np.diff(errs) < 0), f'not monotone: {errs}'

    r2_exp, slope = _r2(nzs, np.log(errs))
    r2_alg, _ = _r2(np.log(nzs), np.log(errs))
    assert r2_exp > r2_alg, (
        f'log(err) vs log(Nz) fits better (R2 {r2_alg:.4f}) than log(err) vs Nz '
        f'(R2 {r2_exp:.4f}) -- convergence looks ALGEBRAIC.  errors {errs}')
    assert r2_exp > 0.99, f'exponential fit poor (R2 {r2_exp:.4f}): {errs}'
    assert slope < -0.5, f'decay rate {slope:.3f} per mode is too weak'


def test_z_convergence_reaches_round_off():
    """The rate test above stops at Nz = 24 to stay off the floor; this pins
    that the floor exists and is round-off, not a stalled plateau."""
    assert z_error(32) < 1e-12


def test_z_error_is_not_limited_by_aliasing():
    """With 3/2 dealiasing the quadratic product g*g' is computed exactly once
    resolved, so the error floor must be round-off rather than an aliasing
    plateau.  A plateau above ~1e-13 would mean the padding is too small."""
    assert z_error(32) < 1e-12
