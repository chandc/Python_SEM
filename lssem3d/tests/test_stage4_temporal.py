"""STAGE 4: temporal order of RKW3/Crank-Nicolson, and what limits it.

    uv run --quiet python -m pytest lssem3d/tests/test_stage4_temporal.py -q

The plan's gate reads "RKW3 slope 3.0 +/- 0.15 ... measuring 2.0 means the
alpha/beta/gamma/zeta table is mis-transcribed OR Crank-Nicolson is limiting".
Those are two different diagnoses with two different fixes, and a single
convergence run on the full PDE cannot tell them apart.  So split the model
problem instead, on the scalar

    u' = (lam_e + lam_i) u,        u(0) = 1,     exact  u(t) = e^{(lam_e+lam_i)t}

which is precisely the problem the coefficients were designed for, and run it
three ways:

    lam_i = 0   -- pure explicit RK.  Order 3 REQUIRED.  This is the term the
                   gamma/zeta table controls, so a failure here IS a
                   mis-transcription and nothing else.
    lam_e = 0   -- pure implicit.  Crank-Nicolson alone; order 2 expected.
    both        -- the real configuration.

Measuring 2 in the mixed case is therefore NOT a bug to hunt: it is CN doing
what CN does, and the explicit-only column proves the table is right.  Recording
which of the two it is, is the entire point of this file.

The scalar problem also runs in microseconds, so the order can be measured over
a wide dt range where a PDE run would be dominated by spatial error.
"""
import numpy as np
import pytest
from lssem3d import timestep as T
from lssem3d import solver3d as S3


def step(u, dt, lam_e, lam_i):
    """One RKW3/CN step of u' = (lam_e + lam_i) u, written out in full.

        u^k = u^{k-1} + dt[ g_k N^{k-1} + z_k N^{k-2} + a_k L^{k-1} + b_k L^k ]

    with N = lam_e*u explicit and L = lam_i*u implicit.  Note zeta_0 = 0, so the
    scheme is self-starting -- no history is needed across steps.
    """
    n_prev = 0.0
    for k in range(T.NSTAGE):
        nk = lam_e*u
        lk = lam_i*u
        rhs = u + dt*(T.GAMMA[k]*nk + T.ZETA[k]*n_prev + T.ALPHA[k]*lk)
        u = rhs/(1.0 - dt*T.BETA[k]*lam_i)
        n_prev = nk
    return u


def integrate(lam_e, lam_i, dt, tend=1.0):
    n = int(round(tend/dt))
    u = 1.0
    for _ in range(n):
        u = step(u, dt, lam_e, lam_i)
    return u


def observed_order(lam_e, lam_i, dts=(0.05, 0.025, 0.0125, 0.00625)):
    """Least-squares slope of log|error| vs log dt."""
    exact = np.exp(lam_e + lam_i)
    errs = np.array([abs(integrate(lam_e, lam_i, dt) - exact) for dt in dts])
    assert np.all(errs > 0), 'exact hit -- cannot measure a rate'
    return np.polyfit(np.log(np.array(dts)), np.log(errs), 1)[0], errs


# ------------------------------------------------- the diagnostic triple

def test_explicit_only_is_third_order():
    """THE gate.  Order 3 here means the gamma/zeta table is transcribed right.

    Real negative lam_e (a decaying mode) rather than imaginary, so the error is
    a clean power of dt and not an oscillation the fit would smear.
    """
    p, errs = observed_order(-1.5, 0.0)
    assert abs(p - 3.0) < 0.15, f'explicit-only order {p:.3f}, errors {errs}'


def test_implicit_only_is_second_order():
    """Crank-Nicolson alone.  Order 2 is correct, not a defect -- recorded so
    the mixed result below is not misread as a bug."""
    p, errs = observed_order(0.0, -1.5)
    assert abs(p - 2.0) < 0.15, f'implicit-only order {p:.3f}, errors {errs}'


def test_mixed_order_is_limited_by_crank_nicolson():
    """The configuration the solver actually runs.

    Asserts the ~2 that CN imposes AND, in the same breath, that the explicit
    path is still 3 -- so this file can never be read as "the scheme is broken"
    when it is simply CN-limited.
    """
    p_mix, errs = observed_order(-1.0, -0.5)
    p_exp, _ = observed_order(-1.5, 0.0)
    assert abs(p_exp - 3.0) < 0.15, (
        f'explicit path is order {p_exp:.3f} -- the table IS mis-transcribed, '
        f'and the mixed order {p_mix:.3f} is not merely CN-limited')
    assert 1.85 < p_mix < 2.5, f'mixed order {p_mix:.3f}, errors {errs}'


def test_consistency_relation_is_what_buys_third_order():
    """Negative control on the gate above.

    alpha_k + beta_k == gamma_k + zeta_k is asserted in exact arithmetic at
    import.  Break it by a plausible typo and the explicit-only order must
    collapse -- otherwise `test_explicit_only_is_third_order` is not actually
    sensitive to the table and proves nothing.
    """
    good = T.GAMMA
    try:
        T.GAMMA = (8/15, 5/12, 0.7)            # was 3/4
        p, _ = observed_order(-1.5, 0.0)
    finally:
        T.GAMMA = good
    assert p < 2.5, (
        f'a corrupted gamma still gave order {p:.3f} -- the order test is '
        f'insensitive to the coefficient table')


# ------------------------------------------ the rkw3_step contract itself

def test_rkw3_step_omits_alpha_and_says_so():
    """`solver3d.rkw3_step` assembles rhs = U + dt(gamma N + zeta N_prev) --
    the alpha_k L^{k-1} term is NOT in it, and is left to solve_stage.

    That is a real trap: a solve_stage that forgets alpha loses an order
    SILENTLY, with no shape error and a perfectly plausible field.  Pin the
    contract so the omission is a documented interface, not an accident.
    """
    seen = {}

    def rhs_explicit(U):
        return 2.0*U

    def solve_stage(rhs, c, k):
        seen[k] = (rhs.copy(), c)
        return rhs

    U0 = np.ones((1, 1))
    dt = 0.1
    S3.rkw3_step(U0, dt, rhs_explicit, solve_stage)
    # stage 0: zeta_0 = 0, so rhs must be exactly U + dt*gamma_0*N, no alpha
    rhs0, c0 = seen[0]
    assert np.allclose(rhs0, U0 + dt*T.GAMMA[0]*2.0*U0), (
        'stage-0 rhs is not U + dt*gamma_0*N -- an alpha term appeared')
    assert np.isclose(c0, 1.0/(T.BETA[0]*dt)), (
        f'implicit coefficient {c0} is not 1/(beta_0*dt)')


def test_zeta_zero_makes_the_step_self_starting():
    """zeta_0 = 0, so the first stage never reads history.  That is why a step
    needs no startup procedure -- worth pinning, since a non-zero zeta_0 would
    make every run depend on an uninitialised register."""
    assert T.ZETA[0] == 0.0
    a = integrate(-1.0, -0.5, 0.05)
    b = 1.0
    for _ in range(20):                      # same thing, history never seeded
        b = step(b, 0.05, -1.0, -0.5)
    assert abs(a - b) < 1e-15


def test_implicit_coeff_matches_the_stage_used_in_the_solve():
    """c = 1/(beta_k dt), and the worst stage is k=2 with 1/beta = 6."""
    dt = 0.01
    for k in range(T.NSTAGE):
        assert np.isclose(T.implicit_coeff(dt, k), 1.0/(T.BETA[k]*dt))
    assert np.isclose(T.a_mass_worst(dt), 6.0/dt)
    assert np.argmax([T.implicit_coeff(dt, k) for k in range(T.NSTAGE)]) == 2
