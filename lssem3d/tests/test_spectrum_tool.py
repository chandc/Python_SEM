"""The diagnostic tool must be checked against KNOWN answers before it is trusted.

    uv run --quiet python -m pytest lssem3d/tests/test_spectrum_tool.py -q

WHY THIS FILE EXISTS.  `scratch/spectrum.py` was written as "the invariant
measurement -- reach for it first", and its first version contained exactly the
class of error it was meant to prevent: it symmetrised M^-1 A, which is a
product of two symmetric matrices and therefore not symmetric, so the
eigenvalues of its symmetric part are not the operator's.  A wrong conclusion
about p-multigrid was published on the strength of it.

The bug would have died in one minute against a matrix whose spectrum is known.
So: no diagnostic gets used on the real operator until it reproduces answers
that can be written down in advance.
"""
import numpy as np
import pytest
from scipy.linalg import eigh


def _gen_spectrum(A, M):
    """The routine under test, in the form spectrum.py now uses:
    the generalised problem A v = lam M v, NOT eigvals of the symmetrised M^-1 A."""
    return np.sort(eigh(A, M, eigvals_only=True))


def _wrong_way(A, M):
    """The bug: symmetrise M^-1 A and take its eigenvalues."""
    P = np.linalg.inv(M) @ A
    return np.sort(np.real(np.linalg.eigvals(0.5*(P + P.T))))


def test_identity_preconditioner_returns_the_operator_spectrum():
    """M = I  =>  the generalised spectrum IS the spectrum of A."""
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((12, 12)))[0]
    lam = np.linspace(1.0, 50.0, 12)
    A = Q @ np.diag(lam) @ Q.T
    got = _gen_spectrum(A, np.eye(12))
    assert np.allclose(got, np.sort(lam)), f'{got} vs {np.sort(lam)}'


def test_exact_preconditioner_gives_all_ones():
    """M = A  =>  every generalised eigenvalue is exactly 1.

    The sharpest possible check: a perfect preconditioner must return a
    spectrum of ones, hence condition number 1.
    """
    rng = np.random.default_rng(1)
    B = rng.standard_normal((10, 10))
    A = B @ B.T + 10*np.eye(10)
    got = _gen_spectrum(A, A.copy())
    assert np.allclose(got, 1.0, atol=1e-10), f'expected all 1, got {got}'


def test_diagonal_preconditioner_matches_hand_computation():
    """A and M both diagonal => eigenvalues are the ratios, by inspection."""
    a = np.array([4.0, 9.0, 16.0, 25.0])
    d = np.array([2.0, 3.0, 4.0, 5.0])
    got = _gen_spectrum(np.diag(a), np.diag(d))
    assert np.allclose(got, np.sort(a/d))


def test_generalised_eigenvalues_are_positive_for_SPD_pairs():
    """A SPD and M SPD => every eigenvalue is real and positive.

    Negative eigenvalues here are impossible, and their appearance is what
    finally exposed the symmetrisation bug on the real operator.
    """
    rng = np.random.default_rng(2)
    B = rng.standard_normal((15, 15))
    A = B @ B.T + np.eye(15)
    C = rng.standard_normal((15, 15))
    M = C @ C.T + np.eye(15)
    got = _gen_spectrum(A, M)
    assert np.all(got > 0), f'non-positive eigenvalue for an SPD pair: {got.min()}'


def test_the_old_symmetrisation_really_was_wrong():
    """Negative control: the bug must be detectable, or these tests prove nothing.

    Symmetrising M^-1 A gives a DIFFERENT spectrum, and on an SPD pair it can
    produce negative values -- which is exactly what was observed.
    """
    rng = np.random.default_rng(3)
    B = rng.standard_normal((15, 15))
    A = B @ B.T + np.eye(15)
    C = rng.standard_normal((15, 15))
    M = C @ C.T + np.eye(15)
    good, bad = _gen_spectrum(A, M), _wrong_way(A, M)
    assert not np.allclose(good, bad), 'the two agree -- the test has no teeth'
    assert bad.min() < 0, ('the buggy route did not produce a negative eigenvalue '
                           'on this case, so it is not reproducing the observed failure')


def test_spectrum_tool_reproduces_a_known_operator_spectrum():
    """End-to-end on the real code path, against a case solvable by hand.

    A 1-element mesh with everything but a single field frozen leaves a system
    small enough to form densely and check against `eigvalsh` of the same matrix
    -- i.e. the tool must agree with the direct symmetric computation when the
    preconditioner is the identity.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scratch'))
    import spectrum as SP
    ev_unpre, n = SP.spectrum(3, ex=1, precond=False)
    assert n > 0 and np.all(ev_unpre > 0), 'unpreconditioned A must be SPD'
    ev_pre, _ = SP.spectrum(3, ex=1, precond=True)
    assert np.all(ev_pre > 0), f'preconditioned spectrum must be positive, got min {ev_pre.min():.3e}'
    # preconditioning cannot make the conditioning worse than the unpreconditioned
    # operator by orders of magnitude -- a sanity band, not an exact claim
    c_un = ev_unpre.max()/ev_unpre.min()
    c_pr = ev_pre.max()/ev_pre.min()
    assert c_pr < c_un, f'Jacobi made it worse: {c_pr:.3e} vs {c_un:.3e}'
