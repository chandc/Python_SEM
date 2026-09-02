"""A IS the Hessian of the least-squares functional -- the gate everything rests on.

3D_FORMULATION.md sec 4 states the operator as

    A = M Q^T Q L0^T (rho W) L0 M

that is, A is the Hessian of  J = sum_r rho_r |R_r|^2 W  over the eight rows
R_0..R_7.  If that identity does not hold, A is not the normal operator of the
stated functional, and every FOSLS result -- norm equivalence, the ellipticity
constant c2/c1, J as an a-posteriori error estimator -- describes a DIFFERENT
operator from the one being solved.  Nothing downstream is meaningful without it.

This is the 3D counterpart of FOSLS_2D_PLAN sec F0, which gated the 2D operator at
2.2e-16 asymmetry before any FOSLS measurement was trusted.

TWO IMPLEMENTATION FACTS THESE TESTS PIN, both easy to break silently:

  wq belongs to the FORWARD operator only.  apply_LT is the UNWEIGHTED
  transpose, so L^T(W L u) is the normal operator of the INTEGRAL.  Weighting
  both sides would minimise a nodal sum instead.

  rho appears exactly ONCE.  apply_L applies the row weights; apply_LT has no rw
  argument at all.  Using apply_L for both sides of the inner product squares
  rho -- an error that reads 3.2e-03, which is too large for round-off and too
  small to look structural.  That is how it was found.

THE INNER PRODUCT IS MULTIPLICITY-WEIGHTED and that is load-bearing:
gather_scatter appears once in A, so the assembled operator is symmetric in
solver3d._dot(., ., mw) and NOT in the naive one.
"""
import numpy as np
import pytest

from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel

from lssem3d import backend, bc as BC, operator as OP, solver3d as S3

NU, C = 1.0/180.0, 525.0
TOL = 1e-12


def _case(rowweight, N=6, ex=2, ey=2, nk=3):
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    D = diff_matrix(N)
    kz = np.arange(nk, dtype=float)
    mask = BC.build_mask(m, nk, pin_p=True)
    rw = OP.momentum_row_weights(C) if rowweight else None
    mw = S3.multiplicity_weight(m, mask.shape)
    return m, D, kz, mask, rw, mw


def _rand(m, mask, seed):
    rng = np.random.default_rng(seed)
    return S3.make_continuous(m, rng.standard_normal(mask.shape))*mask


def _pieces(rowweight):
    m, D, kz, mask, rw, mw = _case(rowweight)
    A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, NU, C, m, mask,
                               m.wq, 0.0, rw)
    # forward operator WITH wq and rw ...
    L = lambda x: OP.apply_L(x*mask, D, m.facx, m.facy, kz, NU, C, m.wq, 0.0, rw)
    # ... and the bare L0, carrying NEITHER, because apply_LT carries neither.
    L0 = lambda x: OP.apply_L(x*mask, D, m.facx, m.facy, kz, NU, C, None, 0.0, None)
    return m, mask, mw, A, L, L0


@pytest.fixture(autouse=True)
def _numpy_backend():
    backend.set_backend('numpy')
    yield
    backend.set_backend('numpy')


@pytest.mark.parametrize('rowweight', [True, False])
def test_A_is_the_hessian_of_J(rowweight):
    """<v, A u> == <L0 v, rho W L0 u>.  The FOSLS identity itself."""
    m, mask, mw, A, L, L0 = _pieces(rowweight)
    u, v = _rand(m, mask, 0), _rand(m, mask, 1)
    lhs = float(np.sum(S3._dot(v, A(u), mw)))
    rhs = float(np.sum(S3._dot(L0(v), L(u))))
    rel = abs(lhs - rhs)/max(abs(lhs), 1e-300)
    assert rel < TOL, (
        f'A is not the Hessian of J (rel {rel:.3e}, row weights={rowweight}): '
        f'<v,Au>={lhs:.12e} vs <L0 v, rho W L0 u>={rhs:.12e}')


@pytest.mark.parametrize('rowweight', [True, False])
def test_A_is_symmetric(rowweight):
    """Follows from the Hessian property, but tested separately: a masking or
    gather-scatter error can break symmetry while leaving the identity intact."""
    m, mask, mw, A, _, _ = _pieces(rowweight)
    u, v = _rand(m, mask, 0), _rand(m, mask, 1)
    a = float(np.sum(S3._dot(v, A(u), mw)))
    b = float(np.sum(S3._dot(u, A(v), mw)))
    rel = abs(a - b)/max(abs(a), 1e-300)
    assert rel < TOL, f'A not symmetric (rel {rel:.3e}, row weights={rowweight})'


@pytest.mark.parametrize('rowweight', [True, False])
def test_A_is_positive_definite(rowweight):
    """<u, A u> > 0 on the free space -- what makes CG admissible."""
    m, mask, mw, A, _, _ = _pieces(rowweight)
    for seed in range(4):
        u = _rand(m, mask, seed)
        q = float(np.sum(S3._dot(u, A(u), mw)))
        assert q > 0.0, f'<u,Au> = {q:.6e} <= 0 (seed {seed}, rw={rowweight})'


def test_wq_belongs_to_the_forward_operator_only():
    """Weighting BOTH sides minimises a nodal sum, not the integral.  Guard the
    contract by showing the doubly-weighted form is measurably different."""
    m, mask, mw, A, L, _ = _pieces(True)
    u, v = _rand(m, mask, 0), _rand(m, mask, 1)
    good = float(np.sum(S3._dot(v, A(u), mw)))
    doubled = float(np.sum(S3._dot(L(v), L(u))))       # wq and rw on both sides
    assert abs(good - doubled)/abs(good) > 1e-6, (
        'the doubly-weighted product is indistinguishable from the correct one, '
        'so this test can no longer detect the error it exists to catch')
