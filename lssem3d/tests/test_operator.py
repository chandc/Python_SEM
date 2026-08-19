"""Per-mode operator: adjointness, the k_z=0 reduction, and the real/complex map.

    uv run --quiet python -m pytest lssem3d/tests -q

The adjoint test is the one that matters.  CG requires L^T L symmetric; if the
transpose is wrong the solver still runs and still produces plausible iterates,
it just converges to the wrong thing -- the failure mode this project has hit
more than once.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP

N, EX, NK = 4, 2, 3          # NK: trailing mode axis, per the standard layout
NU, C = 0.01, 3.0
KZS = [0.0, 1.0, 3.7, -2.5]


@pytest.fixture(scope='module')
def geom():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    return m, diff_matrix(N)


def _rand(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape)


# ------------------------------------------------------ real/complex mapping

def test_real_complex_round_trip():
    Ur = _rand((3, N+1, N+1, OP.NVAR_R, NK), 0)
    assert np.abs(OP.to_real(OP.to_complex(Ur)) - Ur).max() == 0.0


# ---------------------------------------------------------------- adjointness

@pytest.mark.parametrize('kz', KZS)
def test_adjoint_identity(geom, kz):
    """<L a, b> == <a, L^T b> in the plain Euclidean inner product.

    Note this uses the split-REAL form, so the inner product needs no
    conjugation -- which is exactly the reason sec 1.2 of the plan chose it.
    """
    m, D = geom
    a = _rand((m.nelem, N+1, N+1, OP.NVAR_R, NK), 1)
    b = _rand((m.nelem, N+1, N+1, OP.NROW_R, NK), 2)
    La = OP.apply_L(a, D, m.facx, m.facy, kz, NU, C)   # wq=None -> L0
    LTb = OP.apply_LT(b, D, m.facx, m.facy, kz, NU, C)
    lhs = float(np.sum(La*b))
    rhs = float(np.sum(a*LTb))
    rel = abs(lhs - rhs)/max(abs(lhs), 1e-300)
    assert rel < 1e-12, f'kz={kz}: <La,b>={lhs:.12e} <a,LTb>={rhs:.12e} rel={rel:.2e}'


@pytest.mark.parametrize('kz', KZS)
def test_LtL_symmetric(geom, kz):
    """The matrix CG actually sees, L^T L, is symmetric."""
    m, D = geom
    a = _rand((m.nelem, N+1, N+1, OP.NVAR_R, NK), 3)
    b = _rand((m.nelem, N+1, N+1, OP.NVAR_R, NK), 4)
    f = lambda x: OP.apply_LT(OP.apply_L(x, D, m.facx, m.facy, kz, NU, C, m.wq),
                              D, m.facx, m.facy, kz, NU, C)
    s1, s2 = float(np.sum(b*f(a))), float(np.sum(a*f(b)))
    assert abs(s1 - s2)/max(abs(s1), 1e-300) < 1e-12, \
        'L0^T W L0 must stay symmetric with the quadrature weights in place'


@pytest.mark.parametrize('kz', KZS)
def test_weighting_actually_changes_the_operator(kz):
    """Negative control: without wq this would be a different functional.

    Guards against the weights being silently dropped again -- the failure is
    invisible otherwise, since the unweighted operator is also symmetric.
    """
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    a = _rand((m.nelem, N+1, N+1, OP.NVAR_R, NK), 8)
    plain = OP.apply_L(a, D, m.facx, m.facy, kz, NU, C)
    wtd = OP.apply_L(a, D, m.facx, m.facy, kz, NU, C, m.wq)
    assert np.abs(wtd - plain).max() > 1e-6


# ------------------------------------------------------- the k_z = 0 structure

def test_kz0_decouples_into_2d_plus_transverse(geom):
    """At k_z = 0, (u,v,oz,p) and (w,ox,oy) must not talk to each other.

    This is the structural half of Stage 1: the 3D system has to contain the 2D
    one as an exact sub-block before any comparison of numbers is meaningful.
    """
    m, D = geom
    shape = (m.nelem, N+1, N+1, OP.NVAR, NK)
    inplane = [OP.U_, OP.V_, OP.OZ_, OP.P_]
    transverse = [OP.W_, OP.OX_, OP.OY_]
    # rows carrying the 2D system, and rows carrying the transverse one
    rows_2d, rows_tr = [0, 3, 4, 5], [1, 2, 6, 7]

    U = np.zeros(shape, dtype=complex)
    for f in transverse:                       # excite ONLY transverse fields
        U[..., f, :] = _rand(shape[:-2] + (NK,), 10 + f)
    R = OP.apply_L0_complex(U, D, m.facx, m.facy, 0.0, NU, C)
    leak = max(np.abs(R[..., r, :]).max() for r in rows_2d)
    assert leak < 1e-13, f'transverse fields leaked into the 2D rows: {leak:.2e}'

    U = np.zeros(shape, dtype=complex)
    for f in inplane:                          # excite ONLY in-plane fields
        U[..., f, :] = _rand(shape[:-2] + (NK,), 20 + f)
    R = OP.apply_L0_complex(U, D, m.facx, m.facy, 0.0, NU, C)
    leak = max(np.abs(R[..., r, :]).max() for r in rows_tr)
    assert leak < 1e-13, f'in-plane fields leaked into the transverse rows: {leak:.2e}'


def test_kz0_real_input_stays_real(geom):
    """No i*k terms at k_z = 0, so a real state must give a real residual."""
    m, D = geom
    U = _rand((m.nelem, N+1, N+1, OP.NVAR, NK), 5).astype(complex)
    R = OP.apply_L0_complex(U, D, m.facx, m.facy, 0.0, NU, C)
    assert np.abs(R.imag).max() < 1e-300


def test_kz_sign_flips_only_the_imaginary_coupling(geom):
    """+k and -k give conjugate residuals for a real state (Hermitian symmetry)."""
    m, D = geom
    U = _rand((m.nelem, N+1, N+1, OP.NVAR, NK), 6).astype(complex)
    Rp = OP.apply_L0_complex(U, D, m.facx, m.facy, 2.5, NU, C)
    Rm = OP.apply_L0_complex(U, D, m.facx, m.facy, -2.5, NU, C)
    assert np.abs(Rp - np.conj(Rm)).max() < 1e-13


def test_c_enters_only_the_momentum_rows(geom):
    """c multiplies u,v,w in rows 4,5,6 and nothing else.

    Guards against the a_mass coefficient leaking into a constraint row, which
    is what makes the weighting analysis in the 2D study meaningful at all.
    """
    m, D = geom
    U = _rand((m.nelem, N+1, N+1, OP.NVAR, NK), 7).astype(complex)
    args = (D, m.facx, m.facy, 1.3, NU)
    d = OP.apply_L0_complex(U, *args, 5.0) - OP.apply_L0_complex(U, *args, 2.0)
    for r in (0, 1, 2, 3, 7):
        assert np.abs(d[..., r, :]).max() < 1e-300, f'c leaked into row {r}'
    for r, f in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        assert np.abs(d[..., r, :] - 3.0*U[..., f, :]).max() < 1e-12
