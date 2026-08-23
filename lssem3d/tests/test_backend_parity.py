"""Every backend must be the NumPy backend, to round-off.  numba AND torch.

The NumPy path in operator.py is the reference: it is what every physics
validation in 3D_STATUS.md was measured with.  The fused kernels are a
performance rewrite of it, so the only thing that makes them safe is that they
agree -- on every row, every field, every optional argument.

WHAT IS DELIBERATELY EXERCISED HERE, because each was a way to be silently wrong:

  * kap != 0.  lssem2d shipped `_check_ac_backend` because a backend that
    quietly ignores the artificial-compressibility term still produces a
    plausible, converging run -- with the wrong continuity equation.
  * rw != 1, including the 1e-4 row-7 weight.  A backend that dropped the row
    weights would look 5x SLOWER (the conditioning gain vanishes) rather than
    wrong, which is easy to misread as "numba did not help".
  * wq = None, which is the unweighted operator the symmetry tests use.
  * k_z = 0 alongside k_z != 0.  Half the i*k terms vanish at k_z = 0, so a sign
    error in the imaginary coupling is invisible in the M2 cavity case.
  * A non-uniform mesh, so facx != facy and a swapped x/y factor cannot pass.
"""
import numpy as np
import pytest

from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem3d import backend
from lssem3d import operator as OP

BACKENDS = [b for b in ('numba', 'torch') if backend.available(b)]

pytestmark = pytest.mark.skipif(not BACKENDS, reason='no alt backend installed')

N, EX, EY = 5, 3, 2


@pytest.fixture(scope='module')
def case():
    m = build_channel(2.0, 1.0, EX, EY, N, bcs=(1, 1, 1, 1))   # facx != facy
    return m, diff_matrix(N)


def _impl(name):
    backend.set_backend(name)
    return OP._IMPL_L, OP._IMPL_LT


def test_kernel_constants_match_operator():
    """The kernel modules restate NVAR/NROW and the field indices rather than
    importing operator.py -- that import is a CYCLE (operator._bind_backend
    imports them).  Restating means they can DRIFT, so pin them here."""
    for name in BACKENDS:
        K = __import__(f'lssem3d.kernels_{name}', fromlist=['x'])
        assert (K.NV, K.NR) == (OP.NVAR, OP.NROW), f'{name}: NV/NR drifted'
    if 'torch' in BACKENDS:
        from lssem3d import kernels_torch as KT
        assert (KT.U_, KT.V_, KT.W_, KT.OX_, KT.OY_, KT.OZ_, KT.P_) == \
               (OP.U_, OP.V_, OP.W_, OP.OX_, OP.OY_, OP.OZ_, OP.P_)


def test_torch_actually_uses_the_device_and_dtype_it_claims():
    """L15: verify what you are measuring.  MLX silently cast float64 -> float32
    and inverted a conclusion; Legate silently resolved a CPU-only build that
    still reported a GPU.  torch does neither, but asserting costs nothing."""
    if 'torch' not in BACKENDS:
        pytest.skip('torch not installed')
    import torch
    from lssem3d import kernels_torch as KT
    dev = KT.device()
    seen = {}
    orig = KT._apply_L

    def spy(U, *a, **kw):
        seen['device'], seen['dtype'] = U.device.type, U.dtype
        return orig(U, *a, **kw)
    KT._apply_L = spy
    try:
        m = build_channel(1.0, 1.0, 2, 2, 4, bcs=(1, 1, 1, 1))
        KT.apply_L(np.zeros((m.nelem, 5, 5, OP.NVAR_R, 2)), diff_matrix(4),
                   m.facx, m.facy, np.zeros(2), 0.01, 1.0)
    finally:
        KT._apply_L = orig
    assert seen['dtype'] is torch.float64, f'not float64: {seen["dtype"]}'
    assert seen['device'] == dev.type, f'ran on {seen["device"]}, wanted {dev.type}'


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    backend.set_backend('numpy')


def _random_state(m, nk, nrow=OP.NVAR_R, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m.nelem, m.N+1, m.N+1, nrow, nk))


KZS = [np.array([0.0]),
       np.array([0.0, 1.0, -2.5]),
       np.array([3.0, -7.25])]


@pytest.mark.parametrize('name', BACKENDS)
@pytest.mark.parametrize('kz', KZS)
@pytest.mark.parametrize('kap', [0.0, 0.35])
@pytest.mark.parametrize('use_rw', [False, True])
@pytest.mark.parametrize('use_wq', [False, True])
def test_apply_L_matches_numpy(case, name, kz, kap, use_rw, use_wq):
    m, D = case
    U = _random_state(m, len(kz))
    c, nu = 37.0, 1.0/180.0
    wq = m.wq if use_wq else None
    rw = OP.momentum_row_weights(c) if use_rw else None
    ref = OP._apply_L_numpy(U, D, m.facx, m.facy, kz, nu, c, wq, kap, rw)
    got = _impl(name)[0](U, D, m.facx, m.facy, kz, nu, c, wq, kap, rw)
    assert got.shape == ref.shape
    scale = np.abs(ref).max()
    assert np.abs(got - ref).max() <= 1e-12*scale


@pytest.mark.parametrize('name', BACKENDS)
@pytest.mark.parametrize('kz', KZS)
@pytest.mark.parametrize('kap', [0.0, 0.35])
def test_apply_LT_matches_numpy(case, name, kz, kap):
    m, D = case
    R = _random_state(m, len(kz), nrow=OP.NROW_R, seed=1)
    c, nu = 37.0, 1.0/180.0
    ref = OP._apply_LT_numpy(R, D, m.facx, m.facy, kz, nu, c, kap)
    got = _impl(name)[1](R, D, m.facx, m.facy, kz, nu, c, kap)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() <= 1e-12*np.abs(ref).max()


def test_normal_operator_matches_end_to_end(case):
    """The composition is what the solver actually calls -- gather-scatter,
    mask and multiplicity included."""
    from lssem3d import bc as BC, solver3d as S3
    m, D = case
    nk = 3
    kz = np.array([0.0, 1.0, 2.0])
    c, nu = 37.0, 1.0/180.0
    mask = BC.build_mask(m, nk, pin_p=True)
    U = S3.make_continuous(m, _random_state(m, nk, seed=2))*mask
    rw = OP.momentum_row_weights(c)
    args = (D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw)

    backend.set_backend('numpy')
    ref = S3.normal_op(U, *args)
    for name in BACKENDS:
        backend.set_backend(name)
        got = S3.normal_op(U, *args)
        err = np.abs(got - ref).max()/np.abs(ref).max()
        assert err <= 1e-12, f'{name}: {err:.3e}'


def test_switching_backend_actually_rebinds():
    """A no-op set_backend would make every test above vacuous."""
    backend.set_backend('numpy')
    assert OP._IMPL_L is OP._apply_L_numpy
    for name in BACKENDS:
        backend.set_backend(name)
        assert OP._IMPL_L is not OP._apply_L_numpy, name
        assert OP._IMPL_LT is not OP._apply_LT_numpy, name


def test_unavailable_backend_raises_rather_than_falling_back():
    """A silent fallback would turn a missing dependency into a mysterious
    slowdown.  Two distinct failures, and they must stay distinct:

      unknown NAME          -> ValueError
      known name, no LIB    -> ImportError

    ('cuda' used to be the example of an unknown name here; it is a real backend
    now, which is how this test caught the rename.)
    """
    with pytest.raises(ValueError):
        backend.set_backend('opencl')
    for name in ('numba', 'torch', 'cuda'):
        if not backend.available(name):
            with pytest.raises(ImportError):
                backend.set_backend(name)
            break
