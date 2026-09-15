"""The numba kernels must reproduce the NumPy reference to round-off.

This is the gate that stops the fused kernels drifting from the reference
implementation.  It matters more than it looks: an earlier draft of these
kernels used the pre-fix least-squares row weighting ((fac1/dt)*u + N(u) instead
of fac1*u + dt*N(u)).  That difference is invisible on well-resolved cases such
as the cavity -- it only diverges on under-resolved ones like the BFS -- so the
parity check is run at several dt, including dt=0 (steady form), rather than at
one convenient value.
"""
import numpy as np
import pytest

from lssem2d import backend
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import (SolverState, _apply_L_numpy, _apply_LT_numpy,
                           apply_L, apply_LT)
from lssem2d.mesh import build_channel

numba_only = pytest.mark.skipif(not backend.available('numba'),
                                reason="numba not installed")


def _make_case(N, EX, EY, dt, seed=0, dtau=None):
    mesh = build_channel(1.0, 1.0, EX, EY, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/389.0, dt=dt, fac1=1.5, dtau=dtau)
    rng = np.random.default_rng(seed)
    shp = (mesh.nelem, N+1, N+1)
    U = rng.standard_normal(shp + (4,))
    fu = np.ascontiguousarray(rng.standard_normal(shp))
    fv = np.ascontiguousarray(rng.standard_normal(shp))
    st.update_linearisation(fu, fv)
    return st, U, fu, fv


def _rel(a, b):
    scale = max(np.max(np.abs(a)), 1e-300)
    return np.max(np.abs(a - b)) / scale


@numba_only
@pytest.mark.parametrize("N,EX,EY", [(7, 6, 6), (8, 6, 6), (5, 4, 4), (9, 3, 5)])
@pytest.mark.parametrize("dt", [0.1, 1.0, 0.0])
def test_numba_matches_numpy(N, EX, EY, dt):
    st, U, fu, fv = _make_case(N, EX, EY, dt)

    su_ref = _apply_L_numpy(st, U, fu, fv).copy()
    c_ref = _apply_LT_numpy(st, su_ref, fu, fv).copy()

    from lssem2d import kernels_numba as K
    su_nb = K.apply_L(st, U, fu, fv).copy()
    c_nb = K.apply_LT(st, su_ref, fu, fv).copy()

    assert _rel(su_ref, su_nb) < 1e-13, f"apply_L mismatch at dt={dt}"
    assert _rel(c_ref, c_nb) < 1e-13, f"apply_LT mismatch at dt={dt}"


@numba_only
@pytest.mark.parametrize("dtau", [1.0, 0.1])
@pytest.mark.parametrize("dt", [0.5, 0.0])
def test_numba_matches_numpy_with_pseudo_time(dt, dtau):
    """The pseudo-time term must reach the fused kernels too.

    It enters as an addition to the mass coefficient in both apply_L and
    apply_LT, so a wrapper that forgot it would leave the kernels solving the
    unaugmented operator while the NumPy path solved the augmented one.
    """
    st, U, fu, fv = _make_case(8, 4, 4, dt, dtau=dtau)

    su_ref = _apply_L_numpy(st, U, fu, fv).copy()
    c_ref = _apply_LT_numpy(st, su_ref, fu, fv).copy()

    from lssem2d import kernels_numba as K
    su_nb = K.apply_L(st, U, fu, fv).copy()
    c_nb = K.apply_LT(st, su_ref, fu, fv).copy()

    assert _rel(su_ref, su_nb) < 1e-13, f"apply_L mismatch, dt={dt} dtau={dtau}"
    assert _rel(c_ref, c_nb) < 1e-13, f"apply_LT mismatch, dt={dt} dtau={dtau}"


@numba_only
def test_strided_linearisation_velocities():
    """newton_step passes fu = U[..., 0], a strided view.  It must still match."""
    st, U, fu, fv = _make_case(7, 4, 4, 0.1)
    fu_s, fv_s = U[..., 0], U[..., 1]           # deliberately strided
    assert not fu_s.flags.c_contiguous
    st.update_linearisation(np.ascontiguousarray(fu_s), np.ascontiguousarray(fv_s))

    su_ref = _apply_L_numpy(st, U, fu_s, fv_s).copy()
    from lssem2d import kernels_numba as K
    su_nb = K.apply_L(st, U, fu_s, fv_s).copy()
    assert _rel(su_ref, su_nb) < 1e-13


@numba_only
def test_set_backend_switches_dispatch():
    """The public apply_L must follow set_backend, and switching must be reversible."""
    st, U, fu, fv = _make_case(6, 4, 4, 0.1)
    original = backend.get_backend()
    try:
        backend.set_backend('numpy')
        a = apply_L(st, U, fu, fv).copy()
        backend.set_backend('numba')
        b = apply_L(st, U, fu, fv).copy()
        backend.set_backend('numpy')
        c = apply_L(st, U, fu, fv).copy()
        assert _rel(a, b) < 1e-13
        assert np.array_equal(a, c)            # back to the reference exactly
    finally:
        backend.set_backend(original)


def test_default_backend_is_numpy_and_unknown_rejected():
    original = backend.get_backend()
    try:
        assert backend.set_backend('numpy') == 'numpy'
        assert backend.available('numpy') is True
        with pytest.raises(ValueError):
            backend.set_backend('cuda')
    finally:
        backend.set_backend(original)


@numba_only
def test_adjointness_preserved_under_numba():
    """<x, Ay> == <y, Ax> in the multiplicity-weighted inner product.

    Test vectors must be PROJECTED first: random local arrays are not continuous
    and so are not in the operator's domain -- testing with them shows a spurious
    ~4% asymmetry in BOTH backends.
    """
    from lssem2d.assembly import gather_scatter
    from lssem2d.solver import apply_A

    st, _, fu, fv = _make_case(7, 4, 4, 0.1)
    mesh = st.mesh
    mult = gather_scatter(mesh, np.ones((mesh.nelem, mesh.N+1, mesh.N+1, 4)))
    mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
    gm = st.get_global_mask(pin_p=False)
    rng = np.random.default_rng(3)

    def project(a):
        return gather_scatter(mesh, a)*mw*gm

    x = project(rng.standard_normal(mult.shape))
    y = project(rng.standard_normal(mult.shape))
    original = backend.get_backend()
    try:
        backend.set_backend('numba')
        xy = np.sum(x*apply_A(st, y, fu, fv, pin_p=False)*mw)
        yx = np.sum(y*apply_A(st, x, fu, fv, pin_p=False)*mw)
    finally:
        backend.set_backend(original)
    assert abs(xy-yx)/max(abs(xy), 1e-300) < 1e-12
