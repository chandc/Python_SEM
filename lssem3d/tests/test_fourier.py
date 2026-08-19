"""Stage 0 of 3D_DEVELOPMENT_PLAN.md: the Fourier utilities, before any solver.

    uv run --quiet python -m pytest lssem3d/tests -q
"""
import numpy as np
import pytest
from lssem3d import fourier as F
from lssem3d import timestep as T

NZ, LZ = 32, 2.0*np.pi


def _field(nz=NZ, shape=(3, 4, 4)):
    rng = np.random.default_rng(0)
    return rng.standard_normal(shape + (nz,))


def test_round_trip():
    u = _field()
    assert np.abs(F.to_physical(F.to_modes(u), NZ) - u).max() < 1e-14


def test_ddz_is_spectrally_exact():
    z = np.arange(NZ)*LZ/NZ
    kz = F.wavenumbers(NZ, LZ)
    for m in (1, 3, 7, NZ//2 - 1):          # resolved modes only
        u = np.sin(2*np.pi*m*z/LZ)
        d = F.to_physical(F.ddz(F.to_modes(u), kz), NZ)
        exact = (2*np.pi*m/LZ)*np.cos(2*np.pi*m*z/LZ)
        assert np.abs(d - exact).max() < 1e-12, f'mode {m}'


def test_wavenumbers_match_analytic():
    kz = F.wavenumbers(NZ, LZ)
    assert kz.shape == (NZ//2 + 1,)
    assert np.allclose(kz, 2*np.pi*np.arange(NZ//2 + 1)/LZ)


def test_hermitian_modes_are_real():
    uh = F.to_modes(_field())
    F.assert_hermitian_ok(uh, NZ)           # raises if not
    for k in F.real_mode_indices(NZ):
        assert np.abs(uh[..., k].imag).max() < 1e-12


def test_real_mode_indices_parity():
    assert F.real_mode_indices(8) == (0, 4)
    assert F.real_mode_indices(9) == (0,)


# ------------------------------------------------------------- dealiasing

def test_dealiased_product_has_only_sum_and_difference_modes():
    """A product of two single modes must give EXACTLY k1+k2 and |k1-k2|.

    This is the test the 3/2 rule exists to pass; test_aliased_product_fails
    below is its negative control.
    """
    z = np.arange(NZ)*LZ/NZ
    k1, k2 = 5, 7
    a = np.cos(2*np.pi*k1*z/LZ)
    b = np.cos(2*np.pi*k2*z/LZ)
    ah, bh = F.to_modes(a), F.to_modes(b)
    prod = F.dealias_forward(ah, NZ)*F.dealias_forward(bh, NZ)
    ph = np.abs(F.dealias_backward(prod, NZ))
    expected = {k1 + k2, abs(k1 - k2)}
    for k in range(len(ph)):
        if k in expected:
            assert ph[k] > 0.1, f'expected mode {k} missing'
        else:
            assert ph[k] < 1e-10, f'spurious energy in mode {k}: {ph[k]:.2e}'


def test_aliased_product_fails_without_padding():
    """Negative control: without the 3/2 rule the same test must FAIL.

    Without this, test_dealiased_product... could pass for the wrong reason and
    nobody would know the padding was inert.
    """
    z = np.arange(NZ)*LZ/NZ
    k1, k2 = 11, 13                          # k1+k2 = 24 > NZ/2 = 16 -> aliases
    a, b = np.cos(2*np.pi*k1*z/LZ), np.cos(2*np.pi*k2*z/LZ)
    ph = np.abs(F.to_modes(a*b))             # no padding
    expected = {k1 + k2, abs(k1 - k2)}
    spurious = max(ph[k] for k in range(len(ph)) if k not in expected)
    assert spurious > 1e-6, 'aliasing did not occur; the control is not testing anything'


def test_padded_size_is_even_and_big_enough():
    for nz in (8, 16, 32, 48, 64):
        m = F.padded_size(nz)
        assert m % 2 == 0 and m >= 1.5*nz


# -------------------------------------------------------------- timestep

def test_rkw3_coefficients_consistent():
    """alpha_k + beta_k == gamma_k + zeta_k at every stage."""
    for k in range(T.NSTAGE):
        assert abs(T.ALPHA[k] + T.BETA[k] - (T.GAMMA[k] + T.ZETA[k])) < 1e-14


def test_rkw3_stage_weights_sum_to_one():
    """The explicit weights advance a full step."""
    assert abs(sum(T.GAMMA) + sum(T.ZETA) - 1.0) < 1e-14


def test_implicit_coeff_is_worse_than_bdf2():
    """Guards the correction in 3D_DEVELOPMENT_PLAN.md sec 0.4.

    RKW3/CN does NOT relieve the a_mass problem.  At matched CFL the worst stage
    sits ~15% ABOVE BDF2, and an implementation that assumed 1.5/dt would
    under-budget the stability margin by a factor of four.
    """
    dt_ab2 = 0.01
    dt_rk = dt_ab2*(T.cfl_limit()/0.5)
    assert T.a_mass_worst(dt_rk) > 1.5/dt_ab2
    assert 1.1 < T.a_mass_worst(dt_rk)/(1.5/dt_ab2) < 1.25
    # and for the SAME dt it is 4x worse
    assert abs(T.a_mass_worst(dt_ab2)/(1.5/dt_ab2) - 4.0) < 1e-9
