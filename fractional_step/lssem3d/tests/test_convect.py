"""Explicit convection: analytic cases, and the 2D reduction at k_z = 0.

    uv run --quiet python -m pytest lssem3d/tests -q

Each test compares against a hand-computable u.grad u rather than against
another code path, so a systematic error in the pipeline cannot cancel itself.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import convect as CV
from lssem3d import fourier as FR
from lssem3d import operator as OP

N, EX, NZ, LZ = 4, 2, 16, 2.0*np.pi


@pytest.fixture(scope='module')
def geom():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    return m, diff_matrix(N), FR.wavenumbers(NZ, LZ)


def _state(m, nmode):
    return np.zeros((m.nelem, N+1, N+1, OP.NVAR, nmode), dtype=complex)


def _zline():
    return np.arange(NZ)*LZ/NZ


def test_uniform_flow_has_no_convection(geom):
    """Constant u: every derivative vanishes, so u.grad u == 0 exactly."""
    m, D, kz = geom
    Uh = _state(m, len(kz))
    Uh[..., OP.U_, 0] = 1.0*NZ          # k=0 coefficient of a constant field
    Uh[..., OP.V_, 0] = -0.5*NZ
    Nh = CV.convective(Uh, D, m.facx, m.facy, kz, NZ)
    assert np.abs(Nh).max() < 1e-10


def test_shear_in_z_only_is_self_advecting_free(geom):
    """u = (f(z), 0, 0): u_x = u_y = 0 and w = 0, so u.grad u == 0.

    Catches a d/dz term wrongly wired into the x-momentum component.
    """
    m, D, kz = geom
    z = _zline()
    f = np.sin(2*np.pi*3*z/LZ)
    Uh = _state(m, len(kz))
    Uh[..., OP.U_, :] = FR.to_modes(f)[None, None, None, :]
    Nh = CV.convective(Uh, D, m.facx, m.facy, kz, NZ)
    assert np.abs(Nh).max() < 1e-10


def test_w_of_z_gives_w_dwdz(geom):
    """u = (0, 0, g(z)) -> N_z = g g', N_x = N_y = 0.

    With g = sin(kz), g g' = k sin(kz)cos(kz) = (k/2) sin(2kz) -- a single mode
    at 2k, which the 3/2 rule must represent exactly.
    """
    m, D, kz = geom
    z = _zline()
    kmode = 3
    k = 2*np.pi*kmode/LZ
    g = np.sin(k*z)
    Uh = _state(m, len(kz))
    Uh[..., OP.W_, :] = FR.to_modes(g)[None, None, None, :]
    Nh = CV.convective(Uh, D, m.facx, m.facy, kz, NZ)
    assert np.abs(Nh[..., 0, :]).max() < 1e-10, 'spurious N_x'
    assert np.abs(Nh[..., 1, :]).max() < 1e-10, 'spurious N_y'
    got = FR.to_physical(Nh[0, 0, 0, 2, :], NZ)
    exact = g*(k*np.cos(k*z))
    assert np.abs(got - exact).max() < 1e-10


def test_kz0_reduces_to_2d_convection(geom):
    """With no z-dependence, N_x, N_y must equal the plain 2D u.grad u.

    Compared against a directly computed 2D expression, not against another
    call into the same pipeline.
    """
    m, D, kz = geom
    rng = np.random.default_rng(0)
    u2 = rng.standard_normal((m.nelem, N+1, N+1))
    v2 = rng.standard_normal((m.nelem, N+1, N+1))
    Uh = _state(m, len(kz))
    Uh[..., OP.U_, 0] = u2*NZ
    Uh[..., OP.V_, 0] = v2*NZ
    Nh = CV.convective(Uh, D, m.facx, m.facy, kz, NZ)

    from lssem2d.operators import dUdx, dUdy
    ex = u2*dUdx(u2, D, m.facx) + v2*dUdy(u2, D, m.facy)
    ey = u2*dUdx(v2, D, m.facx) + v2*dUdy(v2, D, m.facy)
    gx = (Nh[..., 0, 0]/NZ).real
    gy = (Nh[..., 1, 0]/NZ).real
    assert np.abs(gx - ex).max() < 1e-10
    assert np.abs(gy - ey).max() < 1e-10
    assert np.abs(Nh[..., 2, :]).max() < 1e-10, 'w-momentum should vanish'


def test_convection_is_real_for_real_fields(geom):
    """Hermitian symmetry survives the padded round trip."""
    m, D, kz = geom
    z = _zline()
    Uh = _state(m, len(kz))
    for f, s in ((OP.U_, 1.0), (OP.V_, 0.7), (OP.W_, -0.3)):
        Uh[..., f, :] = FR.to_modes(s*np.cos(2*np.pi*2*z/LZ))[None, None, None, :]
    Nh = CV.convective(Uh, D, m.facx, m.facy, kz, NZ)
    for c in range(3):
        FR.assert_hermitian_ok(Nh[..., c, :], NZ, tol=1e-9)


# ------------------------------------------------------------------- CFL

def test_cfl_scales_linearly_with_dt_and_velocity(geom):
    m, D, _ = geom
    U = np.zeros((m.nelem, N+1, N+1, OP.NVAR, NZ))
    U[..., OP.U_, :] = 2.0
    c1 = CV.cfl(U, D, m.facx, m.facy, LZ, NZ, 0.01)
    assert abs(CV.cfl(U, D, m.facx, m.facy, LZ, NZ, 0.02) - 2*c1) < 1e-12
    U[..., OP.U_, :] = 4.0
    assert abs(CV.cfl(U, D, m.facx, m.facy, LZ, NZ, 0.01) - 2*c1) < 1e-12


def test_cfl_reads_the_right_FIELD_not_a_mode(geom):
    """Each of u, v, w must contribute, and nothing else may.

    The original cfl() wrote U_phys[..., OP.V_], which indexes the MODE axis --
    the field axis is -2.  That is silent, and for field 0 it even returns a
    plausible number, so the test above passed while the function was wrong.
    Caught only when a single-mode array made the index go out of bounds.
    """
    m, D, _ = geom
    base = np.zeros((m.nelem, N+1, N+1, OP.NVAR, NZ))
    ref = None
    for f in (OP.U_, OP.V_, OP.W_):
        U = base.copy(); U[..., f, :] = 1.0
        c = CV.cfl(U, D, m.facx, m.facy, LZ, NZ, 0.01)
        assert c > 0, f'field {f} did not contribute to the CFL'
        if f != OP.W_:                      # x,y share the same spacing
            if ref is None:
                ref = c
            else:
                assert abs(c - ref) < 1e-12
    # a NON-velocity field must contribute nothing
    U = base.copy(); U[..., OP.P_, :] = 100.0
    assert CV.cfl(U, D, m.facx, m.facy, LZ, NZ, 0.01) == 0.0


def test_cfl_works_with_a_single_mode(geom):
    """nmode = 1 is the k_z = 0 case the M2 gate runs; the mode-axis bug made
    this raise IndexError rather than return a wrong answer, which is the only
    reason it was noticed."""
    m, D, _ = geom
    U = np.zeros((m.nelem, N+1, N+1, OP.NVAR, 1))
    U[..., OP.U_, :] = 1.0
    assert CV.cfl(U, D, m.facx, m.facy, LZ, 1, 0.01) > 0


def test_max_dt_inverts_cfl(geom):
    m, D, _ = geom
    U = np.zeros((m.nelem, N+1, N+1, OP.NVAR, NZ))
    U[..., OP.U_, :] = 1.5
    U[..., OP.W_, :] = 0.4
    target = 3.0**0.5                       # RKW3 limit
    dt = CV.max_dt_for_cfl(U, D, m.facx, m.facy, LZ, NZ, target)
    assert abs(CV.cfl(U, D, m.facx, m.facy, LZ, NZ, dt) - target) < 1e-10


def test_zero_velocity_gives_unbounded_dt(geom):
    m, D, _ = geom
    U = np.zeros((m.nelem, N+1, N+1, OP.NVAR, NZ))
    assert CV.max_dt_for_cfl(U, D, m.facx, m.facy, LZ, NZ, 1.0) == float('inf')


def test_cfl_tightens_with_polynomial_order():
    """Min GLL spacing shrinks as ~1/N^2, so the CFL-limited dt must fall with N.

    The original cfl() took the order from shape[-2], which is NVAR = 7 in the
    standard layout, making the limit INDEPENDENT of N -- it reported the same
    dt at N = 6, 8 and 10.  That is a silently over-permissive time step.
    """
    prev = None
    for NN in (4, 6, 8, 10):
        m = build_channel(1.0, 1.0, 2, 2, NN, bcs=(1, 1, 1, 2))
        D = diff_matrix(NN)
        U = np.zeros((m.nelem, NN+1, NN+1, OP.NVAR, 1))
        U[..., OP.U_, :] = 1.0
        dt = CV.max_dt_for_cfl(U, D, m.facx, m.facy, LZ, 1, 1.0)
        if prev is not None:
            assert dt < prev, f'N={NN}: dt {dt:.3e} not below N-1 value {prev:.3e}'
        prev = dt


def test_cfl_rejects_a_wrong_shaped_array():
    """The layout assertion fires rather than silently using the wrong axis."""
    m = build_channel(1.0, 1.0, 2, 2, 4, bcs=(1, 1, 1, 2))
    D = diff_matrix(4)
    with pytest.raises(AssertionError):
        CV.cfl(np.zeros((m.nelem, 5, 5, OP.NVAR)), D, m.facx, m.facy, LZ, 1, 0.01)
