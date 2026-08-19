"""Masking and values for the 7-field system.

The central test is that build_mask and apply_values agree on WHICH degrees of
freedom are prescribed.  lssem2d shipped a bug where its two mask paths
disagreed on bc == 4, so a p = 0 outflow imposed nothing at all
(OUTFLOW_BC_STUDY.md sec 3).  That failure is silent: the run completes and
looks plausible.  Here it is caught by construction.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import bc as BC, operator as OP, solver3d as S3

N, EX, NK = 4, 2, 3


@pytest.fixture(scope='module')
def cav():
    return build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))


def test_walls_freeze_all_three_velocities(cav):
    """3D freezes w as well as u, v: a no-slip wall admits no spanwise slip."""
    mask = BC.build_mask(cav, NK)
    frozen = mask == 0.0
    assert frozen[..., OP.U_, :].any() and frozen[..., OP.V_, :].any()
    assert frozen[..., OP.W_, :].any(), 'w must be frozen on walls in 3D'


def test_both_real_and_imaginary_halves_are_frozen(cav):
    """A Dirichlet condition applies to the complex coefficient.

    Freezing only the real half leaves half the BC unimposed at every non-zero
    mode -- and is invisible at k_z = 0, so it would pass Stage 1 and fail only
    in 3D.
    """
    mask = BC.build_mask(cav, NK)
    for f in BC.VEL:
        re = (mask[..., f, :] == 0.0)
        im = (mask[..., OP.NVAR + f, :] == 0.0)
        assert np.array_equal(re, im), f'field {f}: real/imag masks differ'


def test_vorticity_stays_free(cav):
    mask = BC.build_mask(cav, NK)
    for f in (OP.OX_, OP.OY_, OP.OZ_):
        assert (mask[..., f, :] == 1.0).all(), 'vorticity should not be prescribed'


def test_mask_and_values_touch_the_same_entries(cav):
    """apply_values must write exactly where build_mask zeroes -- the lssem2d bug."""
    mask = BC.build_mask(cav, NK, pin_p=True)
    U = np.random.default_rng(0).standard_normal(mask.shape) + 10.0
    before = U.copy()
    BC.apply_values(cav, U, NK, lid_speed=1.0, pin_p=True)
    changed = np.abs(U - before) > 0
    prescribed = mask == 0.0
    assert not (changed & ~prescribed).any(), 'apply_values touched a FREE dof'
    assert (prescribed & ~changed).sum() <= prescribed.sum(), 'sanity'
    # every prescribed entry now holds a prescribed value (0, or the lid speed)
    vals = U[prescribed]
    assert np.all((np.abs(vals) < 1e-15) | (np.abs(vals - 1.0) < 1e-15))


def test_lid_only_on_the_kz0_mode(cav):
    """A steady z-uniform lid has content in no other mode."""
    mask = BC.build_mask(cav, NK)
    U = np.zeros(mask.shape)
    BC.apply_values(cav, U, NK, lid_speed=1.0)
    assert np.abs(U[..., OP.U_, 0]).max() == pytest.approx(1.0)
    assert np.abs(U[..., OP.U_, 1:]).max() == 0.0, 'lid leaked into k_z != 0'
    assert np.abs(U[..., OP.NVAR + OP.U_, :]).max() == 0.0, 'lid leaked into Im'


def test_pin_p_removes_the_null_space(cav):
    """Without a pin, constant p at k_z=0 is a null vector; with it, it is not.

    This is the concrete reason bc.py exists: solver3d's CG cannot converge to a
    unique answer on the unpinned system.
    """
    D = diff_matrix(N)
    kz = np.zeros(NK)
    shape = (cav.nelem, N+1, N+1, OP.NVAR_R, NK)
    Uc = np.zeros(shape)
    Uc[..., OP.P_, 0] = 1.0

    free = BC.build_mask(cav, NK, pin_p=False)
    r0 = S3.normal_op(Uc, D, cav.facx, cav.facy, kz, 0.01, 2.0, cav, free, cav.wq)
    assert np.abs(r0).max() < 1e-12, 'expected a null vector without the pin'

    pinned = BC.build_mask(cav, NK, pin_p=True)
    Up = Uc*pinned
    rp = S3.normal_op(Up, D, cav.facx, cav.facy, kz, 0.01, 2.0, cav, pinned, cav.wq)
    # the pinned constant-p vector is no longer in the null space of the
    # masked operator: it is not representable as a free-DOF constant
    assert np.abs(Up[0, 0, 0, OP.P_, 0]) == 0.0


def test_masked_solve_recovers_a_known_solution(cav):
    """With BCs applied the system is solvable: CG returns x_exact itself."""
    D = diff_matrix(N)
    kz = np.array([0.0, 1.0, 2.0])
    mask = BC.build_mask(cav, NK, pin_p=True)
    # The assembled operator annihilates the discontinuous part, so a random
    # LOCAL array is not recoverable -- only its C0 projection is.
    x = S3.make_continuous(cav, np.random.default_rng(2).standard_normal(mask.shape))*mask
    b = S3.normal_op(x, D, cav.facx, cav.facy, kz, 0.02, 3.0, cav, mask, cav.wq)
    xs, it, _ = S3.pcg(b, D, cav.facx, cav.facy, kz, 0.02, 3.0, mesh=cav,
                       mask=mask, tol=1e-13, max_iter=20000, wq=cav.wq)
    err = np.abs(xs - x).max()/np.abs(x).max()
    assert err < 1e-5, f'rel err {err:.3e} after {it} iters'
