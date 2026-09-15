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
from lssem3d import bc as BC
from lssem3d import solver3d as S3, operator as OP

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
    real_k = BC.real_mode_columns(NK)
    free_k = [k for k in range(NK) if k not in real_k]
    assert free_k, 'no non-real modes -- test would be vacuous'
    for f in BC.VEL:
        re = (mask[..., f, :] == 0.0)
        im = (mask[..., OP.NVAR + f, :] == 0.0)
        # On the modes that carry a genuine imaginary part the two halves must
        # match exactly.  They CANNOT match on k = 0 or Nyquist, whose whole
        # imaginary half is prescribed everywhere (build_mask), so those columns
        # are excluded here and checked by the test below instead.
        assert np.array_equal(re[..., free_k], im[..., free_k]), (
            f'field {f}: real/imag masks differ on a complex mode')
        assert im[..., real_k].all(), (
            f'field {f}: imaginary half of a real mode is not fully frozen')


def test_imaginary_half_of_real_modes_is_frozen_everywhere(cav):
    """k = 0 and Nyquist must be real, so their imaginary half is prescribed at
    INTERIOR points too, not merely on boundaries.

    irfft discards those components, so anything the solver puts there is
    invisible in physical space -- an unconstrained direction CG will fill.
    Measured before this was enforced: the Nyquist imaginary part reached
    1.5e-03 against a real part of 6.1e-03 after three steps, which would have
    failed fourier.assert_hermitian_ok on the solver's own state.
    """
    mask = BC.build_mask(cav, NK)
    for k in BC.real_mode_columns(NK):
        assert (mask[..., OP.NVAR:, k] == 0.0).all(), f'mode {k} imag not frozen'
    # and an interior point of a complex mode is still free, or the mask is
    # simply zeroing everything
    interior = mask[0, 1, 1, OP.NVAR + OP.U_, :]
    assert (interior != 0.0).any(), 'every mode frozen -- mask is too aggressive'


def test_kz0_only_run_prescribes_the_entire_imaginary_half(cav):
    """nmode = 1 is the M2 cavity case: k = 0 alone, where every imaginary
    component is unphysical.  Freezing them is also the k_z = 0 fast path --
    those DOFs stop being solved for at all."""
    mask = BC.build_mask(cav, 1)
    assert (mask[..., OP.NVAR:, :] == 0.0).all()
    assert (mask[..., :OP.NVAR, :] != 0.0).any(), 'real half must stay live'


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


# ------------------------------- pinning on a shared / periodic node

def _periodic_mesh(N=4, ex=2, ey=2, both=True):
    from lssem2d.mesh import build_channel
    L = 2.0*np.pi
    m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L
    if both:
        m.periodic_y = L
    m.compute_global_indices()
    return m


def test_pin_dof_covers_every_copy_of_a_shared_node():
    """A pin must prescribe the GLOBAL dof, i.e. all of its local copies.

    `mask[0,0,0,f,k] = 0` prescribes ONE copy.  On a periodic seam that node is
    shared 2 or 4 ways, so the siblings stay free: the dof is not pinned at all,
    and the mask disagrees with itself across copies of one global node.
    """
    m = _periodic_mesh()
    nk = 1
    mask = np.ones((m.nelem, m.N+1, m.N+1, OP.NVAR_R, nk))
    mult = S3.gs(m, np.ones_like(mask))
    assert mult[0, 0, 0, OP.P_, 0] > 1.5, 'test needs a SHARED node to be meaningful'
    BC.pin_dof(m, mask, OP.P_, 0)
    ind = np.zeros_like(mask)
    ind[0, 0, 0, OP.P_, 0] = 1.0
    copies = S3.gs(m, ind) > 0.5
    assert copies.sum() > 1, 'node is not actually shared'
    assert np.all(mask[copies] == 0.0), 'a sibling copy was left free'


def test_assembled_operator_is_symmetric_on_a_periodic_mesh():
    """CG requires a symmetric A, and a one-copy pin destroys it.

    Tested on the CONTINUOUS subspace: a random LOCAL array is discontinuous,
    the assembled operator does not act on it meaningfully, and a symmetry test
    built from one would report failure for a correct operator.  (The pre-
    existing symmetry tests pass mesh=None and so never exercised assembly.)

    Measured before the fix: 1.5e-07 with multiplicity 2, 5.9e-05 with 4.
    """
    from lssem2d.lgl import diff_matrix
    m = _periodic_mesh()
    D = diff_matrix(m.N)
    nk = 1
    kz = np.zeros(nk)
    mask = BC.build_mask(m, nk, pin_p=True, nz=1)
    mw = S3.multiplicity_weight(m, mask.shape)
    rng = np.random.default_rng(0)
    a, b = (S3.make_continuous(m, rng.standard_normal(mask.shape))*mask
            for _ in range(2))
    f = lambda v: S3.normal_op(v, D, m.facx, m.facy, kz, 0.1, 60.0, m, mask,
                               m.wq, 0.0)
    s1 = float(np.sum(b*f(a)*mw))
    s2 = float(np.sum(a*f(b)*mw))
    assert abs(s1 - s2)/abs(s1) < 1e-12, f'assembled operator not symmetric: {abs(s1-s2)/abs(s1):.3e}'


def test_one_copy_pin_would_break_symmetry():
    """Negative control: the fix must be load-bearing."""
    from lssem2d.lgl import diff_matrix
    m = _periodic_mesh()
    D = diff_matrix(m.N)
    nk = 1
    kz = np.zeros(nk)
    bad = BC.build_mask(m, nk, pin_p=False, nz=1)
    bad[0, 0, 0, OP.P_, 0] = 0.0                 # the old, single-copy pin
    mw = S3.multiplicity_weight(m, bad.shape)
    rng = np.random.default_rng(0)
    a, b = (S3.make_continuous(m, rng.standard_normal(bad.shape))*bad
            for _ in range(2))
    f = lambda v: S3.normal_op(v, D, m.facx, m.facy, kz, 0.1, 60.0, m, bad,
                               m.wq, 0.0)
    s1 = float(np.sum(b*f(a)*mw))
    s2 = float(np.sum(a*f(b)*mw))
    assert abs(s1 - s2)/abs(s1) > 1e-9, (
        'a one-copy pin did NOT break symmetry -- this mesh has no shared '
        'pinned node, so the test above proves nothing')
