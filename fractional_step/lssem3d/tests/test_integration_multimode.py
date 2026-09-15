"""End-to-end: several RKW3/CN steps with MANY modes live, serial vs parallel.

    uv run --quiet python -m pytest lssem3d/tests/test_integration_multimode.py -q

GAP THIS CLOSES.  Every other test exercises one piece: the operator, the
transform, convection, a single linear solve.  The time-stepping driver itself
lives in `scratch/cavity3d_kz0.py` -- a script, not the library -- so the
assembled sequence (explicit convection -> defect-corrected stage RHS -> solve ->
update, three times) has never been run under test at all.  And the M2 gate that
did exercise it ran at k_z = 0, where EVERY i*k_z term vanishes and only one
mode exists.  So nothing has ever tested the driver with the mode axis populated.

The stage assembly below deliberately mirrors the driver, including the defect
correction for inhomogeneous BCs -- solving `A U = L^T W f` with a masked A is
well-posed but wrong, and produced a motionless cavity (3D_STATUS.md sec 2.1).

WHAT IS ASSERTED, and why each is not redundant:
  * the field stays finite and bounded -- catches the forward-Euler-on-convection
    blow-up class (NaN at step 36 in the M2 run);
  * Hermitian symmetry survives every step -- a real field must stay real, and
    the split-real representation makes it easy to violate silently;
  * serial and mode-parallel trajectories agree -- the parallel path is exercised
    through the whole driver, not just one pcg call;
  * modes with no initial content stay empty -- L cannot manufacture z-coupling
    that the physics does not have.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (operator as OP, solver3d as S3, bc as BC, convect as CV,
                     fourier as FR, timestep as T, parallel as PAR)

N, EX, NZ, LZ = 4, 2, 8, 2.0*np.pi
RE = 100.0
NU = 1.0/RE
DT = 2e-3
KAP = 1.0/(T.BETA[2]*DT)          # AC coefficient tied to the worst stage


@pytest.fixture(scope='module')
def setup():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    nk = NZ//2 + 1
    D = diff_matrix(N)
    kz = FR.wavenumbers(NZ, LZ)
    mask = BC.build_mask(m, nk, pin_p=True, nz=NZ)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    # one preconditioner per stage: c differs per stage, and the diagonal is
    # dominated by c
    Minv = [S3.jacobi_inverse(S3.jacobi_diagonal(
        shape, D, m.facx, m.facy, kz, NU, T.implicit_coeff(DT, k), m, mask,
        m.wq, KAP), mask) for k in range(T.NSTAGE)]
    return dict(m=m, D=D, kz=kz, nk=nk, mask=mask, Minv=Minv)


def initial_state(s, seed=0):
    """A smooth, z-dependent, BC-respecting start.  Only modes 0..2 seeded, so
    the emptiness of modes 3..4 is a live negative control."""
    m, nk = s['m'], s['nk']
    U = np.zeros((m.nelem, N+1, N+1, OP.NVAR_R, nk))
    rng = np.random.default_rng(seed)
    for k in range(3):
        for f in (OP.U_, OP.V_, OP.W_):
            U[..., f, k] = 0.05*rng.standard_normal((m.nelem, N+1, N+1))
    U *= s['mask']
    BC.apply_values(m, U, nk, lid_speed=1.0, pin_p=True)
    return U


def stage_solve(s, U, Nprev, k, pcg):
    """One RKW3/CN stage, mirroring the driver including defect correction."""
    m, D, kz, mask = s['m'], s['D'], s['kz'], s['mask']
    c = T.implicit_coeff(DT, k)
    Uc = OP.to_complex(U)
    Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, NZ)
    R0 = OP.apply_L0_complex(Uc, D, m.facx, m.facy, kz, NU, 0.0, KAP)
    Lk = -R0[..., 4:7, :]

    fc = np.zeros_like(Uc, shape=Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]))
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        i = row - 4
        fc[..., row, :] = c*(Uc[..., fld, :] + DT*(
            T.GAMMA[k]*Nk[..., i, :] + T.ZETA[k]*Nprev[..., i, :]
            + T.ALPHA[k]*Lk[..., i, :]))
    fc[..., 0, :] = KAP*Uc[..., OP.P_, :]              # artificial compressibility
    f = np.concatenate([fc.real, fc.imag], axis=-2)

    wqR = m.wq[..., None, None]           # wq is already (nelem, n, n)
    r = OP.apply_LT(
        OP.apply_L(U, D, m.facx, m.facy, kz, NU, c, m.wq, KAP) - f*wqR,
        D, m.facx, m.facy, kz, NU, c, KAP)
    b = -S3.gs(m, r)*mask
    # Jacobi preconditioning, as the driver uses.  Without it these solves do
    # not converge in any sane iteration count -- the first version of this test
    # ran 800 unpreconditioned iterations and stopped at residual 1.1e-01, which
    # silently turned the serial-vs-parallel comparison into a comparison of two
    # unconverged states.
    dU, it, res = pcg(b, D, m.facx, m.facy, kz, NU, c, mesh=m, mask=mask,
                      M_inv=s['Minv'][k], tol=1e-10, max_iter=4000,
                      wq=m.wq, kap=KAP)
    assert it < 4000, f'stage {k} hit max_iter, worst residual {res.max():.2e}'
    return U + dU, Nk


def advance(s, U, nstep, pcg):
    Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
    for _ in range(nstep):
        for k in range(T.NSTAGE):
            U, Nprev = stage_solve(s, U, Nprev, k, pcg)
    return U


def _serial(b, *a, **kw):
    return S3.pcg(b, *a, **kw)


def _parallel(b, *a, **kw):
    return PAR.pcg(b, *a, workers=4, **kw)


# ------------------------------------------------------------------ tests

def test_multimode_step_stays_finite_and_bounded(setup):
    """The blow-up class: explicit convection under a scheme with no imaginary-
    axis interval gave NaN at step 36 in the M2 run.  A few steps at Re=100
    should stay well-behaved, and the lid speed bounds the physical velocity."""
    s = setup
    U = advance(s, initial_state(s), 3, _serial)
    assert np.all(np.isfinite(U)), 'non-finite field'
    for f in (OP.U_, OP.V_, OP.W_):
        assert np.abs(U[..., f, :]).max() < 5.0, f'field {f} unbounded'


def test_field_stays_real_through_the_steps(setup):
    """Hermitian symmetry: a real physical field must survive the split-real
    round trip.  Mode 0 and, for even Nz, the Nyquist mode must have ZERO
    imaginary part -- those are the two that are their own conjugates."""
    s = setup
    U = advance(s, initial_state(s), 2, _serial)
    Uc = OP.to_complex(U)
    for f in (OP.U_, OP.V_, OP.W_, OP.P_):
        assert np.abs(Uc[..., f, 0].imag).max() < 1e-10, (
            f'field {f}: k_z = 0 mode acquired an imaginary part')
        assert np.abs(Uc[..., f, -1].imag).max() < 1e-10, (
            f'field {f}: Nyquist mode acquired an imaginary part')


def test_parallel_and_serial_trajectories_agree(setup):
    """The parallel solver driven through the WHOLE integrator, not one solve.

    Errors here compound across 9 stage solves, so a per-mode slicing bug that a
    single-solve test might absorb inside its tolerance shows up as drift.
    """
    s = setup
    U0 = initial_state(s)
    Us = advance(s, U0.copy(), 3, _serial)
    Up = advance(s, U0.copy(), 3, _parallel)
    scale = max(np.abs(Us).max(), 1e-30)
    assert np.abs(Up - Us).max()/scale < 1e-6, (
        f'trajectories diverged by {np.abs(Up-Us).max()/scale:.3e} relative')


def test_unseeded_modes_stay_empty(setup):
    """Negative control on the z-coupling.

    The initial state seeds modes 0-2 only.  The LINEAR operator cannot move
    energy between modes, and convection is quadratic, so modes 3-4 may receive
    only what products of seeded modes generate -- never more than the seeded
    amplitude.  A mode that fills to O(1) means the mode axis is being mixed by
    something that has no right to mix it (a mis-sliced kz, or an FFT applied on
    the wrong axis).
    """
    s = setup
    U = advance(s, initial_state(s), 2, _serial)
    seeded = np.abs(U[..., :3]).max()
    for k in (3, 4):
        top = np.abs(U[..., k]).max()
        assert top < seeded, (
            f'mode {k} reached {top:.3e}, at or above the seeded scale '
            f'{seeded:.3e} -- the mode axis is being mixed')


def test_lid_boundary_values_are_held_every_step(setup):
    """Defect correction must keep the inhomogeneous BC exactly, not merely
    approximately.  Dropping it converged to a motionless cavity while looking
    perfectly healthy (3D_STATUS.md sec 2.1)."""
    s = setup
    U = advance(s, initial_state(s), 2, _serial)
    ref = np.zeros_like(U)
    BC.apply_values(s['m'], ref, s['nk'], lid_speed=1.0, pin_p=True)
    held = ref != 0.0
    assert held.any(), 'no prescribed entries -- test is vacuous'
    assert np.abs(U[held] - ref[held]).max() < 1e-8, 'boundary values drifted'
