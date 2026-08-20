"""Batched per-mode solve and the RKW3 driver.

    uv run --quiet python -m pytest lssem3d/tests -q
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import solver3d as S3, operator as OP, fourier as FR

N, EX, NZ, LZ = 3, 2, 8, 2.0*np.pi
NU, C = 0.05, 4.0


@pytest.fixture(scope='module')
def geom():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    return m, diff_matrix(N), FR.wavenumbers(NZ, LZ)


def _shape(m, nk):
    return (m.nelem, N+1, N+1, OP.NVAR_R, nk)


def test_normal_op_is_symmetric(geom):
    m, D, kz = geom
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal(_shape(m, len(kz))) for _ in range(2))
    f = lambda v: S3.normal_op(v, D, m.facx, m.facy, kz, NU, C)
    s1, s2 = float(np.sum(b*f(a))), float(np.sum(a*f(b)))
    assert abs(s1 - s2)/abs(s1) < 1e-12


def test_operator_has_a_null_space_at_kz0(geom):
    """Constant pressure at k_z = 0 is annihilated: row 4 sees p_x = 0 and row 6
    sees i*k*p = 0.  So L^T L is SINGULAR without a pressure pin, exactly as in
    2D (which is why lssem2d carries pin_p).  Recorded as a test because it
    dictates what the CG tests below may legitimately assert.
    """
    m, D, kz = geom
    U = np.zeros(_shape(m, len(kz)))
    U[..., OP.P_, 0] = 1.0                       # constant p, k_z = 0 only
    r = S3.normal_op(U, D, m.facx, m.facy, kz, NU, C)
    assert np.abs(r[..., 0]).max() < 1e-12, 'expected constant p at kz=0 to be a null vector'


def test_pcg_drives_the_residual_down(geom):
    """b = A x_exact, then solve.  The assertion is on the RESIDUAL, not on
    x - x_exact: the system is singular (see above), so CG returns a solution
    modulo the null space and the iterate legitimately differs from x_exact.
    """
    m, D, kz = geom
    x_exact = np.random.default_rng(1).standard_normal(_shape(m, len(kz)))
    b = S3.normal_op(x_exact, D, m.facx, m.facy, kz, NU, C)
    x, it, res = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, tol=1e-10,
                        max_iter=5000)
    r = b - S3.normal_op(x, D, m.facx, m.facy, kz, NU, C)
    rel = np.sqrt(np.sum(r*r))/np.sqrt(np.sum(b*b))
    assert rel < 1e-8, f'relative residual {rel:.3e} after {it} iters'


def test_pcg_is_exact_on_the_nonsingular_part(geom):
    """Pinning the pressure removes the null space, and then CG must recover
    x_exact itself -- the stronger statement, available once the system is
    actually nonsingular."""
    m, D, kz = geom
    mask = np.ones(_shape(m, len(kz)))
    mask[..., OP.P_, :] = 0.0                    # freeze p entirely: crude but
    mask[..., OP.NVAR + OP.P_, :] = 0.0          # sufficient to kill the null space
    x_exact = np.random.default_rng(4).standard_normal(_shape(m, len(kz)))*mask
    b = S3.normal_op(x_exact, D, m.facx, m.facy, kz, NU, C, mask=mask)
    x, it, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, mask=mask, tol=1e-13,
                      max_iter=8000)
    err = np.abs(x - x_exact).max()/np.abs(x_exact).max()
    assert err < 1e-6, f'rel err {err:.3e} after {it} iters'


def test_jacobi_preconditioner_reduces_iterations(geom):
    m, D, kz = geom
    x_exact = np.random.default_rng(2).standard_normal(_shape(m, len(kz)))
    b = S3.normal_op(x_exact, D, m.facx, m.facy, kz, NU, C)
    _, it_plain, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, tol=1e-10,
                            max_iter=5000)
    diag = S3.jacobi_diagonal(_shape(m, len(kz)), D, m.facx, m.facy, kz, NU, C)
    M_inv = np.where(np.abs(diag) > 1e-300, 1.0/np.where(diag == 0, 1, diag), 0.0)
    _, it_pc, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, M_inv=M_inv,
                         tol=1e-10, max_iter=5000)
    assert it_pc < it_plain, f'Jacobi did not help: {it_pc} vs {it_plain}'


def test_every_mode_converges_under_batching(geom):
    """Each mode must individually reach the tolerance in the batched solve.

    NOT "the batched and standalone residuals are equal" -- they legitimately
    differ, because the batched loop runs until ALL modes are converged, so a
    mode that converges early keeps iterating and ends up better converged than
    it would alone.  What matters operationally is that batching starves no
    mode, which is what this asserts.
    """
    m, D, kz = geom
    x_exact = np.random.default_rng(3).standard_normal(_shape(m, len(kz)))
    b = S3.normal_op(x_exact, D, m.facx, m.facy, kz, NU, C)
    tol = 1e-10
    x, it, res = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, tol=tol, max_iter=5000)
    bn = np.sqrt(np.sum(b*b, axis=(1, 2, 3)).sum(axis=0))
    for k in range(len(kz)):
        assert res[k] <= tol*bn[k]*1.001 + 1e-12, \
            f'mode {k} left unconverged: {res[k]:.3e} vs target {tol*bn[k]:.3e}'


def test_a_mode_is_unaffected_by_other_modes_data(geom):
    """Changing mode j's right-hand side must not change mode k's answer.

    This is the decoupling itself, tested directly rather than inferred.
    """
    m, D, kz = geom
    rng = np.random.default_rng(7)
    b = S3.normal_op(rng.standard_normal(_shape(m, len(kz))),
                     D, m.facx, m.facy, kz, NU, C)
    x1, _, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, tol=1e-12, max_iter=6000)
    b2 = b.copy()
    b2[..., 1] *= 3.0                              # disturb ONLY mode 1
    x2, _, _ = S3.pcg(b2, D, m.facx, m.facy, kz, NU, C, tol=1e-12, max_iter=6000)
    for k in (0, 2, 3):
        d = np.abs(x1[..., k] - x2[..., k]).max()
        scale = max(np.abs(x1[..., k]).max(), 1e-30)
        assert d/scale < 1e-4, f'mode {k} moved when mode 1 changed: {d/scale:.2e}'


# --------------------------------------------------------------- RKW3 driver

def test_rkw3_is_third_order_on_a_linear_ode():
    """du/dt = -u, solved with the driver, must converge at slope 3.

    End-to-end check of the coefficient table: a transcription error that
    preserves consistency (alpha+beta == gamma+zeta) still drops the order, and
    only a convergence study catches it.
    """
    lam = -1.0

    def integrate(dt, T=1.0):
        u = np.ones((1, 1, 1, 1, 1))
        Np = None
        for _ in range(int(round(T/dt))):
            # explicit part carries the whole operator; implicit part is identity
            u, Np = S3.rkw3_step(u, dt,
                                 rhs_explicit=lambda U: lam*U,
                                 solve_stage=lambda rhs, c, k: rhs,
                                 N_prev=Np)
        return float(u.ravel()[0])

    exact = np.exp(lam)
    dts = [0.1, 0.05, 0.025, 0.0125]
    errs = [abs(integrate(d) - exact) for d in dts]
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    assert 2.7 < slope < 3.3, f'RKW3 order {slope:.2f}, errors {errs}'


def test_rkw3_uses_beta_for_the_implicit_coefficient():
    """The c handed to solve_stage must be 1/(beta_k*dt), 4-6x larger than
    fac1/dt.  Guards plan sec 0.4 at the call site, not just in the docs."""
    from lssem3d.timestep import BETA
    seen = []
    S3.rkw3_step(np.zeros((1, 1, 1, 1, 1)), 0.01,
                 rhs_explicit=lambda U: U,
                 solve_stage=lambda rhs, c, k: (seen.append(c), rhs)[1])
    assert seen == [1.0/(b*0.01) for b in BETA]
    assert max(seen) == pytest.approx(6.0/0.01)


def test_normal_op_symmetric_with_quadrature_weights(geom):
    """The operator CG actually solves -- L0^T W L0 -- must stay symmetric."""
    m, D, kz = geom
    rng = np.random.default_rng(11)
    a, b = (rng.standard_normal(_shape(m, len(kz))) for _ in range(2))
    f = lambda v: S3.normal_op(v, D, m.facx, m.facy, kz, NU, C, wq=m.wq)
    s1, s2 = float(np.sum(b*f(a))), float(np.sum(a*f(b)))
    assert abs(s1 - s2)/abs(s1) < 1e-12


def test_pcg_converges_with_weights(geom):
    """CG on the weighted operator still drives its residual down."""
    m, D, kz = geom
    x = np.random.default_rng(12).standard_normal(_shape(m, len(kz)))
    b = S3.normal_op(x, D, m.facx, m.facy, kz, NU, C, wq=m.wq)
    xs, it, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, tol=1e-10,
                       max_iter=8000, wq=m.wq)
    r = b - S3.normal_op(xs, D, m.facx, m.facy, kz, NU, C, wq=m.wq)
    rel = np.sqrt(np.sum(r*r))/np.sqrt(np.sum(b*b))
    assert rel < 1e-7, f'relative residual {rel:.3e} after {it} iters'


# --------------------------------------------- the assembled Jacobi diagonal

def _jac_geom():
    """A mesh with genuinely shared nodes, and BCs, so the assembly matters."""
    from lssem3d import bc as BC
    m = build_channel(1.0, 1.0, 2, 2, 4, bcs=(1, 1, 1, 2))
    nz = 8
    nk = nz//2 + 1
    kz = FR.wavenumbers(nz, LZ)
    mask = BC.build_mask(m, nk, pin_p=True, nz=nz)
    return m, diff_matrix(4), kz, mask, (m.nelem, 5, 5, OP.NVAR_R, nk)


def test_jacobi_diagonal_is_exact_at_EVERY_free_dof():
    """diag(A) against ground truth at every free dof of every field.

    THE PREVIOUS VERSION OF THIS TEST SPOT-CHECKED FIVE NODES ON THE VELOCITY
    FIELD AND PASSED WHILE THE ROUTINE WAS 1.4% WRONG.  Two reasons it missed:

      * velocity-row contamination shrinks like 1/c^2, so at production a_mass it
        is invisible; the error lives on the c-independent pressure and vorticity
        rows, which were never sampled.
      * the probe vector -- local index (i,j) set in EVERY element -- is
        DISCONTINUOUS at an interface, so gs() folds intra-element off-diagonal
        couplings into the reading.  Comparing that against another gs'd probe
        cannot see it: both carry the contamination in proportion.  Lesson L1,
        committed by the same author who wrote L1.

    Ground truth here is the only probe that measures the true assembled
    diagonal: a CONTINUOUS unit vector for one global node (gs of a one-hot
    local array is exactly that), pushed through the real solve operator and
    read back at the node.  Sweeping all fields makes the c-independent rows
    unavoidable.
    """
    m, D, kz, mask, shape = _jac_geom()
    nu, c, kap = 0.01, 50.0, 50.0
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=kap)
    diag = S3.jacobi_diagonal(shape, D, m.facx, m.facy, kz, nu, c, **kw)
    mult = S3.gs(m, np.ones(shape))
    n_checked = n_interface = 0
    worst = 0.0
    for f in range(OP.NVAR_R):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for k in (0, 1):
                    if mask[0, i, j, f, k] == 0.0:
                        continue
                    ed = np.zeros(shape)
                    ed[0, i, j, f, k] = 1.0
                    want = S3.normal_op(S3.gs(m, ed), D, m.facx, m.facy, kz,
                                        nu, c, **kw)[0, i, j, f, k]
                    if abs(want) < 1e-14:
                        continue
                    n_checked += 1
                    if mult[0, i, j, f, k] > 1.5:
                        n_interface += 1
                    worst = max(worst, abs(diag[0, i, j, f, k] - want)/abs(want))
    assert n_checked > 200, f'only {n_checked} dofs checked'
    assert n_interface > 50, f'only {n_interface} INTERFACE dofs -- the bug lived there'
    assert worst < 1e-12, f'worst relative error {worst:.3e} over {n_checked} dofs'


def test_probing_the_assembled_operator_would_contaminate_the_diagonal():
    """Negative control: the fix must be load-bearing.

    Probing WITH the mesh (the old behaviour) folds off-diagonal couplings into
    interface nodes.  If this reproduced the correct answer, the test above
    would be passing for free.
    """
    m, D, kz, mask, shape = _jac_geom()
    nu, c, kap = 0.01, 50.0, 50.0
    good = S3.jacobi_diagonal(shape, D, m.facx, m.facy, kz, nu, c, m, mask,
                              m.wq, kap)
    bad = np.zeros(shape)
    n = shape[1]
    for f in range(OP.NVAR_R):
        for i in range(n):
            for j in range(n):
                e = np.zeros(shape)
                e[:, i, j, f, :] = 1.0
                bad[:, i, j, f, :] = S3.normal_op(
                    e, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq,
                    kap)[:, i, j, f, :]
    bad = S3.gs(m, bad)
    mult = S3.gs(m, np.ones(shape))
    iface = (mult > 1.5) & (np.abs(good) > 1e-12)
    rel = np.abs(bad[iface] - good[iface])/np.abs(good[iface])
    assert rel.max() > 1e-3, (
        f'assembled-probe contamination is only {rel.max():.3e} -- the fix is '
        f'not load-bearing and the exactness test proves nothing')


def test_assembled_diagonal_reduces_cg_iterations():
    """The correction is not cosmetic: it buys ~1.4x fewer iterations."""
    m, D, kz, mask, shape = _jac_geom()
    nu, c, kap = 0.01, 50.0, 50.0
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=kap)
    b = S3.gs(m, np.random.default_rng(0).standard_normal(shape)*mask)*mask
    its = {}
    for lab, asm in (('raw', False), ('assembled', True)):
        d = S3.jacobi_diagonal(shape, D, m.facx, m.facy, kz, nu, c,
                               assemble=asm, **kw)
        Mi = 1.0/np.maximum(d, 1e-30)
        _, it, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, c, m, mask, Mi,
                          1e-8, 20000, None, m.wq, kap)
        its[lab] = it
    assert its['assembled'] < its['raw'], (
        f"assembled {its['assembled']} not better than raw {its['raw']}")
