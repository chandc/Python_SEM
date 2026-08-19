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


def test_jacobi_diagonal_is_the_assembled_diagonal():
    """diag(A) at a shared node is the SUM over the elements that own it.

    Checked against an exact single-DOF probe: `gs` of a one-hot local array is
    precisely the global basis function (1 at every copy of that node, 0
    elsewhere), so `normal_op` of it, read back at the node, IS the assembled
    diagonal entry -- no thresholding and no multiplicity bookkeeping to get
    wrong.
    """
    m, D, kz, mask, shape = _jac_geom()
    nu, c, kap = 0.01, 50.0, 50.0
    diag = S3.jacobi_diagonal(shape, D, m.facx, m.facy, kz, nu, c, m, mask,
                              m.wq, kap)
    f, k = OP.U_, 1
    checked = 0
    for (e, i, j) in [(0, 2, 2), (0, 4, 2), (0, 2, 4), (0, 4, 4), (3, 0, 0)]:
        if mask[e, i, j, f, k] == 0.0:
            continue
        ed = np.zeros(shape)
        ed[e, i, j, f, k] = 1.0
        Eg = S3.gs(m, ed)                     # exact global basis function
        want = S3.normal_op(Eg, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq,
                            kap)[e, i, j, f, k]
        got = diag[e, i, j, f, k]
        assert abs(got - want) <= 1e-9*abs(want), (
            f'node (e={e},i={i},j={j}): got {got:.6f}, assembled {want:.6f}')
        checked += 1
    assert checked >= 4, 'too few live nodes checked'


def test_unassembled_diagonal_is_wrong_at_shared_nodes():
    """Negative control: the old behaviour really was wrong, and by the factor
    claimed.  On this uniform mesh the raw probe returns diag/multiplicity --
    1/2 on an edge, 1/4 at a corner -- so 1/diag over-weighted every
    element-boundary node.  Without this, the test above could pass against an
    assembly that does nothing.
    """
    m, D, kz, mask, shape = _jac_geom()
    nu, c, kap = 0.01, 50.0, 50.0
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=kap)
    asm = S3.jacobi_diagonal(shape, D, m.facx, m.facy, kz, nu, c, **kw)
    raw = S3.jacobi_diagonal(shape, D, m.facx, m.facy, kz, nu, c,
                             assemble=False, **kw)
    mult = S3.gs(m, np.ones(shape))
    live = (mask != 0.0) & (np.abs(asm) > 1e-12)
    ratio = np.where(live, raw/np.where(asm == 0, 1, asm), 1.0)
    interior = live & (mult < 1.5)
    edge = live & (mult > 1.5) & (mult < 2.5)
    corner = live & (mult > 3.5)
    assert interior.any() and edge.any() and corner.any()
    # Interior nodes have one owner, so assembly is a no-op there: exact.
    assert np.abs(ratio[interior] - 1.0).max() < 1e-9, 'interior must be equal'
    # Shared nodes are 1/multiplicity only when the owning elements contribute
    # EQUALLY.  They nearly do on a uniform mesh, but not exactly: a boundary
    # element carries a different mask, so its contribution differs slightly.
    # Hence a tolerance on the ratio, plus the exact statement that assembly
    # strictly increases the diagonal wherever a node is shared.
    assert np.abs(ratio[edge] - 0.5).max() < 1e-2, 'edge should be ~half'
    assert np.abs(ratio[corner] - 0.25).max() < 1e-2, 'corner should be ~quarter'
    shared = live & (mult > 1.5)
    assert (raw[shared] < asm[shared]).all(), (
        'assembly must strictly increase the diagonal at every shared node')


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
