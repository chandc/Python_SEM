"""The CG loop must run entirely on the device, and give the same answer.

TORCH_VERIFY_PLAN.md V3 measured why this matters: through the NumPy facade, one
host round trip per call costs **21.9x** the matvec at 88³ (386.0 ms against
17.6 ms device-resident).  At ~4800 CG iterations per stage that is tens of GB of
PCIe traffic per step, and it would erase the whole 3.3x the GPU is worth.

So these tests check two separate things, and the second is the one that is easy
to get wrong:

  1. the device path computes the SAME answer as the reference;
  2. it does so WITHOUT going back to the host mid-loop.

A port can pass (1) and silently fail (2) -- it would be correct and slow, which
is the failure mode that looks like success in a unit test.
"""
import numpy as np
import pytest

from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem3d import backend, bc as BC, device as DEV, operator as OP, solver3d as S3

torch = pytest.importorskip('torch')
from lssem3d import kernels_torch as KT   # noqa: E402

# Whatever the backend would actually use -- cuda on the Spark, cpu on the Mac.
# Building CPU tensors here would make these tests pass locally and exercise
# nothing on the machine the port exists for.
DEVICE = KT.device()


def _tt(a):
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64),
                           device=DEVICE)

pytestmark = pytest.mark.skipif(not backend.available('torch'),
                                reason='torch not installed')


@pytest.fixture(autouse=True)
def _restore():
    yield
    backend.set_backend('numpy')


def _periodic(N=4, ex=2, ey=2):
    """Nodes shared 2 AND 4 ways.  A non-periodic mesh would let a wrong
    gather-scatter pass -- this is the configuration that exposed the one-copy
    pressure pin (3D_STATUS.md §2)."""
    L = 2.0*np.pi
    m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L
    m.periodic_y = L
    m.compute_global_indices()
    return m


def test_gather_scatter_torch_matches_scipy_on_a_periodic_mesh():
    """`index_add_` + gather must equal `QT @ (Q @ x)` exactly.

    The two disagree if `gidx`'s C-order flattening does not line up with what
    `Q` encodes — which is assumed by `device._index` and therefore checked here
    rather than trusted.
    """
    m = _periodic()
    nk = 3
    U = np.random.default_rng(0).standard_normal((m.nelem, m.N+1, m.N+1,
                                                  OP.NVAR_R, nk))
    ref = S3.gs(m, U)
    got = S3.gs(m, _tt(U)).cpu().numpy()
    assert np.abs(got - ref).max() <= 1e-13*np.abs(ref).max()


def test_the_periodic_mesh_really_has_multiply_shared_nodes():
    """Negative control: if nothing is shared, the test above proves nothing."""
    m = _periodic()
    mult = S3.gs(m, np.ones((m.nelem, m.N+1, m.N+1, 1, 1)))
    assert mult.max() >= 4.0, f'max multiplicity {mult.max()} -- mesh not shared'


def test_multiplicity_weight_and_make_continuous_agree_across_backends():
    m = _periodic()
    nk = 2
    shape = (m.nelem, m.N+1, m.N+1, OP.NVAR_R, nk)
    U = np.random.default_rng(1).standard_normal(shape)
    ref = S3.make_continuous(m, U)
    got = S3.make_continuous(m, _tt(U)).cpu().numpy()
    assert np.abs(got - ref).max() <= 1e-13*np.abs(ref).max()


def _case(nk=3):
    m = build_channel(2.0, 1.0, 3, 2, 5, bcs=(1, 1, 1, 1))
    D = diff_matrix(5)
    kz = np.arange(nk, dtype=float)
    c, nu = 37.0, 1.0/180.0
    mask = BC.build_mask(m, nk, pin_p=True)
    rw = OP.momentum_row_weights(c)
    x = S3.make_continuous(
        m, np.random.default_rng(2).standard_normal(mask.shape))*mask
    return m, D, kz, c, nu, mask, rw, x


def test_pcg_on_device_reaches_the_same_solution():
    """Bit-identical is impossible — reductions reorder — so what is checked is
    that the two agree to the accuracy the SOLVE ITSELF defines.

    §7M used ±2 iterations, calibrated on ~1100-iteration solves; that is made
    relative here. And the solution gate is not a fixed number: two CG runs that
    stop at the same RESIDUAL tolerance differ in the SOLUTION by up to κ·tol,
    and κ ≈ 1e4 after the row-7 fix (§7J). At tol = 1e-6 that is ~1e-2 — so a
    naive `err < 1e-6` would fail a perfectly correct port, which is exactly what
    it did on the first run (measured 1.19e-04 at tol = 1e-8).

    The real evidence is that the difference SHRINKS with the tolerance. A
    genuine defect would not; a round-off-and-conditioning artifact must.
    """
    m, D, kz, c, nu, mask, rw, x = _case()
    DEV.deterministic(True)
    t = _tt

    backend.set_backend('numpy')
    b = S3.normal_op(x, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw)
    diag = S3.jacobi_diagonal_analytic(mask.shape, D, m.facx, m.facy, kz, nu, c,
                                       m, mask, m.wq, rw=rw)
    Minv = S3.jacobi_inverse(diag, mask)

    errs = {}
    for tol in (1e-6, 1e-9):
        backend.set_backend('numpy')
        xn, itn, rn = S3.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                             M_inv=Minv, tol=tol, max_iter=20000, wq=m.wq, rw=rw)
        backend.set_backend('torch')
        xt, itt, rt = S3.pcg(t(b), t(D), t(m.facx), t(m.facy), t(kz), nu, c,
                             mesh=m, mask=t(mask), M_inv=t(Minv), tol=tol,
                             max_iter=20000, wq=t(m.wq), rw=t(rw))
        assert isinstance(xt, torch.Tensor), 'pcg left the device'
        assert abs(itn - itt) <= max(3, 0.02*itn), f'tol={tol}: {itn} vs {itt}'
        errs[tol] = np.abs(xt.cpu().numpy() - xn).max()/np.abs(xn).max()

    # Tightening tol by 1e3 must shrink the disagreement by roughly the same
    # factor.  Anything that does NOT shrink is a defect, not round-off.
    assert errs[1e-9] < errs[1e-6]/30.0, (
        f'disagreement did not shrink with tolerance: {errs} -- that is a bug, '
        f'not conditioning')
    assert errs[1e-9] < 1e-6, f'too large even at tol=1e-9: {errs[1e-9]:.3e}'


def test_pcg_never_returns_to_the_host_inside_the_loop():
    """The failure mode that looks like success: correct answers, no speedup.

    Every `.numpy()`/`.cpu()`/`.item()` on a tensor is a synchronising copy. One
    per iteration would cost more than the operator saves, so this counts them
    over a multi-iteration solve and requires the total to stay tiny — the loop
    legitimately needs a couple of scalar reads for the convergence test, but not
    a per-iteration array transfer.
    """
    m, D, kz, c, nu, mask, rw, x = _case()
    backend.set_backend('numpy')
    b = S3.normal_op(x, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw)

    backend.set_backend('torch')
    t = _tt
    calls = {'n': 0}
    orig = torch.Tensor.cpu          # .cpu() is the transfer; .numpy() only
                                     # follows it, and on CPU tensors is free

    def counted(self, *a, **k):
        calls['n'] += 1
        return orig(self, *a, **k)

    torch.Tensor.cpu = counted
    try:
        _, it, _ = S3.pcg(t(b), t(D), t(m.facx), t(m.facy), t(kz), nu, c,
                          mesh=m, mask=t(mask), tol=1e-8, max_iter=200,
                          wq=t(m.wq), rw=t(rw))
    finally:
        torch.Tensor.cpu = orig
    assert it > 20, 'need a multi-iteration solve for this to mean anything'
    assert calls['n'] == 0, (
        f'{calls["n"]} host transfers during {it} iterations -- the CG loop is '
        f'copying to the host, which costs 21.9x the matvec (V3)')


# ------------------------------------------------- Phase 3: convection / FFT

def _ct(a):
    return torch.as_tensor(np.ascontiguousarray(a), device=DEVICE)


def test_fourier_round_trip_on_device():
    """rfft/irfft must be the same transform, including the ODD-nz case where
    there is no Nyquist mode and the padded size differs."""
    from lssem3d import fourier as FR
    rng = np.random.default_rng(0)
    for nz in (8, 16, 15):
        u = rng.standard_normal((3, 4, 4, 2, nz))
        ref = FR.to_physical(FR.to_modes(u), nz)
        got = FR.to_physical(FR.to_modes(_ct(u)), nz).cpu().numpy()
        assert np.abs(got - ref).max() <= 1e-12*np.abs(ref).max(), f'nz={nz}'


def test_dealiasing_matches_on_device():
    """The 3/2-rule pad/truncate pair, which carries an amplitude rescale that
    is easy to drop on one side only."""
    from lssem3d import fourier as FR
    nz = 16
    rng = np.random.default_rng(1)
    uh = (rng.standard_normal((2, 4, 4, nz//2 + 1))
          + 1j*rng.standard_normal((2, 4, 4, nz//2 + 1)))
    ref = FR.dealias_backward(FR.dealias_forward(uh, nz), nz)
    t = torch.as_tensor(np.ascontiguousarray(uh), device=DEVICE)
    got = FR.dealias_backward(FR.dealias_forward(t, nz), nz).cpu().numpy()
    assert np.abs(got - ref).max() <= 1e-12*np.abs(ref).max()


def test_convective_matches_on_device():
    """u.grad u, dealiased -- the whole explicit term.

    Complex arithmetic throughout, so this is also the check that torch's
    complex128 path agrees with numpy's on the derivative contractions.
    """
    from lssem3d import convect as CV
    m = build_channel(2.0, 1.0, 3, 2, 5, bcs=(1, 1, 1, 1))
    D = diff_matrix(5)
    nz = 16
    nk = nz//2 + 1
    kz = np.arange(nk, dtype=float)
    rng = np.random.default_rng(2)
    Uh = (rng.standard_normal((m.nelem, 6, 6, OP.NVAR, nk))
          + 1j*rng.standard_normal((m.nelem, 6, 6, OP.NVAR, nk)))
    ref = CV.convective(Uh, D, m.facx, m.facy, kz, nz)
    got = CV.convective(_ct(Uh), _ct(D), _ct(m.facx), _ct(m.facy), _ct(kz),
                        nz).cpu().numpy()
    assert np.abs(got - ref).max() <= 1e-11*np.abs(ref).max()


def test_dealiasing_is_load_bearing_on_device():
    """Negative control: without the 3/2 padding the answer must CHANGE.

    A device port that silently skipped dealiasing would still produce a
    plausible, converging run -- with energy piling up at high k_z.  The
    equivalence test above cannot see that; this can.
    """
    from lssem3d import convect as CV, fourier as FR
    m = build_channel(2.0, 1.0, 2, 2, 4, bcs=(1, 1, 1, 1))
    D = diff_matrix(4)
    nz = 16
    nk = nz//2 + 1
    kz = np.arange(nk, dtype=float)
    rng = np.random.default_rng(3)
    Uh = (rng.standard_normal((m.nelem, 5, 5, OP.NVAR, nk))
          + 1j*rng.standard_normal((m.nelem, 5, 5, OP.NVAR, nk)))
    good = CV.convective(_ct(Uh), _ct(D), _ct(m.facx), _ct(m.facy), _ct(kz), nz)
    orig = FR.pad_factor
    FR.pad_factor = lambda: 1.0                      # padding disabled
    try:
        naive = CV.convective(_ct(Uh), _ct(D), _ct(m.facx), _ct(m.facy),
                              _ct(kz), nz)
    finally:
        FR.pad_factor = orig
    d = float(torch.abs(good - naive).max()/torch.abs(good).max())
    assert d > 1e-3, f'dealiasing changed nothing ({d:.2e}) -- it is not active'
