"""Mode-parallel solve: exactness where it is claimed, tolerance where it is not.

    uv run --quiet python -m pytest lssem3d/tests/test_parallel.py -q

The two claims in `parallel.py` are DIFFERENT in strength and are tested
differently on purpose:

  apply_op  -- bitwise identical to serial.  The mode axis carries no cross-mode
               work, so chunking it is exact, and anything less than bitwise
               equality means a slice went astray (kz, mask or M_inv not sliced
               with the data is the obvious failure, and it would still produce
               a plausible-looking field).

  pcg       -- NOT bitwise identical, and asserting that it were would be wrong:
               serial `pcg` runs every mode until the WORST mode converges, so
               easy modes accumulate extra iterations that chunking removes.
               What must hold is the per-mode residual tolerance, checked here
               against the operator directly rather than against serial output.
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d import parallel as PAR

N, EX, NZ, LZ = 4, 2, 16, 2.0*np.pi
NU, C, KAP = 0.01, 50.0, 50.0
WORKERS = [1, 2, 3, 5, 8]


@pytest.fixture(scope='module')
def prob():
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    nk = NZ//2 + 1
    mask = BC.build_mask(m, nk, pin_p=True)
    rng = np.random.default_rng(0)
    U = rng.standard_normal((m.nelem, N+1, N+1, OP.NVAR_R, nk))*mask
    return dict(m=m, D=diff_matrix(N), kz=FR.wavenumbers(NZ, LZ), nk=nk,
                mask=mask, U=U, b=S3.gs(m, U)*mask)


def _args(p):
    return dict(mesh=p['m'], mask=p['mask'], wq=p['m'].wq, kap=KAP)


# ------------------------------------------------------- chunking itself

@pytest.mark.parametrize('nk', [1, 2, 5, 17, 65])
@pytest.mark.parametrize('w', [1, 3, 8, 16])
def test_chunks_tile_the_mode_axis_exactly(nk, w):
    """Every mode in exactly one chunk, in order, none empty.

    A dropped or duplicated mode is the failure mode that would silently
    corrupt one wavenumber and leave the rest of the field looking right.
    """
    cs = PAR.mode_chunks(nk, w)
    covered = np.concatenate([np.arange(nk)[s] for s in cs])
    assert np.array_equal(covered, np.arange(nk))
    assert all(s.stop > s.start for s in cs)
    assert len(cs) <= max(1, min(w, nk))


def test_worker_count_never_exceeds_mode_count():
    """More workers than modes would make empty chunks -- pure overhead."""
    assert PAR.n_workers(3, cap=16) == 3
    assert PAR.n_workers(64, cap=4) == 4
    assert PAR.n_workers(0, cap=8) >= 1


# ------------------------------------------- apply_op: the exactness claim

@pytest.mark.parametrize('w', WORKERS)
def test_apply_op_is_bitwise_identical_to_serial(prob, w):
    p = prob
    ref = S3.normal_op(p['U'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                       NU, C, p['m'], p['mask'], p['m'].wq, KAP)
    got = PAR.apply_op(p['U'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                       NU, C, workers=w, **_args(p))
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), f'workers={w}: not bitwise equal'


def test_apply_op_would_catch_an_unsliced_wavenumber(prob):
    """Negative control: the test above only has teeth if kz actually matters.

    If every mode gave the same answer, slicing kz wrongly would be invisible
    and `test_apply_op_is_bitwise_identical_to_serial` would prove nothing.
    """
    p = prob
    ref = S3.normal_op(p['U'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                       NU, C, p['m'], p['mask'], p['m'].wq, KAP)
    shuffled = p['kz'][::-1].copy()
    bad = S3.normal_op(p['U'], p['D'], p['m'].facx, p['m'].facy, shuffled,
                       NU, C, p['m'], p['mask'], p['m'].wq, KAP)
    assert not np.allclose(bad, ref), 'kz has no effect -- the test is vacuous'


# ------------------------------------------------ pcg: the tolerance claim

@pytest.mark.parametrize('w', WORKERS)
def test_parallel_pcg_solves_every_mode_to_tolerance(prob, w):
    """Check A x = b per mode against the operator, not against serial output."""
    p = prob
    tol = 1e-9
    x, its, res = PAR.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                          NU, C, tol=tol, max_iter=4000, workers=w, **_args(p))
    r = p['b'] - PAR.apply_op(x, p['D'], p['m'].facx, p['m'].facy, p['kz'],
                              NU, C, workers=1, **_args(p))
    mw = S3.multiplicity_weight(p['m'], p['b'].shape)
    rn = np.sqrt(np.sum(r*r*mw, axis=S3.SPATIAL).sum(axis=0))
    bn = np.sqrt(np.sum(p['b']*p['b']*mw, axis=S3.SPATIAL).sum(axis=0))
    live = bn > 1e-300
    assert np.all(rn[live]/bn[live] < 1e-6), (
        f'workers={w}: worst relative residual '
        f'{(rn[live]/bn[live]).max():.3e} after {its} iters')
    assert res.shape == (p['nk'],)


@pytest.mark.parametrize('w', [2, 5])
def test_parallel_pcg_agrees_with_serial_to_solver_tolerance(prob, w):
    """Same linear system, so the SOLUTIONS must agree -- to the tolerance both
    were solved at, not bitwise (different modes stop at different iterations).
    """
    p = prob
    kw = dict(tol=1e-11, max_iter=4000)
    xs, _, _ = S3.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                      NU, C, p['m'], p['mask'], None, kw['tol'],
                      kw['max_iter'], None, p['m'].wq, KAP)
    xp, _, _ = PAR.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                       NU, C, workers=w, **kw, **_args(p))
    scale = max(np.abs(xs).max(), 1e-300)
    assert np.abs(xp - xs).max()/scale < 1e-6


def test_parallel_pcg_handles_a_preconditioner_and_initial_guess(prob):
    """M_inv and x0 both carry a mode axis and both must be sliced WITH the
    data; slicing one and not the other silently mismatches mode to operator."""
    p = prob
    Minv = 1.0/np.maximum(S3.jacobi_diagonal(
        p['b'].shape, p['D'], p['m'].facx, p['m'].facy, p['kz'], NU, C,
        p['m'], p['mask'], p['m'].wq, KAP), 1e-30)
    x0 = 0.1*np.random.default_rng(1).standard_normal(p['b'].shape)*p['mask']
    x, _, _ = PAR.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'], NU, C,
                      M_inv=Minv, x0=x0, tol=1e-9, max_iter=4000, workers=4,
                      **_args(p))
    r = p['b'] - PAR.apply_op(x, p['D'], p['m'].facx, p['m'].facy, p['kz'],
                              NU, C, workers=1, **_args(p))
    assert np.abs(r).max() < 1e-6*max(np.abs(p['b']).max(), 1e-30)


def test_single_worker_matches_serial_bitwise(prob):
    """workers=1 must be a genuine passthrough, so the parallel path can be
    left on unconditionally without perturbing a reference run."""
    p = prob
    a = S3.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'], NU, C,
               p['m'], p['mask'], None, 1e-10, 2000, None, p['m'].wq, KAP)
    b = PAR.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'], NU, C,
                tol=1e-10, max_iter=2000, workers=1, **_args(p))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_chunking_does_not_cost_iterations(prob):
    """Chunked solves should need no MORE iterations than serial: splitting can
    only remove the wait for the worst mode, never add work."""
    p = prob
    kw = dict(tol=1e-10, max_iter=4000)
    _, it_s, _ = S3.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                        NU, C, p['m'], p['mask'], None, kw['tol'],
                        kw['max_iter'], None, p['m'].wq, KAP)
    _, it_p, _ = PAR.pcg(p['b'], p['D'], p['m'].facx, p['m'].facy, p['kz'],
                         NU, C, workers=4, **kw, **_args(p))
    assert it_p <= it_s
