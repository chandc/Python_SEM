"""Vertex-patch Schwarz preconditioner and the momentum-weighting options.

    uv run --quiet python -m pytest lssem3d/tests/test_vertex_schwarz.py -q
"""
import numpy as np
import pytest
from lssem2d.mesh import build_channel
from lssem3d import bc as BC, operator as OP, fourier as FR, solver3d as S3
from lssem3d.precond import VertexSchwarz3D, _Level

N, EX, EY, NZ, LZ, NU = 4, 2, 2, 8, 0.34*np.pi, 1.0/180.0


@pytest.fixture(scope='module')
def geom():
    m = build_channel(np.pi, 2.0, EX, EY, N, bcs=(0, 0, 1, 1))
    m.periodic_x = np.pi
    m.compute_global_indices()
    nk = NZ//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=NZ)
    BC.pin_dof(m, mask, OP.P_, 0)
    BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    return m, nk, mask, FR.wavenumbers(NZ, LZ)


def _rand_state(m, nk, mask, seed):
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    return S3.gs(m, np.random.default_rng(seed).standard_normal(shape))*mask


# ------------------------------------------------------------ weighting family

def test_weighting_exponents():
    c = 100.0
    assert np.allclose(OP.momentum_row_weights(c)[4:7], 1.0/c**2)          # legacy default
    assert np.allclose(OP.momentum_row_weights(c, weighting='balanced')[4:7], 1.0/c)
    assert np.allclose(OP.momentum_row_weights(c, weighting='unit')[4:7], 1.0)
    assert np.allclose(OP.momentum_row_weights(c, mom_exp=1.5)[4:7], c**-1.5)
    # only the momentum rows change
    a, b = OP.momentum_row_weights(c), OP.momentum_row_weights(c, weighting='balanced')
    assert np.allclose(a[:4], b[:4]) and a[7] == b[7]
    with pytest.raises(ValueError):
        OP.momentum_row_weights(c, weighting='nonsense')


def test_weighting_env(monkeypatch):
    monkeypatch.setattr(OP, 'DEFAULT_WEIGHTING', 'balanced')
    assert np.allclose(OP.momentum_row_weights(10.0)[4:7], 0.1)
    monkeypatch.setattr(OP, '_ENV_MOM_EXP', '0.5')
    assert np.allclose(OP.momentum_row_weights(4.0)[4:7], 0.5)


# ------------------------------------------------------ vertex-patch Schwarz

@pytest.mark.parametrize('weighting', ['legacy', 'balanced'])
def test_condensed_equals_dense_and_symmetric(geom, weighting):
    m, nk, mask, kz = geom
    c = 5405.4
    rw = OP.momentum_row_weights(c, weighting=weighting)
    Mc = VertexSchwarz3D(m, nk, NZ, NU, c, kz, 0.0, rw, mask=mask, coarse='element', condense=True)
    Md = VertexSchwarz3D(m, nk, NZ, NU, c, kz, 0.0, rw, mask=mask, coarse='element', condense=False)
    r = _rand_state(m, nk, mask, 0)
    zc, zd = Mc(r), Md(r)
    assert np.abs(zc - zd).max() < 1e-10*np.abs(zd).max()
    assert Mc.bytes < 0.5*Md.bytes
    a, b = _rand_state(m, nk, mask, 1), _rand_state(m, nk, mask, 2)
    s1, s2 = np.sum(b*Mc(a)*Mc.mw), np.sum(a*Mc(b)*Mc.mw)
    assert abs(s1 - s2) < 1e-10*abs(s1)
    assert np.abs(zc*(1 - mask)).max() == 0.0          # prescribed dofs stay zero


@pytest.mark.parametrize('weighting,c', [('legacy', 1.0), ('legacy', 5405.4), ('balanced', 5405.4)])
def test_patch_pcg_beats_jacobi(geom, weighting, c):
    m, nk, mask, kz = geom
    rw = OP.momentum_row_weights(c, weighting=weighting)
    lev = _Level(m, nk, NZ, NU, c, kz, 0.0, rw, False, mask=mask)
    M = VertexSchwarz3D(m, nk, NZ, NU, c, kz, 0.0, rw, mask=mask)
    x_ex = _rand_state(m, nk, mask, 3)
    b = lev.A(x_ex)
    sol = lambda Minv: S3.pcg(b, lev.D, m.facx, m.facy, kz, NU, c, mesh=m, mask=mask,
                              M_inv=Minv, tol=1e-10, max_iter=5000, wq=m.wq, rw=rw)
    xj, itj, _ = sol(lev.M_inv)
    xp, itp, _ = sol(M)
    r = b - lev.A(xp)
    assert np.sqrt(np.sum(r*r)) < 1e-8*np.sqrt(np.sum(b*b))
    assert itp <= 40, f'{itp} patch iterations'
    assert itp*5 < itj, f'patch {itp} vs jacobi {itj}'


def test_batched_equals_reference(geom):
    """Factor-sharing + batched torch apply reproduces the reference preconditioner."""
    pytest.importorskip('torch')
    from lssem3d.precond import VertexSchwarzBatched3D
    m, nk, mask, kz = geom
    c = 5405.4
    rw = OP.momentum_row_weights(c)
    ref = VertexSchwarz3D(m, nk, NZ, NU, c, kz, 0.0, rw, mask=mask)
    bat = VertexSchwarzBatched3D(m, nk, NZ, NU, c, kz, 0.0, rw, mask=mask, device='cpu')
    r = _rand_state(m, nk, mask, 5)
    zr, zb = ref(r), bat(r)
    assert np.abs(zb - zr).max() < 1e-12*np.abs(zr).max()
    assert bat.n_etypes[1] <= 3 and bat.n_ptypes[1] <= 8              # sharing found the few distinct types (k != 0)
    # dense device coarse solve (the CUDA default) must match the sparse host one
    bd = VertexSchwarzBatched3D(m, nk, NZ, NU, c, kz, 0.0, rw, mask=mask, device='cpu', coarse_dense=True)
    zd = bd(r)
    assert np.abs(zd - zr).max() < 1e-11*np.abs(zr).max()
