"""HelmholtzPMG cache: a reload must be BIT-IDENTICAL, not merely close.

The cache exists so a checkpoint restart skips the coarse factorisation
(~90 min at 20x20 N=8 on the host).  That is only safe if the reloaded
preconditioner is the same operator to the last bit -- CG's trajectory is
chaotic in the preconditioner, so "close" would silently change iteration
counts across a restart and make restarted runs incomparable with straight
ones.  Hence np.array_equal below, not allclose.

Also checks the key: a cache written for one (mesh, lam, ...) must be
REBUILT, not trusted, when any parameter changes.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel
from lssem3d import solver3d as S3, fourier as FR, hpmg
from lssem3d import project as PJ

N, NZ = 4, 8
NK = NZ//2 + 1


def make():
    m = build_channel(np.pi, 2.0, 2, 4, N, bcs=(0, 0, 1, 1))
    m.periodic_x = np.pi
    m.compute_global_indices()
    return m, FR.wavenumbers(NZ, 0.34*np.pi)


def build(m, kz, cache_path):
    return hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, NK, NZ, wall=False,
                             pin_kz0=True, deg=6, cache_path=cache_path)


def test_reload_bit_identical(tmp_path):
    m, kz = make()
    path = str(tmp_path/'hpmg.npz')
    P1 = build(m, kz, path)              # cold: builds, writes the cache
    assert os.path.exists(path)
    P2 = build(m, kz, path)              # warm: loads it
    mask = P1.lv[0].mask_h
    rng = np.random.default_rng(1)
    r = S3.gs(m, rng.standard_normal(mask.shape))*mask
    z1, z2 = P1(r), P2(r)
    assert np.array_equal(z1, z2)
    assert np.abs(z1).max() > 0.0        # not vacuously equal-zero


def test_cached_products_match_fresh(tmp_path):
    # the stored arrays themselves, not just the composed application
    m, kz = make()
    path = str(tmp_path/'hpmg.npz')
    P1 = build(m, kz, path)
    P2 = build(m, kz, path)
    assert np.array_equal(P1.coarse.Bb_h, P2.coarse.Bb_h)
    assert np.array_equal(P1.coarse.Ab_h, P2.coarse.Ab_h)
    for l1, l2 in zip(P1.lv, P2.lv):
        assert np.array_equal(l1.mask_h, l2.mask_h)
        assert np.array_equal(l1.Minv_h, l2.Minv_h)


def test_key_mismatch_rebuilds(tmp_path):
    m, kz = make()
    path = str(tmp_path/'hpmg.npz')
    build(m, kz, path)
    key0 = np.load(path)['key'].item()
    # different lam table -> different key -> rebuild and overwrite
    P = hpmg.HelmholtzPMG(m, N, (kz*2)**2, 1.0, 1, NK, NZ, wall=False,
                          pin_kz0=True, deg=6, cache_path=path)
    assert np.load(path)['key'].item() != key0
    mask = P.lv[0].mask_h
    rng = np.random.default_rng(1)
    r = S3.gs(m, rng.standard_normal(mask.shape))*mask
    fresh = hpmg.HelmholtzPMG(m, N, (kz*2)**2, 1.0, 1, NK, NZ, wall=False,
                              pin_kz0=True, deg=6)
    assert np.array_equal(P(r), fresh(r))


def test_no_cache_path_unchanged(tmp_path):
    # cache_path=None is the old constructor, bit for bit
    m, kz = make()
    P0 = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, NK, NZ, wall=False,
                           pin_kz0=True, deg=6)
    P1 = build(m, kz, str(tmp_path/'hpmg.npz'))
    mask = P0.lv[0].mask_h
    rng = np.random.default_rng(1)
    r = S3.gs(m, rng.standard_normal(mask.shape))*mask
    assert np.array_equal(P0(r), P1(r))
