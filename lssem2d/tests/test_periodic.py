"""Streamwise periodicity: the two ends of the domain must be ONE set of nodes.

Needed for the Chan (1996) channel validations, which impose periodicity in the
streamwise direction (Stokes decay; Orr-Sommerfeld growth at Re=7500).

Everything downstream -- gather_scatter, the Dirichlet masks, the multiplicity
weights used for the CG inner product -- is driven by mesh.gidx, so periodicity
is implemented by wrapping the coordinate when gidx is built.  Only connectivity
wraps; xnod/facx/wq keep their true values so geometry and quadrature are
untouched.

The failure this guards against is silent: an unmerged seam leaves the domain
open, and a channel that is secretly not periodic still runs and still produces
a plausible-looking decay rate.
"""
import numpy as np
import pytest

from lssem2d.assembly import gather_scatter
from lssem2d.mesh import build_channel

LX, LY = 2.0*np.pi, 1.0
N, EX, EY = 6, 2, 4


def _mesh(periodic=True):
    m = build_channel(LX, LY, EX, EY, N, bcs=(0, 0, 1, 1))
    if periodic:
        m.periodic_x = LX
        m.compute_global_indices()
    return m


def test_seam_merges_into_one_set_of_nodes():
    """Every global id at x = Lx must also appear at x = 0."""
    m = _mesh()
    n = N + 1
    lo, hi = set(), set()
    for e in range(m.nelem):
        for i in range(n):
            for j in range(n):
                if abs(m.xnod[e, i]) < 1e-8:
                    lo.add(m.gidx[e, i, j])
                elif abs(m.xnod[e, i] - LX) < 1e-8:
                    hi.add(m.gidx[e, i, j])
    assert lo and hi
    assert hi <= lo, "seam did not merge: the domain is open"


def test_periodic_removes_exactly_one_column_of_dofs():
    """Wrapping must remove the duplicated seam column and nothing else."""
    open_m = _mesh(periodic=False)
    per_m = _mesh(periodic=True)
    n = N + 1
    n_open = open_m.gidx.max() + 1
    n_per = per_m.gidx.max() + 1
    # one column of n_y unique nodes is shared rather than duplicated
    ny_unique = EY * (n - 1) + 1
    assert n_open - n_per == ny_unique


def test_gather_scatter_sees_multiplicity_two_on_the_seam():
    """Direct stiffness must sum across the seam like any interior interface."""
    m = _mesh()
    ones = np.ones((m.nelem, N + 1, N + 1))
    mult = gather_scatter(m, ones)
    n = N + 1
    seen = []
    for e in range(m.nelem):
        for i in range(n):
            if abs(m.xnod[e, i]) < 1e-8 or abs(m.xnod[e, i] - LX) < 1e-8:
                seen.append(mult[e, i, n // 2])
    assert seen, "no seam nodes found"
    assert np.allclose(seen, 2.0), f"seam multiplicity {sorted(set(seen))}, expected 2"


def test_geometry_is_not_wrapped():
    """Only connectivity wraps -- coordinates, jacobians and weights must not."""
    open_m = _mesh(periodic=False)
    per_m = _mesh(periodic=True)
    assert np.array_equal(open_m.xnod, per_m.xnod)
    assert np.array_equal(open_m.facx, per_m.facx)
    assert np.allclose(open_m.wq, per_m.wq)
    assert per_m.xnod.max() == pytest.approx(LX)


def test_a_periodic_function_is_continuous_across_the_seam():
    """cos(x) evaluated at both faces must land on the same global nodes.

    This is the check that would catch a wrap that merged the wrong nodes: the
    ids can match while the geometry is misaligned in y.
    """
    m = _mesh()
    n = N + 1
    vals = {}
    for e in range(m.nelem):
        for i in range(n):
            for j in range(n):
                g = m.gidx[e, i, j]
                v = np.cos(m.xnod[e, i]) * np.sin(np.pi * m.ynod[e, j])
                if g in vals:
                    assert abs(vals[g] - v) < 1e-12, \
                        f"global node {g} carries two different values"
                else:
                    vals[g] = v


def test_seam_guard_fires_when_the_domain_does_not_span_L():
    """A wrong periodic_x must raise, not silently produce an open domain."""
    m = build_channel(LX, LY, EX, EY, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX * 0.5          # domain does not span this
    with pytest.raises(ValueError, match="periodic_x"):
        m.compute_global_indices()


def test_non_periodic_is_untouched():
    """Default behaviour must be bit-identical to before."""
    a = build_channel(LX, LY, EX, EY, N, bcs=(0, 0, 1, 1))
    b = build_channel(LX, LY, EX, EY, N, bcs=(0, 0, 1, 1))
    b.compute_global_indices()
    assert np.array_equal(a.gidx, b.gidx)
