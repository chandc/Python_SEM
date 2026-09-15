"""Dong open-boundary-condition (bc == 6) tests -- Stage 1 of the ladder in
OUTFLOW_DONG_OBC_PLAN.md, plus a fast Stage-0 run.

Pins: (a) the assembled operator with the B^T B boundary term stays symmetric
positive semi-definite in the PCG inner product; (b) a NEGATIVE CONTROL that
the rows actually change the operator (the lssem3d lesson: a symmetry test
alone passes vacuously if the new term is silently dropped); (c) the Jacobi
diagonal matches the true diagonal of A; (d) the D0 = 0, switch-off form is
traction-free  -p + nu*du/dx = 0  and reproduces plane Poiseuille exactly.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import lssem2d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
from lssem2d import obc
import lssem2d.solver as S
from lssem2d.assembly import gather_scatter


def make_state(bc_E=6, E_x=2, E_y=2, N=6, dt=0.5, D0=0.0, delta=None, w_obc=1.0,
               seed=0):
    lssem2d.set_backend('numpy')
    mesh = build_channel(L_x=2.0, L_y=1.0, E_x=E_x, E_y=E_y, N=N,
                         bcs=(3, bc_E, 1, 1))
    D = diff_matrix(N)
    st = SolverState(mesh, D, nu=0.01, dt=dt, fac1=1.5, w_mom=1.0, w_mass=1.0)
    st.obc_w = w_obc
    st.obc_D0 = D0
    st.obc_delta = delta
    rng = np.random.default_rng(seed)
    shape = (mesh.nelem, N + 1, N + 1)
    st.update_linearisation(rng.standard_normal(shape), rng.standard_normal(shape))
    if not hasattr(st, 'multiplicity_weight'):
        mult = gather_scatter(mesh, np.ones(shape + (4,)))
        st.multiplicity_weight = 1.0 / np.where(mult < 1e-10, 1.0, mult)
    return st, rng


def test_non_east_edge_raises():
    lssem2d.set_backend('numpy')
    mesh = build_channel(L_x=2.0, L_y=1.0, E_x=2, E_y=1, N=4, bcs=(6, 4, 1, 1))
    st = SolverState(mesh, diff_matrix(4), nu=0.01, dt=0.5)
    with pytest.raises(NotImplementedError):
        obc.setup_obc(st)


def test_inactive_mesh_is_noop():
    st, rng = make_state(bc_E=4)
    assert not obc.obc_active(st)


@pytest.mark.parametrize("D0", [0.0, 2.0])
def test_apply_A_symmetric_and_psd(D0):
    """<A x, y>_w == <x, A y>_w and <A x, x>_w >= 0 with the boundary term."""
    st, rng = make_state(D0=D0)
    fu = st.dfu_dx * 0 + 1.0   # placeholder; apply_A uses state's cached grads
    shape = (st.mesh.nelem, st.mesh.N + 1, st.mesh.N + 1)
    f_u = rng.standard_normal(shape)
    f_v = rng.standard_normal(shape)
    st.update_linearisation(f_u, f_v)
    # A is symmetric on CONTINUOUS fields (the image of gather-scatter, where
    # every PCG vector lives) -- raw random vectors are discontinuous at the
    # element interfaces and see only the one-sided halves of the stencil.
    w = st.multiplicity_weight
    cont = lambda z: gather_scatter(st.mesh, z) * w
    x = cont(rng.standard_normal(shape + (4,)))
    y = cont(rng.standard_normal(shape + (4,)))
    Ax = S.apply_A(st, x, f_u, f_v)
    Ay = S.apply_A(st, y, f_u, f_v)
    d1 = np.sum(Ax * y * w)
    d2 = np.sum(x * Ay * w)
    assert abs(d1 - d2) / max(abs(d1), 1e-15) < 1e-11
    for _ in range(20):
        z = cont(rng.standard_normal(shape + (4,)))
        Az = S.apply_A(st, z, f_u, f_v)
        assert np.sum(Az * z * w) >= -1e-10


def test_negative_control_rows_change_operator():
    """bc = 6 vs bc = 0 (both mask nothing) must give DIFFERENT operators --
    the whole difference is the B^T B term, so if this passes vacuously the
    boundary rows are not being applied at all."""
    st6, rng = make_state(bc_E=6, seed=3)
    st0, _ = make_state(bc_E=0, seed=3)
    shape = (st6.mesh.nelem, st6.mesh.N + 1, st6.mesh.N + 1)
    f_u = rng.standard_normal(shape)
    f_v = rng.standard_normal(shape)
    st6.update_linearisation(f_u, f_v)
    st0.update_linearisation(f_u, f_v)
    x = rng.standard_normal(shape + (4,))
    d = S.apply_A(st6, x, f_u, f_v) - S.apply_A(st0, x, f_u, f_v)
    assert np.abs(d).max() > 1e-8


@pytest.mark.parametrize("D0", [0.0, 1.5])
def test_jacobi_matches_true_diagonal(D0):
    """compute_jacobi vs unit-vector probing of apply_A.  E_y = 1 so outlet
    edge nodes are not shared between elements; probed at element-interior
    nodes plus non-corner outlet-edge nodes, where a local unit vector IS a
    global unit vector."""
    st, rng = make_state(bc_E=6, E_x=2, E_y=1, N=5, D0=D0)
    n = st.mesh.N + 1
    shape = (st.mesh.nelem, n, n)
    f_u = rng.standard_normal(shape)
    f_v = rng.standard_normal(shape)
    st.update_linearisation(f_u, f_v)
    M_inv = S.compute_jacobi(st, f_u, f_v)
    mask = st.get_global_mask()
    probes = [(e, i, j, k) for e in range(st.mesh.nelem)
              for i in range(1, n - 1) for j in range(1, n - 1) for k in range(4)]
    e_out = [e for e in range(st.mesh.nelem) if st.mesh.bc[e, 1] == 6]
    probes += [(e, n - 1, j, k) for e in e_out
               for j in range(1, n - 1) for k in range(4)]
    for (e, i, j, k) in probes:
        if mask[e, i, j, k] < 0.5:
            continue
        U = np.zeros(shape + (4,))
        U[e, i, j, k] = 1.0
        a_ii = S.apply_A(st, U, f_u, f_v)[e, i, j, k]
        assert M_inv[e, i, j, k] > 0
        rel = abs(a_ii - 1.0 / M_inv[e, i, j, k]) * M_inv[e, i, j, k]
        assert rel < 1e-10, (e, i, j, k, rel)


def test_stage0_poiseuille_traction_free():
    """Stage 0: D0 = 0, switch off => -p + nu*du/dx = 0 on the outlet.  Exact
    Poiseuille is representable, du/dx = 0 there, so p = 0 weakly and
    dp = 12L/Re must come back as a PREDICTION (no pin anywhere)."""
    lssem2d.set_backend('numpy')
    L, RE = 2.0, 100.0
    mesh = build_channel(L_x=L, L_y=1.0, E_x=2, E_y=1, N=7, bcs=(3, 6, 1, 1))
    N = mesh.N
    D = diff_matrix(N)
    w = lgl_weights(N)
    st = SolverState(mesh, D, nu=1.0 / RE, dt=0.5, fac1=1.0, w_mom=1.0, w_mass=1.0)
    inl = lambda x, y, t: 6.0 * np.asarray(y) * (1.0 - np.asarray(y))
    n = N + 1
    U = np.zeros((mesh.nelem, n, n, 4))
    h = [U.copy()]
    conv = False
    for s in range(400):
        prev = h[0].copy()
        U = S.step_bdf(st, h, time=(s + 1) * 0.5, max_newton=5,
                       newton_tol=1e-13, newton_factor=1e-6, custom_inlet=inl,
                       cgsfac=1e-4, cg_tol=1e-10, cg_max_iter=50000,
                       line_search=True)
        assert np.all(np.isfinite(U))
        if np.abs(U - prev).max() < 1e-11:
            conv = True
            break
    assert conv, "did not converge in 400 steps"
    # pressure drop, inlet mean minus outlet mean
    def pmean(i, xt):
        num = den = 0.0
        for e in range(mesh.nelem):
            if abs(mesh.xnod[e, i] - xt) < 1e-9:
                num += np.sum(U[e, i, :, 2] * w) * (0.5 * mesh.hy[e])
                den += mesh.hy[e]
        return num / den
    dp = pmean(0, mesh.xnod.min()) - pmean(n - 1, mesh.xnod.max())
    assert abs(dp - 12.0 * L / RE) < 1e-5, dp
    ye = mesh.ynod[:, None, :]
    err = np.abs(U[..., 0] - 6.0 * ye * (1.0 - ye)).max()
    assert err < 1e-6, err
    # p is only weakly zero at the outlet -- assert the level is actually fixed
    assert np.abs([U[e, -1, :, 2] for e in range(mesh.nelem)
                   if mesh.bc[e, 1] == 6]).max() < 1e-5
