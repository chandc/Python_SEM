"""Dong's convective-like energy-stable open boundary condition (OBC).

Implements OUTFLOW_DONG_OBC_PLAN.md.  The condition, for an outflow plane at
x = x_max with outward normal n = (1, 0)  (Dong 2015, eq. 4, arXiv:1506.01320):

    R_x = nu*D0*du/dt - p + nu*du/dx - Ex(u) = 0
    R_y = nu*D0*dv/dt     + nu*dv/dx - Ey(u) = 0

with the backflow switch

    E(n,u) = 1/2 [ |u|^2 n + (n.u) u ] * Theta0(n,u)
    Theta0 = 1/2 (1 - tanh(u / (delta*U0)))     -> 1 where u < 0, 0 where u > 0

It is one vector condition = TWO scalar conditions per boundary point, exactly
the ADN count OUTFLOW_BC_STUDY.md sec 7b establishes, so it replaces the P+Z
pair one-for-one.

HOW IT ENTERS THE SOLVER.  As a boundary term in the least-squares functional,

    J -> J + w_obc^2 * int_{Gamma_out} (R_x^2 + R_y^2) ds

Nothing is imposed strongly: u, v, p AND om are all left free on a bc == 6
edge (the code needs no branch in bc.apply_mask / get_global_mask -- an
unhandled code masks nothing, which is exactly right here).  The rows are
linear in (u, v, p) except E, which is treated EXPLICITLY (lagged, u* = the
BDF extrapolation 2u^n - u^{n-1}), reproducing Dong's own treatment.  The
lagged E and the BDF history of the du/dt term live in a per-time-step
right-hand side built by build_bhist().

CONVENTIONS.  apply_B emits  ws * w_obc * R_linear  (surface-weighted rows,
mirroring apply_L's wq * R), so the functional contribution is
sum(rb^2 / ws) and apply_BT applies  w_obc * (dR/dU)^T  to rb -- together
they produce w_obc^2 * R * dR/dU * ws, the exact gradient of the boundary
term.  ws = (hy/2)*w_j is the 1D LGL surface weight along the edge.

SCOPE.  East edges only (every outflow in this repo is one).  bc == 6 on any
other edge raises.  Dong's theorem is NOT inherited: energy stability is
proved for the primitive-variable weak form, and a least-squares minimisation
of the residual enforces the condition only approximately -- expect the
condition, not the guarantee (OUTFLOW_DONG_OBC_PLAN.md sec 2).

State attributes (all optional; defaults give the D0 = 0, switch-off
traction-free form  -p + nu*du/dx = 0,  which is Stage 0 of the test ladder):

    obc_w      weight w_obc of the boundary rows vs the volume rows [1.0]
    obc_D0     Dong's D0; 1/D0 plays the role of a convection velocity [0.0]
    obc_delta  switch width delta; None disables Theta0 entirely [None]
    obc_U0     characteristic velocity U0 in the switch [1.0]
"""
import numpy as np
from .lgl import lgl_weights


def _params(state):
    return (float(getattr(state, 'obc_w', 1.0)),
            float(getattr(state, 'obc_D0', 0.0)),
            getattr(state, 'obc_delta', None),
            float(getattr(state, 'obc_U0', 1.0)))


def _cb(state):
    """Boundary time-derivative coefficient c_b = nu*D0*fac1/dt.

    Uses fac1/dt directly (not w_mass*fac1/dt): the boundary row carries its
    own weight w_obc, and folding w_mass in as well would double-count.  Only
    w_mom = w_mass has been exercised; the legacy weighting is untested here.
    """
    _, D0, _, _ = _params(state)
    if D0 == 0.0 or state.dt == 0:
        return 0.0
    return state.nu * D0 * state.fac1 / state.dt


def setup_obc(state):
    """Find and cache the Dong-OBC edges of state.mesh.  Returns the element
    list (empty when the mesh has no bc == 6 edge, which disables every hook).
    """
    elems = getattr(state, '_obc_elems', None)
    if elems is not None:
        return elems
    m = state.mesh
    for e in range(m.nelem):
        for d in (0, 2, 3):
            if m.bc[e, d] == 6:
                raise NotImplementedError(
                    "bc == 6 (Dong OBC) is implemented for East edges only; "
                    f"element {e} carries it on edge {d}")
    elems = [e for e in range(m.nelem) if m.bc[e, 1] == 6]
    state._obc_elems = elems
    if elems:
        w = lgl_weights(m.N)
        if getattr(m, 'curvilinear', False):
            # CURVILINEAR EDGE GEOMETRY.  Three things change and all three are
            # needed even when the outflow edge itself is straight:
            #
            #   d/dx along the edge is rx*u_r + sx*u_s, not facx*u_r.  sx is NOT
            #     zero on a straight vertical edge unless the whole element is a
            #     rectangle -- it is -y_r/J, and y_r points INTO the domain.
            #   rx varies along the edge instead of being one constant.
            #   the surface weight is the true arc length |d(x,y)/ds| * w_j,
            #     not (hy/2)*w_j, which is a bounding box and not an edge.
            #
            # The NORMAL is still assumed to be +x and is checked below.  A
            # genuinely curved outflow needs the true normal in the Dong rows
            # themselves, which is plan step 8; this covers every outflow whose
            # edge is axis aligned, which is what the O-ring-in-a-box topology
            # was chosen to give.
            D = state.D
            xs = np.matmul(m.X, D.T)
            ys = np.matmul(m.Y, D.T)
            state._obc_rx = np.array([m.rx[e, -1, :] for e in elems])
            state._obc_sx = np.array([m.sx[e, -1, :] for e in elems])
            state._obc_ws = np.array([np.hypot(xs[e, -1, :], ys[e, -1, :])*w
                                      for e in elems])
            for idx, e in enumerate(elems):
                nx, ny = m.rx[e, -1, :], m.ry[e, -1, :]
                tilt = np.abs(ny)/np.maximum(np.hypot(nx, ny), 1e-300)
                if tilt.max() > 1e-8 or np.any(nx <= 0):
                    raise NotImplementedError(
                        f'Dong OBC on element {e}: the outflow edge normal is '
                        f'not +x (max |n_y| = {tilt.max():.3e}).  The rows here '
                        f'are written for n = (1, 0); a curved outflow needs '
                        f'true boundary normals -- CURVILINEAR_2D_PLAN.md '
                        f'step 8.')
        else:
            # surface (edge) quadrature weight ws_j = (hy/2)*w_j, per element
            state._obc_ws = np.array([0.5 * m.hy[e] * w for e in elems])
            state._obc_rx = np.array([m.facx[e]*np.ones(m.N + 1) for e in elems])
            state._obc_sx = None          # identically zero: skip the term
    return elems


def obc_active(state):
    return len(setup_obc(state)) > 0


def apply_B(state, U):
    """LINEAR part of the two Dong rows on every bc == 6 edge.

    Returns rb[idx, j, 0:2] = ws * w_obc * (R_x_lin, R_y_lin) with
        R_x_lin = c_b*u - p + nu*du/dx
        R_y_lin = c_b*v     + nu*dv/dx
    evaluated at the edge nodes i = N of element state._obc_elems[idx].
    The lagged E terms and the BDF history belong to the right-hand side
    (build_bhist), NOT here: apply_B is also the operator applied to the
    Newton increment inside apply_A, where only the linear part acts.
    """
    elems = state._obc_elems
    m, D, nu = state.mesh, state.D, state.nu
    a_obc, _, _, _ = _params(state)
    cb = _cb(state)
    rb = np.empty((len(elems), m.N + 1, 2))
    sxa = state._obc_sx
    for idx, e in enumerate(elems):
        rx = state._obc_rx[idx]
        u_x = rx * (D[-1, :] @ U[e, :, :, 0])      # du/dx along the edge
        v_x = rx * (D[-1, :] @ U[e, :, :, 1])
        if sxa is not None:                        # curvilinear cross term
            u_x = u_x + sxa[idx] * (U[e, -1, :, 0] @ D.T)
            v_x = v_x + sxa[idx] * (U[e, -1, :, 1] @ D.T)
        ws = state._obc_ws[idx]
        rb[idx, :, 0] = ws * a_obc * (cb * U[e, -1, :, 0] - U[e, -1, :, 2] + nu * u_x)
        rb[idx, :, 1] = ws * a_obc * (cb * U[e, -1, :, 1] + nu * v_x)
    return rb


def apply_BT(state, rb, c):
    """Add  w_obc * (dR/dU)^T rb  into the local gradient array c, in place.

    dR_x/du = c_b*delta_{i,N} + nu*facx*D[N,i],  dR_x/dp = -delta_{i,N},
    dR_y/dv = c_b*delta_{i,N} + nu*facx*D[N,i];  the lagged E has no
    derivative.  Together with apply_B this is the exact gradient of the
    boundary term of the functional, so the assembled operator stays
    symmetric (tests/test_obc.py pins this).
    """
    elems = state._obc_elems
    m, D, nu = state.mesh, state.D, state.nu
    a_obc, _, _, _ = _params(state)
    cb = _cb(state)
    sxa = state._obc_sx
    for idx, e in enumerate(elems):
        gx = state._obc_rx[idx]
        rx = rb[idx, :, 0]
        ry = rb[idx, :, 1]
        # nu * d/dx adjoint: scatter along i with the D row of the edge
        c[e, :, :, 0] += (a_obc * nu) * np.outer(D[-1, :], gx * rx)
        c[e, :, :, 1] += (a_obc * nu) * np.outer(D[-1, :], gx * ry)
        if sxa is not None:                # adjoint of the tangential term
            c[e, -1, :, 0] += (a_obc * nu) * ((sxa[idx] * rx) @ D)
            c[e, -1, :, 1] += (a_obc * nu) * ((sxa[idx] * ry) @ D)
        if cb != 0.0:
            c[e, -1, :, 0] += (a_obc * cb) * rx
            c[e, -1, :, 1] += (a_obc * cb) * ry
        c[e, -1, :, 2] -= a_obc * rx
    return c


def _E_terms(state, us, vs):
    """Dong's backflow term at edge velocities (us, vs), normal n = (1, 0):
    E_x = 1/2[(u^2+v^2) + u*u]Theta0,  E_y = 1/2[u*v]Theta0."""
    _, _, delta, U0 = _params(state)
    th = 0.5 * (1.0 - np.tanh(us / (float(delta) * U0)))
    return 0.5 * (2.0 * us**2 + vs**2) * th, 0.5 * (us * vs) * th


def build_bhist(state, U_history, alpha):
    """Per-time-step right-hand side of the boundary rows: BDF history of the
    nu*D0*du/dt term plus the LAGGED backflow term E(n, u*), u* = the BDF
    extrapolation (2u^n - u^{n-1} for BDF2, u^n for BDF1).  Mirrors what
    step_bdf's su_history does for the volume momentum rows.  Stored on
    state._obc_bhist; the residual is then apply_B(U) - bhist.
    """
    elems = state._obc_elems
    m = state.mesh
    a_obc, D0, delta, U0 = _params(state)
    bh = np.zeros((len(elems), m.N + 1, 2))
    for idx, e in enumerate(elems):
        ws = state._obc_ws[idx]
        if D0 != 0.0 and state.dt != 0:
            hu = np.zeros(m.N + 1)
            hv = np.zeros(m.N + 1)
            for mm in range(len(alpha)):
                hu += alpha[mm] * U_history[mm][e, -1, :, 0]
                hv += alpha[mm] * U_history[mm][e, -1, :, 1]
            fac = state.nu * D0 / state.dt
            bh[idx, :, 0] += fac * hu
            bh[idx, :, 1] += fac * hv
        if delta is not None and not getattr(state, 'obc_picard', False):
            # LAGGED (Dong's own explicit treatment): E at the BDF
            # extrapolation u* = 2u^n - u^{n-1}.  WARNING: measured to blow up
            # on the short BFS at dt = 1 (scratch/dong_bfs.py) -- an explicit
            # boundary term carries its own CFL-like limit, exactly the
            # OUTFLOW_BC_STUDY.md remedy-E failure.  Use obc_picard = True
            # with sub-iterations for large-dt steady-seeking runs.
            if len(U_history) >= 2:
                us = 2.0 * U_history[0][e, -1, :, 0] - U_history[1][e, -1, :, 0]
                vs = 2.0 * U_history[0][e, -1, :, 1] - U_history[1][e, -1, :, 1]
            else:
                us = U_history[0][e, -1, :, 0]
                vs = U_history[0][e, -1, :, 1]
            Ex, Ey = _E_terms(state, us, vs)
            bh[idx, :, 0] += Ex
            bh[idx, :, 1] += Ey
        bh[idx, :, 0] *= ws * a_obc
        bh[idx, :, 1] *= ws * a_obc
    state._obc_bhist = bh
    return bh


def residual_B(state, U):
    """Boundary-row residual rb = apply_B(U) - bhist [- ws*w_obc*E(U)].

    With obc_picard set, the backflow term is evaluated at the CURRENT
    iterate instead of the lagged extrapolation -- semi-implicit (Picard: E
    enters the residual but not the Jacobian), so with sub-iterations it
    converges to the fully implicit condition and the explicit-lag CFL limit
    disappears.  Zero-history fallback when step_bdf has not built one
    (D0 = 0 and switch off need none)."""
    rb = apply_B(state, U)
    bh = getattr(state, '_obc_bhist', None)
    if bh is not None and bh.shape == rb.shape:
        rb = rb - bh
    a_obc, _, delta, _ = _params(state)
    if delta is not None and getattr(state, 'obc_picard', False):
        for idx, e in enumerate(state._obc_elems):
            Ex, Ey = _E_terms(state, U[e, -1, :, 0], U[e, -1, :, 1])
            ws = state._obc_ws[idx]
            rb[idx, :, 0] -= ws * a_obc * Ex
            rb[idx, :, 1] -= ws * a_obc * Ey
    return rb


def merit_B(state, rb):
    """Boundary contribution to the least-squares merit: sum(rb^2/ws), the
    discrete  w_obc^2 * int_edge R^2 ds  (rb carries one factor of ws)."""
    return float(np.sum(rb * rb / state._obc_ws[..., None]))


def jacobi_add(state, diag_A):
    """Add the boundary rows' contribution to the diagonal of A, in place.
    Row j of edge idx contributes, at local node (i, j):
        field u:  (w_obc*(c_b*delta_{i,N} + nu*facx*D[N,i]))^2 * ws_j
        field v:  same
        field p:  w_obc^2 * ws_j   (at i = N only)
    Omitting this would not make the preconditioner wrong, just worse --
    exactly on the flows this condition targets (plan sec 3.4)."""
    elems = state._obc_elems
    m, D, nu = state.mesh, state.D, state.nu
    a_obc, _, _, _ = _params(state)
    cb = _cb(state)
    sxa = state._obc_sx
    n = m.N + 1
    for idx, e in enumerate(elems):
        gx = state._obc_rx[idx]
        ws = state._obc_ws[idx]
        # d(R_r)/d(u_ij) = nu*gx[r]*D[N,i]*delta_rj            (normal part)
        #                + [i == N]*( c_b*delta_rj + nu*sx[r]*D[r,j] )
        # and the diagonal of A at (i, j) is sum_r (dR_r/du_ij)^2 * ws_r.
        #
        # For i < N only r = j contributes.  AT i = N THE TANGENTIAL TERM COUPLES
        # EVERY EDGE NODE TO EVERY OTHER, so rows r != j contribute there too --
        # omitting them left this 5e-05 wrong on a deformed mesh while staying
        # exact on an affine one, where sx == 0 kills the whole term.
        contrib = (a_obc * nu * np.outer(D[-1, :], gx))**2 * ws[None, :]
        dN = nu * gx * D[-1, -1] + cb
        extra = 0.0
        if sxa is not None:
            T = nu * sxa[idx][:, None] * D          # T[r, j] = nu*sx[r]*D[r,j]
            dN = dN + np.diag(T)
            off = (a_obc * T)**2 * ws[:, None]
            off[np.arange(n), np.arange(n)] = 0.0   # r = j already in dN
            extra = off.sum(axis=0)
        contrib[-1, :] = (a_obc * dN)**2 * ws + extra
        diag_A[e, :, :, 0] += contrib
        diag_A[e, :, :, 1] += contrib
        diag_A[e, -1, :, 2] += a_obc**2 * ws
    return diag_A
