import numpy as np
from .operators import dUdx, dUdy, DxT, DyT
from . import backend

def ls_coeffs(state):
    """Least-squares row coefficients for the two momentum equations.

    The momentum rows are   a_mass*u + a_flux*N(u)   and the continuity and
    vorticity rows are unweighted, so **a_flux is the least-squares weight of
    momentum against the constraints**.

    The momentum row carries TWO independent weights:

        w_mass * [fac1*u - sum_m alpha_m u^{n-m}] / dt   +   w_mom * N(u)

    giving

      both None (legacy):  a_mass = fac1,            a_flux = dt,     hist = 1
      w_mom only:          a_mass = fac1/dt,         a_flux = w_mom,  hist = 1/dt
      both set:            a_mass = w_mass*fac1/dt,  a_flux = w_mom,  hist = w_mass/dt

    w_mass scales the mass and history terms TOGETHER, so they still cancel
    identically at steady state (BDF gives fac1 = sum alpha_m) and the steady
    momentum residual is exactly `w_mom * N(u)` regardless of w_mass.  The
    continuity and vorticity rows carry weight 1, so the steady functional is

        J = int[ w_mom^2 (N_1^2 + N_2^2) + (div u)^2 + (om + u_y - v_x)^2 ]

    WHAT THE TWO PARAMETERS MEAN
      w_mom             NUMERICAL.  The least-squares weight of momentum against
                        the constraints -- which equation the solver may violate.
                        No physical content.
      w_mass / w_mom    PHYSICAL.  A rescaling of time:  dt_eff = dt*w_mom/w_mass.
                        Divide the row by w_mom and the bracket appears over
                        dt*w_mom/w_mass, which is the step the scheme actually
                        takes.

    So w_mass alone has no physical meaning; it is half of a time rescaling.
    **For time-accurate runs set w_mass = w_mom** (then dt_eff = dt).  For
    steady-state runs the group cancels, the converged answer does not depend on
    w_mass at all, and w_mass/w_mom is a pseudo-time step chosen for convergence
    rather than accuracy -- larger w_mass means a smaller pseudo-step, more
    diagonal dominance in the velocity block, slower but more stable.

    Two parameters are needed because (a_mass, a_flux) has two degrees of freedom
    and each single-parameter form reaches only a line through that plane:
    w_mom alone pins a_mass = fac1/dt to the time step; scaling the whole row (an
    earlier, wrong version of this function) pins the ratio a_mass/a_flux and so
    factors out entirely.

    Note w_mass = dt, w_mom = 1 gives (fac1, 1, 1), which IS legacy at dt=1
    whatever the nominal dt says -- the scheme then takes unit steps.  That is
    the configuration measured to reproduce the analytic Poiseuille solution, but
    the nominal dt becomes decorative, so do not read it as "dt=0.5 with dt=1
    accuracy".

    Why the weighting matters at all: pressure appears ONLY in the momentum
    rows, so the pressure block of L^T L scales as a_flux^2.  When a_flux is
    small the pressure is under-weighted, L^T L develops a near-null space in p,
    and the exact solution stops being the UNIQUE minimiser.  Measured on
    Poiseuille, whose exact solution is exactly representable: legacy at dt=0.05
    (hence a_flux=0.05) gives 98% velocity error and an outlet pressure varying
    by 4.3x the total streamwise pressure drop, while a_flux=1 gives 5e-4 and a
    flat outlet pressure.

    An earlier version of this function multiplied the mass and history terms by
    w_mom as well (a_mass = w*fac1/dt, hist = w/dt).  That made w_mom factor out
    of the entire row, so it could only rescale the row and never change the
    balance between the time-derivative term and N(u).  It is fixed here.

    Returns (a_mass, a_flux, hist_scale).  a_mass and hist_scale must stay in
    step -- if they disagree the mass terms no longer cancel at steady state and
    the solver silently converges to a modified equation.
    """
    dtl = state.dt if state.dt != 0 else 1.0     # dt == 0 => pure steady form
    f1 = state.fac1 if state.dt != 0 else 0.0
    w_f = getattr(state, 'w_mom', None)
    w_m = getattr(state, 'w_mass', None)
    if w_f is None and w_m is None:
        return f1, dtl, 1.0                      # legacy, untouched
    if w_f is None:
        w_f = 1.0
    if w_m is None:
        w_m = 1.0
    # s computed as w_m/dtl and then multiplied into f1, NOT (w_m*f1)/dtl: when
    # w_mass == dt the ratio is exactly 1.0 and a_mass comes back bit-identical
    # to fac1.  The other grouping round-trips through a division and loses an ulp.
    s = float(w_m)/dtl
    return f1*s, float(w_f), s


def ls_pseudo(state):
    """Pseudo-time coefficient kappa for the momentum rows.  0.0 when disabled.

    The 1996 F77 source (reference/tj_channel_1996.f) carries a pseudo-time
    derivative that both lssem_baseline.f90 and this port dropped:

        su(ij,1) = fac1*u*facem + dt*( u + fu*dudx + ... )*facem     <- the bare u
        c1(ij,ne) = (dt+fac1)*su(ij,1) + ...                         <- its transpose

    Chan (1996) p.2 gives the operator with 1/dtau on the momentum diagonals and
    u0/dtau on the right-hand side: "dt is the physical time step whereas dtau is
    the pseudo-time step".  The row is multiplied through by dt, so (dt/dtau)*u
    against the code's dt*u fixes dtau = 1 there.  In our coefficients that is

        kappa = a_flux / dtau

    so the momentum row becomes  (a_mass + kappa)*u + a_flux*N(u)  and the
    transpose collocation coefficient becomes (a_mass + kappa)/a_flux -- exactly
    the Fortran's (dt+fac1).  `dtau = None` gives kappa = 0 and the dtau -> inf
    limit, which is what every result before 2026-08-09 used.

    IT IS NOT w_mass.  Both add a multiple of the identity to the momentum rows,
    but w_mass pairs with the previous TIME LEVEL (via hist_scale), so raising it
    moves the converged steady state.  kappa pairs with the previous SUB-ITERATE,
    which the caller cancels out of the residual, so it modifies the Jacobian and
    leaves the equations being solved alone.

    It does NOT leave the answer exactly invariant.  Each sub-iteration minimises
    ||L_k u - (f + kappa u0)||^2 about u0, and at the fixed point u = u0 the
    stationarity condition is

        L^T R + kappa E^T R = 0        rather than        L^T R = 0

    so the converged state is perturbed by O(kappa * R).  Where the residual is
    driven to zero (Poiseuille reaches J = 5.94e-27) the two coincide and the
    term is free; where it is not (the BFS sits at J ~ 3.7e-05) the perturbation
    is proportional to the discretisation residual and vanishes under refinement.
    Consistent, SUPG-like -- but it has to be measured, not assumed.  See
    PSEUDO_TIME_DESIGN.md sections 3 and 6.
    """
    dtau = getattr(state, 'dtau', None)
    if dtau is None:
        return 0.0
    _, a_flux, _ = ls_coeffs(state)
    return a_flux / float(dtau)


class SolverState:
    """Holds mesh, operator matrices, and cached linearisation data for the LSSEM solver."""
    def __init__(self, mesh, D, nu, dt, fac1=1.0, w_mom=None, w_mass=None, dtau=None):
        self.mesh = mesh
        self.D = D
        self.nu = nu
        self.dt = dt
        self.fac1 = fac1
        # Momentum-row weights; see ls_coeffs() for the full contract.
        #   w_mom   weight on N(u)  -- the LS weight vs the constraints
        #   w_mass  weight on the time-derivative group (mass AND history)
        # Both None => legacy behaviour, unchanged.  w_mass/w_mom rescales time:
        # dt_eff = dt*w_mom/w_mass, so set w_mass = w_mom for time-accurate runs.
        self.w_mom = w_mom
        self.w_mass = w_mass
        # Pseudo-time step from Chan (1996); see ls_pseudo() for the contract.
        # None => the dtau -> inf limit, i.e. no pseudo-time term (the default,
        # and what the F90 baseline does).  dtau = 1 reproduces the 1996 F77 code.
        self.dtau = dtau
        
        self.dfu_dx = None
        self.dfu_dy = None
        self.dfv_dx = None
        self.dfv_dy = None
        
        # Preallocated work arrays for apply_L and apply_LT
        nelem, n = mesh.nelem, mesh.N + 1
        self.su = np.zeros((nelem, n, n, 4))
        self.c = np.zeros((nelem, n, n, 4))
        self.tmp_x = np.zeros((nelem, n, n))
        self.tmp_y = np.zeros((nelem, n, n))
        self.u_x = np.zeros((nelem, n, n))
        self.u_y = np.zeros((nelem, n, n))
        self.v_x = np.zeros((nelem, n, n))
        self.v_y = np.zeros((nelem, n, n))
        self.p_x = np.zeros((nelem, n, n))
        self.p_y = np.zeros((nelem, n, n))
        self.om_x = np.zeros((nelem, n, n))
        self.om_y = np.zeros((nelem, n, n))
        self.U_t = np.empty((nelem, 4, n, n))

        # Contiguous work buffers for apply_L / apply_LT (avoid strided U[...,k] views).
        # These MUST be allocated here, not in get_global_mask(): that method early-returns
        # when the mask is cached, so allocating there makes any call path that reaches
        # apply_L before the mask is built fail with AttributeError.
        self.c = np.empty((nelem, n, n, 4))
        self.u_c = np.empty((nelem, n, n))
        self.v_c = np.empty((nelem, n, n))
        self.p_c = np.empty((nelem, n, n))
        self.om_c = np.empty((nelem, n, n))
        self.su0_c = np.empty((nelem, n, n))
        self.su1_c = np.empty((nelem, n, n))
        self.su2_c = np.empty((nelem, n, n))
        self.su3_c = np.empty((nelem, n, n))
        self.su_out = np.empty((nelem, n, n, 4))
        self.c0_c = np.empty((nelem, n, n))
        self.c1_c = np.empty((nelem, n, n))
        self.c2_c = np.empty((nelem, n, n))
        self.c3_c = np.empty((nelem, n, n))
        self.c_out = np.empty((nelem, n, n, 4))

    def update_linearisation(self, fu, fv):
        """Precompute gradients of linearisation velocities fu, fv."""
        self.dfu_dx = dUdx(fu, self.D, self.mesh.facx)
        self.dfu_dy = dUdy(fu, self.D, self.mesh.facy)
        self.dfv_dx = dUdx(fv, self.D, self.mesh.facx)
        self.dfv_dy = dUdy(fv, self.D, self.mesh.facy)
        
    def get_global_mask(self, pin_p=False):
        if hasattr(self, '_cached_mask_pin') and self._cached_mask_pin == pin_p:
            return self._global_mask
            
        # NOTE: this duplicates bc.apply_mask's logic and MUST stay in step with
        # it.  The two disagreed on bc == 4 until 2026-08-12: apply_mask froze the
        # pressure DOF, this did not, and since newton_step builds
        # `b = -c_gs * mask_global` from HERE, the update moved p straight off the
        # value apply_bc had just written.  bc = 4 therefore imposed nothing --
        # measured max|p| = 4.87e-01 on a plane where p = 0 was claimed.  See
        # OUTFLOW_BC_STUDY.md sec 3.
        mask_local = np.ones((self.mesh.nelem, self.mesh.N + 1, self.mesh.N + 1, 4))
        for e in range(self.mesh.nelem):
            bc_W = self.mesh.bc[e, 0]
            if bc_W in (1, 2, 3): mask_local[e, 0, :, 0:2] = 0.0
            elif bc_W == 4: mask_local[e, 0, :, 2] = 0.0
            elif bc_W == 5: mask_local[e, 0, :, 1] = 0.0; mask_local[e, 0, :, 3] = 0.0

            bc_E = self.mesh.bc[e, 1]
            if bc_E in (1, 2, 3): mask_local[e, -1, :, 0:2] = 0.0
            elif bc_E == 4: mask_local[e, -1, :, 2] = 0.0
            elif bc_E == 5: mask_local[e, -1, :, 1] = 0.0; mask_local[e, -1, :, 3] = 0.0

            bc_S = self.mesh.bc[e, 2]
            if bc_S in (1, 2, 3): mask_local[e, :, 0, 0:2] = 0.0
            elif bc_S == 4: mask_local[e, :, 0, 2] = 0.0
            elif bc_S == 5: mask_local[e, :, 0, 1] = 0.0; mask_local[e, :, 0, 3] = 0.0

            bc_N = self.mesh.bc[e, 3]
            if bc_N in (1, 2, 3): mask_local[e, :, -1, 0:2] = 0.0
            elif bc_N == 4: mask_local[e, :, -1, 2] = 0.0
            elif bc_N == 5: mask_local[e, :, -1, 1] = 0.0; mask_local[e, :, -1, 3] = 0.0

        if pin_p:
            e_p, i_p, j_p = pin_p if isinstance(pin_p, tuple) else (0, 0, 0)
            mask_local[e_p, i_p, j_p, 2] = 0.0
            
        from .assembly import gather_scatter
        mult = gather_scatter(self.mesh, np.ones_like(mask_local))
        mask_gs = gather_scatter(self.mesh, mask_local)
        self._global_mask = (mask_gs > mult - 0.01).astype(float)
        self._cached_mask_pin = pin_p
        return self._global_mask

def _apply_L_numpy(state, U, fu, fv):
    """
    Apply the VVP operator L.  NumPy reference implementation.
    Returns the four weighted equation residuals su[e, i, j, k].

    tests/test_backend_parity.py validates the numba kernels against this.
    """
    u, v, p, om = state.u_c, state.v_c, state.p_c, state.om_c
    np.copyto(u, U[..., 0])
    np.copyto(v, U[..., 1])
    np.copyto(p, U[..., 2])
    np.copyto(om, U[..., 3])
    
    # Compute spatial derivatives in-place
    u_x = dUdx(u, state.D, state.mesh.facx, out=state.u_x)
    u_y = dUdy(u, state.D, state.mesh.facy, out=state.u_y)
    
    v_x = dUdx(v, state.D, state.mesh.facx, out=state.v_x)
    v_y = dUdy(v, state.D, state.mesh.facy, out=state.v_y)
    
    p_x = dUdx(p, state.D, state.mesh.facx, out=state.p_x)
    p_y = dUdy(p, state.D, state.mesh.facy, out=state.p_y)
    
    om_x = dUdx(om, state.D, state.mesh.facx, out=state.om_x)
    om_y = dUdy(om, state.D, state.mesh.facy, out=state.om_y)
    
    dfu_dx, dfu_dy = state.dfu_dx, state.dfu_dy
    dfv_dx, dfv_dy = state.dfv_dx, state.dfv_dy
    
    # Least-squares row weighting must match the Fortran reference: the momentum rows
    # are  fac1*u + dt*(...),  NOT  (fac1/dt)*u + (...).  The two are the same equation
    # but differ by a factor dt as LEAST-SQUARES rows, so the (fac1/dt) form over-weights
    # momentum by 1/dt relative to continuity and the vorticity definition and leaves
    # continuity under-enforced.  See lssem_baseline.f90 rhs().
    #
    # An earlier version of this comment said the error was "harmless when the
    # residual is ~0 (cavity, Poiseuille)".  That is true for the CAVITY, which is
    # lid-driven -- the forcing is a velocity BC, so a poorly weighted pressure
    # degrades p without wrecking u (measured: 5.9x accuracy spread over dt).  It
    # is badly WRONG for Poiseuille, which is pressure-driven: the weighting there
    # corrupts the driving force and gives 98% velocity error at dt=0.05, on a
    # problem whose exact solution is exactly representable (1875x spread over dt).
    # See POISEUILLE_DT_STUDY.md and WEIGHT_VS_TIMESTEP_STUDY.md.
    a_mass, a_flux, _ = ls_coeffs(state)
    # Pseudo-time (Chan 1996) folds straight into the mass coefficient: the row is
    # (a_mass + kappa)*u + a_flux*N(u).  kappa = 0 unless state.dtau is set, so
    # this is bit-identical to the previous behaviour by default.  The caller must
    # cancel kappa*u out of the RESIDUAL (solver.newton_step does); here it is
    # unconditional because apply_L is also the operator applied to the increment.
    a_mass = a_mass + ls_pseudo(state)
    wq = state.mesh.wq

    su0, su1, su2, su3 = state.su0_c, state.su1_c, state.su2_c, state.su3_c

    su0[...] = (a_mass * u + a_flux * (fu * u_x + fv * u_y + u * dfu_dx + v * dfu_dy + p_x + state.nu * om_y)) * wq
    su1[...] = (a_mass * v + a_flux * (fu * v_x + fv * v_y + u * dfv_dx + v * dfv_dy + p_y - state.nu * om_x)) * wq
    
    su2[...] = (u_x + v_y) * wq
    su3[...] = (om + u_y - v_x) * wq
    
    su = state.su_out
    np.copyto(su[..., 0], su0)
    np.copyto(su[..., 1], su1)
    np.copyto(su[..., 2], su2)
    np.copyto(su[..., 3], su3)
    
    return su

def _apply_LT_numpy(state, su, fu, fv):
    """
    Apply the transpose VVP operator L^T.  NumPy reference implementation.
    """
    su1, su2, su3, su4 = state.su0_c, state.su1_c, state.su2_c, state.su3_c
    
    su3_scaled = su[..., 2]
    np.copyto(su1, su[..., 0])
    np.copyto(su2, su[..., 1])
    np.copyto(su3, su[..., 2])
    np.copyto(su4, su[..., 3])

    # Exact transpose of the row-weighted apply_L.  Row m of the new operator is
    # R_new = S R_old with S = diag(a_flux, a_flux, 1, 1), so L_new^T = L_old^T . S:
    # pre-scale the two momentum components and the rest of this routine is unchanged.
    # inv_dt below is a_mass/a_flux, so inv_dt * a_flux = a_mass as required.
    a_mass, a_flux, _ = ls_coeffs(state)
    a_mass = a_mass + ls_pseudo(state)      # matches the Fortran's (dt+fac1)
    su1 *= a_flux
    su2 *= a_flux

    dfu_dx, dfu_dy = state.dfu_dx, state.dfu_dy
    dfv_dx, dfv_dy = state.dfv_dx, state.dfv_dy

    inv_dt = a_mass / a_flux if a_flux != 0.0 else 0.0
    
    c0, c1, c2, c3 = state.c0_c, state.c1_c, state.c2_c, state.c3_c
    tmp = state.tmp_x
    tmp2 = state.tmp_y
    
    # c_1
    c0[...] = inv_dt * su1 + dfu_dx * su1 + dfv_dx * su2
    DxT(su3_scaled, state.D, state.mesh.facx, out=tmp)
    c0[...] += tmp
    DxT(fu * su1, state.D, state.mesh.facx, out=tmp)
    c0[...] += tmp
    DyT(su4, state.D, state.mesh.facy, out=tmp)
    c0[...] += tmp
    DyT(fv * su1, state.D, state.mesh.facy, out=tmp)
    c0[...] += tmp
    
    # c_2
    c1[...] = inv_dt * su2 + dfu_dy * su1 + dfv_dy * su2
    DxT(su4, state.D, state.mesh.facx, out=tmp)
    c1[...] -= tmp
    DxT(fu * su2, state.D, state.mesh.facx, out=tmp)
    c1[...] += tmp
    DyT(su3_scaled, state.D, state.mesh.facy, out=tmp)
    c1[...] += tmp
    DyT(fv * su2, state.D, state.mesh.facy, out=tmp)
    c1[...] += tmp
    
    # c_3
    DxT(su1, state.D, state.mesh.facx, out=tmp)
    DyT(su2, state.D, state.mesh.facy, out=tmp2)
    c2[...] = tmp + tmp2
    
    # c_4
    DyT(su1, state.D, state.mesh.facy, out=tmp)
    DxT(su2, state.D, state.mesh.facx, out=tmp2)
    c3[...] = su4 + state.nu * tmp - state.nu * tmp2
    
    c = state.c_out
    np.copyto(c[..., 0], c0)
    np.copyto(c[..., 1], c1)
    np.copyto(c[..., 2], c2)
    np.copyto(c[..., 3], c3)

    return c


# ---------------------------------------------------------------------------
#  Backend dispatch
#
#  The public names apply_L / apply_LT are kept in place so every existing call
#  site and test exercises whichever backend is active without modification.
#  Resolution happens once per backend switch, not per call.
# ---------------------------------------------------------------------------
_IMPL_L = None
_IMPL_LT = None


def _bind_backend(name):
    global _IMPL_L, _IMPL_LT
    if name == 'numba':
        from .kernels_numba import apply_L as _L, apply_LT as _LT
    else:
        _L, _LT = _apply_L_numpy, _apply_LT_numpy
    _IMPL_L, _IMPL_LT = _L, _LT


backend.register(_bind_backend)


def apply_L(state, U, fu, fv):
    """Apply the VVP operator L using the active backend (see lssem2d.backend)."""
    return _IMPL_L(state, U, fu, fv)


def apply_LT(state, su, fu, fv):
    """Apply the transpose VVP operator L^T using the active backend."""
    return _IMPL_LT(state, su, fu, fv)
