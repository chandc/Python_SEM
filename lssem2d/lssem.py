import numpy as np
from .operators import dUdx, dUdy, DxT, DyT

class SolverState:
    """Holds mesh, operator matrices, and cached linearisation data for the LSSEM solver."""
    def __init__(self, mesh, D, nu, dt, fac1=1.0):
        self.mesh = mesh
        self.D = D
        self.nu = nu
        self.dt = dt
        self.fac1 = fac1
        
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
            
        mask_local = np.ones((self.mesh.nelem, self.mesh.N + 1, self.mesh.N + 1, 4))
        for e in range(self.mesh.nelem):
            bc_W = self.mesh.bc[e, 0]
            if bc_W in (1, 2, 3): mask_local[e, 0, :, 0:2] = 0.0
            elif bc_W == 5: mask_local[e, 0, :, 1] = 0.0; mask_local[e, 0, :, 3] = 0.0
            
            bc_E = self.mesh.bc[e, 1]
            if bc_E in (1, 2, 3): mask_local[e, -1, :, 0:2] = 0.0
            elif bc_E == 5: mask_local[e, -1, :, 1] = 0.0; mask_local[e, -1, :, 3] = 0.0
            
            bc_S = self.mesh.bc[e, 2]
            if bc_S in (1, 2, 3): mask_local[e, :, 0, 0:2] = 0.0
            elif bc_S == 5: mask_local[e, :, 0, 1] = 0.0; mask_local[e, :, 0, 3] = 0.0
            
            bc_N = self.mesh.bc[e, 3]
            if bc_N in (1, 2, 3): mask_local[e, :, -1, 0:2] = 0.0
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

def apply_L(state, U, fu, fv):
    """
    Apply the VVP operator L.
    Returns the four weighted equation residuals su[e, i, j, k].
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
    # continuity under-enforced.  Harmless when the residual is ~0 (cavity, Poiseuille);
    # it diverges on under-resolved cases such as the BFS.  See lssem_baseline.f90 rhs().
    dtl = state.dt if state.dt != 0 else 1.0          # dt==0 => steady form
    f1 = state.fac1 if state.dt != 0 else 0.0
    wq = state.mesh.wq

    su0, su1, su2, su3 = state.su0_c, state.su1_c, state.su2_c, state.su3_c

    su0[...] = (f1 * u + dtl * (fu * u_x + fv * u_y + u * dfu_dx + v * dfu_dy + p_x + state.nu * om_y)) * wq
    su1[...] = (f1 * v + dtl * (fu * v_x + fv * v_y + u * dfv_dx + v * dfv_dy + p_y - state.nu * om_x)) * wq
    
    su2[...] = (u_x + v_y) * wq
    su3[...] = (om + u_y - v_x) * wq
    
    su = state.su_out
    np.copyto(su[..., 0], su0)
    np.copyto(su[..., 1], su1)
    np.copyto(su[..., 2], su2)
    np.copyto(su[..., 3], su3)
    
    return su

def apply_LT(state, su, fu, fv):
    """
    Apply the transpose VVP operator L^T.
    """
    su1, su2, su3, su4 = state.su0_c, state.su1_c, state.su2_c, state.su3_c
    
    su3_scaled = su[..., 2]
    np.copyto(su1, su[..., 0])
    np.copyto(su2, su[..., 1])
    np.copyto(su3, su[..., 2])
    np.copyto(su4, su[..., 3])

    # Exact transpose of the row-weighted apply_L.  Row m of the new operator is
    # R_new = S R_old with S = diag(dt, dt, 1, 1), so L_new^T = L_old^T . S:
    # pre-scale the two momentum components and the rest of this routine is unchanged.
    # (The inv_dt below then correctly becomes fac1, since inv_dt * dt = fac1.)
    dtl = state.dt if state.dt != 0 else 1.0
    su1 *= dtl
    su2 *= dtl
    
    dfu_dx, dfu_dy = state.dfu_dx, state.dfu_dy
    dfv_dx, dfv_dy = state.dfv_dx, state.dfv_dy
    
    inv_dt = state.fac1 / state.dt if state.dt != 0 else 0.0
    
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
