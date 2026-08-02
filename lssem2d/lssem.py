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
        mask_gs = gather_scatter(self.mesh, mask_local)
        self._global_mask = (mask_gs > 0.99).astype(float)
        self._cached_mask_pin = pin_p
        return self._global_mask

def apply_L(state, U, fu, fv):
    """
    Apply the VVP operator L.
    Returns the four weighted equation residuals su[e, i, j, k].
    
    state: SolverState instance
    U: shape (nelem, n, n, 4) - fields (u, v, p, omega)
    fu, fv: current linearisation velocities, shape (nelem, n, n)
    """
    u, v, p, om = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    
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
    
    inv_dt = state.fac1 / state.dt if state.dt != 0 else 0.0
    wq = state.mesh.wq
    
    su = state.su
    # r1 = inv_dt * u + fu * u_x + fv * u_y + u * dfu_dx + v * dfu_dy + p_x + state.nu * om_y
    su[..., 0] = (inv_dt * u + fu * u_x + fv * u_y + u * dfu_dx + v * dfu_dy + p_x + state.nu * om_y) * wq
    
    # r2 = inv_dt * v + fu * v_x + fv * v_y + u * dfv_dx + v * dfv_dy + p_y - state.nu * om_x
    su[..., 1] = (inv_dt * v + fu * v_x + fv * v_y + u * dfv_dx + v * dfv_dy + p_y - state.nu * om_x) * wq
    
    # r3 = u_x + v_y
    su[..., 2] = (u_x + v_y) * wq
    
    # r4 = om + u_y - v_x
    su[..., 3] = (om + u_y - v_x) * wq
    
    return su

def apply_LT(state, su, fu, fv):
    """
    Apply the transpose VVP operator L^T.
    Returns c[e,i,j,k] = sum_m (dR_m/dU_k)^T su_m
    
    su: weighted residuals, shape (nelem, n, n, 4)
    """
    su1, su2, su3, su4 = su[..., 0], su[..., 1], su[..., 2], su[..., 3]
    
    dfu_dx, dfu_dy = state.dfu_dx, state.dfu_dy
    dfv_dx, dfv_dy = state.dfv_dx, state.dfv_dy
    
    inv_dt = state.fac1 / state.dt if state.dt != 0 else 0.0
    
    c = state.c
    tmp = state.tmp_x
    tmp2 = state.tmp_y
    
    # c_1 = (fac1/dt)*su1 + dfu_dx*su1 + dfv_dx*su2 
    #       + Dx^T(su3) + Dx^T(fu*su1) + Dy^T(su4) + Dy^T(fv*su1)
    c[..., 0] = inv_dt * su1 + dfu_dx * su1 + dfv_dx * su2
    DxT(su3, state.D, state.mesh.facx, out=tmp)
    c[..., 0] += tmp
    np.multiply(fu, su1, out=tmp)
    DxT(tmp, state.D, state.mesh.facx, out=tmp2)
    c[..., 0] += tmp2
    DyT(su4, state.D, state.mesh.facy, out=tmp)
    c[..., 0] += tmp
    np.multiply(fv, su1, out=tmp)
    DyT(tmp, state.D, state.mesh.facy, out=tmp2)
    c[..., 0] += tmp2
    
    # c_2 = (fac1/dt)*su2 + dfu_dy*su1 + dfv_dy*su2 
    #       - Dx^T(su4) + Dx^T(fu*su2) + Dy^T(su3) + Dy^T(fv*su2)
    c[..., 1] = inv_dt * su2 + dfu_dy * su1 + dfv_dy * su2
    DxT(su4, state.D, state.mesh.facx, out=tmp)
    c[..., 1] -= tmp
    np.multiply(fu, su2, out=tmp)
    DxT(tmp, state.D, state.mesh.facx, out=tmp2)
    c[..., 1] += tmp2
    DyT(su3, state.D, state.mesh.facy, out=tmp)
    c[..., 1] += tmp
    np.multiply(fv, su2, out=tmp)
    DyT(tmp, state.D, state.mesh.facy, out=tmp2)
    c[..., 1] += tmp2
    
    # c_3 = Dx^T(su1) + Dy^T(su2)
    DxT(su1, state.D, state.mesh.facx, out=c[..., 2])
    DyT(su2, state.D, state.mesh.facy, out=tmp)
    c[..., 2] += tmp
    
    # c_4 = su4 - nu*Dx^T(su2) + nu*Dy^T(su1)
    c[..., 3] = su4
    DxT(su2, state.D, state.mesh.facx, out=tmp)
    c[..., 3] -= state.nu * tmp
    DyT(su1, state.D, state.mesh.facy, out=tmp)
    c[..., 3] += state.nu * tmp
    
    return c
