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
        
    def update_linearisation(self, fu, fv):
        """Precompute gradients of linearisation velocities fu, fv."""
        self.dfu_dx = dUdx(fu, self.D, self.mesh.facx)
        self.dfu_dy = dUdy(fu, self.D, self.mesh.facy)
        self.dfv_dx = dUdx(fv, self.D, self.mesh.facx)
        self.dfv_dy = dUdy(fv, self.D, self.mesh.facy)

def apply_L(state, U, fu, fv):
    """
    Apply the VVP operator L.
    Returns the four weighted equation residuals su[e, i, j, k].
    
    state: SolverState instance
    U: shape (nelem, n, n, 4) - fields (u, v, p, omega)
    fu, fv: current linearisation velocities, shape (nelem, n, n)
    """
    u, v, p, om = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    
    # Compute spatial derivatives
    u_x = dUdx(u, state.D, state.mesh.facx)
    u_y = dUdy(u, state.D, state.mesh.facy)
    
    v_x = dUdx(v, state.D, state.mesh.facx)
    v_y = dUdy(v, state.D, state.mesh.facy)
    
    p_x = dUdx(p, state.D, state.mesh.facx)
    p_y = dUdy(p, state.D, state.mesh.facy)
    
    om_x = dUdx(om, state.D, state.mesh.facx)
    om_y = dUdy(om, state.D, state.mesh.facy)
    
    # Retrieve cached gradients
    dfu_dx, dfu_dy = state.dfu_dx, state.dfu_dy
    dfv_dx, dfv_dy = state.dfv_dx, state.dfv_dy
    
    # Transient term multiplier
    inv_dt = state.fac1 / state.dt if state.dt != 0 else 0.0
    
    # Residuals
    r1 = inv_dt * u + fu * u_x + fv * u_y + u * dfu_dx + v * dfu_dy + p_x + state.nu * om_y
    r2 = inv_dt * v + fu * v_x + fv * v_y + u * dfv_dx + v * dfv_dy + p_y - state.nu * om_x
    r3 = u_x + v_y
    r4 = om + u_y - v_x
    
    # Weight by quadrature weights wq
    wq = state.mesh.wq
    
    su = np.zeros_like(U)
    su[..., 0] = r1 * wq
    su[..., 1] = r2 * wq
    su[..., 2] = r3 * wq
    su[..., 3] = r4 * wq
    
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
    
    c = np.zeros_like(su)
    
    # c_1 = (fac1/dt)*su1 + dfu_dx*su1 + dfv_dx*su2 
    #       + Dx^T(su3) + Dx^T(fu*su1) + Dy^T(su4) + Dy^T(fv*su1)
    c[..., 0] = (
        inv_dt * su1 
        + dfu_dx * su1 
        + dfv_dx * su2 
        + DxT(su3, state.D, state.mesh.facx) 
        + DxT(fu * su1, state.D, state.mesh.facx) 
        + DyT(su4, state.D, state.mesh.facy) 
        + DyT(fv * su1, state.D, state.mesh.facy)
    )
    
    # c_2 = (fac1/dt)*su2 + dfu_dy*su1 + dfv_dy*su2 
    #       - Dx^T(su4) + Dx^T(fu*su2) + Dy^T(su3) + Dy^T(fv*su2)
    c[..., 1] = (
        inv_dt * su2 
        + dfu_dy * su1 
        + dfv_dy * su2 
        - DxT(su4, state.D, state.mesh.facx) 
        + DxT(fu * su2, state.D, state.mesh.facx) 
        + DyT(su3, state.D, state.mesh.facy) 
        + DyT(fv * su2, state.D, state.mesh.facy)
    )
    
    # c_3 = Dx^T(su1) + Dy^T(su2)
    c[..., 2] = (
        DxT(su1, state.D, state.mesh.facx) 
        + DyT(su2, state.D, state.mesh.facy)
    )
    
    # c_4 = su4 - nu*Dx^T(su2) + nu*Dy^T(su1)
    c[..., 3] = (
        su4 
        - state.nu * DxT(su2, state.D, state.mesh.facx) 
        + state.nu * DyT(su1, state.D, state.mesh.facy)
    )
    
    return c
