import numpy as np
from .lssem import apply_L, apply_LT
from .assembly import gather_scatter
from .bc import apply_mask, apply_bc

def apply_A(state, dU, fu, fv, pin_p=False):
    """
    Applies the full linearised LHS operator A = M Q^T Q L^T L M.
    dU: perturbation (nelem, n, n, 4)
    fu, fv: current linearisation velocities
    """
    # 1. Mask known DOFs (Dirichlet BCs)
    dU_m = apply_mask(state.mesh, dU, pin_p=pin_p)
    
    # 2. Forward VVP operator L
    su = apply_L(state, dU_m, fu, fv)
    
    # 3. Transpose VVP operator L^T
    c = apply_LT(state, su, fu, fv)
    
    # 4. Direct stiffness summation Q^T Q
    c_gs = gather_scatter(state.mesh, c)
    
    # 5. Mask again
    return apply_mask(state.mesh, c_gs, pin_p=pin_p)

def compute_jacobi(state, fu, fv, pin_p=False):
    """
    Computes the exact diagonal of the assembled VVP operator A by applying 
    the local element operator L to unit vectors. 
    Returns M_inv (the inverse of the diagonal).
    """
    nelem = state.mesh.nelem
    n = state.mesh.N + 1
    diag_A = np.zeros((nelem, n, n, 4))
    
    U_unit = np.zeros((nelem, n, n, 4))
    wq = state.mesh.wq[..., None]
    
    for k in range(4):
        for i in range(n):
            for j in range(n):
                U_unit.fill(0)
                U_unit[:, i, j, k] = 1.0
                su = apply_L(state, U_unit, fu, fv)
                diag_A[:, i, j, k] = np.sum(su**2 / wq, axis=(1, 2, 3))
                
    diag_A = gather_scatter(state.mesh, diag_A)
    mask_field = apply_mask(state.mesh, np.ones_like(diag_A), pin_p=pin_p)
    
    M_inv = np.zeros_like(diag_A)
    valid = mask_field > 0.5
    M_inv[valid] = 1.0 / diag_A[valid]
    
    return M_inv

def cg_solve(state, b, fu, fv, pin_p=False, max_iter=5000, tol=1e-6):
    """
    Matrix-free right-preconditioned BiCGSTAB solver for A * dU = b.
    A(v) = assemble(mask * apply_LT(apply_L(mask * v)))
    Preconditioner M is the exact diagonal of A.
    """
    b_norm = np.sqrt(np.sum(b * b))
    if b_norm < 1e-14:
        return np.zeros_like(b), 0
        
    M_inv = compute_jacobi(state, fu, fv, pin_p=pin_p)
    
    x = np.zeros_like(b)
    r = b.copy()
    
    r0_star = r.copy()
    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    v = np.zeros_like(b)
    p = np.zeros_like(b)
    
    for i in range(max_iter):
        rho = np.sum(r0_star * r)
        if abs(rho) < 1e-20:
            break
            
        beta = (rho / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        y = M_inv * p
        v = apply_A(state, y, fu, fv, pin_p=pin_p)
        
        denominator = np.sum(r0_star * v)
        if abs(denominator) < 1e-20:
            x = x + alpha * y
            return x, i + 1
        alpha = rho / denominator
        
        s = r - alpha * v
        
        s_norm = np.sqrt(np.sum(s * s))
        if s_norm < tol * b_norm:
            x = x + alpha * y
            return x, i + 1
            
        z = M_inv * s
        t = apply_A(state, z, fu, fv, pin_p=pin_p)
        
        denominator_t = np.sum(t * t)
        if abs(denominator_t) < 1e-20:
            x = x + alpha * y
            return x, i + 1
        omega = np.sum(t * s) / denominator_t
        
        if abs(omega) < 1e-20:
            x = x + alpha * y
            return x, i + 1
        
        x = x + alpha * y + omega * z
        r = s - omega * t
        
        rho_prev = rho
        
        r_norm = np.sqrt(np.sum(r * r))
        if r_norm < tol * b_norm:
            return x, i + 1
            
    print(f"Warning: BiCGSTAB did not converge after {max_iter} iterations. Relative residual: {np.sqrt(np.sum(r*r))/b_norm:.2e}")
    return x, max_iter

def newton_step(state, U, su_history, time=0.0, f_known=None, custom_inlet=None, custom_lid=None, exact_solution=None, pin_p=False):
    """
    Performs one Newton sub-iteration.
    U: current guess for the new time step, shape (nelem, n, n, 4)
    su_history: weighted historical terms sum(alpha_m / dt U^{n-m} * wq)
    f_known: optional analytical forcing term for MMS (unweighted), shape (nelem, n, n, 4)
    """
    # 0. Enforce Dirichlet boundary conditions EXACTLY each sub-iteration
    U = apply_bc(state.mesh, U, time=time, custom_inlet=custom_inlet, custom_lid=custom_lid, exact_solution=exact_solution, pin_p=pin_p)
    
    # 1. Compute nonlinear residual: R = L(U) - su_history (- f_known * wq)
    # The true nonlinear residual has u*u_x + v*u_y. apply_L is the linearised operator
    # which computes fu*u_x + u*dfu_dx. By passing fu=u/2, fv=v/2, we get exactly the true residual!
    state.update_linearisation(U[..., 0] / 2.0, U[..., 1] / 2.0)
    su_nl = apply_L(state, U, U[..., 0] / 2.0, U[..., 1] / 2.0) - su_history
    if f_known is not None:
        wq = state.mesh.wq[..., None]
        su_nl -= f_known * wq
        
    # Now set the linearisation velocities to U for the implicit solve A * dU = b
    fu = U[..., 0]
    fv = U[..., 1]
    state.update_linearisation(fu, fv)
    
    # 2. Form RHS: b = - Q^T Q L^T (R)
    c = apply_LT(state, su_nl, fu, fv)
    c_gs = gather_scatter(state.mesh, c)
    b = -apply_mask(state.mesh, c_gs, pin_p=pin_p)
    
    # 3. Solve A * dU = b
    dU, cg_iters = cg_solve(state, b, fu, fv, pin_p=pin_p)
    
    # 4. Update U
    U_new = U + dU
    
    return U_new, dU, cg_iters

def step_bdf(state, U_history, time=0.0, max_newton=5, newton_tol=1e-6, f_known=None, custom_inlet=None, custom_lid=None, exact_solution=None, pin_p=False):
    """
    Advances one time step using BDF1 or BDF2.
    U_history: list of previous states [U_n, U_{n-1}]. 
               If length 1, uses BDF1. If length 2, uses BDF2.
    """
    dt = state.dt
    wq = state.mesh.wq[..., None]
    
    if len(U_history) == 1:
        # BDF1 (Backward Euler)
        state.fac1 = 1.0
        alpha = [1.0]
    else:
        # BDF2
        state.fac1 = 1.5
        alpha = [2.0, -0.5]
        
    # Build historical source term
    su_history = np.zeros_like(U_history[0])
    for m in range(len(alpha)):
        su_history += (alpha[m] / dt) * U_history[m] * wq
        
    # Initial guess is the previous state
    U = U_history[0].copy()
    
    for i in range(max_newton):
        U, dU, cg_iters = newton_step(state, U, su_history, time=time, f_known=f_known, custom_inlet=custom_inlet, custom_lid=custom_lid, exact_solution=exact_solution, pin_p=pin_p)
        
        du_norm = np.max(np.abs(dU))
        if du_norm < newton_tol:
            break
            
    return U
