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
    mask = state.get_global_mask(pin_p=pin_p)
    dU_m = dU * mask
    
    # 2. Forward VVP operator L
    su = apply_L(state, dU_m, fu, fv)
    
    # 3. Transpose VVP operator L^T
    c = apply_LT(state, su, fu, fv)
    
    # 4. Direct stiffness summation Q^T Q
    c_gs = gather_scatter(state.mesh, c)
    
    # 5. Mask again
    return c_gs * mask

def compute_jacobi_old(state, fu, fv, pin_p=False):
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
    mask_field = state.get_global_mask(pin_p=pin_p)
    
    M_inv = np.zeros_like(diag_A)
    valid = mask_field > 0.5
    M_inv[valid] = 1.0 / diag_A[valid]
    
    return M_inv

def compute_jacobi(state, fu, fv, pin_p=False):
    """
    Computes the exact diagonal of the assembled VVP operator A by applying 
    the local element operator L to unit vectors.
    Uses the fully vectorised `dge` formulation.
    Returns M_inv (the inverse of the diagonal).
    """
    m = state.mesh; n = m.N + 1; D = state.D; nu = state.nu
    invdt = state.fac1 / state.dt if state.dt != 0 else 0.0
    I = np.eye(n)
    
    # geometry-independent basis arrays, index [i,j,k1,k2]
    phi  = np.einsum('ik,jl->ijkl', I, I)
    phix = np.einsum('ki,jl->ijkl', D, I)
    phiy = np.einsum('ik,lj->ijkl', I, D)

    fx = m.facx[:, None, None, None, None]
    fy = m.facy[:, None, None, None, None]
    P  = phi[None]
    Px = phix[None] * fx
    Py = phiy[None] * fy
    
    # linearisation quantities at the quadrature point (k1,k2) -> axes (e,1,1,k1,k2)
    e_ = lambda a: a[:, None, None, :, :]
    uo, vo = e_(fu), e_(fv)
    ux, uy = e_(state.dfu_dx), e_(state.dfu_dy)
    vx, vy = e_(state.dfv_dx), e_(state.dfv_dy)

    # Rows 1-2 (momentum) carry the dt weighting applied in lssem.apply_L, so every
    # a_mk = dR_m/dU_k for m in {1,2} is scaled by dt.  Rows 3-4 (continuity, vorticity)
    # are unweighted.  Keeping these consistent with apply_L is mandatory: scaling the
    # operator without scaling this diagonal destroys the preconditioner.
    from .lssem import ls_coeffs
    a_mass, a_flux, _ = ls_coeffs(state)
    conv = uo * Px + vo * Py
    a11 = a_mass * P + a_flux * (conv + ux * P)
    a22 = a_mass * P + a_flux * (conv + vy * P)
    a12 = P * uy * a_flux
    a21 = P * vx * a_flux
    a13, a14 = Px * a_flux,  nu * Py * a_flux
    a23, a24 = Py * a_flux, -nu * Px * a_flux
    
    a31, a32 = Px, Py
    
    a41, a42, a44 = Py, -Px, P

    w = m.wq[:, None, None, :, :]
    diag_A = np.empty((m.nelem, n, n, 4))
    diag_A[..., 0] = np.sum((a11**2 + a21**2 + a31**2 + a41**2) * w, axis=(3, 4))
    diag_A[..., 1] = np.sum((a12**2 + a22**2 + a32**2 + a42**2) * w, axis=(3, 4))
    diag_A[..., 2] = np.sum((a13**2 + a23**2) * w, axis=(3, 4))
    diag_A[..., 3] = np.sum((a14**2 + a24**2 + a44**2) * w, axis=(3, 4))
                
    diag_A = gather_scatter(state.mesh, diag_A)
    mask_field = state.get_global_mask(pin_p=pin_p)
    
    M_inv = np.zeros_like(diag_A)
    valid = mask_field > 0.5
    M_inv[valid] = 1.0 / diag_A[valid]
    
    return M_inv

def pcg_solve(state, b, fu, fv, M_inv, multiplicity_weight, pin_p=False, max_iter=5000, tol=1e-6, cgsfac=0.0, precond=None):
    """
    Matrix-free right-preconditioned Conjugate Gradient solver for A * dU = b.
    A(v) = assemble(mask * apply_LT(apply_L(mask * v)))
    Uses a multiplicity-weighted inner product to guarantee symmetry.
    Absolute convergence test, matching the Fortran: ||r|| < tol.

    precond: optional callable z = precond(r).  Defaults to the diagonal
    (Jacobi) preconditioner z = M_inv * r.  See lssem2d.precond for the
    Chebyshev and p-multigrid alternatives.
    """
    _M = precond if precond is not None else (lambda _r: M_inv * _r)
    b_norm = np.sqrt(np.sum(b * b * multiplicity_weight))
    if b_norm < tol:                       # already converged - nothing to do
        return np.zeros_like(b), 0
        
    target = max(cgsfac * b_norm, tol) if cgsfac > 0 else tol   # ABSOLUTE

    x = np.zeros_like(b)
    r = b.copy()
    z = _M(r)
    p = z.copy()
    
    rho_prev = np.sum(r * z * multiplicity_weight)
    
    for i in range(max_iter):
        Ap = apply_A(state, p, fu, fv, pin_p=pin_p)
        
        alpha_denom = np.sum(p * Ap * multiplicity_weight)
        if abs(alpha_denom) < 1e-20:
            return x, i + 1
            
        alpha = rho_prev / alpha_denom
        x = x + alpha * p
        r = r - alpha * Ap
        
        # Check recursive residual
        r_norm = np.sqrt(np.sum(r * r * multiplicity_weight))
        if r_norm < target:                                  # <-- absolute check
            # TRUE-RESIDUAL SAFEGUARD
            true_r = b - apply_A(state, x, fu, fv, pin_p=pin_p)
            true_r_norm = np.sqrt(np.sum(true_r * true_r * multiplicity_weight))
            if true_r_norm < target:                         # <-- absolute check
                return x, i + 1
            # Drift detected! Restart search direction with true residual
            r = true_r
            z = _M(r)                                        # restart cleanly
            p = z.copy()
            rho_prev = np.sum(r * z * multiplicity_weight)
            continue
        
        z = _M(r)
        rho = np.sum(r * z * multiplicity_weight)
        if abs(rho_prev) < 1e-20:
            return x, i + 1
            
        beta = rho / rho_prev
        p = z + beta * p
        rho_prev = rho
        
    print(f"Warning: PCG did not converge after {max_iter} iterations. Relative residual: {np.sqrt(np.sum(r*r*multiplicity_weight))/b_norm:.2e}")
    return x, max_iter

def newton_step(state, U, su_history, M_inv, multiplicity_weight, time=0.0, f_known=None, custom_inlet=None, custom_lid=None, exact_solution=None, pin_p=False, cg_max_iter=5000, cgsfac=0.0):
    """
    Performs one Newton sub-iteration.
    U: current guess for the new time step, shape (nelem, n, n, 4)
    su_history: weighted historical terms sum(alpha_m / dt U^{n-m} * wq)
    f_known: optional analytical forcing term for MMS (unweighted), shape (nelem, n, n, 4)
    """
    # 0. Enforce Dirichlet boundary conditions EXACTLY each sub-iteration
    U = apply_bc(state.mesh, U, time=time, custom_inlet=custom_inlet, custom_lid=custom_lid, exact_solution=exact_solution, pin_p=pin_p)
    
    # 0.5. Make U continuous at re-entrant corners where boundary edges meet internal edges
    mask_local = apply_mask(state.mesh, np.ones_like(U), pin_p=pin_p)
    mask_global = state.get_global_mask(pin_p=pin_p)
    
    U_correct = U * (1.0 - mask_local)
    count_bnd = gather_scatter(state.mesh, 1.0 - mask_local)
    U_correct_global = gather_scatter(state.mesh, U_correct)
    
    valid = count_bnd > 0.5
    U_correct_global[valid] /= count_bnd[valid]
    U[valid] = U_correct_global[valid]
    
    # 1. Compute nonlinear residual: R = L(U) - su_history (- f_known * wq)
    # The true nonlinear residual has u*u_x + v*u_y. apply_L is the linearised operator
    # which computes fu*u_x + u*dfu_dx. By passing fu=u/2, fv=v/2, we get exactly the true residual!
    state.update_linearisation(U[..., 0] / 2.0, U[..., 1] / 2.0)
    # 2. Compute Exact Diagonal of A (Jacobi Preconditioner)
    M_inv = compute_jacobi(state, U[..., 0] / 2.0, U[..., 1] / 2.0, pin_p=pin_p)
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
    b = -c_gs * mask_global
    
    # 3. Solve A * dU = b
    dU, cg_iters = pcg_solve(state, b, fu, fv, M_inv, multiplicity_weight, pin_p=pin_p, max_iter=cg_max_iter, cgsfac=cgsfac)
    
    # 4. Update U
    U_new = U + dU
    
    return U_new, dU, cg_iters

def step_bdf(state, U_history, time=0.0, max_newton=5, newton_tol=1e-6, newton_factor=0.1, f_known=None, custom_inlet=None, custom_lid=None, exact_solution=None, pin_p=False, cg_max_iter=5000, cgsfac=0.0, verbose=False):
    """
    Advances one time step using BDF1 or BDF2.
    U_history: list of previous states [U_n, U_{n-1}]. 
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
        
    # Build historical source term (only applies to u and v momentum equations).
    # No 1/dt: apply_L now emits momentum rows as fac1*u + dt*(...), so the history
    # term must carry the same dt weighting or the residual mixes two scalings.
    # hist_scale keeps the history term matching a_mass in apply_L.  If they
    # disagree the mass terms stop cancelling at steady state and the solver
    # converges to a modified equation (this is exactly how dt=0 through this
    # routine ends up solving N(u) = 1.5u).
    from .lssem import ls_coeffs
    _, _, hist_scale = ls_coeffs(state)
    su_history = np.zeros_like(U_history[0])
    for m in range(len(alpha)):
        su_history[..., 0] += hist_scale * alpha[m] * U_history[m][..., 0] * state.mesh.wq
        su_history[..., 1] += hist_scale * alpha[m] * U_history[m][..., 1] * state.mesh.wq
        
    # Compute and cache multiplicity weights for the PCG solver inner product
    if not hasattr(state, 'multiplicity_weight'):
        mult = gather_scatter(state.mesh, np.ones_like(U_history[0]))
        state.multiplicity_weight = 1.0 / np.where(mult < 1e-10, 1.0, mult)
        
    # Compute and cache the exact Jacobi preconditioner once per time step
    # We use U_n for the linearization velocities
    fu = U_history[0][..., 0]
    fv = U_history[0][..., 1]
    state.update_linearisation(fu, fv)
    state.M_inv = compute_jacobi(state, fu, fv, pin_p=pin_p)
        
    # Initial guess is the previous state
    U = U_history[0].copy()
    
    du_norm_0 = None
    
    for i in range(max_newton):
        U, dU, cg_iters = newton_step(state, U, su_history, state.M_inv, state.multiplicity_weight, time=time, f_known=f_known, custom_inlet=custom_inlet, custom_lid=custom_lid, exact_solution=exact_solution, pin_p=pin_p, cg_max_iter=cg_max_iter, cgsfac=cgsfac)
        
        du_norm = np.max(np.abs(dU))
        if du_norm_0 is None:
            du_norm_0 = du_norm
            
        if verbose:
            print(f"  Newton {i}: change = {du_norm:.2e}, PCG iters = {cg_iters}")
            
        if du_norm < newton_tol or (du_norm_0 > 0 and du_norm / du_norm_0 <= newton_factor):
            break
            
    # BDF history handoff (mutate in-place)
    U_history.insert(0, U)
    if len(U_history) > 2:
        U_history.pop()
        
    return U
