import numpy as np
import pytest
import sympy as sp
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L

def test_apply_L_manufactured():
    # a) Method of manufactured solutions using sympy
    x, y = sp.symbols('x y')
    
    # Define arbitrary smooth fields as polynomials to avoid truncation error
    u_sym = x**2 * y
    v_sym = x * y**2
    p_sym = x**3 + y**3
    om_sym = x * y
    
    # Linearisation velocities
    fu_sym = x
    fv_sym = y
    
    # Parameters
    nu = 0.1
    inv_dt = 2.0  # fac1 / dt
    
    # Exact residuals
    u_x, u_y = sp.diff(u_sym, x), sp.diff(u_sym, y)
    v_x, v_y = sp.diff(v_sym, x), sp.diff(v_sym, y)
    p_x, p_y = sp.diff(p_sym, x), sp.diff(p_sym, y)
    om_x, om_y = sp.diff(om_sym, x), sp.diff(om_sym, y)
    fu_x, fu_y = sp.diff(fu_sym, x), sp.diff(fu_sym, y)
    fv_x, fv_y = sp.diff(fv_sym, x), sp.diff(fv_sym, y)
    
    r1_sym = inv_dt * u_sym + fu_sym * u_x + fv_sym * u_y + u_sym * fu_x + v_sym * fu_y + p_x + nu * om_y
    r2_sym = inv_dt * v_sym + fu_sym * v_x + fv_sym * v_y + u_sym * fv_x + v_sym * fv_y + p_y - nu * om_x
    r3_sym = u_x + v_y
    r4_sym = om_sym + u_y - v_x
    
    # Compile sympy functions for fast evaluation
    lambdify_args = (x, y)
    eval_u = sp.lambdify(lambdify_args, u_sym, 'numpy')
    eval_v = sp.lambdify(lambdify_args, v_sym, 'numpy')
    eval_p = sp.lambdify(lambdify_args, p_sym, 'numpy')
    eval_om = sp.lambdify(lambdify_args, om_sym, 'numpy')
    
    eval_fu = sp.lambdify(lambdify_args, fu_sym, 'numpy')
    eval_fv = sp.lambdify(lambdify_args, fv_sym, 'numpy')
    
    eval_r1 = sp.lambdify(lambdify_args, r1_sym, 'numpy')
    eval_r2 = sp.lambdify(lambdify_args, r2_sym, 'numpy')
    eval_r3 = sp.lambdify(lambdify_args, r3_sym, 'numpy')
    eval_r4 = sp.lambdify(lambdify_args, r4_sym, 'numpy')
    
    # Set up mesh and numerical state
    N = 8
    mesh = build_channel(L_x=2.0, L_y=2.0, E_x=2, E_y=2, N=N)
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=nu, dt=1.0, fac1=inv_dt)
    
    X, Y = mesh.xnod, mesh.ynod
    X_3d = X[:, :, None]
    Y_3d = Y[:, None, :]
    
    # Broadcast all fields explicitly to (nelem, n, n)
    target_shape = (mesh.nelem, N + 1, N + 1)
    
    U = np.zeros((mesh.nelem, N + 1, N + 1, 4))
    U[..., 0] = np.broadcast_to(eval_u(X_3d, Y_3d), target_shape)
    U[..., 1] = np.broadcast_to(eval_v(X_3d, Y_3d), target_shape)
    U[..., 2] = np.broadcast_to(eval_p(X_3d, Y_3d), target_shape)
    U[..., 3] = np.broadcast_to(eval_om(X_3d, Y_3d), target_shape)
    
    fu = np.broadcast_to(eval_fu(X_3d, Y_3d), target_shape)
    fv = np.broadcast_to(eval_fv(X_3d, Y_3d), target_shape)
    
    state.update_linearisation(fu, fv)
    
    su_num = apply_L(state, U, fu, fv)
    
    # Exact weighted residuals
    wq = mesh.wq
    su_exact = np.zeros_like(U)
    su_exact[..., 0] = eval_r1(X_3d, Y_3d) * wq
    su_exact[..., 1] = eval_r2(X_3d, Y_3d) * wq
    su_exact[..., 2] = eval_r3(X_3d, Y_3d) * wq
    su_exact[..., 3] = eval_r4(X_3d, Y_3d) * wq
    
    np.testing.assert_allclose(su_num, su_exact, atol=1e-10, rtol=1e-10)

def test_stokes_spectral_convergence():
    # b) With (u,v,p,om) an exact solution of the steady Stokes problem and fac1=0,
    # ||su|| is at truncation-error level and decreases spectrally with N.
    
    # Exact Stokes solution: 
    # u = sin(x) cos(y)
    # v = -cos(x) sin(y)
    # p = cos(x) cos(y)
    # om = 2 sin(x) sin(y)
    # With nu = 0.5, this exactly satisfies steady Stokes.
    
    nu = 0.5
    errors = []
    
    Ns = [4, 6, 8, 10, 12]
    for N in Ns:
        mesh = build_channel(L_x=2.0, L_y=2.0, E_x=2, E_y=2, N=N)
        D = diff_matrix(N)
        state = SolverState(mesh, D, nu=nu, dt=1.0, fac1=0.0)
        
        X = mesh.xnod[:, :, None]
        Y = mesh.ynod[:, None, :]
        
        target_shape = (mesh.nelem, N + 1, N + 1)
        
        # Exact Stokes solution (irrotational flow):
        # u = sin(x) exp(-y)
        # v = cos(x) exp(-y)
        # p = 0
        # om = 0
        
        U = np.zeros((mesh.nelem, N + 1, N + 1, 4))
        U[..., 0] = np.broadcast_to(np.sin(X) * np.exp(-Y), target_shape)
        U[..., 1] = np.broadcast_to(np.cos(X) * np.exp(-Y), target_shape)
        U[..., 2] = np.zeros(target_shape)
        U[..., 3] = np.zeros(target_shape)
        
        # Stokes means nonlinear terms are zero (i.e. fu=0, fv=0)
        fu = np.zeros(target_shape)
        fv = np.zeros(target_shape)
        state.update_linearisation(fu, fv)
        
        su_num = apply_L(state, U, fu, fv)
        
        # Unweighted residual (L2 norm)
        # divide by wq to get the actual pointwise residual, then take max
        residual = su_num / mesh.wq[..., None]
        err = np.max(np.abs(residual))
        errors.append(err)
        
    # Check that error is decreasing and gets very small
    assert errors[-1] < 1e-10
    assert errors[-1] < errors[0]
    
    # Check for spectral (exponential) convergence
    # On a log plot, error should decrease linearly or faster with N
    log_err = np.log10(errors)
    slope = (log_err[-1] - log_err[0]) / (Ns[-1] - Ns[0])
    
    # Slope should be significantly negative (spectral convergence)
    assert slope < -0.5
