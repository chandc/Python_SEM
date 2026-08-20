import pytest
import numpy as np
import sympy as sp
from lssem2d.mesh import build_channel, build_bfs
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L
from lssem2d.solver import step_bdf
from lssem2d.bc import apply_mask

def test_mms_convergence():
    """
    Manufactured Solution (spectral convergence).
    Pick a smooth exact (u,v,p,om) that satisfies no-slip on [0,1]x[0,1].
    Measure the L2 error vs N = 4,6,8,10,12.
    """
    x, y = sp.symbols('x y')
    nu_sym = 0.1
    
    # Smooth exact solution, zero on boundaries of [0, 1] x [0, 1]
    u_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    v_sym = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    p_sym = sp.cos(sp.pi * x) * sp.cos(sp.pi * y)
    om_sym = sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y)
    
    u_x = sp.diff(u_sym, x); u_y = sp.diff(u_sym, y)
    v_x = sp.diff(v_sym, x); v_y = sp.diff(v_sym, y)
    p_x = sp.diff(p_sym, x); p_y = sp.diff(p_sym, y)
    om_x = sp.diff(om_sym, x); om_y = sp.diff(om_sym, y)
    
    f1_sym = u_sym * u_x + v_sym * u_y + p_x + nu_sym * om_y
    f2_sym = u_sym * v_x + v_sym * v_y + p_y - nu_sym * om_x
    f3_sym = u_x + v_y
    f4_sym = om_sym + u_y - v_x
    
    funcs = [sp.lambdify((x, y), expr, 'numpy') for expr in (u_sym, v_sym, p_sym, om_sym, f1_sym, f2_sym, f3_sym, f4_sym)]
    u_fn, v_fn, p_fn, om_fn, f1_fn, f2_fn, f3_fn, f4_fn = funcs
    
    errors = []
    Ns = [4, 6, 8, 10, 12]
    
    for N in Ns:
        mesh = build_channel(1.0, 1.0, 2, 2, N, bcs=(1, 1, 1, 1)) # All no-slip
        D = diff_matrix(N)
        state = SolverState(mesh, D, nu=nu_sym, dt=1e6, fac1=1.0) # approx steady state
        
        X = np.zeros((mesh.nelem, N+1, N+1))
        Y = np.zeros((mesh.nelem, N+1, N+1))
        for e in range(mesh.nelem):
            X[e, :, :], Y[e, :, :] = np.meshgrid(mesh.xnod[e, :], mesh.ynod[e, :], indexing='ij')
            
        U_exact = np.zeros((mesh.nelem, N+1, N+1, 4))
        U_exact[..., 0] = u_fn(X, Y)
        U_exact[..., 1] = v_fn(X, Y)
        U_exact[..., 2] = p_fn(X, Y)
        U_exact[..., 3] = om_fn(X, Y)
        
        f_known = np.zeros_like(U_exact)
        f_known[..., 0] = f1_fn(X, Y)
        f_known[..., 1] = f2_fn(X, Y)
        f_known[..., 2] = f3_fn(X, Y)
        f_known[..., 3] = f4_fn(X, Y)
        
        # Start from exact solution to measure truncation error directly (which is what solver returns)
        # Or start from 0 and do 2-3 Newton iterations.
        # Starting from U_exact means the Newton update dU is exactly the truncation error!
        U_0 = U_exact.copy()
        
        # We must gather-scatter the exact solution because MMS isn't perfectly continuous in finite N,
        # wait, the exact analytical solution IS perfectly continuous!
        
        # Run 1 BDF step (which uses newton_step).
        # We need to pin pressure because all boundaries are Dirichlet velocity, so p is singular.
        # But for MMS, pressure is uniquely defined by U_exact if we start from U_exact!
        # Actually, let's just do 1 Newton step to get the residual/update.
        from lssem2d.solver import newton_step
        from lssem2d.assembly import gather_scatter
        su_history = np.zeros_like(U_0)

        # First Newton step.  M_inv is ignored by newton_step (it rebuilds the
        # preconditioner internally); multiplicity_weight is required.  This
        # call was missing both positionals for the life of the file -- a
        # latent TypeError, never hit because this script is not
        # pytest-collected (no test_ prefix).
        mult = gather_scatter(state.mesh, np.ones_like(U_0))
        mw = 1.0 / np.where(mult < 1e-10, 1.0, mult)
        U_1, dU, iters = newton_step(state, U_0, su_history, None, mw, time=0.0, f_known=f_known, pin_p=True)
        
        err = np.sqrt(np.sum(dU**2) / np.sum(U_exact**2))
        errors.append(err)
        print(f"N={N}, err={err:.2e}, iters={iters}")
        
    # Check spectral convergence
    assert errors[-1] < 1e-5
    assert errors[0] / errors[-1] > 100 # At least 2 orders of magnitude drop

def test_kovasznay():
    """
    Kovasznay Flow (Re=40).
    L2 velocity error < 1e-6 at N=10 on a 4x4 mesh.
    Domain is [-0.5, 1.0] x [-0.5, 0.5] as standard, but we'll use a channel geometry.
    Wait, let's just use [0, 2] x [-0.5, 0.5] to match typical literature.
    """
    Re = 40.0
    nu = 1.0 / Re
    
    # Kovasznay lambda
    lam = Re/2.0 - np.sqrt(Re**2 / 4.0 + 4*np.pi**2)
    
    def exact_solution(x, y, t=0):
        u = 1.0 - np.exp(lam * x) * np.cos(2 * np.pi * y)
        v = (lam / (2 * np.pi)) * np.exp(lam * x) * np.sin(2 * np.pi * y)
        p = -0.5 * np.exp(2 * lam * x)
        
        u_y = np.exp(lam * x) * np.sin(2 * np.pi * y) * 2 * np.pi
        v_x = (lam**2 / (2 * np.pi)) * np.exp(lam * x) * np.sin(2 * np.pi * y)
        om = v_x - u_y
        
        return u, v, p, om
        
    N = 10
    # Channel from x in [-0.5, 1.5], y in [-0.5, 0.5] (Length 2, Height 1)
    mesh = build_channel(2.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 1)) # All Dirichlet
    
    # Shift domain to [-0.5, 1.5] x [-0.5, 0.5]
    mesh.x0 -= 0.5
    mesh.y0 -= 0.5
    mesh.setup_derived()
    
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=nu, dt=1e6, fac1=1.0) # steady state
    
    # Compare with exact solution
    X = np.zeros((mesh.nelem, N+1, N+1))
    Y = np.zeros((mesh.nelem, N+1, N+1))
    for e in range(mesh.nelem):
        X[e, :, :], Y[e, :, :] = np.meshgrid(mesh.xnod[e, :], mesh.ynod[e, :], indexing='ij')
        
    u_ex, v_ex, p_ex, om_ex = exact_solution(X, Y)
    
    # We solve starting from the exact solution to measure truncation error
    # This avoids the need for pseudo-transient continuation or a preconditioner
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    U_0[..., 0] = u_ex
    U_0[..., 1] = v_ex
    U_0[..., 2] = p_ex
    U_0[..., 3] = om_ex
    
    su_history = np.zeros_like(U_0)
    from lssem2d.solver import step_bdf
    
    U_final = step_bdf(state, [U_0], time=0.0, max_newton=5, newton_tol=1e-8, exact_solution=exact_solution, pin_p=True)
    
    # The U_final will be U_0 + dU (the truncation error)
    # We compare U_final to u_ex directly
    
    # Wait, pressure is only determined up to a constant when all boundaries are Dirichlet.
    # We should shift p_ex and p to have zero mean.
    p = U_final[..., 2]
    p -= np.mean(p)
    p_ex -= np.mean(p_ex)
    
    u_err = np.sqrt(np.sum((U_final[..., 0] - u_ex)**2) / np.sum(u_ex**2))
    v_err = np.sqrt(np.sum((U_final[..., 1] - v_ex)**2) / max(np.sum(v_ex**2), 1e-12))
    
    print(f"Kovasznay (N={N}): u_err={u_err:.2e}, v_err={v_err:.2e}")
    assert u_err < 1e-6
    assert v_err < 1e-6

def test_lid_driven_cavity():
    """
    Lid-Driven Cavity (Re=100)
    centreline velocity profiles match Ghia et al. (1982) to within 2%.
    """
    Re = 100.0
    nu = 1.0 / Re
    N = 8
    
    # 4x4 elements on [0, 1] x [0, 1]
    # Boundaries: W=1, E=1, S=1, N=2 (lid)
    mesh = build_channel(1.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    
    # Large dt for pseudo-steady solve
    state = SolverState(mesh, D, nu=nu, dt=1e6, fac1=1.0)
    
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    U_history = [U_0]
    
    from lssem2d.solver import step_bdf
    for step in range(5):
        U_new = step_bdf(state, U_history, time=0.0, max_newton=10, newton_tol=1e-5, pin_p=True)
        diff = np.max(np.abs(U_history[0] - U_history[1]))
        if diff < 1e-5:
            break
            
    # Gather (y, u) points at x ~ 0.5
    y_pts = []
    u_pts = []
    for e in range(mesh.nelem):
        for i in range(N+1):
            if abs(mesh.xnod[e, i] - 0.5) < 1e-5:
                y_pts.extend(mesh.ynod[e, :])
                u_pts.extend(U_history[0][e, i, :, 0])
                
    idx = np.argsort(y_pts)
    y_pts = np.array(y_pts)[idx]
    u_pts = np.array(u_pts)[idx]
    
    ghia_y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    ghia_u = np.array([1.0, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0])
    
    u_interp = np.interp(ghia_y, y_pts, u_pts)
    err = np.max(np.abs(u_interp - ghia_u))
    print(f"Cavity Re=100 max error against Ghia: {err:.4f}")
    assert err < 0.02
