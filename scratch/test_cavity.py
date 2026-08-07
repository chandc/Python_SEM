import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lssem2d.config import Config

def test_cavity():
    cfg = Config("cavity.toml")
    
    Re = cfg.get("simulation", "Re", 100.0)
    nu = 1.0 / Re
    N = cfg.get("mesh", "N", 8)
    el_x = cfg.get("mesh", "elements_x", 4)
    el_y = cfg.get("mesh", "elements_y", 4)
    L_x = cfg.get("mesh", "L_x", 1.0)
    L_y = cfg.get("mesh", "L_y", 1.0)
    
    # el_x by el_y elements on [0, L_x] x [0, L_y]
    # Boundaries: W=1, E=1, S=1, N=2 (lid)
    mesh = build_channel(L_x, L_y, el_x, el_y, N, bcs=(1, 1, 1, 2))
    
    D = diff_matrix(N)
    
    dt = cfg.get("simulation", "dt", 1e6)
    state = SolverState(mesh, D, nu=nu, dt=dt, fac1=1.0)
    
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    
    t0 = time.time()
    from lssem2d.solver import step_bdf
    
    # We may need multiple bdf steps if Newton stalls, 
    # but let's try a single step with 15 Newton iterations.
    U_history = [U_0]
    
    max_steps = cfg.get("simulation", "max_steps", 5)
    max_newton = cfg.get("solver", "max_newton", 10)
    newton_tol = cfg.get("solver", "newton_tol", 1e-5)
    newton_factor = cfg.get("solver", "newton_factor", 0.1)
    verbose = cfg.get("solver", "verbose", True)
    cgsfac = cfg.get("solver", "cgsfac", 0.0)
    
    for step in range(max_steps):
        print(f"Time step {step}")
        U_new = step_bdf(state, U_history, time=0.0, max_newton=max_newton, newton_tol=newton_tol, newton_factor=newton_factor, pin_p=True, verbose=verbose, cgsfac=cgsfac)
        # Check change against previous state
        diff = np.max(np.abs(U_history[0] - U_history[1]))
        print(f"  -> Step diff (steady state convergence): {diff:.2e}")
            
        if diff < 1e-5:
            print("Reached steady state!")
            break
            
    print(f"Time: {time.time()-t0:.2f}s")
    
    # Extract centerline u at x=0.5
    # For a 4x4 mesh on [0, 1], elements 0,1,2,3 are bottom row.
    # The x=0.5 line is the boundary between the 2nd and 3rd columns of elements!
    # Element columns: x in [0, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]
    # So x=0.5 corresponds to the EAST boundary of column 1, or WEST of column 2.
    
    # Let's gather all (y, u) points at x ~ 0.5
    y_pts = []
    u_pts = []
    for e in range(mesh.nelem):
        for i in range(N+1):
            if abs(mesh.xnod[e, i] - 0.5) < 1e-5:
                y_pts.extend(mesh.ynod[e, :])
                u_pts.extend(U_history[0][e, i, :, 0])
                
    # Sort
    idx = np.argsort(y_pts)
    y_pts = np.array(y_pts)[idx]
    u_pts = np.array(u_pts)[idx]
    
    # Ghia data (y is the same for all Re)
    ghia_y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    ghia_u_100 = np.array([1.0, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0])
    ghia_u_1000 = np.array([1.0, 0.6593, 0.5749, 0.5112, 0.4660, 0.3330, 0.1872, 0.0570, -0.0608, -0.1065, -0.2781, -0.3829, -0.2973, -0.2222, -0.2020, -0.1811, 0.0])
    ghia_u_10000 = np.array([1.0, 0.4722, 0.4778, 0.4801, 0.4789, 0.3476, 0.1569, -0.0019, -0.1118, -0.1605, -0.3429, -0.4600, -0.4077, -0.3016, -0.2521, -0.2065, 0.0])
    
    Re = cfg.get("simulation", "Re", 100.0)
    if Re == 100.0:
        ghia_u = ghia_u_100
    elif Re == 1000.0:
        ghia_u = ghia_u_1000
    elif Re == 10000.0:
        ghia_u = ghia_u_10000
    else:
        # Fallback if Re is something else
        ghia_u = ghia_u_100
        print(f"Warning: No exact Ghia data for Re={Re}, defaulting to Re=100")
    
    # Interpolate our result to Ghia y points
    u_interp = np.interp(ghia_y, y_pts, u_pts)
    
    err = np.max(np.abs(u_interp - ghia_u))
    np.savez("cavity_re1000_data.npz", 
             U_steady=U_history[0], 
             xnod=mesh.xnod, 
             ynod=mesh.ynod, 
             u_pts=u_pts, 
             y_pts=y_pts, 
             ghia_u=ghia_u, 
             ghia_y=ghia_y, 
             Re=Re)
    print("Simulation data saved to cavity_re1000_data.npz")

    
if __name__ == "__main__":
    test_cavity()
