import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf
import time

def solve_cavity():
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
    
    t0 = time.time()
    from lssem2d.solver import step_bdf
    
    # We may need multiple bdf steps if Newton stalls, 
    # but let's try a single step with 15 Newton iterations.
    U_history = [U_0]
    
    for step in range(5):
        print(f"Pseudo-step {step}")
        U_new = step_bdf(state, U_history, time=0.0, max_newton=10, newton_tol=1e-5, pin_p=True)
        # Check change
        diff = np.max(np.abs(U_new - U_history[0]))
        print(f"  Change: {diff:.2e}")
        U_history = [U_new]
        if diff < 1e-5:
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
    
    # Ghia Re=100
    ghia_y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    ghia_u = np.array([1.0, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0])
    
    # Interpolate our result to Ghia y points
    u_interp = np.interp(ghia_y, y_pts, u_pts)
    
    err = np.max(np.abs(u_interp - ghia_u))
    print(f"Max error against Ghia: {err:.4f}")
    
if __name__ == "__main__":
    solve_cavity()
