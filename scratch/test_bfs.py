import numpy as np
from lssem2d.mesh import build_bfs
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf
import time

def solve_bfs():
    # Re=389 (Armaly geometry). Average inlet velocity = 2/3 (if u_max = 1)
    # Channel height h=1 (inlet), expansion H=2.
    # Armaly defines Re = U_max * (2*h) / nu = 1.0 * 2.0 / nu. Wait, actually Armaly uses 
    # U_avg * (2h) / nu or U_max? Let's just try nu = 1/389 and nu = 2/389 and nu = 2/3 * 2 / 389.
    # Most standard definition: Re = U_max * (2h) / nu, or U_avg * (2h) / nu.
    # Let's set nu = 1.0 / 389.0 to start, with U_max = 1.0. If the length is not 8.0, we can adjust.
    
    # Wait, usually for BFS, the characteristic velocity is U_max and length is h.
    # So Re = U_max * h / nu. So nu = 1.0 / 389.0 is standard.
    nu = 1.0 / 389.0
    N = 6
    
    mesh = build_bfs(N)
    D = diff_matrix(N)
    from lssem2d.lgl import lgl_weights
    wq = lgl_weights(N)
    
    # Pseudo-transient continuation (dt=0.1)
    state = SolverState(mesh, D, nu=nu, dt=0.1, fac1=1.0)
    
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    U_history = [U_0]
    
    def custom_inlet(x, y, t):
        # parabolic profile, max 1.0 at y=0.5. y in [0, 1]
        return 4.0 * y * (1.0 - y)
        
    for step in range(200):
        U_new = step_bdf(state, U_history, time=step*0.1, max_newton=5, newton_tol=1e-4, pin_p=True, custom_inlet=custom_inlet)
        diff = np.max(np.abs(U_new - U_history[0]))
        print(f"Step {step}, Change: {diff:.2e}")
        U_history = [U_new]
        if diff < 1e-4:
            break
            
    # Now find reattachment on bottom wall (y=-1, which is j=0 in the bottom elements)
    # The bottom elements are e in [2 + 10, 2 + 20) = [12, 21].
    wall_x = []
    wall_tau = []
    for e in range(12, 22):
        # calculate du/dy at y=-1 (j=0)
        # using the differentiation matrix D
        # u(y) = sum D_{0, m} u_m
        for i in range(N+1):
            x = mesh.xnod[e, i]
            # y-derivative at j=0
            # map derivative: du/dy = du/deta * (2/H)
            # H of the element is 1.0
            du_deta = np.dot(D[0, :], U_history[0][e, i, :, 0])
            du_dy = du_deta * 2.0
            wall_x.append(x)
            wall_tau.append(du_dy)
            
    # Sort by x
    idx = np.argsort(wall_x)
    wall_x = np.array(wall_x)[idx]
    wall_tau = np.array(wall_tau)[idx]
    
    print("x:", wall_x)
    print("tau:", wall_tau)
    
    # Find zero crossing
    x_reattach = 0.0
    for i in range(len(wall_x)-1):
        if wall_tau[i] < 0 and wall_tau[i+1] > 0:
            # Linear interpolate
            frac = -wall_tau[i] / (wall_tau[i+1] - wall_tau[i])
            x_reattach = wall_x[i] + frac * (wall_x[i+1] - wall_x[i])
            break
            
    print(f"Reattachment length: {x_reattach:.2f}")
    
    # Also calculate mass flux Q(x)
    # Integrate u dy at various x stations
    # Actually just check at boundaries of elements
    for e in [12, 15, 18]:
        # Top and bottom elements share x
        x_st = mesh.xnod[e, -1]
        
        # Integral = sum w_m * u_m * J_y
        Q = 0.0
        # Bottom element
        for j in range(N+1):
            Q += wq[j] * 0.5 * U_history[0][e, -1, j, 0]
        # Top element (e - 10)
        for j in range(N+1):
            Q += wq[j] * 0.5 * U_history[0][e-10, -1, j, 0]
            
        print(f"Mass flux at x={x_st:.2f}: {Q:.5f}")
        
    # Inlet mass flux
    Q_in = 0.0
    for j in range(N+1):
        Q_in += wq[j] * 0.5 * custom_inlet(0, mesh.ynod[0, j], 0)
    print(f"Inlet Mass flux: {Q_in:.5f}")
    
if __name__ == "__main__":
    solve_bfs()
