import numpy as np
from lssem2d.mesh import build_bfs
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf

def test():
    nu = 1.0 / 100.0
    N = 4
    mesh = build_bfs(N)
    D = diff_matrix(N)
    
    state = SolverState(mesh, D, nu=nu, dt=0.05, fac1=1.0)
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    U_history = [U_0]
    
    def custom_inlet(x, y, t):
        return 4.0 * y * (1.0 - y)
        
    for step in range(5):
        U_new = step_bdf(state, U_history, time=step*0.05, max_newton=3, newton_tol=1e-3, pin_p=True, custom_inlet=custom_inlet)
        diff = np.max(np.abs(U_history[0] - U_history[1]))
        print(f"Step {step}, Change: {diff:.2e}")

if __name__ == "__main__":
    test()
