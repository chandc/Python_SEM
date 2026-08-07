import numpy as np
import time
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf
from lssem2d.config import Config

def test_cavity():
    cfg = Config("cavity.toml")
    Re = 1000.0
    nu = 1.0 / Re
    N = 8
    el_x, el_y = 4, 4
    
    mesh = build_channel(1.0, 1.0, el_x, el_y, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=nu, dt=0.1, fac1=1.0)
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    
    U_history = [U_0]
    
    t0 = time.time()
    for step in range(20):
        U_new = step_bdf(state, U_history, time=0.0, max_newton=3, newton_tol=1e-5, newton_factor=0.1, pin_p=True, verbose=False, cgsfac=0.3)
        diff = np.max(np.abs(U_history[0] - U_history[1]))
    
    t1 = time.time()
    total_time = t1 - t0
    print(f"Total time for 20 steps: {total_time:.3f}s")
    print(f"Time per step:           {total_time/20 * 1000:.1f}ms")

if __name__ == "__main__":
    test_cavity()
