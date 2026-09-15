import numpy as np
import time
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L
from lssem2d.assembly import gather_scatter
from lssem2d.bc import apply_mask

def test_compute_jacobi():
    N = 4
    mesh = build_channel(1.0, 1.0, 2, 2, N)
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=1.0, dt=1.0)
    
    fu = np.zeros((mesh.nelem, N+1, N+1))
    fv = np.zeros((mesh.nelem, N+1, N+1))
    state.update_linearisation(fu, fv)
    
    nelem = mesh.nelem
    n = N + 1
    diag_A = np.zeros((nelem, n, n, 4))
    
    U_unit = np.zeros((nelem, n, n, 4))
    wq = mesh.wq[..., None]
    
    t0 = time.time()
    for k in range(4):
        for i in range(n):
            for j in range(n):
                U_unit.fill(0)
                U_unit[:, i, j, k] = 1.0
                
                su = apply_L(state, U_unit, fu, fv)
                diag_A[:, i, j, k] = np.sum(su**2 * wq, axis=(1, 2, 3))
                
    t1 = time.time()
    print(f"Time to compute diagonal explicitly: {t1-t0:.4f}s")
    
    diag_A = gather_scatter(mesh, diag_A)
    
    mask_field = apply_mask(mesh, np.ones_like(diag_A), pin_p=False)
    
    M_inv = np.zeros_like(diag_A)
    valid = mask_field > 0.5
    M_inv[valid] = 1.0 / diag_A[valid]
    
    print("M_inv stats:")
    print("Min:", np.min(M_inv[valid]))
    print("Max:", np.max(M_inv[valid]))

if __name__ == "__main__":
    test_compute_jacobi()
