import numpy as np
import time
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L
from lssem2d.assembly import gather_scatter
from lssem2d.solver import compute_jacobi

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

def main():
    N = 10
    print(f"Testing performance and accuracy for N={N} ...")
    mesh = build_channel(1.0, 1.0, 10, 10, N)
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=1.0, dt=1.0)
    
    # Dummy linearisation state
    fu = np.random.rand(mesh.nelem, N+1, N+1)
    fv = np.random.rand(mesh.nelem, N+1, N+1)
    state.update_linearisation(fu, fv)
    
    # 1. Test Old Implementation
    t0 = time.time()
    M_inv_old = compute_jacobi_old(state, fu, fv)
    t1 = time.time()
    old_time = t1 - t0
    
    # 2. Test New DGE Implementation
    t0 = time.time()
    M_inv_new = compute_jacobi(state, fu, fv, extra_shape=False)
    t1 = time.time()
    new_time = t1 - t0
    
    # 3. Compare Accuracy
    diff = np.max(np.abs(M_inv_old - M_inv_new))
    
    print(f"Old approach time: {old_time:.4f}s")
    print(f"New DGE time:      {new_time:.4f}s")
    print(f"Speedup:           {old_time/new_time:.1f}x")
    print(f"Max difference:    {diff:.2e}")
    
    if diff < 1e-10:
        print("Accuracy verified: EXACT match!")
    else:
        print("WARNING: Difference found between implementations!")

if __name__ == "__main__":
    main()
