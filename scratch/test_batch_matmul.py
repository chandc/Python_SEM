import numpy as np
import time
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.operators import dUdx, dUdy, DxT, DyT

def test_apply_L():
    nelem = 400
    N = 8
    n = N + 1
    D = diff_matrix(N)
    facx = np.random.rand(nelem)
    facy = np.random.rand(nelem)
    
    U = np.random.rand(nelem, n, n, 4)
    
    # Old way
    t0 = time.time()
    for _ in range(1000):
        u, v, p, om = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
        u_x = dUdx(u, D, facx)
        u_y = dUdy(u, D, facy)
        v_x = dUdx(v, D, facx)
        v_y = dUdy(v, D, facy)
        p_x = dUdx(p, D, facx)
        p_y = dUdy(p, D, facy)
        om_x = dUdx(om, D, facx)
        om_y = dUdy(om, D, facy)
    t1 = time.time()
    old_time = t1 - t0
    
    # New way
    t0 = time.time()
    for _ in range(1000):
        U_t = np.moveaxis(U, 3, 1) # view
        U_x_all = np.matmul(D, U_t) * facx[:, None, None, None]
        U_y_all = np.matmul(U_t, D.T) * facy[:, None, None, None]
        u_x2, v_x2, p_x2, om_x2 = U_x_all[:, 0], U_x_all[:, 1], U_x_all[:, 2], U_x_all[:, 3]
        u_y2, v_y2, p_y2, om_y2 = U_y_all[:, 0], U_y_all[:, 1], U_y_all[:, 2], U_y_all[:, 3]
    t1 = time.time()
    new_time = t1 - t0
    
    # New way (contiguous)
    t0 = time.time()
    for _ in range(1000):
        U_t = np.ascontiguousarray(np.moveaxis(U, 3, 1))
        U_x_all = np.matmul(D, U_t) * facx[:, None, None, None]
        U_y_all = np.matmul(U_t, D.T) * facy[:, None, None, None]
    t1 = time.time()
    new_time_contig = t1 - t0
    
    # New way (contiguous, pre-allocated)
    t0 = time.time()
    U_t_prealloc = np.empty((nelem, 4, n, n))
    for _ in range(1000):
        U_t_prealloc[...] = np.moveaxis(U, 3, 1)
        U_x_all = np.matmul(D, U_t_prealloc) * facx[:, None, None, None]
        U_y_all = np.matmul(U_t_prealloc, D.T) * facy[:, None, None, None]
        u_x3, v_x3, p_x3, om_x3 = U_x_all[:, 0], U_x_all[:, 1], U_x_all[:, 2], U_x_all[:, 3]
        u_y3, v_y3, p_y3, om_y3 = U_y_all[:, 0], U_y_all[:, 1], U_y_all[:, 2], U_y_all[:, 3]
    t1 = time.time()
    new_time_prealloc = t1 - t0
    
    print(f"apply_L Old:      {old_time:.4f}s")
    print(f"apply_L New:      {new_time:.4f}s")
    print(f"apply_L Contig:   {new_time_contig:.4f}s")
    print(f"apply_L Prealloc: {new_time_prealloc:.4f}s")
    print("Match:", np.allclose(u_x, u_x2) and np.allclose(om_y, om_y2))

def test_apply_LT():
    nelem = 400
    N = 8
    n = N + 1
    D = diff_matrix(N)
    facx = np.random.rand(nelem)
    facy = np.random.rand(nelem)
    
    su1 = np.random.rand(nelem, n, n)
    su2 = np.random.rand(nelem, n, n)
    su3 = np.random.rand(nelem, n, n)
    su4 = np.random.rand(nelem, n, n)
    fu = np.random.rand(nelem, n, n)
    fv = np.random.rand(nelem, n, n)
    
    S_x = np.empty((nelem, 5, n, n))
    S_y = np.empty((nelem, 5, n, n))
    
    t0 = time.time()
    for _ in range(1000):
        dx1 = DxT(su3, D, facx)
        dx2 = DxT(fu*su1, D, facx)
        dx3 = DxT(su4, D, facx)
        dx4 = DxT(fu*su2, D, facx)
        dx5 = DxT(su1, D, facx)
        
        dy1 = DyT(su4, D, facy)
        dy2 = DyT(fv*su1, D, facy)
        dy3 = DyT(su3, D, facy)
        dy4 = DyT(fv*su2, D, facy)
        dy5 = DyT(su2, D, facy)
    t1 = time.time()
    old_time = t1 - t0
    
    t0 = time.time()
    for _ in range(1000):
        S_x[:, 0] = su3
        np.multiply(fu, su1, out=S_x[:, 1])
        S_x[:, 2] = su4
        np.multiply(fu, su2, out=S_x[:, 3])
        S_x[:, 4] = su1
        
        dS_x = np.matmul(D.T, S_x) * facx[:, None, None, None]
        
        S_y[:, 0] = su4
        np.multiply(fv, su1, out=S_y[:, 1])
        S_y[:, 2] = su3
        np.multiply(fv, su2, out=S_y[:, 3])
        S_y[:, 4] = su2
        
        dS_y = np.matmul(S_y, D) * facy[:, None, None, None]
        
    t0 = time.time()
    for _ in range(1000):
        # New way: stack
        S_x_stacked = np.stack([su3, fu*su1, su4, fu*su2, su1], axis=1)
        dS_x = np.matmul(D.T, S_x_stacked) * facx[:, None, None, None]
        
        S_y_stacked = np.stack([su4, fv*su1, su3, fv*su2, su2], axis=1)
        dS_y = np.matmul(S_y_stacked, D) * facy[:, None, None, None]
    t1 = time.time()
    new_time_stack = t1 - t0
    
    print(f"\napply_LT Old:   {old_time:.4f}s")
    print(f"apply_LT Stack: {new_time_stack:.4f}s")
    print("Match:", np.allclose(dx5, dS_x[:, 4]) and np.allclose(dy5, dS_y[:, 4]))

if __name__ == "__main__":
    test_apply_L()
    test_apply_LT()
