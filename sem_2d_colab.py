"""
Standalone 2D Matrix-Free Spectral Element Method (SEM) Solver
Designed for Google Colab to benchmark NumPy vs. PyTorch (CPU & CUDA).

Test Problem: 2D Poisson Equation on [-1, 1]^2
Exact Solution: u(x,y) = sin(4*pi*x) * sin(4*pi*y)
"""

import numpy as np
import torch
import time
from numpy.polynomial.legendre import Legendre
import argparse

# ---------------------------------------------------------
# 1. GLL Matrix Generator
# ---------------------------------------------------------
def gll(p):
    """Computes Gauss-Lobatto-Legendre nodes and weights."""
    if p == 0:
        return np.array([0.0]), np.array([2.0])
    L_p = Legendre([0]*p + [1])
    dL_p = L_p.deriv()
    roots = dL_p.roots()
    roots = np.sort(np.real(roots))
    x = np.concatenate(([-1.0], roots, [1.0]))
    L_p_x = L_p(x)
    w = 2.0 / (p * (p + 1) * L_p_x**2)
    return x, w

def lagrange_derivative_matrix(p, x):
    """Computes the 1D derivative matrix D_{ij} = l'_j(x_i)."""
    D = np.zeros((p + 1, p + 1))
    L_p = Legendre([0]*p + [1])
    L_p_x = L_p(x)
    for i in range(p + 1):
        for j in range(p + 1):
            if i != j:
                D[i, j] = (L_p_x[i] / L_p_x[j]) / (x[i] - x[j])
            else:
                if i == 0:
                    D[i, j] = -p * (p + 1) / 4.0
                elif i == p:
                    D[i, j] = p * (p + 1) / 4.0
                else:
                    D[i, j] = 0.0
    return D

def get_1d_matrices(p):
    """Returns the 1D Mass (diagonal) and Stiffness matrices on [-1, 1]."""
    x, w = gll(p)
    D = lagrange_derivative_matrix(p, x)
    M = np.diag(w)
    K = D.T @ M @ D
    return x, w, M, K

# ---------------------------------------------------------
# 2. Main 2D Solver Benchmark
# ---------------------------------------------------------
def run_benchmark(p=15, E_x=5, E_y=15, max_iters=2000, tol=1e-11):
    print(f"===========================================================")
    print(f"2D SEM Solver Benchmark (Matrix-Free PCG)")
    print(f"Polynomial Degree (p): {p}")
    print(f"Elements: {E_x} x {E_y} ({E_x * E_y} total)")
    print(f"Global DOFs: {(E_x * p + 1) * (E_y * p + 1)}")
    print(f"===========================================================\n")
    
    # Generate 1D GLL matrices
    x_1d, w_1d, M_1d, K_1d = get_1d_matrices(p)
    
    # 2D Domain setup
    L_x, L_y = 2.0, 2.0
    dx, dy = L_x / E_x, L_y / E_y
    
    # Map reference element [-1, 1] to physical element [0, dx]
    # Actually, global domain is [-1, 1]^2
    # So physical element length is dx, dy.
    # The Jacobian is (dx/2) and (dy/2)
    J_x = dx / 2.0
    J_y = dy / 2.0
    
    # Scaled 1D Matrices
    M_1dx = np.diag(w_1d) * J_x
    M_1dy = np.diag(w_1d) * J_y
    K_1dx = K_1d / J_x
    K_1dy = K_1d / J_y
    
    # Global mesh coordinates
    x_edges = np.linspace(-1.0, 1.0, E_x + 1)
    y_edges = np.linspace(-1.0, 1.0, E_y + 1)
    
    # Test Problem
    def exact_u(x, y):
        return np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
    
    def forcing_f(x, y):
        return 32 * np.pi**2 * np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
    
    # Allocate and populate local forcing and exact solution
    F_local_np = np.zeros((E_x, E_y, p+1, p+1))
    u_exact_local_np = np.zeros((E_x, E_y, p+1, p+1))
    
    for e_x in range(E_x):
        for e_y in range(E_y):
            # Map [-1, 1] to local element coords
            x_local_coords = x_edges[e_x] + (x_1d + 1.0) * J_x
            y_local_coords = y_edges[e_y] + (x_1d + 1.0) * J_y
            for i in range(p + 1):
                for j in range(p + 1):
                    xg = x_local_coords[i]
                    yg = y_local_coords[j]
                    u_exact_local_np[e_x, e_y, i, j] = exact_u(xg, yg)
                    F_local_np[e_x, e_y, i, j] = forcing_f(xg, yg) * (w_1d[i] * J_x) * (w_1d[j] * J_y)
                    
    # Direct Stiffness Summation (NumPy)
    def dss_np(v):
        v_new = v.copy()
        v_new[:-1, :, p, :] += v[1:, :, 0, :]
        v_new[1:, :, 0, :] = v_new[:-1, :, p, :]
        v_new[:, :-1, :, p] += v_new[:, 1:, :, 0]
        v_new[:, 1:, :, 0] = v_new[:, :-1, :, p]
        
        # Dirichlet BCs
        v_new[0, :, 0, :] = 0.0
        v_new[-1, :, p, :] = 0.0
        v_new[:, 0, :, 0] = 0.0
        v_new[:, -1, :, p] = 0.0
        return v_new

    # Assembly of right hand side
    b_np = dss_np(F_local_np)
    
    # Diagonal Preconditioner
    diag_M_x = np.diag(M_1dx)
    diag_M_y = np.diag(M_1dy)
    diag_K_x = np.diag(K_1dx)
    diag_K_y = np.diag(K_1dy)
    D_local_np = np.zeros((E_x, E_y, p+1, p+1))
    for e_x in range(E_x):
        for e_y in range(E_y):
            for i in range(p+1):
                for j in range(p+1):
                    D_local_np[e_x, e_y, i, j] = diag_K_x[i] * diag_M_y[j] + diag_M_x[i] * diag_K_y[j]
                    
    D_global_np = dss_np(D_local_np)
    inv_D_np = np.zeros_like(D_global_np)
    mask = D_global_np > 1e-14
    inv_D_np[mask] = 1.0 / D_global_np[mask]
    
    # Weight matrix for global inner products
    W_np = np.ones((E_x, E_y, p+1, p+1))
    W_np[1:, :, 0, :] = 0.0
    W_np[:, 1:, :, 0] = 0.0
    
    # ---------------------------------------------------------
    # NumPy Solver
    # ---------------------------------------------------------
    def apply_K_np(u):
        v_local = K_1dx @ u @ M_1dy.T + M_1dx @ u @ K_1dy.T
        return dss_np(v_local)

    def cg_solve_np(b):
        x = np.zeros_like(b)
        r = b.copy()
        z = r * inv_D_np
        p_vec = z.copy()
        rsold = np.sum(r * z * W_np)
        
        iters = 0
        for _ in range(max_iters):
            iters += 1
            Ap = apply_K_np(p_vec)
            pAp = np.sum(p_vec * Ap * W_np)
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * inv_D_np
            rsnew = np.sum(r * z_new * W_np)
            
            if np.sqrt(rsnew) < tol:
                break
                
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
            
        return x, iters

    print("Running NumPy CPU Benchmark...")
    # Warmup
    _ = cg_solve_np(b_np)
    
    t0 = time.time()
    u_np, iters_np = cg_solve_np(b_np)
    t1 = time.time()
    
    err_np = np.max(np.abs((u_np - u_exact_local_np)))
    time_np = t1 - t0
    print(f"  [NumPy] Iters: {iters_np:4d} | Time: {time_np:.5f}s | Max Error: {err_np:.2e}\n")
    
    
    # ---------------------------------------------------------
    # PyTorch Solver Factory
    # ---------------------------------------------------------
    def run_pytorch(device_name):
        device = torch.device(device_name)
        
        pt_M_1dx = torch.tensor(M_1dx, dtype=torch.float64, device=device)
        pt_K_1dx = torch.tensor(K_1dx, dtype=torch.float64, device=device)
        pt_M_1dy = torch.tensor(M_1dy, dtype=torch.float64, device=device)
        pt_K_1dy = torch.tensor(K_1dy, dtype=torch.float64, device=device)
        pt_inv_D = torch.tensor(inv_D_np, dtype=torch.float64, device=device)
        pt_W = torch.tensor(W_np, dtype=torch.float64, device=device)
        pt_b = torch.tensor(b_np, dtype=torch.float64, device=device)
        
        pt_mask = torch.zeros((E_x, E_y, p+1, p+1), dtype=torch.bool, device=device)
        pt_mask[0, :, 0, :] = True
        pt_mask[-1, :, p, :] = True
        pt_mask[:, 0, :, 0] = True
        pt_mask[:, -1, :, p] = True

        # Ensure CUDA syncs properly for timing
        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize()

        def dss_pt(v):
            update_x_shift = torch.cat((v[1:, :, 0:1, :], torch.zeros(1, E_y, 1, p+1, dtype=torch.float64, device=device)), dim=0)
            update_x_unshift = torch.cat((torch.zeros(1, E_y, 1, p+1, dtype=torch.float64, device=device), v[:-1, :, p:p+1, :]), dim=0)
            v_new = v.clone()
            v_new[:, :, p:p+1, :] += update_x_shift
            v_new[:, :, 0:1, :] += update_x_unshift
            
            update_y_shift = torch.cat((v_new[:, 1:, :, 0:1], torch.zeros(E_x, 1, p+1, 1, dtype=torch.float64, device=device)), dim=1)
            update_y_unshift = torch.cat((torch.zeros(E_x, 1, p+1, 1, dtype=torch.float64, device=device), v_new[:, :-1, :, p:p+1]), dim=1)
            
            v_final = v_new + update_y_shift
            v_final[:, :, :, 0:1] = v_new[:, :, :, 0:1] + update_y_unshift[:, :, :, 0:1] # Add unshift properly
            
            # Quicker DSS using PyTorch tensors directly
            # Re-write DSS to avoid complex cats in tight loop if possible, 
            # but this closely matches Numpy logic
            v_fast = v.clone()
            v_fast[:-1, :, p, :] += v[1:, :, 0, :]
            v_fast[1:, :, 0, :] = v_fast[:-1, :, p, :]
            v_fast[:, :-1, :, p] += v_fast[:, 1:, :, 0]
            v_fast[:, 1:, :, 0] = v_fast[:, :-1, :, p]
            
            return torch.where(pt_mask, torch.tensor(0.0, dtype=torch.float64, device=device), v_fast)

        def apply_K_pt(u):
            v_local = torch.matmul(torch.matmul(pt_K_1dx, u), pt_M_1dy.T) + torch.matmul(torch.matmul(pt_M_1dx, u), pt_K_1dy.T)
            return dss_pt(v_local)

        def cg_solve_pt(b):
            x = torch.zeros_like(b)
            r = b.clone()
            z = r * pt_inv_D
            p_vec = z.clone()
            rsold = torch.sum(r * z * pt_W)
            
            iters = 0
            for _ in range(max_iters):
                iters += 1
                Ap = apply_K_pt(p_vec)
                pAp = torch.sum(p_vec * Ap * pt_W)
                alpha = 0.0 if pAp < 1e-25 else rsold / pAp
                x = x + alpha * p_vec
                r = r - alpha * Ap
                z_new = r * pt_inv_D
                rsnew = torch.sum(r * z_new * pt_W)
                
                if torch.sqrt(rsnew) < tol:
                    break
                    
                beta = 0.0 if rsold < 1e-25 else rsnew / rsold
                p_vec = z_new + beta * p_vec
                rsold = rsnew
                
            return x, iters

        # Standard PyTorch Benchmark
        print(f"Running PyTorch {device_name.upper()} Benchmark...")
        try:
            # Warmup
            sync()
            _ = cg_solve_pt(pt_b)
            sync()
            
            t0 = time.time()
            u_pt, iters_pt = cg_solve_pt(pt_b)
            sync()
            t1 = time.time()
            
            pt_u_exact = torch.tensor(u_exact_local_np, dtype=torch.float64, device=device)
            err_pt = torch.max(torch.abs(u_pt - pt_u_exact)).item()
            time_pt = t1 - t0
            print(f"  [PyTorch {device_name.upper()}] Iters: {iters_pt:4d} | Time: {time_pt:.5f}s | Max Error: {err_pt:.2e}\n")
            
            # If CUDA, also try torch.compile to fuse kernels and remove Python overhead
            if device_name == 'cuda':
                print(f"Running PyTorch {device_name.upper()} (Compiled) Benchmark...")
                # torch.compile requires PyTorch 2.0+
                if hasattr(torch, 'compile'):
                    cg_solve_pt_compiled = torch.compile(cg_solve_pt)
                    # Warmup (compilation happens here, can take time)
                    sync()
                    _ = cg_solve_pt_compiled(pt_b)
                    sync()
                    
                    t0 = time.time()
                    u_pt_c, iters_pt_c = cg_solve_pt_compiled(pt_b)
                    sync()
                    t1 = time.time()
                    
                    err_pt_c = torch.max(torch.abs(u_pt_c - pt_u_exact)).item()
                    time_pt_c = t1 - t0
                    print(f"  [PyTorch {device_name.upper()} Compiled] Iters: {iters_pt_c:4d} | Time: {time_pt_c:.5f}s | Max Error: {err_pt_c:.2e}\n")
                else:
                    print("  [PyTorch Compiled] Skipped: torch.compile not available.\n")
                    
        except Exception as e:
            print(f"  [PyTorch {device_name.upper()}] Failed: {str(e)}\n")

    # Run PyTorch CPU
    run_pytorch('cpu')
    
    # Run PyTorch CUDA if available
    if torch.cuda.is_available():
        run_pytorch('cuda')
    else:
        print("PyTorch CUDA is NOT available on this machine. (Skipping)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Matrix-Free SEM Colab Benchmark")
    parser.add_argument("-p", "--degree", type=int, default=15, help="Polynomial degree (default: 15)")
    parser.add_argument("-ex", "--elem_x", type=int, default=5, help="Elements in x-direction (default: 5)")
    parser.add_argument("-ey", "--elem_y", type=int, default=15, help="Elements in y-direction (default: 15)")
    args = parser.parse_args()
    
    run_benchmark(p=args.degree, E_x=args.elem_x, E_y=args.elem_y)
