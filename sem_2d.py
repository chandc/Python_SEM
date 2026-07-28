"""
2D Spectral Element Method (SEM) Solver using Matrix-Free Tensor Contraction 
and Direct Stiffness Summation (DSS).

This module solves the 2D Poisson equation:
    -∇²u = f   on Ω = [-1, 1] × [-1, 1]
    u = 0      on ∂Ω (Dirichlet boundaries)

The solution is approximated using high-order Gauss-Lobatto-Legendre (GLL) spectral elements.
By utilizing the tensor-product property of the quad-element basis functions, the global sparse
matrix assembly is completely bypassed. The application of the stiffness matrix A * u is computed
locally via 1D tensor contractions, and C^0 continuity is enforced via Direct Stiffness Summation (DSS).
"""

import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import mlx.core as mx
import torch
import subprocess
import os
import re
from export_matrices import export_matrices

def gll(p):
    if p == 0:
        return np.array([0.0]), np.array([2.0])
    L_p = legendre(p)
    dL_p = L_p.deriv()
    roots = np.roots(dL_p.coeffs)
    roots = np.sort(np.real(roots))
    x = np.concatenate(([-1.0], roots, [1.0]))
    L_p_x = L_p(x)
    w = 2.0 / (p * (p + 1) * L_p_x**2)
    return x, w

def lagrange_derivative_matrix(p, x):
    D = np.zeros((p + 1, p + 1))
    L_p = legendre(p)
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

def build_1d_local_matrices(p, dx):
    """
    Constructs the 1D local Mass and Stiffness matrices for a spectral element.
    
    Equations:
        - Mass matrix (diagonal due to GLL quadrature): 
          M_{ii} = w_i * (dx / 2)
        - Stiffness matrix: 
          K = D^T * M * D * (2/dx)^2
          where D is the Lagrange derivative matrix.
          
    Args:
        p (int): Polynomial degree.
        dx (float): Element width in the physical domain.
        
    Returns:
        tuple: (M_1d, K_1d, x_gll) where M_1d and K_1d are (p+1)x(p+1) matrices.
    """
    x_gll, w_gll = gll(p)
    D = lagrange_derivative_matrix(p, x_gll)
    M_1d = np.diag(w_gll) * (dx / 2.0)
    K_1d = D.T @ M_1d @ D * (2.0 / dx)**2
    return M_1d, K_1d, x_gll

def run_sem_2d(p, E_x=6, E_y=6):
    """
    Runs the full 2D Spectral Element Method solver using a matrix-free algorithm.
    
    Instead of building a global N x N stiffness matrix, the action of the stiffness
    operator is evaluated on localized element arrays of shape (E_x, E_y, p+1, p+1).
    """
    mx.set_default_device(mx.cpu) 
    
    def exact_u(x, y):
        return np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y)
    def forcing_f(x, y):
        return 32.0 * np.pi**2 * np.sin(4.0 * np.pi * x) * np.sin(4.0 * np.pi * y)

    num_elements = E_x * E_y
    L_x, L_y = 2.0, 2.0
    dx = L_x / E_x
    dy = L_y / E_y
    
    N_x_global = E_x * p + 1
    N_y_global = E_y * p + 1
    num_global_nodes = N_x_global * N_y_global
    
    M_1dx, K_1dx, x_gll = build_1d_local_matrices(p, dx)
    M_1dy, K_1dy, y_gll = build_1d_local_matrices(p, dy)
    
    x_local = np.zeros((E_x, E_y, p + 1, p + 1), dtype=np.float64)
    y_local = np.zeros((E_x, E_y, p + 1, p + 1), dtype=np.float64)
    F_local_np = np.zeros((E_x, E_y, p + 1, p + 1), dtype=np.float64)
    
    for ey in range(E_y):
        for ex in range(E_x):
            x_L = -1.0 + ex * dx
            y_L = -1.0 + ey * dy
            x_phys = x_L + (x_gll + 1.0) * (dx / 2.0)
            y_phys = y_L + (y_gll + 1.0) * (dy / 2.0)
            
            X_grid, Y_grid = np.meshgrid(x_phys, y_phys)
            x_local[ex, ey, :, :] = X_grid
            y_local[ex, ey, :, :] = Y_grid
            
            F_mat = np.zeros((p+1, p+1))
            for j in range(p+1):
                for i in range(p+1):
                    F_mat[j, i] = M_1dy[j, j] * M_1dx[i, i] * forcing_f(x_phys[i], y_phys[j])
            F_local_np[ex, ey, :, :] = F_mat

    W_np = np.ones((E_x, E_y, p + 1, p + 1), dtype=np.float64)
    W_np[:-1, :, :, p] /= 2.0
    W_np[1:, :, :, 0] /= 2.0
    W_np[:, :-1, p, :] /= 2.0
    W_np[:, 1:, 0, :] /= 2.0

    def dss_np(v):
        """
        Direct Stiffness Summation (DSS) in NumPy.
        
        Enforces C^0 continuity across element boundaries. Since elements share 
        interface nodes, this function takes the localized un-assembled residuals 
        and sums the contributions from neighboring elements across the X and Y edges.
        
        It operates strictly in-place (conceptually) without needing a global topological
        mapping array (Q).
        
        Args:
            v (ndarray): Local element array of shape (E_x, E_y, p+1, p+1)
        Returns:
            ndarray: The assembled array with continuous interface values.
        """
        v_new = v.copy()
        v_new[:-1, :, :, p] += v[1:, :, :, 0]
        v_new[1:, :, :, 0] += v[:-1, :, :, p]
        
        v_final = v_new.copy()
        v_final[:, :-1, p, :] += v_new[:, 1:, 0, :]
        v_final[:, 1:, 0, :] += v_new[:, :-1, p, :]
        
        v_final[0, :, :, 0] = 0.0
        v_final[-1, :, :, p] = 0.0
        v_final[:, 0, 0, :] = 0.0
        v_final[:, -1, p, :] = 0.0
        return v_final

    F_local_np = dss_np(F_local_np)
    
    def apply_K_np(u):
        """
        Matrix-free application of the Stiffness Matrix operator A * u.
        
        Equation:
            A_e * u_e = K_{1dx} * u_e * M_{1dy}^T + M_{1dx} * u_e * K_{1dy}^T
            
        This tensor contraction evaluates the 2D discrete Laplacian on the local element
        grid in O(p^3) time instead of O(p^4). The result is then assembled via DSS.
        """
        v_local = K_1dx @ u @ M_1dy.T + M_1dx @ u @ K_1dy.T
        return dss_np(v_local)

    diag_M_x = np.diag(M_1dx)
    diag_K_x = np.diag(K_1dx)
    diag_M_y = np.diag(M_1dy)
    diag_K_y = np.diag(K_1dy)
    D_element = np.outer(diag_M_y, diag_K_x) + np.outer(diag_K_y, diag_M_x)
    D_local = np.tile(D_element, (E_x, E_y, 1, 1))
    D_global = dss_np(D_local)
    
    mask_boundary = np.zeros_like(D_global, dtype=bool)
    mask_boundary[0, :, :, 0] = True
    mask_boundary[-1, :, :, p] = True
    mask_boundary[:, 0, 0, :] = True
    mask_boundary[:, -1, p, :] = True
    
    inv_D_np = np.zeros_like(D_global)
    np.divide(1.0, D_global, out=inv_D_np, where=(D_global > 1e-14))
    inv_D_np[mask_boundary] = 0.0

    def cg_solve_np(b, max_iters):
        """
        Preconditioned Conjugate Gradient (PCG) solver for NumPy using Jacobi diagonal scaling.
        """
        x = np.zeros_like(b)
        r = b.copy()
        z = r * inv_D_np
        p_vec = z.copy()
        rsold = np.sum(r * z * W_np)
        for _ in range(max_iters):
            Ap = apply_K_np(p_vec)
            pAp = np.sum(p_vec * Ap * W_np)
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * inv_D_np
            rsnew = np.sum(r * z_new * W_np)
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
        return x

    mx_M_1dx = mx.array(M_1dx, dtype=mx.float64)
    mx_K_1dx = mx.array(K_1dx, dtype=mx.float64)
    mx_M_1dy = mx.array(M_1dy, dtype=mx.float64)
    mx_K_1dy = mx.array(K_1dy, dtype=mx.float64)
    mx_W = mx.array(W_np, dtype=mx.float64)
    F_mlx = mx.array(F_local_np, dtype=mx.float64)
    
    def dss_mlx(v):
        update_p = mx.concatenate([v[1:, :, :, 0:1], mx.zeros((1, E_y, p+1, 1))], axis=0)
        update_0 = mx.concatenate([mx.zeros((1, E_y, p+1, 1)), v[:-1, :, :, p:p+1]], axis=0)
        update_x = mx.concatenate([update_0, mx.zeros((E_x, E_y, p+1, p-1)), update_p], axis=3)
        v_new = v + update_x
        
        update_p_y = mx.concatenate([v_new[:, 1:, 0:1, :], mx.zeros((E_x, 1, 1, p+1))], axis=1)
        update_0_y = mx.concatenate([mx.zeros((E_x, 1, 1, p+1)), v_new[:, :-1, p:p+1, :]], axis=1)
        update_y = mx.concatenate([update_0_y, mx.zeros((E_x, E_y, p-1, p+1)), update_p_y], axis=2)
        v_final = v_new + update_y
        
        mask_boundary = np.zeros((E_x, E_y, p+1, p+1), dtype=bool)
        mask_boundary[0, :, :, 0] = True
        mask_boundary[-1, :, :, p] = True
        mask_boundary[:, 0, 0, :] = True
        mask_boundary[:, -1, p, :] = True
        return mx.where(mx.array(mask_boundary), mx.array(0.0, dtype=mx.float64), v_final)
    
    def apply_K_mlx(u):
        v_local = mx.matmul(mx.matmul(mx_M_1dx, u), mx_K_1dy.T) + mx.matmul(mx.matmul(mx_K_1dx, u), mx_M_1dy.T)
        return dss_mlx(v_local)

    inv_D_mlx = mx.array(inv_D_np, dtype=mx.float64)

    @mx.compile
    def cg_solve_mlx(b, max_iters):
        x = mx.zeros_like(b)
        r = b
        z = r * inv_D_mlx
        p_vec = z
        rsold = mx.sum(r * z * mx_W)
        for _ in range(max_iters):
            Ap = apply_K_mlx(p_vec)
            pAp = mx.sum(p_vec * Ap * mx_W)
            alpha = mx.where(pAp < 1e-25, mx.array(0.0, dtype=mx.float64), rsold / pAp)
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * inv_D_mlx
            rsnew = mx.sum(r * z_new * mx_W)
            beta = mx.where(rsold < 1e-25, mx.array(0.0, dtype=mx.float64), rsnew / rsold)
            p_vec = z_new + beta * p_vec
            rsold = rsnew
        return x

    max_iters = min(400, num_global_nodes)
    
    device = torch.device("cpu")
    pt_M_1dx = torch.tensor(M_1dx, dtype=torch.float64, device=device)
    pt_K_1dx = torch.tensor(K_1dx, dtype=torch.float64, device=device)
    pt_M_1dy = torch.tensor(M_1dy, dtype=torch.float64, device=device)
    pt_K_1dy = torch.tensor(K_1dy, dtype=torch.float64, device=device)
    pt_W = torch.tensor(W_np, dtype=torch.float64, device=device)
    pt_F = torch.tensor(F_local_np, dtype=torch.float64, device=device)
    pt_inv_D = torch.tensor(inv_D_np, dtype=torch.float64, device=device)
    pt_mask = torch.tensor(mask_boundary, device=device)

    def dss_pt(v):
        v_new = v.clone()
        v_new[:-1, :, :, p] += v[1:, :, :, 0]
        v_new[1:, :, :, 0] += v[:-1, :, :, p]
        
        v_final = v_new.clone()
        v_final[:, :-1, p, :] += v_new[:, 1:, 0, :]
        v_final[:, 1:, 0, :] += v_new[:, :-1, p, :]
        
        return torch.where(pt_mask, torch.tensor(0.0, dtype=torch.float64, device=device), v_final)
    
    def apply_K_pt(u):
        v_local = torch.matmul(torch.matmul(pt_M_1dx, u), pt_K_1dy.T) + torch.matmul(torch.matmul(pt_K_1dx, u), pt_M_1dy.T)
        return dss_pt(v_local)

    def cg_solve_pt(b, max_iters):
        x = torch.zeros_like(b)
        r = b.clone()
        z = r * pt_inv_D
        p_vec = z.clone()
        rsold = torch.sum(r * z * pt_W)
        for _ in range(max_iters):
            Ap = apply_K_pt(p_vec)
            pAp = torch.sum(p_vec * Ap * pt_W)
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * pt_inv_D
            rsnew = torch.sum(r * z_new * pt_W)
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
        return x
    
    start = time.perf_counter()
    U_num_np = cg_solve_np(F_local_np, max_iters)
    np_time = time.perf_counter() - start
    
    _ = cg_solve_mlx(F_mlx, max_iters)
    mx.eval(_)
    
    start = time.perf_counter()
    U_num_mlx = cg_solve_mlx(F_mlx, max_iters)
    mx.eval(U_num_mlx)
    mlx_time = time.perf_counter() - start
    
    start = time.perf_counter()
    U_num_pt = cg_solve_pt(pt_F, max_iters)
    pt_time = time.perf_counter() - start
    
    U_num = np.array(U_num_mlx)
    U_ex = exact_u(x_local, y_local)
    error = np.max(np.abs(U_num - U_ex))
    
    fortran_time = None
    matrices_filename = f"matrices_p{p}.txt"
    if not os.path.exists(matrices_filename):
        export_matrices(p, matrices_filename)
        
    try:
        cmd = ["./sem_2d_f90", matrices_filename, str(E_x), str(E_y), str(max_iters)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        match = re.search(r"Fortran Solve Time:\s+([0-9.eE+-]+)", result.stdout)
        if match:
            fortran_time = float(match.group(1))
    except Exception as e:
        print(f"Error running Fortran: {e}")
        fortran_time = 0.0
    
    return mlx_time, np_time, pt_time, fortran_time, error

def main():
    import sys
    E_x = 5
    E_y = 15
    p_values = list(range(3, 16))
    
    print("Compiling Fortran solver...", flush=True)
    subprocess.run(["gfortran", "-O3", "sem_2d.f90", "-o", "sem_2d_f90", "-framework", "Accelerate"], check=True)
    
    mlx_times = []
    np_times = []
    pt_times = []
    fortran_times = []
    errors = []
    
    print(f"Sweeping 2D Polynomial Degree (p) from 3 to 15 at fixed E_x=5, E_y=15", flush=True)
    print(f"{'p':<5} | {'N':<8} | {'Error (L_inf)':<15} | {'MLX (s)':<10} | {'NumPy (s)':<10} | {'PyTorch (s)':<11} | {'Fortran (s)':<10}", flush=True)
    print("-" * 95, flush=True)
    
    with open('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/p_convergence_2d_data.csv', 'w') as f:
        f.write("p,N_global,error_linf,mlx_cpu_s,numpy_s,pytorch_s,fortran_s\n")
        f.flush()
        for p in p_values:
            num_global = (E_x * p + 1) * (E_y * p + 1)
            mlx_t, np_t, pt_t, fortran_t, err = run_sem_2d(p, E_x, E_y)
            mlx_times.append(mlx_t)
            np_times.append(np_t)
            pt_times.append(pt_t)
            fortran_times.append(fortran_t)
            errors.append(err)
            f_t_str = f"{fortran_t:.5f}" if fortran_t is not None else "N/A"
            print(f"{p:<5} | {num_global:<8} | {err:<15.5e} | {mlx_t:<10.5f} | {np_t:<10.5f} | {pt_t:<11.5f} | {f_t_str:<10}", flush=True)
            f.write(f"{p},{num_global},{err},{mlx_t},{np_t},{pt_t},{fortran_t}\n")
            f.flush()
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogy(p_values, errors, 'bo-')
    ax1.set_xlabel('Polynomial Degree (p)')
    ax1.set_ylabel(r'$L_\infty$ Error')
    ax1.set_title(f'2D p-Refinement Convergence (E_x=5, E_y=15)')
    ax1.grid(True, which="both", ls="--")
    
    ax2.plot(p_values, mlx_times, 'ro-', label='MLX (Compiled CPU)')
    ax2.plot(p_values, np_times, 'go-', label='NumPy')
    ax2.plot(p_values, pt_times, 'co-', label='PyTorch (CPU)')
    ax2.plot(p_values, fortran_times, 'mo-', label='Fortran (Accelerate/BLAS)')
    ax2.set_xlabel('Polynomial Degree (p)')
    ax2.set_ylabel('CPU Time (s)')
    ax2.set_title('Computational Cost (Time vs p)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/p_convergence_2d_plot.png'
    plt.savefig(plot_path)
    
    print(f"\nSaved data to p_convergence_2d_data.csv", flush=True)
    print(f"Saved plot to {plot_path}", flush=True)

if __name__ == "__main__":
    main()
