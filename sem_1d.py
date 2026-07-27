import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
import time
import mlx.core as mx

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

def build_local_matrices(p, dx):
    x_gll, w_gll = gll(p)
    D = lagrange_derivative_matrix(p, x_gll)
    M_e = np.diag(w_gll) * (dx / 2.0)
    K_e = D.T @ M_e @ D * (2.0 / dx)**2
    return M_e, K_e, x_gll

def run_sem(p, num_elements=5):
    # Force MLX to run on the CPU in float64
    mx.set_default_device(mx.cpu)
    
    def exact_u(x):
        return np.sin(np.pi * x)
    def forcing_f(x):
        return np.pi**2 * np.sin(np.pi * x)

    L = 2.0               
    dx = L / num_elements 
    num_global_nodes = num_elements * p + 1
    
    M_e_np, K_e_np, x_gll_np = build_local_matrices(p, dx)
    
    Q_np = np.zeros((num_elements * (p + 1), num_global_nodes), dtype=np.float64)
    X_global_np = np.zeros(num_global_nodes, dtype=np.float64)
    F_global_np = np.zeros(num_global_nodes, dtype=np.float64)
    
    for e in range(num_elements):
        x_L = -1.0 + e * dx
        x_phys = x_L + (x_gll_np + 1.0) * (dx / 2.0)
        global_indices = list(range(e * p, (e + 1) * p + 1))
        X_global_np[global_indices] = x_phys
        
        for local_i, global_i in enumerate(global_indices):
            row = e * (p + 1) + local_i
            Q_np[row, global_i] = 1.0
            F_global_np[global_i] += M_e_np[local_i, local_i] * forcing_f(x_phys[local_i])

    # ---------------------------------------------
    # NumPy Matrix-Free Implementation
    # ---------------------------------------------
    F_np = F_global_np.copy()
    F_np[0] = 0.0
    F_np[-1] = 0.0
    
    def apply_K_np(u):
        u_local_flat = Q_np @ u
        u_local = u_local_flat.reshape((num_elements, p + 1))
        v_local = u_local @ K_e_np.T
        v_local_flat = v_local.reshape(-1)
        v_global = Q_np.T @ v_local_flat
        v_global[0] = u[0]
        v_global[-1] = u[-1]
        return v_global

    def cg_solve_np(b, max_iters):
        x = np.zeros_like(b)
        r = b.copy()
        p_vec = r.copy()
        rsold = np.sum(r * r)
        for _ in range(max_iters):
            Ap = apply_K_np(p_vec)
            pAp = np.sum(p_vec * Ap)
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            rsnew = np.sum(r * r)
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = r + beta * p_vec
            rsold = rsnew
        return x

    # ---------------------------------------------
    # MLX Matrix-Free Implementation
    # ---------------------------------------------
    Q = mx.array(Q_np, dtype=mx.float64)
    Q_T = mx.array(Q_np.T, dtype=mx.float64)
    K_e = mx.array(K_e_np, dtype=mx.float64)
    F = mx.array(F_global_np, dtype=mx.float64)
    F = mx.where(mx.arange(num_global_nodes) == 0, mx.array(0.0, dtype=mx.float64), F)
    F = mx.where(mx.arange(num_global_nodes) == num_global_nodes - 1, mx.array(0.0, dtype=mx.float64), F)
    
    def apply_K(u):
        u_local_flat = mx.matmul(Q, u)
        u_local = mx.reshape(u_local_flat, (num_elements, p + 1))
        v_local = mx.matmul(u_local, K_e.T)
        v_local_flat = mx.reshape(v_local, (-1,))
        v_global = mx.matmul(Q_T, v_local_flat)
        mask = (mx.arange(num_global_nodes) == 0) | (mx.arange(num_global_nodes) == num_global_nodes - 1)
        v_global = mx.where(mask, u, v_global)
        return v_global

    @mx.compile
    def cg_solve(b, max_iters):
        x = mx.zeros_like(b)
        r = b
        p_vec = r
        rsold = mx.sum(r * r)
        for _ in range(max_iters):
            Ap = apply_K(p_vec)
            pAp = mx.sum(p_vec * Ap)
            alpha = mx.where(pAp < 1e-25, mx.array(0.0, dtype=mx.float64), rsold / pAp)
            x = x + alpha * p_vec
            r = r - alpha * Ap
            rsnew = mx.sum(r * r)
            beta = mx.where(rsold < 1e-25, mx.array(0.0, dtype=mx.float64), rsnew / rsold)
            p_vec = r + beta * p_vec
            rsold = rsnew
        return x

    # --- Timings ---
    max_iters = min(100, num_global_nodes) 
    
    # MLX Warmup: Must use the exact same max_iters to avoid a cache miss!
    _ = cg_solve(F, max_iters)
    mx.eval(_)
    
    start = time.perf_counter()
    U_numerical_mlx = cg_solve(F, max_iters)
    mx.eval(U_numerical_mlx)
    mlx_time = time.perf_counter() - start
    
    # Numpy doesn't need warmup
    start = time.perf_counter()
    U_numerical_np = cg_solve_np(F_np, max_iters)
    np_time = time.perf_counter() - start
    
    U_num_np = np.array(U_numerical_mlx)
    U_exact = exact_u(X_global_np)
    error = np.max(np.abs(U_num_np - U_exact))
    
    return mlx_time, np_time, error

def main():
    p = 11
    element_values = list(range(5, 21))
    mlx_times = []
    np_times = []
    errors = []
    
    print(f"Sweeping Elements (E) from 5 to 20 with fixed p={p}")
    print(f"{'E':<5} | {'Error (L_inf)':<15} | {'MLX CPU (s)':<15} | {'NumPy (s)':<15}")
    print("-" * 55)
    
    with open('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/e_convergence_data.csv', 'w') as f:
        f.write("E,error_linf,mlx_cpu_s,numpy_s\n")
        for E in element_values:
            mlx_t, np_t, err = run_sem(p, num_elements=E)
            mlx_times.append(mlx_t)
            np_times.append(np_t)
            errors.append(err)
            print(f"{E:<5} | {err:<15.5e} | {mlx_t:<15.5f} | {np_t:<15.5f}")
            f.write(f"{E},{err},{mlx_t},{np_t}\n")
            
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogy(element_values, errors, 'bo-')
    ax1.set_xlabel('Number of Elements (E)')
    ax1.set_ylabel(r'$L_\infty$ Error')
    ax1.set_title(f'h-Refinement Convergence (p={p})')
    ax1.grid(True, which="both", ls="--")
    
    ax2.plot(element_values, mlx_times, 'ro-', label='MLX (Compiled CPU)')
    ax2.plot(element_values, np_times, 'go-', label='NumPy')
    ax2.set_xlabel('Number of Elements (E)')
    ax2.set_ylabel('CPU Time (s)')
    ax2.set_title('Computational Cost (Time vs E)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/e_convergence_plot.png'
    plt.savefig(plot_path)
    
    print(f"\nSaved data to e_convergence_data.csv")
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
