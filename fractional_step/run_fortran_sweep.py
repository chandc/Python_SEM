import os
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the robust ReferenceElement from our OO script
# (This avoids scipy.special.legendre precision bugs at high p)
import sys
sys.path.append('.')
from sem_2d_oo import ReferenceElement

def export_matrices_for_fortran(p, filename):
    ref_el = ReferenceElement(p)
    
    with open(filename, 'w') as f:
        f.write(f"{p}\n")
        f.write("x_gll\n")
        for val in ref_el.x:
            f.write(f"{val:.16e}\n")
        
        f.write("w_gll\n")
        for val in ref_el.w:
            f.write(f"{val:.16e}\n")
            
        f.write("D_matrix\n")
        for i in range(p + 1):
            f.write(" ".join([f"{val:.16e}" for val in ref_el.D[i, :]]) + "\n")

def run_fortran_sweep():
    E_x = 30
    E_y = 30
    p_values = list(range(3, 16))
    max_iters = 2000
    
    binary = "./sem_2d_f90"
    
    # Compile
    print("Compiling Fortran solver...", flush=True)
    # -fexternal-blas routes MATMUL to Accelerate's DGEMM; without it the
    # "-framework Accelerate" link is inert and gfortran uses its own internal
    # matmul, which costs ~3.6x at p=15 (514 -> 144 ms at E=30x30).
    # matmul-limit=6 sets the crossover: DGEMM for (p+1)x(p+1) blocks >= 6x6
    # (p >= 5), where it starts to pay; below that the call overhead dominates
    # and the internal path is faster.
    # -mcpu=native is machine-specific - drop it if this binary must be portable.
    subprocess.run(["gfortran", "-O3", "-mcpu=native", "-funroll-loops",
                    "-fexternal-blas", "-fblas-matmul-limit=6",
                    "sem_2d.f90", "-o", "sem_2d_f90", "-framework", "Accelerate"], check=True)
    
    results = []
    
    for p in p_values:
        mat_file = f"matrices_p{p}.txt"
        export_matrices_for_fortran(p, mat_file)
        
        cmd = [binary, mat_file, str(E_x), str(E_y), str(max_iters)]
        print(f"Running Fortran p={p} Ex={E_x} Ey={E_y}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        time_taken = 0.0
        err = 0.0
        
        match_time = re.search(r"Fortran Solve Time:\s+([0-9.eE+-]+)", result.stdout)
        if match_time:
            time_taken = float(match_time.group(1))
            
        match_err = re.search(r"L_inf Error:\s+([0-9.eE+-]+)", result.stdout)
        if match_err:
            err = float(match_err.group(1))
            
        results.append({
            'p': p,
            'fortran_s': time_taken,
            'fortran_error': err
        })
        
    df_fortran = pd.DataFrame(results)
    df_fortran.to_csv('fortran_sweep_Ex30.csv', index=False)
    print("Saved Fortran sweep to fortran_sweep_Ex30.csv")
    return df_fortran

def generate_consolidated_plot():
    # Load Python results
    df_py = pd.read_csv('p_convergence_oo_data.csv')
    
    # Load Fortran results
    df_f90 = pd.read_csv('fortran_sweep_Ex30.csv')
    
    # Merge on p
    df = pd.merge(df_py, df_f90, on='p')
    
    p_values = df['p'].values
    errors = df['error_linf'].values
    mlx_times = df['mlx_cpu_s'].values
    np_times = df['numpy_s'].values
    pt_times = df['pytorch_s'].values
    fortran_times = df['fortran_s'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogy(p_values, errors, 'bo-')
    ax1.set_xlabel('Polynomial Degree (p)')
    ax1.set_ylabel(r'$L_\infty$ Error')
    ax1.set_title('2D p-Refinement Convergence (E_x=30, E_y=30)')
    ax1.grid(True, which='both', ls='--')
    
    # Performance Plot
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
    plt.savefig('consolidated_sweep_plot.png')
    print("Saved consolidated plot to consolidated_sweep_plot.png")

if __name__ == "__main__":
    run_fortran_sweep()
    generate_consolidated_plot()
