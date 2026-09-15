import numpy as np
from scipy.special import legendre
import os

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

def export_matrices(p=8, filename="matrices_p8.txt"):
    # Reference unit element: [-1, 1] mapped to [0, dx]
    # For a general element, we do this in Fortran, but we can export the reference domain [-1, 1] matrices
    # M_1d_ref = diag(w_gll)
    # K_1d_ref = D^T * M_1d_ref * D
    x_gll, w_gll = gll(p)
    D = lagrange_derivative_matrix(p, x_gll)
    M_1d_ref = np.diag(w_gll)
    K_1d_ref = D.T @ M_1d_ref @ D

    np.set_printoptions(precision=16)
    with open(filename, 'w') as f:
        f.write(f"{p}\n")
        f.write("x_gll\n")
        for val in x_gll:
            f.write(f"{val:.16e}\n")
        
        f.write("w_gll\n")
        for val in w_gll:
            f.write(f"{val:.16e}\n")
            
        f.write("D_matrix\n")
        for i in range(p + 1):
            f.write(" ".join([f"{val:.16e}" for val in D[i, :]]) + "\n")
            
    print(f"Exported matrices to {filename}")

if __name__ == "__main__":
    export_matrices(8, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/matrices_p8.txt')
