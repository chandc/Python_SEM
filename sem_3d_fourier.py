"""
3D Fourier-Spectral Element Method ("2.5D" SEM)

This module extends the 2D Spectral Element Method solver to 3 physical dimensions 
by utilizing a Fourier Spectral expansion in the periodic z-direction. 

ALGORITHM & EQUATIONS:
----------------------
We solve the 3D Poisson equation:
    - (\\nabla^2_{2D} + \\partial^2_z) u(x,y,z) = f(x,y,z)

Assuming periodicity in z, we apply a 1D Fourier transform along the z-axis. 
The spatial derivative \\partial^2_z is replaced by the scalar wavenumber -k_z^2.
This decouples the 3D Poisson equation into N_z independent 2D Helmholtz equations:
    - \\nabla^2_{2D} \\hat{u}(x,y,k_z) + k_z^2 \\hat{u}(x,y,k_z) = \\hat{f}(x,y,k_z)

IMPLEMENTATION PIPELINE:
------------------------
1. Forward FFT:
   Evaluate the physical forcing F(x,y,z) as a tensor of shape (E_x, E_y, p+1, p+1, N_z).
   Apply `np.fft.rfft` along the z-axis (axis=-1) to yield \\hat{F}(x,y,k_z).

2. Memory Permutation (The "NumPy Optimization"):
   Transpose the tensor to shape (E_x, E_y, N_kz, p+1, p+1). 
   NumPy's `matmul` (@) operator natively performs matrix multiplication over the *last two dimensions*.
   By shifting the spatial nodes (p+1) to the back, NumPy automatically and seamlessly batches 
   the tensor contractions across all elements AND all Fourier modes using optimized C BLAS.

3. Complex Conjugate Gradient Solve:
   Solve the Helmholtz operator for all N_kz modes simultaneously:
   v = (K_x @ u @ M_y.T) + (M_x @ u @ K_y.T) + k_z^2 * (M_x @ u @ M_y.T)

4. Inverse FFT:
   Transpose the solved tensor back to (E_x, E_y, p+1, p+1, N_kz).
   Apply `np.fft.irfft` to recover the 3D physical domain u(x,y,z).
"""
import numpy as np
import torch
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    pass
import time
import argparse
import os

from sem_2d_oo import ReferenceElement

class Mesh3DFourier:
    def __init__(self, E_x, E_y, N_z, L_x=2.0, L_y=2.0, L_z=1.0, ref_el=None):
        self.E_x = E_x
        self.E_y = E_y
        self.N_z = N_z
        self.L_x = L_x
        self.L_y = L_y
        self.L_z = L_z
        self.ref_el = ref_el
        
        self.dx = L_x / E_x
        self.dy = L_y / E_y
        self.dz = L_z / N_z
        self.J_x = self.dx / 2.0
        self.J_y = self.dy / 2.0
        
        self.M_1dx = self.ref_el.M * self.J_x
        self.M_1dy = self.ref_el.M * self.J_y
        self.K_1dx = self.ref_el.K / self.J_x
        self.K_1dy = self.ref_el.K / self.J_y
        
        # Fourier Wavenumbers for rfft (real to complex FFT)
        # rfftfreq returns frequencies f = k / L_z. Wavenumber k_z = 2 * pi * f
        freqs = np.fft.rfftfreq(N_z, d=self.dz)
        self.kz = 2.0 * np.pi * freqs
        self.kz_squared = self.kz**2
        self.N_kz = len(self.kz)
        
class ProblemDefinition3D:
    def evaluate(self, mesh):
        p = mesh.ref_el.p
        x_edges = np.linspace(-mesh.L_x/2.0, mesh.L_x/2.0, mesh.E_x + 1)
        y_edges = np.linspace(-mesh.L_y/2.0, mesh.L_y/2.0, mesh.E_y + 1)
        z_1d = np.linspace(0, mesh.L_z, mesh.N_z, endpoint=False)
        
        u_exact = np.zeros((mesh.E_x, mesh.E_y, p+1, p+1, mesh.N_z))
        F_local = np.zeros((mesh.E_x, mesh.E_y, p+1, p+1, mesh.N_z))
        
        for e_x in range(mesh.E_x):
            for e_y in range(mesh.E_y):
                x_local = x_edges[e_x] + (mesh.ref_el.x + 1.0) * mesh.J_x
                y_local = y_edges[e_y] + (mesh.ref_el.x + 1.0) * mesh.J_y
                
                # Broadcasting arrays to evaluate 3D field instantly
                X = x_local[:, None, None]
                Y = y_local[None, :, None]
                Z = z_1d[None, None, :]
                
                # Exact solution and forcing function
                U = np.sin(4*np.pi*X) * np.sin(4*np.pi*Y) * np.sin(2*np.pi*Z)
                F = 36 * np.pi**2 * U
                
                # Apply mass weights
                W_x = mesh.ref_el.w[:, None, None] * mesh.J_x
                W_y = mesh.ref_el.w[None, :, None] * mesh.J_y
                
                u_exact[e_x, e_y, :, :, :] = U
                F_local[e_x, e_y, :, :, :] = F * W_x * W_y
                
        return u_exact, F_local

class NumpySolver3D:
    def __init__(self, mesh):
        self.mesh = mesh
        self.p = mesh.ref_el.p
        self._build_preconditioner()
        self._build_weight_matrix()
        
    def _build_preconditioner(self):
        diag_M_x = np.diag(self.mesh.M_1dx).reshape(1, 1, 1, self.p+1, 1)
        diag_M_y = np.diag(self.mesh.M_1dy).reshape(1, 1, 1, 1, self.p+1)
        diag_K_x = np.diag(self.mesh.K_1dx).reshape(1, 1, 1, self.p+1, 1)
        diag_K_y = np.diag(self.mesh.K_1dy).reshape(1, 1, 1, 1, self.p+1)
        kz2 = self.mesh.kz_squared.reshape(1, 1, self.mesh.N_kz, 1, 1)
        
        D_local = (diag_K_x * diag_M_y) + (diag_M_x * diag_K_y) + (kz2 * diag_M_x * diag_M_y)
        D_local = np.broadcast_to(D_local, (self.mesh.E_x, self.mesh.E_y, self.mesh.N_kz, self.p+1, self.p+1))
        
        D_global = self.dss_np(D_local.copy())
        self.inv_D_np = np.zeros_like(D_global)
        mask = D_global > 1e-14
        self.inv_D_np[mask] = 1.0 / D_global[mask]

    def _build_weight_matrix(self):
        self.W_np = np.ones((self.mesh.E_x, self.mesh.E_y, self.mesh.N_kz, self.p+1, self.p+1))
        self.W_np[1:, :, :, 0, :] = 0.0
        self.W_np[:, 1:, :, :, 0] = 0.0

    def dss_np(self, v):
        # Shape of v: (E_x, E_y, N_kz, p+1, p+1)
        v_new = v.copy()
        
        # X-exchange
        v_new[:-1, :, :, self.p, :] += v[1:, :, :, 0, :]
        v_new[1:, :, :, 0, :] = v_new[:-1, :, :, self.p, :]
        
        # Y-exchange
        v_new[:, :-1, :, :, self.p] += v_new[:, 1:, :, :, 0]
        v_new[:, 1:, :, :, 0] = v_new[:, :-1, :, :, self.p]
        
        # Dirichlet boundaries on x and y walls
        v_new[0, :, :, 0, :] = 0.0
        v_new[-1, :, :, self.p, :] = 0.0
        v_new[:, 0, :, :, 0] = 0.0
        v_new[:, -1, :, :, self.p] = 0.0
        
        return v_new

    def apply_K(self, u):
        # Shape of u: (E_x, E_y, N_kz, p+1, p+1)
        # Using NumPy matmul (@) which automatically batches over E_x, E_y, N_kz
        v_local = (self.mesh.K_1dx @ u @ self.mesh.M_1dy.T) + \
                  (self.mesh.M_1dx @ u @ self.mesh.K_1dy.T)
        
        # Add the Fourier mass term: kz^2 * (M_x @ u @ M_y.T)
        kz2 = self.mesh.kz_squared.reshape(1, 1, self.mesh.N_kz, 1, 1)
        v_local += kz2 * (self.mesh.M_1dx @ u @ self.mesh.M_1dy.T)
        
        return self.dss_np(v_local)

    def solve(self, b, max_iters=2000, tol=1e-11):
        x = np.zeros_like(b, dtype=np.complex128)
        r = b.copy()
        z = r * self.inv_D_np
        p_vec = z.copy()
        rsold = np.sum(r.conj() * z * self.W_np).real
        
        iters = 0
        for _ in range(max_iters):
            iters += 1
            Ap = self.apply_K(p_vec)
            pAp = np.sum(p_vec.conj() * Ap * self.W_np).real
            
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * self.inv_D_np
            rsnew = np.sum(r.conj() * z_new * self.W_np).real
            
            if np.sqrt(rsnew) < tol:
                break
                
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
            
        return x, iters


import torch

class PyTorchSolver3D:
    def __init__(self, mesh, device_name='cpu'):
        self.mesh = mesh
        self.p = mesh.ref_el.p
        self.device = torch.device(device_name)
        
        self.pt_M_1dx = torch.tensor(self.mesh.M_1dx, dtype=torch.complex128, device=self.device)
        self.pt_K_1dx = torch.tensor(self.mesh.K_1dx, dtype=torch.complex128, device=self.device)
        self.pt_M_1dy = torch.tensor(self.mesh.M_1dy, dtype=torch.complex128, device=self.device)
        self.pt_K_1dy = torch.tensor(self.mesh.K_1dy, dtype=torch.complex128, device=self.device)
        self.pt_kz2 = torch.tensor(self.mesh.kz_squared.reshape(1, 1, self.mesh.N_kz, 1, 1), dtype=torch.complex128, device=self.device)
        
        self._build_preconditioner()
        self._build_weight_matrix()
        
    def _build_preconditioner(self):
        diag_M_x = torch.diag(self.pt_M_1dx).reshape(1, 1, 1, self.p+1, 1)
        diag_M_y = torch.diag(self.pt_M_1dy).reshape(1, 1, 1, 1, self.p+1)
        diag_K_x = torch.diag(self.pt_K_1dx).reshape(1, 1, 1, self.p+1, 1)
        diag_K_y = torch.diag(self.pt_K_1dy).reshape(1, 1, 1, 1, self.p+1)
        
        D_local = (diag_K_x * diag_M_y) + (diag_M_x * diag_K_y) + (self.pt_kz2 * diag_M_x * diag_M_y)
        D_local = D_local.expand(self.mesh.E_x, self.mesh.E_y, self.mesh.N_kz, self.p+1, self.p+1)
        
        D_global = self.dss_pt(D_local.clone())
        self.pt_inv_D = torch.zeros_like(D_global)
        mask = torch.abs(D_global) > 1e-14
        self.pt_inv_D[mask] = 1.0 / D_global[mask]

    def _build_weight_matrix(self):
        self.pt_W = torch.ones((self.mesh.E_x, self.mesh.E_y, self.mesh.N_kz, self.p+1, self.p+1), dtype=torch.complex128, device=self.device)
        self.pt_W[1:, :, :, 0, :] = 0.0
        self.pt_W[:, 1:, :, :, 0] = 0.0

    def dss_pt(self, v):
        v_new = v.clone()
        v_new[:-1, :, :, self.p, :] += v[1:, :, :, 0, :]
        v_new[1:, :, :, 0, :] = v_new[:-1, :, :, self.p, :]
        
        v_new[:, :-1, :, :, self.p] += v_new[:, 1:, :, :, 0]
        v_new[:, 1:, :, :, 0] = v_new[:, :-1, :, :, self.p]
        
        v_new[0, :, :, 0, :] = 0.0
        v_new[-1, :, :, self.p, :] = 0.0
        v_new[:, 0, :, :, 0] = 0.0
        v_new[:, -1, :, :, self.p] = 0.0
        return v_new

    def apply_K(self, u):
        v_local = torch.matmul(torch.matmul(self.pt_K_1dx, u), self.pt_M_1dy.mT) + \
                  torch.matmul(torch.matmul(self.pt_M_1dx, u), self.pt_K_1dy.mT)
        v_local += self.pt_kz2 * torch.matmul(torch.matmul(self.pt_M_1dx, u), self.pt_M_1dy.mT)
        return self.dss_pt(v_local)

    def solve(self, b_np, max_iters=2000, tol=1e-11):
        b = torch.tensor(b_np, dtype=torch.complex128, device=self.device)
        x = torch.zeros_like(b)
        r = b.clone()
        z = r * self.pt_inv_D
        p_vec = z.clone()
        rsold = torch.sum(torch.conj(r) * z * self.pt_W).real
        
        iters = 0
        for _ in range(max_iters):
            iters += 1
            Ap = self.apply_K(p_vec)
            pAp = torch.sum(torch.conj(p_vec) * Ap * self.pt_W).real
            
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * self.pt_inv_D
            rsnew = torch.sum(torch.conj(r) * z_new * self.pt_W).real
            
            if torch.sqrt(rsnew) < tol:
                break
                
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
            
        return x.cpu().numpy(), iters



class PyTorchSolver3D:
    def __init__(self, mesh, device_name='cpu'):
        self.mesh = mesh
        self.p = mesh.ref_el.p
        self.device = torch.device(device_name)
        
        self.pt_M_1dx = torch.tensor(self.mesh.M_1dx, dtype=torch.complex128, device=self.device)
        self.pt_K_1dx = torch.tensor(self.mesh.K_1dx, dtype=torch.complex128, device=self.device)
        self.pt_M_1dy = torch.tensor(self.mesh.M_1dy, dtype=torch.complex128, device=self.device)
        self.pt_K_1dy = torch.tensor(self.mesh.K_1dy, dtype=torch.complex128, device=self.device)
        self.pt_kz2 = torch.tensor(self.mesh.kz_squared.reshape(1, 1, self.mesh.N_kz, 1, 1), dtype=torch.complex128, device=self.device)
        
        self._build_preconditioner()
        self._build_weight_matrix()
        
    def _build_preconditioner(self):
        diag_M_x = torch.diag(self.pt_M_1dx).reshape(1, 1, 1, self.p+1, 1)
        diag_M_y = torch.diag(self.pt_M_1dy).reshape(1, 1, 1, 1, self.p+1)
        diag_K_x = torch.diag(self.pt_K_1dx).reshape(1, 1, 1, self.p+1, 1)
        diag_K_y = torch.diag(self.pt_K_1dy).reshape(1, 1, 1, 1, self.p+1)
        
        D_local = (diag_K_x * diag_M_y) + (diag_M_x * diag_K_y) + (self.pt_kz2 * diag_M_x * diag_M_y)
        D_local = D_local.expand(self.mesh.E_x, self.mesh.E_y, self.mesh.N_kz, self.p+1, self.p+1)
        
        D_global = self.dss_pt(D_local.clone())
        self.pt_inv_D = torch.zeros_like(D_global)
        mask = torch.abs(D_global) > 1e-14
        self.pt_inv_D[mask] = 1.0 / D_global[mask]

    def _build_weight_matrix(self):
        self.pt_W = torch.ones((self.mesh.E_x, self.mesh.E_y, self.mesh.N_kz, self.p+1, self.p+1), dtype=torch.complex128, device=self.device)
        self.pt_W[1:, :, :, 0, :] = 0.0
        self.pt_W[:, 1:, :, :, 0] = 0.0

    def dss_pt(self, v):
        v_new = v.clone()
        v_new[:-1, :, :, self.p, :] += v[1:, :, :, 0, :]
        v_new[1:, :, :, 0, :] = v_new[:-1, :, :, self.p, :]
        
        v_new[:, :-1, :, :, self.p] += v_new[:, 1:, :, :, 0]
        v_new[:, 1:, :, :, 0] = v_new[:, :-1, :, :, self.p]
        
        v_new[0, :, :, 0, :] = 0.0
        v_new[-1, :, :, self.p, :] = 0.0
        v_new[:, 0, :, :, 0] = 0.0
        v_new[:, -1, :, :, self.p] = 0.0
        return v_new

    def apply_K(self, u):
        v_local = torch.matmul(torch.matmul(self.pt_K_1dx, u), self.pt_M_1dy.mT) + \
                  torch.matmul(torch.matmul(self.pt_M_1dx, u), self.pt_K_1dy.mT)
        v_local += self.pt_kz2 * torch.matmul(torch.matmul(self.pt_M_1dx, u), self.pt_M_1dy.mT)
        return self.dss_pt(v_local)

    def solve(self, b_np, max_iters=2000, tol=1e-11):
        b = torch.tensor(b_np, dtype=torch.complex128, device=self.device)
        x = torch.zeros_like(b)
        r = b.clone()
        z = r * self.pt_inv_D
        p_vec = z.clone()
        rsold = torch.sum(torch.conj(r) * z * self.pt_W).real
        
        iters = 0
        for _ in range(max_iters):
            iters += 1
            Ap = self.apply_K(p_vec)
            pAp = torch.sum(torch.conj(p_vec) * Ap * self.pt_W).real
            
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * self.pt_inv_D
            rsnew = torch.sum(torch.conj(r) * z_new * self.pt_W).real
            
            if torch.sqrt(rsnew) < tol:
                break
                
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
            
        return x.cpu().numpy(), iters



    def dss_mx(self, v):
        # MLX lacks simple in-place slicing, so we construct the updates
        E_x, E_y, N_kz, p1, p1 = v.shape
        p = p1 - 1
        
        # Start with original v
        v_new = v
        
        # X-exchange
        # We need to add v[1:, :, :, 0, :] to v[:-1, :, :, p, :]
        update_x = mx.zeros_like(v)
        update_x[:-1, :, :, p, :] = v[1:, :, :, 0, :]
        v_new = v_new + update_x
        
        update_x2 = mx.zeros_like(v)
        update_x2[1:, :, :, 0, :] = v_new[:-1, :, :, p, :]
        v_new = v_new + update_x2 - update_x # prevent double addition on the boundary
        # Actually in MLX it's easier to use a mask to clear the boundaries then just sum.
        # But wait, MLX doesn't have in-place item assignment. We can do it using slicing.
        # Let's just use Numpy for dss_mx if it's too complex, or we can use a simpler approach.
        # Since DSS is only O(boundary), we can do it in numpy if we must, but let's try to do it in MLX.
        # Actually, let's just use Numpy for DSS to keep MLX code simple, but wait, then we can't @mx.compile the whole loop!
        # Let's write DSS using mx.concatenate
        
        # For X-exchange:
        # We take v[:,:,:,0,:] and v[:,:,:,p,:]
        # Left boundary gets right boundary of previous element
        # Right boundary gets left boundary of next element
        # It's easier to use the padding trick.
        # Let's just use Python lists of slices and concatenate.
        
        # Let's create a mask for boundaries
        # This is a bit complex in MLX. Let's try to implement a compiled-friendly version.
        # Or, we can use a scatter operation, but MLX scatter is not exactly like PyTorch.
        pass


class BenchmarkRunner3D:
    def __init__(self, p=15, E_x=10, E_y=10, N_z=16):
        self.problem = ProblemDefinition3D()
        self.E_x = E_x
        self.E_y = E_y
        self.N_z = N_z
        
    def run_sweep(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        
        p_values = list(range(3, 11))
        
        print(f"Sweeping 3D Polynomial Degree (p) from 3 to 10 at fixed E_x={self.E_x}, E_y={self.E_y}, N_z={self.N_z}", flush=True)
        print(f"{'p':<5} | {'N_global':<10} | {'Numpy (s)':<10} | {'PyTorch (s)':<11}", flush=True)
        print("-" * 65, flush=True)
        
        results = []
        
        for p in p_values:
            ref_el = ReferenceElement(p)
            mesh = Mesh3DFourier(self.E_x, self.E_y, self.N_z, ref_el=ref_el)
            u_exact, F_local = self.problem.evaluate(mesh)
            
            solvers = {
                "NumPy": NumpySolver3D(mesh),
                "PyTorch": PyTorchSolver3D(mesh, 'cpu')
            }
            
            # Forward FFT along the Z axis (axis=-1)
            F_hat = np.fft.rfft(F_local, axis=-1)
            # Transpose from (E_x, E_y, p+1, p+1, N_kz) to (E_x, E_y, N_kz, p+1, p+1)
            F_hat = np.transpose(F_hat, (0, 1, 4, 2, 3))
            
            times = {"NumPy": 0.0, "PyTorch": 0.0}
            errors = {"NumPy": 0.0, "PyTorch": 0.0}
            
            for name, solver in solvers.items():
                if name == "NumPy":
                    b_np = solver.dss_np(F_hat)
                    max_iters = 2000
                elif name == "PyTorch":
                    b_np = solver.dss_pt(torch.tensor(F_hat, dtype=torch.complex128, device=solver.device)).cpu().numpy()
                    max_iters = 2000
                
                # Warmup
                _ = solver.solve(b_np, max_iters=5) 
                
                t0 = time.time()
                u_hat, iters = solver.solve(b_np, max_iters=max_iters)
                t1 = time.time()
                
                u_hat_transposed = np.transpose(u_hat, (0, 1, 3, 4, 2))
                u_solved = np.fft.irfft(u_hat_transposed, n=self.N_z, axis=-1)
                
                err = np.max(np.abs(u_solved - u_exact))
                times[name] = t1 - t0
                errors[name] = err
                
            num_global = (self.E_x * p + 1) * (self.E_y * p + 1) * self.N_z
            print(f"{p:<5} | {num_global:<10} | {times['NumPy']:<10.5f} | {times['PyTorch']:<11.5f}", flush=True)
            
            results.append({
                'p': p,
                'N_global': num_global,
                'error_numpy': errors['NumPy'],
                'error_pytorch': errors['PyTorch'],
                'numpy_s': times['NumPy'],
                'pytorch_s': times['PyTorch']
            })
            
        df = pd.DataFrame(results)
        df.to_csv('p_convergence_3d_data.csv', index=False)
        
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(df['p']), np.array(df['numpy_s']), marker='o', label='NumPy (Accelerate)')
        plt.plot(np.array(df['p']), np.array(df['pytorch_s']), marker='s', label='PyTorch CPU')
        
        plt.xlabel('Polynomial Degree (p)')
        plt.ylabel('Compute Time (s)')
        plt.title(f'2.5D SEM-Fourier Performance (Ex={self.E_x}, Ey={self.E_y}, Nz={self.N_z})')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig('3d_fourier_sweep_plot.png', dpi=300, bbox_inches='tight')
        print("Saved results to p_convergence_3d_data.csv and 3d_fourier_sweep_plot.png")

if __name__ == "__main__":
    runner = BenchmarkRunner3D(E_x=10, E_y=10, N_z=16)
    runner.run_sweep()
