import re

with open('sem_3d_fourier.py', 'r') as f:
    content = f.read()

pytorch_solver = """
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
        v_local = torch.matmul(torch.matmul(self.pt_K_1dx, u), self.pt_M_1dy.mT) + \\
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
"""

# Insert PyTorchSolver3D before BenchmarkRunner3D
content = content.replace('class BenchmarkRunner3D:', pytorch_solver + '\nclass BenchmarkRunner3D:')

# Add import torch if not there
if 'import torch' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport torch')

# Update BenchmarkRunner3D
runner_code = """
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
        print(f"{'p':<5} | {'N_global':<10} | {'Error (L_inf)':<15} | {'NumPy (s)':<10} | {'PyTorch (s)':<11}", flush=True)
        print("-" * 70, flush=True)
        
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
            err = 0.0
            
            for name, solver in solvers.items():
                if name == "NumPy":
                    b_np = solver.dss_np(F_hat)
                else:
                    b_np = solver.dss_pt(torch.tensor(F_hat, dtype=torch.complex128, device=solver.device)).cpu().numpy()
                
                _ = solver.solve(b_np, max_iters=5) # warmup
                
                t0 = time.time()
                u_hat, iters = solver.solve(b_np)
                t1 = time.time()
                
                u_hat_transposed = np.transpose(u_hat, (0, 1, 3, 4, 2))
                u_solved = np.fft.irfft(u_hat_transposed, n=self.N_z, axis=-1)
                
                err = np.max(np.abs(u_solved - u_exact))
                times[name] = t1 - t0
                
            num_global = (self.E_x * p + 1) * (self.E_y * p + 1) * self.N_z
            print(f"{p:<5} | {num_global:<10} | {err:<15.5e} | {times['NumPy']:<10.5f} | {times['PyTorch']:<11.5f}", flush=True)
            
            results.append({
                'p': p,
                'N_global': num_global,
                'error_linf': err,
                'numpy_s': times['NumPy'],
                'pytorch_s': times['PyTorch']
            })
            
        df = pd.DataFrame(results)
        df.to_csv('p_convergence_3d_data.csv', index=False)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['p'], df['numpy_s'], marker='o', label='NumPy (Accelerate)')
        plt.plot(df['p'], df['pytorch_s'], marker='s', label='PyTorch CPU')
        
        plt.xlabel('Polynomial Degree (p)')
        plt.ylabel('Compute Time (s)')
        plt.title(f'2.5D SEM-Fourier Performance (Ex={self.E_x}, Ey={self.E_y}, Nz={self.N_z})')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig('3d_fourier_sweep_plot.png', dpi=300, bbox_inches='tight')
        print("Saved results to p_convergence_3d_data.csv and 3d_fourier_sweep_plot.png")
"""

# Replace the old BenchmarkRunner3D
content = re.sub(r'class BenchmarkRunner3D:.*', runner_code, content, flags=re.DOTALL)

with open('sem_3d_fourier.py', 'w') as f:
    f.write(content)
