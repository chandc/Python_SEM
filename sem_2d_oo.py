import numpy as np
import torch
import time
import argparse
from numpy.polynomial.legendre import Legendre
try:
    import mlx.core as mx
    has_mlx = True
except ImportError:
    has_mlx = False
    
    class DummyMX:
        def compile(self, f):
            return f
    mx = DummyMX()

class ReferenceElement:
    def __init__(self, p):
        self.p = p
        self.x, self.w = self._get_gll(p)
        self.D = self._get_derivative_matrix(p, self.x)
        self.M = np.diag(self.w)
        self.K = self.D.T @ self.M @ self.D
        
    def _get_gll(self, p):
        if p == 0:
            return np.array([0.0]), np.array([2.0])
        L_p = Legendre([0]*p + [1])
        roots = L_p.deriv().roots()
        roots = np.sort(np.real(roots))
        x = np.concatenate(([-1.0], roots, [1.0]))
        w = 2.0 / (p * (p + 1) * L_p(x)**2)
        return x, w

    def _get_derivative_matrix(self, p, x):
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

class Mesh2D:
    def __init__(self, E_x, E_y, L_x=2.0, L_y=2.0, ref_el=None):
        self.E_x = E_x
        self.E_y = E_y
        self.L_x = L_x
        self.L_y = L_y
        self.ref_el = ref_el
        
        self.dx = L_x / E_x
        self.dy = L_y / E_y
        self.J_x = self.dx / 2.0
        self.J_y = self.dy / 2.0
        
        # Scale reference matrices to physical elements
        self.M_1dx = self.ref_el.M * self.J_x
        self.M_1dy = self.ref_el.M * self.J_y
        self.K_1dx = self.ref_el.K / self.J_x
        self.K_1dy = self.ref_el.K / self.J_y
        
    def get_global_coordinates(self):
        p = self.ref_el.p
        x_edges = np.linspace(-self.L_x/2.0, self.L_x/2.0, self.E_x + 1)
        y_edges = np.linspace(-self.L_y/2.0, self.L_y/2.0, self.E_y + 1)
        
        coords = np.zeros((self.E_x, self.E_y, p+1, p+1, 2))
        for e_x in range(self.E_x):
            for e_y in range(self.E_y):
                x_local = x_edges[e_x] + (self.ref_el.x + 1.0) * self.J_x
                y_local = y_edges[e_y] + (self.ref_el.x + 1.0) * self.J_y
                for i in range(p+1):
                    for j in range(p+1):
                        coords[e_x, e_y, i, j, 0] = x_local[i]
                        coords[e_x, e_y, i, j, 1] = y_local[j]
        return coords

class ProblemDefinition:
    def __init__(self, exact_u_func, forcing_f_func):
        self.exact_u_func = exact_u_func
        self.forcing_f_func = forcing_f_func
        
    def evaluate(self, mesh):
        p = mesh.ref_el.p
        coords = mesh.get_global_coordinates()
        u_exact = np.zeros((mesh.E_x, mesh.E_y, p+1, p+1))
        F_local = np.zeros((mesh.E_x, mesh.E_y, p+1, p+1))
        
        for e_x in range(mesh.E_x):
            for e_y in range(mesh.E_y):
                for i in range(p+1):
                    for j in range(p+1):
                        x, y = coords[e_x, e_y, i, j]
                        u_exact[e_x, e_y, i, j] = self.exact_u_func(x, y)
                        val = self.forcing_f_func(x, y)
                        weight_x = mesh.ref_el.w[i] * mesh.J_x
                        weight_y = mesh.ref_el.w[j] * mesh.J_y
                        F_local[e_x, e_y, i, j] = val * weight_x * weight_y
        return u_exact, F_local

class SEMSolver:
    def __init__(self, mesh):
        self.mesh = mesh
        self.p = mesh.ref_el.p
        self._build_preconditioner()
        self._build_weight_matrix()
        
    def _build_preconditioner(self):
        diag_M_x = np.diag(self.mesh.M_1dx)
        diag_M_y = np.diag(self.mesh.M_1dy)
        diag_K_x = np.diag(self.mesh.K_1dx)
        diag_K_y = np.diag(self.mesh.K_1dy)
        
        D_local = np.zeros((self.mesh.E_x, self.mesh.E_y, self.p+1, self.p+1))
        for e_x in range(self.mesh.E_x):
            for e_y in range(self.mesh.E_y):
                for i in range(self.p+1):
                    for j in range(self.p+1):
                        D_local[e_x, e_y, i, j] = diag_K_x[i] * diag_M_y[j] + diag_M_x[i] * diag_K_y[j]
                        
        D_global = self.dss_np(D_local)
        self.inv_D_np = np.zeros_like(D_global)
        mask = D_global > 1e-14
        self.inv_D_np[mask] = 1.0 / D_global[mask]

    def _build_weight_matrix(self):
        self.W_np = np.ones((self.mesh.E_x, self.mesh.E_y, self.p+1, self.p+1))
        self.W_np[1:, :, 0, :] = 0.0
        self.W_np[:, 1:, :, 0] = 0.0

    def dss_np(self, v):
        v_new = v.copy()
        v_new[:-1, :, self.p, :] += v[1:, :, 0, :]
        v_new[1:, :, 0, :] = v_new[:-1, :, self.p, :]
        v_new[:, :-1, :, self.p] += v_new[:, 1:, :, 0]
        v_new[:, 1:, :, 0] = v_new[:, :-1, :, self.p]
        
        v_new[0, :, 0, :] = 0.0
        v_new[-1, :, self.p, :] = 0.0
        v_new[:, 0, :, 0] = 0.0
        v_new[:, -1, :, self.p] = 0.0
        return v_new

    def solve(self, b, max_iters, tol):
        raise NotImplementedError

class NumpySolver(SEMSolver):
    def __init__(self, mesh):
        super().__init__(mesh)
        
    def apply_K(self, u):
        v_local = self.mesh.K_1dx @ u @ self.mesh.M_1dy.T + self.mesh.M_1dx @ u @ self.mesh.K_1dy.T
        return self.dss_np(v_local)
        
    def solve(self, b, max_iters=2000, tol=1e-11):
        x = np.zeros_like(b)
        r = b.copy()
        z = r * self.inv_D_np
        p_vec = z.copy()
        rsold = np.sum(r * z * self.W_np)
        
        iters = 0
        for _ in range(max_iters):
            iters += 1
            Ap = self.apply_K(p_vec)
            pAp = np.sum(p_vec * Ap * self.W_np)
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * self.inv_D_np
            rsnew = np.sum(r * z_new * self.W_np)
            
            if np.sqrt(rsnew) < tol:
                break
                
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
            
        return x, iters

class PyTorchSolver(SEMSolver):
    def __init__(self, mesh, device_name='cpu'):
        super().__init__(mesh)
        self.device = torch.device(device_name)
        self.pt_M_1dx = torch.tensor(self.mesh.M_1dx, dtype=torch.float64, device=self.device)
        self.pt_K_1dx = torch.tensor(self.mesh.K_1dx, dtype=torch.float64, device=self.device)
        self.pt_M_1dy = torch.tensor(self.mesh.M_1dy, dtype=torch.float64, device=self.device)
        self.pt_K_1dy = torch.tensor(self.mesh.K_1dy, dtype=torch.float64, device=self.device)
        self.pt_inv_D = torch.tensor(self.inv_D_np, dtype=torch.float64, device=self.device)
        self.pt_W = torch.tensor(self.W_np, dtype=torch.float64, device=self.device)
        
        self.pt_mask = torch.zeros((mesh.E_x, mesh.E_y, self.p+1, self.p+1), dtype=torch.bool, device=self.device)
        self.pt_mask[0, :, 0, :] = True
        self.pt_mask[-1, :, self.p, :] = True
        self.pt_mask[:, 0, :, 0] = True
        self.pt_mask[:, -1, :, self.p] = True

    def dss(self, v):
        v_fast = v.clone()
        v_fast[:-1, :, self.p, :] += v[1:, :, 0, :]
        v_fast[1:, :, 0, :] = v_fast[:-1, :, self.p, :]
        v_fast[:, :-1, :, self.p] += v_fast[:, 1:, :, 0]
        v_fast[:, 1:, :, 0] = v_fast[:, :-1, :, self.p]
        return torch.where(self.pt_mask, torch.tensor(0.0, dtype=torch.float64, device=self.device), v_fast)

    def apply_K(self, u):
        v_local = torch.matmul(torch.matmul(self.pt_K_1dx, u), self.pt_M_1dy.T) + torch.matmul(torch.matmul(self.pt_M_1dx, u), self.pt_K_1dy.T)
        return self.dss(v_local)

    def solve(self, b_np, max_iters=2000, tol=1e-11):
        b = torch.tensor(b_np, dtype=torch.float64, device=self.device)
        x = torch.zeros_like(b)
        r = b.clone()
        z = r * self.pt_inv_D
        p_vec = z.clone()
        rsold = torch.sum(r * z * self.pt_W)
        
        iters = 0
        for _ in range(max_iters):
            iters += 1
            Ap = self.apply_K(p_vec)
            pAp = torch.sum(p_vec * Ap * self.pt_W)
            alpha = 0.0 if pAp < 1e-25 else rsold / pAp
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * self.pt_inv_D
            rsnew = torch.sum(r * z_new * self.pt_W)
            
            if torch.sqrt(rsnew) < tol:
                break
                
            beta = 0.0 if rsold < 1e-25 else rsnew / rsold
            p_vec = z_new + beta * p_vec
            rsold = rsnew
            
        return x, iters

class MLXSolver(SEMSolver):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.mx_M_1dx = mx.array(self.mesh.M_1dx, dtype=mx.float64)
        self.mx_K_1dx = mx.array(self.mesh.K_1dx, dtype=mx.float64)
        self.mx_M_1dy = mx.array(self.mesh.M_1dy, dtype=mx.float64)
        self.mx_K_1dy = mx.array(self.mesh.K_1dy, dtype=mx.float64)
        self.mx_inv_D = mx.array(self.inv_D_np, dtype=mx.float64)
        self.mx_W = mx.array(self.W_np, dtype=mx.float64)
        
        mask = np.zeros((mesh.E_x, mesh.E_y, self.p+1, self.p+1), dtype=bool)
        mask[0, :, 0, :] = True
        mask[-1, :, self.p, :] = True
        mask[:, 0, :, 0] = True
        mask[:, -1, :, self.p] = True
        self.mx_mask = mx.array(mask)

    def dss(self, v):
        E_x, E_y = self.mesh.E_x, self.mesh.E_y
        p = self.p
        
        update_p = mx.concatenate([v[1:, :, 0:1, :], mx.zeros((1, E_y, 1, p+1))], axis=0)
        update_0 = mx.concatenate([mx.zeros((1, E_y, 1, p+1)), v[:-1, :, p:p+1, :]], axis=0)
        update_x = mx.concatenate([update_0, mx.zeros((E_x, E_y, p-1, p+1)), update_p], axis=2)
        v_new = v + update_x
        
        update_p_y = mx.concatenate([v_new[:, 1:, :, 0:1], mx.zeros((E_x, 1, p+1, 1))], axis=1)
        update_0_y = mx.concatenate([mx.zeros((E_x, 1, p+1, 1)), v_new[:, :-1, :, p:p+1]], axis=1)
        update_y = mx.concatenate([update_0_y, mx.zeros((E_x, E_y, p+1, p-1)), update_p_y], axis=3)
        v_final = v_new + update_y
        
        return mx.where(self.mx_mask, mx.array(0.0, dtype=mx.float64), v_final)

    def apply_K(self, u):
        v_local = mx.matmul(mx.matmul(self.mx_K_1dx, u), self.mx_M_1dy.T) + mx.matmul(mx.matmul(self.mx_M_1dx, u), self.mx_K_1dy.T)
        return self.dss(v_local)

    @mx.compile
    def _compiled_cg(self, b, max_iters):
        x = mx.zeros_like(b)
        r = b
        z = r * self.mx_inv_D
        p_vec = z
        rsold = mx.sum(r * z * self.mx_W)
        
        for _ in range(max_iters):
            Ap = self.apply_K(p_vec)
            pAp = mx.sum(p_vec * Ap * self.mx_W)
            alpha = mx.where(pAp < 1e-25, mx.array(0.0, dtype=mx.float64), rsold / pAp)
            x = x + alpha * p_vec
            r = r - alpha * Ap
            z_new = r * self.mx_inv_D
            rsnew = mx.sum(r * z_new * self.mx_W)
            beta = mx.where(rsold < 1e-25, mx.array(0.0, dtype=mx.float64), rsnew / rsold)
            p_vec = z_new + beta * p_vec
            rsold = rsnew
            
        return x

    def solve(self, b_np, max_iters=2000, tol=1e-11):
        b = mx.array(b_np, dtype=mx.float64)
        x = self._compiled_cg(b, max_iters)
        return x, max_iters

class BenchmarkRunner:
    def __init__(self, p=15, E_x=5, E_y=15):
        self.ref_el = ReferenceElement(p)
        self.mesh = Mesh2D(E_x, E_y, ref_el=self.ref_el)
        self.problem = ProblemDefinition(
            exact_u_func=lambda x, y: np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y),
            forcing_f_func=lambda x, y: 32 * np.pi**2 * np.sin(4 * np.pi * x) * np.sin(4 * np.pi * y)
        )
        self.u_exact, self.F_local = self.problem.evaluate(self.mesh)
        
    def run_all(self):
        print(f"===========================================================")
        print(f"2D SEM Solver Benchmark (Object-Oriented Architecture)")
        print(f"Polynomial Degree (p): {self.ref_el.p}")
        print(f"Elements: {self.mesh.E_x} x {self.mesh.E_y} ({self.mesh.E_x * self.mesh.E_y} total)")
        print(f"===========================================================\n")
        
        solvers = [
            ("NumPy CPU", NumpySolver(self.mesh)),
            ("PyTorch CPU", PyTorchSolver(self.mesh, 'cpu'))
        ]
        
        if has_mlx:
            solvers.append(("MLX CPU (Compiled)", MLXSolver(self.mesh)))
        
        if torch.cuda.is_available():
            solvers.append(("PyTorch CUDA", PyTorchSolver(self.mesh, 'cuda')))
            
        b_np = solvers[0][1].dss_np(self.F_local)
        
        for name, solver in solvers:
            print(f"Running {name} Benchmark...")
            try:
                _ = solver.solve(b_np, max_iters=10)
                
                t0 = time.time()
                u, iters = solver.solve(b_np, max_iters=2000)
                t1 = time.time()
                
                if isinstance(u, torch.Tensor):
                    pt_u_exact = torch.tensor(self.u_exact, dtype=torch.float64, device=u.device)
                    err = torch.max(torch.abs(u - pt_u_exact)).item()
                elif has_mlx and isinstance(u, mx.array):
                    u_np = np.array(u)
                    err = np.max(np.abs(u_np - self.u_exact))
                else:
                    u_np = u
                    err = np.max(np.abs(u_np - self.u_exact))
                    
                print(f"  [{name}] Iters: {iters} | Time: {t1-t0:.5f}s | Max Error: {err:.2e}\n")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  [{name}] Failed: {str(e)}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--degree", type=int, default=15)
    parser.add_argument("-ex", "--elem_x", type=int, default=5)
    parser.add_argument("-ey", "--elem_y", type=int, default=15)
    args = parser.parse_args()
    
    runner = BenchmarkRunner(p=args.degree, E_x=args.elem_x, E_y=args.elem_y)
    runner.run_all()
