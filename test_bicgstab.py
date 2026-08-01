import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.bc import apply_mask
from lssem2d.solver import apply_A

def bicgstab(state, b, fu, fv, tol=1e-10, max_iter=2000):
    x = np.zeros_like(b)
    r = b - apply_A(state, x, fu, fv)
    r = apply_mask(state.mesh, r)
    r0_star = r.copy()
    
    rho_prev = 1.0
    alpha = 1.0
    omega = 1.0
    
    p = np.zeros_like(b)
    v = np.zeros_like(b)
    
    b_norm = np.sqrt(np.sum(b*b))
    
    for i in range(max_iter):
        rho = np.sum(r0_star * r)
        if abs(rho) < 1e-20:
            break
            
        beta = (rho / rho_prev) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        v = apply_A(state, p, fu, fv)
        alpha = rho / np.sum(r0_star * v)
        
        s = r - alpha * v
        s_norm = np.sqrt(np.sum(s*s))
        if s_norm < tol * b_norm:
            x = x + alpha * p
            return x, i+1
            
        t = apply_A(state, s, fu, fv)
        omega = np.sum(t * s) / np.sum(t * t)
        
        x = x + alpha * p + omega * s
        r = s - omega * t
        
        rho_prev = rho
        
        r_norm = np.sqrt(np.sum(r*r))
        if r_norm < tol * b_norm:
            return x, i+1
            
    return x, max_iter

N = 4
mesh = build_channel(1.0, 1.0, 2, 2, N)
mesh.y0 -= 1.0
mesh.ynod -= 1.0

D = diff_matrix(N)
state = SolverState(mesh, D, nu=0.1, dt=1.0, fac1=1.0)
fu = np.random.randn(mesh.nelem, N+1, N+1)
fv = np.random.randn(mesh.nelem, N+1, N+1)
state.update_linearisation(fu, fv)

np.random.seed(42)
x_exact = np.random.randn(mesh.nelem, N+1, N+1, 4)
x_exact = apply_mask(mesh, x_exact)

b = apply_A(state, x_exact, fu, fv)
b = apply_mask(mesh, b)

x_num, iters = bicgstab(state, b, fu, fv, tol=1e-10)

b_num = apply_A(state, x_num, fu, fv)
r_norm = np.sqrt(np.sum((b - b_num)**2))
b_norm = np.sqrt(np.sum(b**2))
print("BiCGSTAB Iters:", iters)
print("Rel Res:", r_norm / b_norm)
