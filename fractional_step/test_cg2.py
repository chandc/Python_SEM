import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.bc import apply_mask
from lssem2d.solver import apply_A, cg_solve
from lssem2d.assembly import gather_scatter

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
# Make it continuous!
x_exact = gather_scatter(mesh, x_exact)
x_exact = apply_mask(mesh, x_exact)

b = apply_A(state, x_exact, fu, fv)
b = apply_mask(mesh, b)

x_num, iters = cg_solve(state, b, fu, fv, tol=1e-10)

b_num = apply_A(state, x_num, fu, fv)
r_norm = np.sqrt(np.sum((b - b_num)**2))
b_norm = np.sqrt(np.sum(b**2))
print("CG Iters:", iters)
print("Rel Res:", r_norm / b_norm)
