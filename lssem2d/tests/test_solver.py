import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.bc import apply_mask
from lssem2d.solver import apply_A, cg_solve

def test_cg_solve():
    # a) test_cg_solve: manufacture a positive-definite RHS, solve it, check A*x == b.
    N = 4
    mesh = build_channel(L_x=1.0, L_y=1.0, E_x=2, E_y=2, N=N)
    
    # We must properly apply the bc mask to define the physical boundary walls
    # build_channel goes from y=0 to y=1, so we must shift it to y=-1 to y=1
    mesh.y0 -= 1.0
    mesh.ynod -= 1.0
    
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=0.1, dt=1.0, fac1=1.0)
    
    target_shape = (mesh.nelem, N + 1, N + 1)
    fu = np.random.randn(*target_shape)
    fv = np.random.randn(*target_shape)
    state.update_linearisation(fu, fv)
    
    # Manufacture a known solution x_exact
    np.random.seed(42)
    x_exact = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    # Make it continuous so that b is in the correct subspace
    from lssem2d.assembly import gather_scatter
    x_exact = gather_scatter(mesh, x_exact)
    # Mask it so it satisfies Dirichlet BCs
    x_exact = apply_mask(mesh, x_exact)
    
    # Form RHS b = A * x_exact
    b = apply_A(state, x_exact, fu, fv)
    # Ensure RHS is masked as well (A operator output is already masked, but just to be sure)
    b = apply_mask(mesh, b)
    
    # Solve for x
    x_num, iters = cg_solve(state, b, fu, fv, tol=1e-10, max_iter=2000)
    
    # Check that A*x == b (the residual was driven to zero)
    # Note: CG solves it iteratively, so we check if the tolerance was met.
    b_num = apply_A(state, x_num, fu, fv)
    
    # The solver returns when ||r|| < tol * ||b||
    b_norm = np.sqrt(np.sum(b * b))
    residual = b - b_num
    r_norm = np.sqrt(np.sum(residual * residual))
    
    assert r_norm < 1e-6 * b_norm
    
    # The matrix A = L^T M L is positive semi-definite and highly singular here because
    # East/West boundaries are unconstrained (no periodicity or inlet/outlet BCs applied in this test).
    # Thus, x_num and x_exact will differ by a null space mode, but A*x = b is strictly satisfied!
