import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT

def inner(A, B):
    """Unweighted dot product, since apply_L folds wq into its output."""
    return np.sum(A * B)

@pytest.mark.parametrize("N", [4, 8])
def test_full_operator_adjoint(N):
    # a) FULL-OPERATOR ADJOINT TEST. For random U, S:
    # <apply_L(U), S>  ==  <U, apply_LT(S)>     to 1e-12 relative
    # on a mesh with non-uniform elements, with fu,fv random and non-zero
    
    mesh = build_channel(L_x=3.0, L_y=1.5, E_x=2, E_y=3, N=N)
    # Make mesh non-uniform artificially
    mesh.hx = np.random.uniform(0.5, 1.5, mesh.nelem)
    mesh.hy = np.random.uniform(0.5, 1.5, mesh.nelem)
    mesh.setup_derived()
    
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=0.1, dt=0.5, fac1=1.5)
    
    np.random.seed(42 + N)
    target_shape = (mesh.nelem, N + 1, N + 1)
    fu = np.random.randn(*target_shape)
    fv = np.random.randn(*target_shape)
    state.update_linearisation(fu, fv)
    
    U = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    S_unweighted = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    
    # Weight S since apply_LT expects a weighted residual (dual vector)
    wq = mesh.wq
    S_weighted = S_unweighted.copy()
    for k in range(4):
        S_weighted[..., k] *= wq
    
    LU = apply_L(state, U, fu, fv)
    LTS = apply_LT(state, S_weighted, fu, fv)
    
    # <W L U, S> == <U, L^T (W S)>
    dot1 = inner(LU, S_unweighted)
    dot2 = inner(U, LTS)
    
    assert abs(dot1 - dot2) / max(abs(dot1), 1e-15) < 1e-12

@pytest.mark.parametrize("N", [4, 8])
def test_symmetry_and_positivity(N):
    # b) SYMMETRY TEST. Define A(U) = apply_LT(apply_L(U)). For random U, V:
    # <A(U), V> == <U, A(V)> to 1e-12 relative
    # c) POSITIVITY. <A(U), U> >= 0 for 100 random U.
    
    mesh = build_channel(L_x=2.0, L_y=2.0, E_x=2, E_y=2, N=N)
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=0.01, dt=1.0, fac1=1.0)
    
    target_shape = (mesh.nelem, N + 1, N + 1)
    fu = np.random.randn(*target_shape)
    fv = np.random.randn(*target_shape)
    state.update_linearisation(fu, fv)
    
    def A(U):
        return apply_LT(state, apply_L(state, U, fu, fv), fu, fv).copy()
        
    U = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    V = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    
    AU = A(U)
    AV = A(V)
    
    dot1 = inner(AU, V)
    dot2 = inner(U, AV)
    
    assert abs(dot1 - dot2) / max(abs(dot1), 1e-15) < 1e-12
    
    # c) POSITIVITY
    for _ in range(100):
        Ur = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
        val = inner(A(Ur), Ur)
        # Note: tolerance for zero
        assert val > -1e-13

def test_dense_cross_check():
    # d) DENSE CROSS-CHECK on a tiny case (1 element, N=2): build L explicitly column by
    # column by applying apply_L to unit vectors, form L^T numerically, and verify
    # apply_LT reproduces it to 1e-12.
    
    N = 2
    mesh = build_channel(L_x=1.0, L_y=1.0, E_x=1, E_y=1, N=N)
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=1.0, dt=1.0, fac1=1.0)
    
    target_shape = (mesh.nelem, N + 1, N + 1)
    fu = np.random.randn(*target_shape)
    fv = np.random.randn(*target_shape)
    state.update_linearisation(fu, fv)
    
    dofs = mesh.nelem * (N + 1) * (N + 1) * 4
    shape = (mesh.nelem, N + 1, N + 1, 4)
    
    L_matrix = np.zeros((dofs, dofs))
    
    # Build L column by column
    for j in range(dofs):
        U = np.zeros(dofs)
        U[j] = 1.0
        U_reshaped = U.reshape(shape)
        
        LU = apply_L(state, U_reshaped, fu, fv)
        L_matrix[:, j] = LU.flatten()
        
    # The true algebraic transpose of the matrix operator
    LT_matrix = L_matrix.T
    
    # Verify apply_LT matches LT_matrix column by column
    for i in range(dofs):
        S = np.zeros(dofs)
        S[i] = 1.0
        S_reshaped = S.reshape(shape).copy()
        
        # apply_LT expects a weighted residual
        wq = mesh.wq
        for k in range(4):
            S_reshaped[..., k] *= wq
            
        LTS_num = apply_LT(state, S_reshaped, fu, fv).flatten()
        LTS_exact = LT_matrix[:, i]
        
        np.testing.assert_allclose(LTS_num, LTS_exact, atol=1e-12, rtol=1e-12)
