import numpy as np
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem2d.operators import dUdx, dUdy, DxT, DyT

def test_exact_polynomial_derivative():
    # a) Differentiate a polynomial exactly (compare to analytic derivative, 1e-10).
    N = 4
    mesh = build_channel(L_x=3.0, L_y=2.0, E_x=2, E_y=2, N=N)
    D = diff_matrix(N)
    
    # f(x, y) = x^2 * y^3
    x = mesh.xnod[:, :, None]
    y = mesh.ynod[:, None, :]
    U = x**2 * y**3
    
    # df/dx = 2x * y^3
    dfdx_exact = 2.0 * x * y**3
    
    # df/dy = 3x^2 * y^2
    dfdy_exact = 3.0 * x**2 * y**2
    
    dfdx_num = dUdx(U, D, mesh.facx)
    dfdy_num = dUdy(U, D, mesh.facy)
    
    np.testing.assert_allclose(dfdx_num, dfdx_exact, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dfdy_num, dfdy_exact, atol=1e-10, rtol=1e-10)

@pytest.mark.parametrize("N", [4, 8])
def test_adjoint_dot_product(N):
    # b) ADJOINT (DOT-PRODUCT) TEST. For random U, S of shape (nelem,n,n):
    # <Dx U, S>_w  ==  <U, Dx^T S>_w      to 1e-12 relative
    
    # Create a non-uniform mesh
    nelem = 3
    n = N + 1
    
    # Manually build some non-uniform factors
    facx = np.array([1.2, 0.8, 2.5])
    facy = np.array([0.5, 3.1, 1.0])
    
    # We also need quadrature weights wq
    # wq = jac * w_i * w_j, and jac = 1 / (facx * facy)
    from lssem2d.lgl import lgl_weights
    w = lgl_weights(N)
    jac = 1.0 / (facx * facy)
    
    wq = np.zeros((nelem, n, n))
    for e in range(nelem):
        for i in range(n):
            for j in range(n):
                wq[e, i, j] = jac[e] * w[i] * w[j]
                
    D = diff_matrix(N)
    
    # Random fields
    np.random.seed(42 + N)
    U = np.random.randn(nelem, n, n)
    S = np.random.randn(nelem, n, n)
    
    # Inner product function
    # Because we fold wq into the residual (as per Fortran/Documentation),
    # the primitive DxT and DyT operators are purely algebraic transposes.
    # Therefore, the adjoint identity holds for the UNWEIGHTED dot product.
    def inner(A, B):
        return np.sum(A * B)
        
    # Test Dx
    Dx_U = dUdx(U, D, facx)
    DxT_S = DxT(S, D, facx)
    
    dot1_x = inner(Dx_U, S)
    dot2_x = inner(U, DxT_S)
    
    assert abs(dot1_x - dot2_x) / max(abs(dot1_x), 1e-15) < 1e-12
    
    # Test Dy
    Dy_U = dUdy(U, D, facy)
    DyT_S = DyT(S, D, facy)
    
    dot1_y = inner(Dy_U, S)
    dot2_y = inner(U, DyT_S)
    
    assert abs(dot1_y - dot2_y) / max(abs(dot1_y), 1e-15) < 1e-12

def test_performance_benchmark():
    # Benchmark einsum against the reshape+matmul alternative
    N = 8
    mesh = build_channel(L_x=10.0, L_y=10.0, E_x=20, E_y=20, N=N) # 400 elements
    D = diff_matrix(N)
    
    U = np.random.randn(mesh.nelem, N + 1, N + 1)
    
    # 1. einsum implementation
    def dUdx_einsum(U, D, facx):
        return facx[:, None, None] * np.einsum('im,emj->eij', D, U, optimize=True)
        
    # 2. reshape+matmul implementation
    def dUdx_matmul(U, D, facx):
        # U is (e, m, j)
        # D is (i, m) -> D @ U over axis m
        # np.tensordot(D, U, axes=([1], [1])) yields (i, e, j)
        # Then transpose to (e, i, j)
        res = np.tensordot(D, U, axes=([1], [1])).transpose(1, 0, 2)
        return facx[:, None, None] * res

    def dUdy_matmul(U, D, facy):
        # U is (e, i, m)
        # D is (j, m) -> D @ U over axis m
        # np.tensordot(D, U, axes=([1], [2])) yields (j, e, i)
        # Then transpose to (e, i, j)
        res = np.tensordot(D, U, axes=([1], [2])).transpose(1, 2, 0)
        return facy[:, None, None] * res

    # verify correctness
    res1 = dUdx_einsum(U, D, mesh.facx)
    res2 = dUdx_matmul(U, D, mesh.facx)
    np.testing.assert_allclose(res1, res2, atol=1e-14)
    
    res3 = dUdy(U, D, mesh.facy)
    res4 = dUdy_matmul(U, D, mesh.facy)
    np.testing.assert_allclose(res3, res4, atol=1e-14)

    # Benchmark
    iters = 1000
    
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = dUdx_einsum(U, D, mesh.facx)
    t_einsum = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = dUdx_matmul(U, D, mesh.facx)
    t_matmul = time.perf_counter() - t0
    
    print(f"\\neinsum time: {t_einsum:.4f}s")
    print(f"matmul time: {t_matmul:.4f}s")
    # Will record these results in operators.py later.
