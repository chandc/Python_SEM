import numpy as np
import pytest
import sys
import os

# Add the parent directory to sys.path so we can import lssem2d modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.lgl import lgl_nodes, lgl_weights, diff_matrix

@pytest.mark.parametrize("N", [2, 4, 8, 16])
def test_lgl_nodes(N):
    # a) Nodes are symmetric about 0 and include ±1 exactly.
    xi = lgl_nodes(N)
    
    assert xi[0] == -1.0
    assert xi[-1] == 1.0
    
    # Check symmetry
    np.testing.assert_allclose(xi, -xi[::-1], atol=1e-15, rtol=1e-15)

@pytest.mark.parametrize("N", [2, 4, 8, 16])
def test_lgl_quadrature(N):
    # b) Quadrature is exact for polynomials up to degree 2N-1:
    # integrate x^m on [-1,1] for m=0..2N-1, compare to the analytic value, tol 1e-12.
    xi = lgl_nodes(N)
    w = lgl_weights(N)
    
    for m in range(2 * N):
        # Analytic integral of x^m from -1 to 1 is:
        # [x^(m+1) / (m+1)]_-1^1 = (1 - (-1)^(m+1)) / (m+1)
        if m % 2 == 0:
            analytic = 2.0 / (m + 1)
        else:
            analytic = 0.0
            
        numerical = np.sum(w * (xi ** m))
        assert abs(numerical - analytic) < 1e-12

@pytest.mark.parametrize("N", [2, 4, 8, 16])
def test_diff_matrix(N):
    # c) D differentiates exactly: for f = x^m (m <= N), D @ f equals m*x^(m-1) to 1e-10.
    xi = lgl_nodes(N)
    D = diff_matrix(N)
    
    for m in range(N + 1):
        f = xi ** m
        df_num = D @ f
        if m == 0:
            df_ana = np.zeros_like(xi)
        else:
            df_ana = m * (xi ** (m - 1))
            
        np.testing.assert_allclose(df_num, df_ana, atol=1e-10, rtol=1e-10)
        
    # d) Row sums of D are zero to 1e-12 (differentiates a constant to zero).
    row_sums = np.sum(D, axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-12, rtol=1e-12)
