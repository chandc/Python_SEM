import numpy as np
import pytest
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel
from lssem2d.assembly import gather_scatter

def test_gather_scatter_multiplicities():
    # a) Set a field to 1.0 everywhere. After GS, interior corner nodes must equal 4.0,
    # edge nodes 2.0, interior nodes 1.0. (Assumes a structured multielement grid).
    N = 4
    mesh = build_channel(L_x=2.0, L_y=2.0, E_x=3, E_y=3, N=N)
    
    U = np.ones((mesh.nelem, N + 1, N + 1))
    U_gs = gather_scatter(mesh, U)
    
    # We can deduce the multiplicities expected just by looking at the location.
    # An interior element (e=4, the center of 3x3) has:
    # - 4 corner nodes (multiplicity 4)
    # - 4*(N-1) edge nodes (multiplicity 2)
    # - (N-1)^2 interior nodes (multiplicity 1)
    
    # Element 4 is the middle element (0,1,2; 3,4,5; 6,7,8)
    e_mid = 4
    U_mid = U_gs[e_mid]
    
    # Corners
    assert U_mid[0, 0] == 4.0
    assert U_mid[0, -1] == 4.0
    assert U_mid[-1, 0] == 4.0
    assert U_mid[-1, -1] == 4.0
    
    # Edges (excluding corners)
    np.testing.assert_allclose(U_mid[0, 1:-1], 2.0)
    np.testing.assert_allclose(U_mid[-1, 1:-1], 2.0)
    np.testing.assert_allclose(U_mid[1:-1, 0], 2.0)
    np.testing.assert_allclose(U_mid[1:-1, -1], 2.0)
    
    # Interior
    np.testing.assert_allclose(U_mid[1:-1, 1:-1], 1.0)
    
    # Also test the 4D version works identically
    U4 = np.ones((mesh.nelem, N + 1, N + 1, 3))
    U4_gs = gather_scatter(mesh, U4)
    np.testing.assert_allclose(U4_gs[..., 0], U_gs)
    np.testing.assert_allclose(U4_gs[..., 1], U_gs)
    np.testing.assert_allclose(U4_gs[..., 2], U_gs)


def test_gather_scatter_symmetry():
    # b) Random symmetry test: <U, Q^T Q V> == <Q^T Q U, V> to machine precision.
    N = 4
    mesh = build_channel(L_x=1.0, L_y=1.0, E_x=4, E_y=5, N=N)
    
    # Test 3D
    np.random.seed(42)
    U = np.random.randn(mesh.nelem, N + 1, N + 1)
    V = np.random.randn(mesh.nelem, N + 1, N + 1)
    
    U_gs = gather_scatter(mesh, U)
    V_gs = gather_scatter(mesh, V)
    
    dot1 = np.sum(U * V_gs)
    dot2 = np.sum(U_gs * V)
    assert abs(dot1 - dot2) / max(abs(dot1), 1e-15) < 1e-13
    
    # Test 4D
    U4 = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    V4 = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    
    U4_gs = gather_scatter(mesh, U4)
    V4_gs = gather_scatter(mesh, V4)
    
    dot1_4 = np.sum(U4 * V4_gs)
    dot2_4 = np.sum(U4_gs * V4)
    assert abs(dot1_4 - dot2_4) / max(abs(dot1_4), 1e-15) < 1e-13


def test_gather_scatter_performance():
    # c) Performance gate: For N=8, 400 elements, one GS must take < 2ms on numpy.
    N = 8
    # E_x = 20, E_y = 20 -> 400 elements
    mesh = build_channel(L_x=1.0, L_y=1.0, E_x=20, E_y=20, N=N)
    assert mesh.nelem == 400
    
    # Usually the solver handles 4 fields at once
    U = np.random.randn(mesh.nelem, N + 1, N + 1, 4)
    
    # Warm up
    _ = gather_scatter(mesh, U)
    
    # Time 100 iterations
    n_iters = 100
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = gather_scatter(mesh, U)
    t1 = time.perf_counter()
    
    avg_ms = (t1 - t0) * 1000.0 / n_iters
    print(f"\nAverage gather_scatter time (4 fields): {avg_ms:.3f} ms")
    
    # Gate check
    assert avg_ms < 2.0, f"Performance gate failed: GS took {avg_ms:.3f} ms, expected < 2ms"
