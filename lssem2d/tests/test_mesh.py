import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel, build_bfs, plot_mesh
from lssem2d.lgl import lgl_nodes

def test_node_coordinates():
    # a) Node coordinates match a hand-computed 2-element mesh.
    # 2 elements in x, 1 in y. Domain [0, 2] x [0, 1]
    # Element 0: [0, 1] x [0, 1], Element 1: [1, 2] x [0, 1]
    mesh = build_channel(L_x=2.0, L_y=1.0, E_x=2, E_y=1, N=2)
    
    assert mesh.nelem == 2
    assert mesh.nterm == 3
    
    xi = lgl_nodes(2) # [-1, 0, 1]
    
    # Element 0 (left)
    np.testing.assert_allclose(mesh.xnod[0, :], 0.5 + 0.5 * xi, atol=1e-14)
    np.testing.assert_allclose(mesh.ynod[0, :], 0.5 + 0.5 * xi, atol=1e-14)
    
    # Element 1 (right)
    np.testing.assert_allclose(mesh.xnod[1, :], 1.5 + 0.5 * xi, atol=1e-14)
    np.testing.assert_allclose(mesh.ynod[1, :], 0.5 + 0.5 * xi, atol=1e-14)

def test_wq_sum():
    # b) sum(wq) over all elements equals the total domain area to 1e-12.
    L_x, L_y = 3.0, 2.5
    E_x, E_y = 3, 2
    mesh = build_channel(L_x, L_y, E_x, E_y, N=4)
    
    total_area = np.sum(mesh.wq)
    expected_area = L_x * L_y
    
    assert abs(total_area - expected_area) < 1e-12
    
    # Check BFS as well
    # Inlet: 2 * 1 = 2
    # Outlet top: 6 * 1 = 6
    # Outlet bot: 6 * 1 = 6
    # Total = 14
    mesh_bfs = build_bfs(N=3)
    total_area_bfs = np.sum(mesh_bfs.wq)
    assert abs(total_area_bfs - 14.0) < 1e-12

def test_neighbour_consistency():
    # c) Neighbour/BC arrays are self-consistent: if neighbour[e,E]=f then neighbour[f,W]=e.
    mesh = build_channel(L_x=2.0, L_y=2.0, E_x=3, E_y=3, N=2)
    
    for e in range(mesh.nelem):
        # W(0), E(1), S(2), N(3)
        # Opposite pairs: W <-> E (0 <-> 1), S <-> N (2 <-> 3)
        pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
        
        for d1, d2 in pairs:
            f = mesh.neighbour[e, d1]
            if f != -1:
                assert mesh.neighbour[f, d2] == e

    mesh_bfs = build_bfs(N=2)
    for e in range(mesh_bfs.nelem):
        pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
        for d1, d2 in pairs:
            f = mesh_bfs.neighbour[e, d1]
            if f != -1:
                assert mesh_bfs.neighbour[f, d2] == e

def test_plot_mesh():
    # d) A plotting helper renders elements + collocation points (visual check only).
    mesh = build_bfs(N=4)
    plot_path = 'mesh_bfs_test.png'
    if os.path.exists(plot_path):
        os.remove(plot_path)
        
    plot_mesh(mesh, filename=plot_path)
    assert os.path.exists(plot_path)
    # clean up
    os.remove(plot_path)
