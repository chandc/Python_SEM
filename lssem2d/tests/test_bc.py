import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel
from lssem2d.bc import apply_mask

def test_apply_mask():
    # a) Set U to 1.0 everywhere. Call apply_mask. Assert U[e, :, 0, 0:2] == 0 for
    # elements touching y=-1. Assert U[e, :, -1, 0:2] == 0 for elements touching y=+1.
    # All other values remain 1.0.
    
    N = 4
    # E_y = 4, so elements 0, 1, 2, 3 in y-direction.
    # South boundary elements will be those with neighbour[e, 2] == -1
    # North boundary elements will be those with neighbour[e, 3] == -1
    mesh = build_channel(L_x=2.0, L_y=2.0, E_x=3, E_y=4, N=N)
    
    # Let's shift y to range [-1, 1] to match the prompt's description precisely
    mesh.y0 -= 1.0
    mesh.ynod -= 1.0
    
    U = np.ones((mesh.nelem, N + 1, N + 1, 4))
    U_masked = apply_mask(mesh, U)
    
    # Check boundaries and interiors
    for e in range(mesh.nelem):
        south_boundary = mesh.neighbour[e, 2] == -1
        north_boundary = mesh.neighbour[e, 3] == -1
        
        if south_boundary:
            # Check u and v are zeroed
            assert np.all(U_masked[e, :, 0, 0] == 0.0)
            assert np.all(U_masked[e, :, 0, 1] == 0.0)
            # Check p and om are unaffected
            assert np.all(U_masked[e, :, 0, 2] == 1.0)
            assert np.all(U_masked[e, :, 0, 3] == 1.0)
        else:
            assert np.all(U_masked[e, :, 0, :] == 1.0)
            
        if north_boundary:
            # Check u and v are zeroed
            assert np.all(U_masked[e, :, -1, 0] == 0.0)
            assert np.all(U_masked[e, :, -1, 1] == 0.0)
            # Check p and om are unaffected
            assert np.all(U_masked[e, :, -1, 2] == 1.0)
            assert np.all(U_masked[e, :, -1, 3] == 1.0)
        else:
            assert np.all(U_masked[e, :, -1, :] == 1.0)
            
        # Interior of the element (j from 1 to N-1) is always unaffected
        assert np.all(U_masked[e, :, 1:-1, :] == 1.0)
