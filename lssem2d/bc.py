import numpy as np


# Physical coordinates of the four element edges: W=0, E=1, S=2, N=3.
#
# ON AN AFFINE MESH the nodes are a tensor product, so a west edge has ONE x and
# a whole row of y, and the callbacks below are handed (scalar, vector) and
# broadcast.  ON A CURVILINEAR MESH that is simply false -- a curved edge varies
# in both coordinates -- so the true nodal fields `mesh.X`, `mesh.Y` are used
# instead and the callback receives (vector, vector).  Every callback in the
# repo is numpy-vectorised, so the shape change is transparent to it.
#
# The affine branch is kept rather than routed through X/Y so that existing
# validations stay bit-identical: X/Y would give the same numbers, but not the
# same floating-point operations.
_EDGE = {0: (np.s_[0, :], 0), 1: (np.s_[-1, :], -1), 2: (np.s_[:, 0], 0), 3: (np.s_[:, -1], -1)}


def edge_xy(mesh, e, side):
    if getattr(mesh, 'curvilinear', False):
        sl = _EDGE[side][0]
        return mesh.X[e][sl], mesh.Y[e][sl]
    k = _EDGE[side][1]
    if side in (0, 1):
        return mesh.xnod[e, k], mesh.ynod[e, :]
    return mesh.xnod[e, :], mesh.ynod[e, k]


def apply_mask(mesh, U, pin_p=False):
    """
    Apply Dirichlet boundary condition mask to the state U.
    Sets known degrees of freedom to zero for iterative solver perturbations.
    
    Fields: 0: u, 1: v, 2: p, 3: om
    
    Edge codes:
    1: no-slip (u=0, v=0)
    2: lid (u=0, v=0)
    3: inlet (u=0, v=0)
    4: outlet (p=0)
    5: symmetry (v=0, om=0)
    """
    U_masked = U.copy()
    
    for e in range(mesh.nelem):
        # W=0, E=1, S=2, N=3
        # W (i=0)
        bc_W = mesh.bc[e, 0]
        if bc_W in (1, 2, 3):
            U_masked[e, 0, :, 0:2] = 0.0
        elif bc_W == 4:
            U_masked[e, 0, :, 2] = 0.0
        elif bc_W == 5:
            U_masked[e, 0, :, 1] = 0.0
            U_masked[e, 0, :, 3] = 0.0
            
        # E (i=N)
        bc_E = mesh.bc[e, 1]
        if bc_E in (1, 2, 3):
            U_masked[e, -1, :, 0:2] = 0.0
        elif bc_E == 4:
            U_masked[e, -1, :, 2] = 0.0
        elif bc_E == 5:
            U_masked[e, -1, :, 1] = 0.0
            U_masked[e, -1, :, 3] = 0.0
            
        # S (j=0)
        bc_S = mesh.bc[e, 2]
        if bc_S in (1, 2, 3):
            U_masked[e, :, 0, 0:2] = 0.0
        elif bc_S == 4:
            U_masked[e, :, 0, 2] = 0.0
        elif bc_S == 5:
            U_masked[e, :, 0, 1] = 0.0
            U_masked[e, :, 0, 3] = 0.0
            
        # N (j=N)
        bc_N = mesh.bc[e, 3]
        if bc_N in (1, 2, 3):
            U_masked[e, :, -1, 0:2] = 0.0
        elif bc_N == 4:
            U_masked[e, :, -1, 2] = 0.0
        elif bc_N == 5:
            U_masked[e, :, -1, 1] = 0.0
            U_masked[e, :, -1, 3] = 0.0
            
    return U_masked

def apply_bc(mesh, U, time=0.0, custom_inlet=None, custom_lid=None, exact_solution=None, pin_p=False):
    """
    Writes exact prescribed values into the state array U for Dirichlet boundaries.
    """
    for e in range(mesh.nelem):
        # Iterate over the 4 boundaries
        # W
        bc = mesh.bc[e, 0]
        if bc in (1, 2, 3, 4, 5) and exact_solution:
            u_ex, v_ex, p_ex, om_ex = exact_solution(*edge_xy(mesh, e, 0), time)
            if bc in (1, 2, 3):
                U[e, 0, :, 0] = u_ex; U[e, 0, :, 1] = v_ex
            elif bc == 5:
                U[e, 0, :, 1] = v_ex; U[e, 0, :, 3] = om_ex
        else:
            if bc == 1:
                U[e, 0, :, 0:2] = 0.0
            elif bc == 2:
                U[e, 0, :, 1] = 0.0
                U[e, 0, :, 0] = custom_lid(*edge_xy(mesh, e, 0), time) if custom_lid else 1.0
            elif bc == 3:
                U[e, 0, :, 1] = 0.0
                if custom_inlet:
                    U[e, 0, :, 0] = custom_inlet(*edge_xy(mesh, e, 0), time)
            elif bc == 4:
                U[e, 0, :, 2] = 0.0
            elif bc == 5:
                U[e, 0, :, 1] = 0.0; U[e, 0, :, 3] = 0.0
            
        # E
        bc = mesh.bc[e, 1]
        if bc in (1, 2, 3, 4, 5) and exact_solution:
            u_ex, v_ex, p_ex, om_ex = exact_solution(*edge_xy(mesh, e, 1), time)
            if bc in (1, 2, 3):
                U[e, -1, :, 0] = u_ex; U[e, -1, :, 1] = v_ex
            elif bc == 4:
                U[e, -1, :, 2] = p_ex
            elif bc == 5:
                U[e, -1, :, 1] = v_ex; U[e, -1, :, 3] = om_ex
        else:
            if bc == 1:
                U[e, -1, :, 0:2] = 0.0
            elif bc == 2:
                U[e, -1, :, 1] = 0.0
                U[e, -1, :, 0] = custom_lid(*edge_xy(mesh, e, 1), time) if custom_lid else 1.0
            elif bc == 3:
                U[e, -1, :, 1] = 0.0
                if custom_inlet:
                    U[e, -1, :, 0] = custom_inlet(*edge_xy(mesh, e, 1), time)
            elif bc == 4:
                U[e, -1, :, 2] = 0.0
            elif bc == 5:
                U[e, -1, :, 1] = 0.0; U[e, -1, :, 3] = 0.0
            
        # S
        bc = mesh.bc[e, 2]
        if bc in (1, 2, 3, 4, 5) and exact_solution:
            u_ex, v_ex, p_ex, om_ex = exact_solution(*edge_xy(mesh, e, 2), time)
            if bc in (1, 2, 3):
                U[e, :, 0, 0] = u_ex; U[e, :, 0, 1] = v_ex
            elif bc == 4:
                U[e, :, 0, 2] = p_ex
            elif bc == 5:
                U[e, :, 0, 1] = v_ex; U[e, :, 0, 3] = om_ex
        else:
            if bc == 1:
                U[e, :, 0, 0:2] = 0.0
            elif bc == 2:
                U[e, :, 0, 1] = 0.0
                U[e, :, 0, 0] = custom_lid(*edge_xy(mesh, e, 2), time) if custom_lid else 1.0
            elif bc == 3:
                U[e, :, 0, 1] = 0.0
                if custom_inlet:
                    U[e, :, 0, 0] = custom_inlet(*edge_xy(mesh, e, 2), time)
            elif bc == 4:
                U[e, :, 0, 2] = 0.0
            elif bc == 5:
                U[e, :, 0, 1] = 0.0; U[e, :, 0, 3] = 0.0
            
        # N
        bc = mesh.bc[e, 3]
        if bc in (1, 2, 3, 4, 5) and exact_solution:
            u_ex, v_ex, p_ex, om_ex = exact_solution(*edge_xy(mesh, e, 3), time)
            if bc in (1, 2, 3):
                U[e, :, -1, 0] = u_ex; U[e, :, -1, 1] = v_ex
            elif bc == 4:
                U[e, :, -1, 2] = p_ex
            elif bc == 5:
                U[e, :, -1, 1] = v_ex; U[e, :, -1, 3] = om_ex
        else:
            if bc == 1:
                U[e, :, -1, 0:2] = 0.0
            elif bc == 2:
                U[e, :, -1, 1] = 0.0
                U[e, :, -1, 0] = custom_lid(*edge_xy(mesh, e, 3), time) if custom_lid else 1.0
            elif bc == 3:
                U[e, :, -1, 1] = 0.0
                if custom_inlet:
                    U[e, :, -1, 0] = custom_inlet(*edge_xy(mesh, e, 3), time)
            elif bc == 4:
                U[e, :, -1, 2] = 0.0
            elif bc == 5:
                U[e, :, -1, 1] = 0.0; U[e, :, -1, 3] = 0.0
            
    return U
