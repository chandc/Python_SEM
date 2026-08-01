import numpy as np

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
            
    if pin_p:
        U_masked[0, 0, 0, 2] = 0.0
        
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
            u_ex, v_ex, p_ex, om_ex = exact_solution(mesh.xnod[e, 0], mesh.ynod[e, :], time)
            if bc in (1, 2, 3):
                U[e, 0, :, 0] = u_ex; U[e, 0, :, 1] = v_ex
            elif bc == 4:
                U[e, 0, :, 2] = p_ex
            elif bc == 5:
                U[e, 0, :, 1] = v_ex; U[e, 0, :, 3] = om_ex
        else:
            if bc == 1:
                U[e, 0, :, 0:2] = 0.0
            elif bc == 2:
                U[e, 0, :, 1] = 0.0
                U[e, 0, :, 0] = custom_lid(mesh.xnod[e, 0], mesh.ynod[e, :], time) if custom_lid else 1.0
            elif bc == 3:
                U[e, 0, :, 1] = 0.0
                if custom_inlet:
                    U[e, 0, :, 0] = custom_inlet(mesh.xnod[e, 0], mesh.ynod[e, :], time)
            elif bc == 4:
                U[e, 0, :, 2] = 0.0
            elif bc == 5:
                U[e, 0, :, 1] = 0.0; U[e, 0, :, 3] = 0.0
            
        # E
        bc = mesh.bc[e, 1]
        if bc in (1, 2, 3, 4, 5) and exact_solution:
            u_ex, v_ex, p_ex, om_ex = exact_solution(mesh.xnod[e, -1], mesh.ynod[e, :], time)
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
                U[e, -1, :, 0] = custom_lid(mesh.xnod[e, -1], mesh.ynod[e, :], time) if custom_lid else 1.0
            elif bc == 3:
                U[e, -1, :, 1] = 0.0
                if custom_inlet:
                    U[e, -1, :, 0] = custom_inlet(mesh.xnod[e, -1], mesh.ynod[e, :], time)
            elif bc == 4:
                U[e, -1, :, 2] = 0.0
            elif bc == 5:
                U[e, -1, :, 1] = 0.0; U[e, -1, :, 3] = 0.0
            
        # S
        bc = mesh.bc[e, 2]
        if bc in (1, 2, 3, 4, 5) and exact_solution:
            u_ex, v_ex, p_ex, om_ex = exact_solution(mesh.xnod[e, :], mesh.ynod[e, 0], time)
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
                U[e, :, 0, 0] = custom_lid(mesh.xnod[e, :], mesh.ynod[e, 0], time) if custom_lid else 1.0
            elif bc == 3:
                U[e, :, 0, 1] = 0.0
                if custom_inlet:
                    U[e, :, 0, 0] = custom_inlet(mesh.xnod[e, :], mesh.ynod[e, 0], time)
            elif bc == 4:
                U[e, :, 0, 2] = 0.0
            elif bc == 5:
                U[e, :, 0, 1] = 0.0; U[e, :, 0, 3] = 0.0
            
        # N
        bc = mesh.bc[e, 3]
        if bc in (1, 2, 3, 4, 5) and exact_solution:
            u_ex, v_ex, p_ex, om_ex = exact_solution(mesh.xnod[e, :], mesh.ynod[e, -1], time)
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
                U[e, :, -1, 0] = custom_lid(mesh.xnod[e, :], mesh.ynod[e, -1], time) if custom_lid else 1.0
            elif bc == 3:
                U[e, :, -1, 1] = 0.0
                if custom_inlet:
                    U[e, :, -1, 0] = custom_inlet(mesh.xnod[e, :], mesh.ynod[e, -1], time)
            elif bc == 4:
                U[e, :, -1, 2] = 0.0
            elif bc == 5:
                U[e, :, -1, 1] = 0.0; U[e, :, -1, 3] = 0.0
            
    if pin_p:
        if exact_solution:
            _, _, p_ex, _ = exact_solution(mesh.xnod[0, 0], mesh.ynod[0, 0], time)
            U[0, 0, 0, 2] = p_ex
        else:
            U[0, 0, 0, 2] = 0.0
            
    return U
