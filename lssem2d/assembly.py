import numpy as np

def gather_scatter(mesh, U):
    """
    Apply the gather-scatter operator Q^T Q to the local array U.
    Sums values across element boundaries (adding contributions from
    neighbouring elements) and scatters the sum back to the local nodes.
    
    mesh: Mesh instance, containing mesh.Q and mesh.QT
    U: local array, shape (nelem, n, n) or (nelem, n, n, k)
    
    Returns:
    U_gs: Gathered-scattered array of the same shape as U.
    """
    original_shape = U.shape
    
    if U.ndim == 3:
        U_flat = U.reshape(-1)
        global_U = mesh.Q @ U_flat
        U_gs_flat = mesh.QT @ global_U
        return U_gs_flat.reshape(original_shape)
        
    elif U.ndim == 4:
        k = U.shape[3]
        U_flat = U.reshape(-1, k)
        global_U = mesh.Q @ U_flat
        U_gs_flat = mesh.QT @ global_U
        return U_gs_flat.reshape(original_shape)
        
    else:
        raise ValueError("U must be 3D or 4D")
