import numpy as np

def gather_scatter(mesh, U):
    """
    Apply the gather-scatter operator Q^T Q to the local array U.
    Sums values across element boundaries (adding contributions from
    neighbouring elements) and scatters the sum back to the local nodes.
    
    mesh: Mesh instance, containing mesh.idx (the global node indices) and mesh.ndof
    U: local array, shape (nelem, n, n) or (nelem, n, n, k)
    
    Returns:
    U_gs: Gathered-scattered array of the same shape as U.
    """
    flat_idx = mesh.gidx.ravel()
    ndof = flat_idx.max() + 1
    
    if U.ndim == 3:
        # Single field case: shape (nelem, n, n)
        # Gather
        global_U = np.bincount(flat_idx, weights=U.ravel(), minlength=ndof)
        # Scatter
        U_gs = global_U[mesh.gidx]
        return U_gs
        
    elif U.ndim == 4:
        # Multi field case: shape (nelem, n, n, k)
        k = U.shape[3]
        global_U = np.zeros((ndof, k))
        
        # Gather (looping over small number of fields is faster with bincount)
        for i in range(k):
            global_U[:, i] = np.bincount(flat_idx, weights=U[..., i].ravel(), minlength=ndof)
            
        # Scatter
        U_gs = global_U[mesh.gidx, :]
        return U_gs
        
    else:
        raise ValueError("U must be 3D or 4D")
