import numpy as np
import matplotlib.pyplot as plt
from .lgl import lgl_nodes, lgl_weights

class Mesh:
    def __init__(self, nelem, N):
        self.nelem = nelem
        self.N = N
        self.nterm = N + 1
        
        # Element origin and size
        self.x0 = np.zeros(nelem)
        self.y0 = np.zeros(nelem)
        self.hx = np.zeros(nelem)
        self.hy = np.zeros(nelem)
        
        # Physical node coordinates
        self.xnod = np.zeros((nelem, self.nterm))
        self.ynod = np.zeros((nelem, self.nterm))
        
        # W, E, S, N neighbour element (-1 = boundary)
        self.neighbour = -np.ones((nelem, 4), dtype=int)
        
        # Edge BC codes (0=interior, 1=no-slip, 2=lid, 3=inlet profile, 4=outlet, 5=symmetry, 6=free)
        self.bc = np.zeros((nelem, 4), dtype=int)
        
        # Derived
        self.jac = np.zeros(nelem)
        self.facx = np.zeros(nelem)
        self.facy = np.zeros(nelem)
        self.wq = np.zeros((nelem, self.nterm, self.nterm))
        
        self.gidx = -np.ones((nelem, self.nterm, self.nterm), dtype=int)
        
        # Precomputed gather-scatter sparse matrices
        self.Q = None
        self.QT = None
        
    def setup_derived(self):
        """Compute Jacobians, mapping factors, physical nodes, and quadrature weights."""
        xi = lgl_nodes(self.N)
        w = lgl_weights(self.N)
        
        self.jac = self.hx * self.hy / 4.0
        self.facx = 2.0 / self.hx
        self.facy = 2.0 / self.hy
        
        for e in range(self.nelem):
            self.xnod[e, :] = self.x0[e] + self.hx[e] / 2.0 * (xi + 1.0)
            self.ynod[e, :] = self.y0[e] + self.hy[e] / 2.0 * (xi + 1.0)
            
            for i in range(self.nterm):
                for j in range(self.nterm):
                    self.wq[e, i, j] = self.jac[e] * w[i] * w[j]
                    
    def compute_global_indices(self):
        """Assign unique global node IDs to gidx[e, i, j], respecting shared boundaries."""
        # Simple algorithm: iterate over elements, assign new IDs to unassigned nodes.
        # But we must identify shared nodes! 
        # A shared node is one that lies on a shared edge (or corner).
        # We can do this robustly by snapping physical coordinates (xnod[e,i], ynod[e,j])
        # to a dictionary or grid, since it's a conforming grid.
        
        # Alternatively, use the `neighbour` array to trace relationships, but coordinate
        # hashing is very simple and robust for this scale.
        
        node_dict = {}
        current_id = 0
        
        for e in range(self.nelem):
            for i in range(self.nterm):
                for j in range(self.nterm):
                    x = self.xnod[e, i]
                    y = self.ynod[e, j]
                    
                    # Rounding to avoid floating point issues (1e-10 is safe for domain sizes ~O(1))
                    key = (round(x, 10), round(y, 10))
                    
                    if key not in node_dict:
                        node_dict[key] = current_id
                        current_id += 1
                    
                    self.gidx[e, i, j] = node_dict[key]
                    
        # Build scipy.sparse gather-scatter matrices Q and QT
        import scipy.sparse as sp
        flat_idx = self.gidx.ravel()
        ndof = flat_idx.max() + 1
        total_local_nodes = len(flat_idx)
        
        row = flat_idx
        col = np.arange(total_local_nodes)
        data = np.ones(total_local_nodes, dtype=float)
        
        self.Q = sp.csr_matrix((data, (row, col)), shape=(ndof, total_local_nodes))
        self.QT = self.Q.T.tocsr()
        
    def global_index(self, e, i, j):
        return self.gidx[e, i, j]

def build_channel(L_x, L_y, E_x, E_y, N, bcs=(0, 0, 1, 1)):
    """
    Build a simple rectangular grid of size L_x * L_y with E_x * E_y elements.
    Domain is [0, L_x] x [0, L_y].
    bcs: tuple of (W, E, S, N) boundary condition codes.
    """
    nelem = E_x * E_y
    mesh = Mesh(nelem, N)
    
    hx = L_x / E_x
    hy = L_y / E_y
    
    e = 0
    # Create elements in a raster scan (y varying fastest or x varying fastest)
    # Let's do (ex, ey) where ex is x-index, ey is y-index
    elem_grid = np.zeros((E_x, E_y), dtype=int)
    for ex in range(E_x):
        for ey in range(E_y):
            mesh.x0[e] = ex * hx
            mesh.y0[e] = ey * hy
            mesh.hx[e] = hx
            mesh.hy[e] = hy
            elem_grid[ex, ey] = e
            e += 1
            
    # Set neighbours
    # W=0, E=1, S=2, N=3
    for ex in range(E_x):
        for ey in range(E_y):
            e = elem_grid[ex, ey]
            if ex > 0:
                mesh.neighbour[e, 0] = elem_grid[ex - 1, ey]
            if ex < E_x - 1:
                mesh.neighbour[e, 1] = elem_grid[ex + 1, ey]
            if ey > 0:
                mesh.neighbour[e, 2] = elem_grid[ex, ey - 1]
            if ey < E_y - 1:
                mesh.neighbour[e, 3] = elem_grid[ex, ey + 1]
                
            # Set BCs
            if ex == 0:
                mesh.bc[e, 0] = bcs[0]
            if ex == E_x - 1:
                mesh.bc[e, 1] = bcs[1]
            if ey == 0:
                mesh.bc[e, 2] = bcs[2]
            if ey == E_y - 1:
                mesh.bc[e, 3] = bcs[3]
                
    mesh.setup_derived()
    mesh.compute_global_indices()
    return mesh

def build_bfs(N):
    """
    Backward-facing step with an upstream inlet channel.
    Let's build a standard geometry:
    Inlet channel: x in [-L_in, 0], y in [0, H]
    Expansion: x in [0, L_out], y in [-H, H] (step down by H)
    """
    # For now, a minimal representative geometry
    # 2 elements for inlet, 6 elements for expansion
    L_in = 2.0
    L_out = 20.0
    H = 1.0
    
    # We will just construct this manually block by block.
    # Block 1 (Inlet): [-2, 0] x [0, 1] (2x1 elements)
    # Block 2 (Step/Outlet top): [0, 20] x [0, 1] (10x1 elements)
    # Block 3 (Step/Outlet bot): [0, 20] x [-1, 0] (10x1 elements)
    E_in_x = 2
    E_out_x = 10
    E_y = 1
    
    nelem = E_in_x * E_y + E_out_x * E_y + E_out_x * E_y
    mesh = Mesh(nelem, N)
    
    hx_in = L_in / E_in_x
    hy_in = H / E_y
    
    hx_out = L_out / E_out_x
    hy_out = H / E_y
    
    # Needs full manual construction; just placing elements and coordinate hash
    # will handle connectivity! We only need neighbours for `neighbour` array consistency.
    # A generic unstructured quadrilateral mesher from block data is better, 
    # but since tests only need a basic BFS, we'll keep it simple.
    
    e = 0
    # Inlet
    elem_inlet = np.zeros((E_in_x, E_y), dtype=int)
    for ex in range(E_in_x):
        for ey in range(E_y):
            mesh.x0[e] = -L_in + ex * hx_in
            mesh.y0[e] = ey * hy_in
            mesh.hx[e] = hx_in
            mesh.hy[e] = hy_in
            elem_inlet[ex, ey] = e
            e += 1
            
    # Outlet Top
    elem_out_top = np.zeros((E_out_x, E_y), dtype=int)
    for ex in range(E_out_x):
        for ey in range(E_y):
            mesh.x0[e] = ex * hx_out
            mesh.y0[e] = ey * hy_out
            mesh.hx[e] = hx_out
            mesh.hy[e] = hy_out
            elem_out_top[ex, ey] = e
            e += 1
            
    # Outlet Bot
    elem_out_bot = np.zeros((E_out_x, E_y), dtype=int)
    for ex in range(E_out_x):
        for ey in range(E_y):
            mesh.x0[e] = ex * hx_out
            mesh.y0[e] = -H + ey * hy_out
            mesh.hx[e] = hx_out
            mesh.hy[e] = hy_out
            elem_out_bot[ex, ey] = e
            e += 1
            
    # Compute neighbours by coordinate matching (W, E, S, N)
    # W(0) = -x, E(1) = +x, S(2) = -y, N(3) = +y
    for i in range(nelem):
        xc = mesh.x0[i] + mesh.hx[i] / 2
        yc = mesh.y0[i] + mesh.hy[i] / 2
        
        for j in range(nelem):
            if i == j: continue
            x_cj = mesh.x0[j] + mesh.hx[j] / 2
            y_cj = mesh.y0[j] + mesh.hy[j] / 2
            
            # Check E
            if abs(yc - y_cj) < 1e-8 and abs((mesh.x0[i] + mesh.hx[i]) - mesh.x0[j]) < 1e-8:
                mesh.neighbour[i, 1] = j
            # Check W
            if abs(yc - y_cj) < 1e-8 and abs(mesh.x0[i] - (mesh.x0[j] + mesh.hx[j])) < 1e-8:
                mesh.neighbour[i, 0] = j
            # Check N
            if abs(xc - x_cj) < 1e-8 and abs((mesh.y0[i] + mesh.hy[i]) - mesh.y0[j]) < 1e-8:
                mesh.neighbour[i, 3] = j
            if abs(xc - x_cj) < 1e-8 and abs(mesh.y0[i] - (mesh.y0[j] + mesh.hy[j])) < 1e-8:
                mesh.neighbour[i, 2] = j
                
    # Assign BCs for BFS
    # Inlet: W edge of inlet elements (bc=3)
    # Outlet: E edge of outlet elements (bc=4)
    # Walls: N/S edges, and the step face (bc=1)
    for i in range(nelem):
        if mesh.neighbour[i, 0] == -1:
            if abs(mesh.x0[i] - (-L_in)) < 1e-8:
                mesh.bc[i, 0] = 3  # Inlet
            else:
                mesh.bc[i, 0] = 1  # Step wall
        if mesh.neighbour[i, 1] == -1:
            mesh.bc[i, 1] = 4  # Outlet
        if mesh.neighbour[i, 2] == -1:
            mesh.bc[i, 2] = 1  # Bottom wall
        if mesh.neighbour[i, 3] == -1:
            mesh.bc[i, 3] = 1  # Top wall

    mesh.setup_derived()
    mesh.compute_global_indices()
    return mesh

def plot_mesh(mesh, filename='mesh_plot.png'):
    """Renders elements + collocation points (visual check)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for e in range(mesh.nelem):
        # Draw element boundary
        x_min, x_max = mesh.x0[e], mesh.x0[e] + mesh.hx[e]
        y_min, y_max = mesh.y0[e], mesh.y0[e] + mesh.hy[e]
        
        ax.plot([x_min, x_max, x_max, x_min, x_min], 
                [y_min, y_min, y_max, y_max, y_min], 'k-', lw=1)
                
        # Draw points
        X, Y = np.meshgrid(mesh.xnod[e, :], mesh.ynod[e, :], indexing='ij')
        ax.plot(X.flatten(), Y.flatten(), 'r.', markersize=2)
        
    ax.set_aspect('equal')
    ax.set_title(f'Mesh (nelem={mesh.nelem}, N={mesh.N})')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
