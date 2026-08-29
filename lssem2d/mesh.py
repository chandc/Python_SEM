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
        
        # Edge BC codes (0=interior/free, 1=no-slip, 2=lid, 3=inlet profile,
        # 4=outlet p=0, 5=symmetry, 6=Dong OBC -- see lssem2d/obc.py and
        # DONG_OBC_RESULTS.md; East edges only)
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

        # Merge coincident nodes with a genuine TOLERANCE, not a rounding grid.
        #
        # round(x, k) defines fixed buckets, so two points 1e-10 apart still land in
        # different buckets whenever they straddle a bucket boundary -- no choice of k
        # fixes that.  Meshes read from a file hit this constantly: coordinates stored
        # with ~10 significant digits and rebuilt as x0 + hx/2*(xi+1) differ by ~1e-10
        # between the two elements sharing an edge, and the mesh ends up SILENTLY
        # disconnected (flow cannot cross the interface).
        #
        # Instead bin to integer cells of size TOL and probe the 3x3 neighbourhood:
        # any two points within TOL are then guaranteed to be found, regardless of
        # where the cell boundaries fall.  TOL sits far above the reconstruction error
        # (~1e-10) and far below the smallest LGL spacing (O(h/p^2), ~1e-3 here).
        TOL = 1.0e-8

        # PERIODICITY.  Everything downstream -- gather_scatter, the masks, the
        # multiplicity weights -- is driven by gidx, so wrapping the coordinate
        # here is the whole change: x = Lx then hashes to the same cell as x = 0
        # and the two ends of the domain merge into one set of global nodes.
        #
        # Only CONNECTIVITY is wrapped.  xnod/facx/wq keep their true values, so
        # geometry, quadrature and derivatives are untouched.
        #
        # Set mesh.periodic_x = Lx (and/or periodic_y = Ly) BEFORE calling this,
        # and leave the corresponding bc codes at 0 so no Dirichlet mask is
        # applied on the seam.
        px = getattr(self, 'periodic_x', None)
        py = getattr(self, 'periodic_y', None)

        for e in range(self.nelem):
            for i in range(self.nterm):
                for j in range(self.nterm):
                    x = self.xnod[e, i]
                    y = self.ynod[e, j]
                    if px:
                        # tolerance-aware wrap: a node sitting a hair below Lx
                        # must land on 0, not on Lx - 1e-12
                        x = x % px
                        if px - x < TOL:
                            x = 0.0
                    if py:
                        y = y % py
                        if py - y < TOL:
                            y = 0.0

                    cx = int(np.floor(x / TOL))
                    cy = int(np.floor(y / TOL))
                    gid = None
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            hit = node_dict.get((cx + dx, cy + dy))
                            if hit is not None:
                                gid = hit
                                break
                        if gid is not None:
                            break
                    if gid is None:
                        gid = current_id
                        current_id += 1
                    node_dict[(cx, cy)] = gid

                    self.gidx[e, i, j] = gid

        # Guard: every neighbour pair must share its entire edge after hashing.
        n = self.nterm
        unmerged = 0
        for e in range(self.nelem):
            for d in range(4):
                j2 = self.neighbour[e, d]
                if j2 < 0:
                    continue
                if d in (0, 1):
                    a = [self.gidx[e, 0 if d == 0 else -1, k] for k in range(n)]
                    b = [self.gidx[j2, -1 if d == 0 else 0, k] for k in range(n)]
                else:
                    a = [self.gidx[e, k, 0 if d == 2 else -1] for k in range(n)]
                    b = [self.gidx[j2, k, -1 if d == 2 else 0] for k in range(n)]
                if a != b:
                    unmerged += 1
        if unmerged:
            raise ValueError(
                f"compute_global_indices: {unmerged} neighbour edges were not merged "
                f"(snap tolerance {TOL:g} too tight for this mesh's coordinate "
                f"precision). The mesh would be silently disconnected.")

        # Periodic seam guard.  The check above only looks at `neighbour` pairs,
        # and the two ends of a periodic domain are NOT neighbours -- so without
        # this, a periodic_x that failed to merge would leave the domain silently
        # open and every result would look plausible.
        for axis, L in (('x', px), ('y', py)):
            if not L:
                continue
            # The declared period must equal the domain extent.  If it is a
            # fraction of it, the wrap folds the domain onto ITSELF -- x = 0, L
            # and 2L all merge -- and the seam test below still passes because
            # the merge, though wrong, is self-consistent.  Check the extent
            # first, where the error is unambiguous.
            c = self.xnod if axis == 'x' else self.ynod
            extent = float(c.max() - c.min())
            if abs(extent - L) > TOL:
                raise ValueError(
                    f"periodic_{axis} = {L} but the domain spans {extent:g} in "
                    f"{axis}. The period must equal the extent, or the wrap "
                    f"folds the domain onto itself.")
            lo, hi = [], []
            for e in range(self.nelem):
                for i in range(n):
                    for j in range(n):
                        c = self.xnod[e, i] if axis == 'x' else self.ynod[e, j]
                        if abs(c) < TOL:
                            lo.append(self.gidx[e, i, j])
                        elif abs(c - L) < TOL:
                            hi.append(self.gidx[e, i, j])
            if not lo or not hi:
                raise ValueError(
                    f"periodic_{axis} = {L}: no nodes found at one of the seam "
                    f"faces ({len(lo)} at 0, {len(hi)} at {L}). Check that the "
                    f"domain really spans [0, {L}] in {axis}.")
            if not set(hi) <= set(lo):
                raise ValueError(
                    f"periodic_{axis} = {L}: the seam did not merge -- "
                    f"{len(set(hi) - set(lo))} global ids at {axis} = {L} have no "
                    f"counterpart at {axis} = 0. The domain would be open.")

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

def build_bfs(N, E_in_x=2, E_out_x=20, E_y=1, L_in=2.0, L_out=20.0,
              H_in=0.5, H_step=0.5, xpow=1.0):
    """
    Backward-facing step with an upstream inlet channel.
    Let's build a standard geometry:
    Inlet channel: x in [-L_in, 0], y in [0, H]
    Expansion: x in [0, L_out], y in [-H, H] (step down by H)
    """
    # Geometry now parametric: Armaly ER=1.94 is H_in=1.0, H_step=0.94.
    
    # We will just construct this manually block by block.
    # Block 1 (Inlet): [-2, 0] x [0.5, 1.0] (2x1 elements)
    # Block 2 (Outlet Top): [0, 20] x [0.5, 1.0] (10x1 elements)
    # Block 3 (Outlet Bot): [0, 20] x [0, 0.5] (10x1 elements)
    
    nelem = E_in_x * E_y + E_out_x * E_y + E_out_x * E_y
    mesh = Mesh(nelem, N)
    
    hx_in = L_in / E_in_x
    hy_in = H_in / E_y
    
    # xpow > 1 clusters outlet columns toward the step (x = 0), the 2D
    # Armaly mesh's recipe (1.6) for the corner singularity and shear layer.
    xedges = L_out*(np.arange(E_out_x + 1)/E_out_x)**xpow
    hy_out = H_step / E_y
    
    e = 0
    # Inlet
    elem_inlet = np.zeros((E_in_x, E_y), dtype=int)
    for ex in range(E_in_x):
        for ey in range(E_y):
            mesh.x0[e] = -L_in + ex * hx_in
            mesh.y0[e] = H_step + ey * hy_in
            mesh.hx[e] = hx_in
            mesh.hy[e] = hy_in
            elem_inlet[ex, ey] = e
            e += 1
            
    # Outlet Top
    elem_out_top = np.zeros((E_out_x, E_y), dtype=int)
    for ex in range(E_out_x):
        for ey in range(E_y):
            mesh.x0[e] = xedges[ex]
            mesh.y0[e] = H_step + ey * hy_in
            mesh.hx[e] = xedges[ex+1] - xedges[ex]
            mesh.hy[e] = hy_in
            elem_out_top[ex, ey] = e
            e += 1
            
    # Outlet Bot
    elem_out_bot = np.zeros((E_out_x, E_y), dtype=int)
    for ex in range(E_out_x):
        for ey in range(E_y):
            mesh.x0[e] = xedges[ex]
            mesh.y0[e] = ey * hy_out
            mesh.hx[e] = xedges[ex+1] - xedges[ex]
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
