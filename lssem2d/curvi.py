"""Curvilinear geometry for the 2D spectral-element least-squares solver.

CURVILINEAR_2D_PLAN.md steps 1-2.  The affine code stores two scalars per
element, `facx = 2/hx` and `facy = 2/hy`, and every derivative is one tensor
contraction scaled by one of them.  Here each scalar becomes a field over the
element and the chain rule brings in a second contraction:

    df/dx = r_x df/dr + s_x df/ds,      df/dy = r_y df/dr + s_y df/ds

with  J = x_r y_s - x_s y_r  and

    r_x =  y_s/J,   r_y = -x_s/J,   s_x = -y_r/J,   s_y =  x_r/J .

COLLOCATION METRICS, NOT ANALYTIC ONES (plan decision D2).  x_r and friends are
obtained by differentiating the nodal coordinates with the SAME matrix `D` the
solver differentiates solutions with.  Using exact analytic derivatives of the
mapping instead would be more accurate pointwise and *worse* discretely: the
identities that make a constant field have zero gradient, and a linear field an
exact one, hold only when the metric is differentiated the same way the solution
is.  Gate G0 measures precisely that, and it is the first thing to fail if this
is done the tempting way.

In two dimensions the metric identity

    d/dr (J r_x) + d/ds (J s_x) = d/dr(y_s) - d/ds(y_r) = 0

holds identically for collocation metrics because mixed derivatives commute on
the tensor-product grid.  In three dimensions it does not, and the conservative
(cross-product) form becomes necessary -- noted for the eventual 3D port, and
deliberately not built here.
"""
import numpy as np

from .lgl import diff_matrix, lgl_nodes, lgl_weights


# ----------------------------------------------------------------- metrics --

def collocation_metrics(X, Y, N):
    """Metric fields from nodal coordinates.

    X, Y : (nelem, n, n) physical coordinates of the GLL nodes.
    Returns dict of (nelem, n, n) arrays: rx, ry, sx, sy, jac.

    The index convention matches `operators.dUdx`/`dUdy`: r acts on the FIRST
    node index (matmul(D, U)) and s on the second (matmul(U, D.T)).
    """
    D = diff_matrix(N)
    xr = np.matmul(D, X)
    xs = np.matmul(X, D.T)
    yr = np.matmul(D, Y)
    ys = np.matmul(Y, D.T)
    J = xr*ys - xs*yr
    if np.any(J <= 0):
        bad = int((J <= 0).sum())
        raise ValueError(f'non-positive Jacobian at {bad} nodes: the mapping '
                         f'folds.  Check the element ordering or the deformation '
                         f'amplitude (min J = {J.min():.3e}).')
    return dict(rx=ys/J, ry=-xs/J, sx=-yr/J, sy=xr/J, jac=J,
                xr=xr, xs=xs, yr=yr, ys=ys)


def attach(mesh, X, Y):
    """Make `mesh` curvilinear with the given nodal coordinates.

    Sets the metric fields and rebuilds `wq` as J*w_i*w_j, then flips
    `mesh.curvilinear`, after which `mesh.facx`/`facy` raise rather than return a
    stale constant (plan decision D1).
    """
    m = collocation_metrics(X, Y, mesh.N)
    w = lgl_weights(mesh.N)
    mesh.X, mesh.Y = np.ascontiguousarray(X), np.ascontiguousarray(Y)
    for k in ('rx', 'ry', 'sx', 'sy'):
        setattr(mesh, k, np.ascontiguousarray(m[k]))
    mesh.jacq = np.ascontiguousarray(m['jac'])
    mesh.wq = mesh.jacq*w[None, :, None]*w[None, None, :]
    mesh.curvilinear = True
    return mesh


def affine_coords(mesh):
    """The (nelem, n, n) coordinates an affine mesh implies, for conversion."""
    X = np.repeat(mesh.xnod[:, :, None], mesh.nterm, axis=2)
    Y = np.repeat(mesh.ynod[:, None, :], mesh.nterm, axis=1)
    return X, Y


# -------------------------------------------------------------- primitives --

def ddx(U, D, mesh, out=None):
    """d/dx on a curvilinear mesh: r_x U_r + s_x U_s."""
    ur = np.matmul(D, U)
    us = np.matmul(U, D.T)
    z = mesh.rx*ur + mesh.sx*us
    if out is not None:
        out[...] = z
        return out
    return z


def ddy(U, D, mesh, out=None):
    """d/dy on a curvilinear mesh: r_y U_r + s_y U_s."""
    ur = np.matmul(D, U)
    us = np.matmul(U, D.T)
    z = mesh.ry*ur + mesh.sy*us
    if out is not None:
        out[...] = z
        return out
    return z


def ddxT(S, D, mesh, out=None):
    """Adjoint of `ddx` in the PLAIN inner product, matching `operators.DxT`.

    Defined by  sum_ij ddx(u)_ij S_ij = sum_ij u_ij ddxT(S)_ij, so

        sum (r_x S)(D u) + sum (s_x S)(u D^T)
      = sum u [ D^T (r_x S) ] + sum u [ (s_x S) D ] .

    UNWEIGHTED ON PURPOSE.  `apply_L` multiplies its rows by `wq` before
    returning, so the quadrature weight is already inside the `su` that
    `apply_LT` receives -- exactly as the affine `DxT` assumes.  Pass `wq*v` to
    test the weighted adjoint.

    A sign or a transposed index here survives every forward test and destroys
    the least-squares operator, which is built from adjoint pairs; gate G0's
    adjoint check and gate G1's rotation invariance are what catch it.
    """
    z = np.matmul(D.T, mesh.rx*S) + np.matmul(mesh.sx*S, D)
    if out is not None:
        out[...] = z
        return out
    return z


def ddyT(S, D, mesh, out=None):
    """Adjoint of `ddy` in the plain inner product, matching `operators.DyT`."""
    z = np.matmul(D.T, mesh.ry*S) + np.matmul(mesh.sy*S, D)
    if out is not None:
        out[...] = z
        return out
    return z


# ---------------------------------------------------------------- builders --

def deform(mesh, amp=0.1, kx=1.0, ky=1.0):
    """Sinusoidally distort a conforming affine mesh, keeping it conforming.

    The perturbation is a function of physical position only, so nodes shared
    between elements move together and the mesh stays watertight -- which is
    what makes the deformed mesh a legitimate test rather than a broken one.
    Boundary nodes move along the boundary is NOT enforced here: the domain
    itself deforms, which is the point.

    `amp` is in units of the smallest element size.
    """
    X, Y = affine_coords(mesh)
    h = min(float(mesh.hx.min()), float(mesh.hy.min()))
    Lx = float(X.max() - X.min())
    Ly = float(Y.max() - Y.min())
    dx = amp*h*np.sin(kx*np.pi*(X - X.min())/Lx)*np.sin(ky*np.pi*(Y - Y.min())/Ly)
    dy = amp*h*np.sin(kx*np.pi*(X - X.min())/Lx)*np.sin(ky*np.pi*(Y - Y.min())/Ly)
    return attach(mesh, X + dx, Y + dy)


def rotate(mesh, theta):
    """Rotate a mesh rigidly by `theta` radians about the origin.

    The Jacobian stays constant per element, so this is still an AFFINE mapping
    -- but it is not axis-aligned, so every metric term and every cross term is
    exercised while the exact solution is simply the unrotated one, rotated.
    That is gate G1, and it is the strongest test in the plan.
    """
    X, Y = affine_coords(mesh)
    c, s = np.cos(theta), np.sin(theta)
    return attach(mesh, c*X - s*Y, s*X + c*Y)


def build_annulus(r_in, r_out, E_r, E_th, N, theta0=0.0, theta1=2*np.pi,
                  bcs=(1, 1, 0, 0)):
    """An annular sector: no straight element edges anywhere.

    Returns a Mesh with curvilinear metrics attached.  Element (i,j) spans
    r in [r_i, r_{i+1}] and theta in [th_j, th_{j+1}]; the reference square maps
    through (r, theta) -> (r cos theta, r sin theta).

    bcs are (inner, outer, start, end) edge codes.  A full annulus
    (theta1 - theta0 = 2 pi) is periodic in theta, which the caller must set up
    through `mesh.periodic_x`-style handling; the default sector avoids it.
    """
    from .mesh import Mesh
    xi = lgl_nodes(N)
    nelem = E_r*E_th
    mesh = Mesh(nelem, N)
    n = N + 1
    X = np.zeros((nelem, n, n))
    Y = np.zeros((nelem, n, n))
    redges = np.linspace(r_in, r_out, E_r + 1)
    tedges = np.linspace(theta0, theta1, E_th + 1)
    for i in range(E_r):
        for j in range(E_th):
            e = i*E_th + j
            r = redges[i] + (redges[i+1] - redges[i])*(xi + 1)/2
            th = tedges[j] + (tedges[j+1] - tedges[j])*(xi + 1)/2
            R, T = np.meshgrid(r, th, indexing='ij')
            X[e], Y[e] = R*np.cos(T), R*np.sin(T)
            # bookkeeping the affine path would have filled in
            mesh.x0[e], mesh.y0[e] = X[e].min(), Y[e].min()
            mesh.hx[e] = redges[i+1] - redges[i]
            mesh.hy[e] = (tedges[j+1] - tedges[j])*0.5*(redges[i] + redges[i+1])
            mesh.bc[e, 0] = bcs[0] if i == 0 else 0          # inner radius
            mesh.bc[e, 1] = bcs[1] if i == E_r - 1 else 0    # outer radius
            mesh.bc[e, 2] = bcs[2] if j == 0 else 0
            mesh.bc[e, 3] = bcs[3] if j == E_th - 1 else 0
            # Face-neighbour table, in the same W/E/S/N order as the bc codes:
            # the reference r direction is the FIRST node index, so W/E are the
            # radial faces and S/N the azimuthal ones.  `compute_global_indices`
            # uses this to verify that every shared edge actually merged; left
            # at -1 the check passes vacuously and a disconnected mesh is silent.
            mesh.neighbour[e, 0] = e - E_th if i > 0 else -1
            mesh.neighbour[e, 1] = e + E_th if i < E_r - 1 else -1
            mesh.neighbour[e, 2] = e - 1 if j > 0 else -1
            mesh.neighbour[e, 3] = e + 1 if j < E_th - 1 else -1
    mesh.xnod = X[:, :, 0].copy()          # r-direction trace, for hashing
    mesh.ynod = Y[:, 0, :].copy()
    attach(mesh, X, Y)
    mesh.E_r, mesh.E_th = E_r, E_th
    return mesh


def build_cylinder(r_cyl=0.5, r_far=25.0, E_r=8, E_th=24, N=8,
                   bcs=(1, 3), stretch=1.0):
    """O-grid around a circular cylinder: the mesh gate G6 needs.

    A single topological annulus from the cylinder surface out to a circular
    far field.  Every element is curved, every element edge on the body is an
    arc, and the body is resolved by the geometry rather than approximated by
    steps -- which is the whole reason the curvilinear port exists.

    GEOMETRIC RADIAL SPACING, r_{i+1}/r_i constant, is the default and is not an
    arbitrary choice.  The azimuthal size of an element grows like r, so radial
    thickness proportional to r keeps the ASPECT RATIO constant through the whole
    mesh -- one number instead of a near-wall value and a far-field value that
    differ by the domain ratio.  `stretch` biases it further toward the wall:
    the radial coordinate is (r/r_cyl)^stretch in the exponent, so stretch > 1
    packs elements into the boundary layer at the cost of aspect ratio.

    THE THETA SEAM CLOSES ITSELF.  theta = 0 and theta = 2 pi produce IDENTICAL
    physical coordinates, and `compute_global_indices` hashes physical
    coordinates on a curvilinear mesh, so the two sides of the seam merge with no
    periodic-wrap machinery at all.  The neighbour table below wraps to match, so
    the disconnection guard still checks it.

    bcs = (body, far field).  The far field is one boundary here: at the steady
    Re = 40 the wake is ~2.2 diameters long, so a circular outer boundary at
    r_far = 25 D sees essentially free stream all the way round.  An unsteady
    case needs an outflow condition on the downstream arc instead, which is
    `obc.py`'s business and needs true boundary normals (plan step 8).
    """
    from .mesh import Mesh
    xi = lgl_nodes(N)
    nelem = E_r*E_th
    mesh = Mesh(nelem, N)
    n = N + 1
    X = np.zeros((nelem, n, n))
    Y = np.zeros((nelem, n, n))

    # radial element edges: geometric, optionally biased toward the wall
    u = np.linspace(0.0, 1.0, E_r + 1)**stretch
    redges = r_cyl*(r_far/r_cyl)**u
    tedges = np.linspace(0.0, 2*np.pi, E_th + 1)

    for i in range(E_r):
        for j in range(E_th):
            e = i*E_th + j
            # geometric within the element too, so the spacing law is smooth
            # across element boundaries rather than piecewise linear
            r = redges[i]*(redges[i+1]/redges[i])**((xi + 1)/2)
            th = tedges[j] + (tedges[j+1] - tedges[j])*(xi + 1)/2
            R, T = np.meshgrid(r, th, indexing='ij')
            X[e], Y[e] = R*np.cos(T), R*np.sin(T)
            mesh.x0[e], mesh.y0[e] = X[e].min(), Y[e].min()
            mesh.hx[e] = redges[i+1] - redges[i]
            mesh.hy[e] = (tedges[j+1] - tedges[j])*0.5*(redges[i] + redges[i+1])
            mesh.bc[e, 0] = bcs[0] if i == 0 else 0           # cylinder surface
            mesh.bc[e, 1] = bcs[1] if i == E_r - 1 else 0     # far field
            mesh.bc[e, 2] = 0                                 # theta: periodic
            mesh.bc[e, 3] = 0
            mesh.neighbour[e, 0] = e - E_th if i > 0 else -1
            mesh.neighbour[e, 1] = e + E_th if i < E_r - 1 else -1
            mesh.neighbour[e, 2] = i*E_th + (j - 1) % E_th    # wraps
            mesh.neighbour[e, 3] = i*E_th + (j + 1) % E_th
    mesh.xnod = X[:, :, 0].copy()
    mesh.ynod = Y[:, 0, :].copy()
    attach(mesh, X, Y)
    mesh.E_r, mesh.E_th = E_r, E_th
    mesh.r_cyl, mesh.r_far, mesh.redges = r_cyl, r_far, redges
    return mesh


def _geom_edges(a0, a1, n, ratio=1.0):
    """n+1 edges from a0 to a1 with geometric growth `ratio` per element."""
    if abs(ratio - 1.0) < 1e-12:
        return np.linspace(a0, a1, n + 1)
    w = ratio**np.arange(n)
    c = np.concatenate(([0.0], np.cumsum(w)))
    return a0 + (a1 - a0)*c/c[-1]



def nested_lateral_edges(H_new, n_extra, a=1.5, H=10.0, ny_side=4, ratio_out=1.45):
    """Lateral edges for a WIDER box that keeps the narrow box's divisions.

    Returns the `ny_side` edges of the reference box unchanged, with `n_extra`
    further elements appended geometrically out to `H_new`.  Feed the result to
    `build_cylinder_box(ys_side=...)`.  Because the reference edges survive
    untouched, every node with |y| <= H is common to both meshes, so a
    domain-size comparison changes the domain and nothing else -- and a solution
    on the narrow mesh transfers to the wide one by direct copy on the shared
    nodes, with free stream in the new strips.
    """
    base = _geom_edges(a, H, ny_side, ratio_out)
    if H_new <= H:
        raise ValueError(f'H_new = {H_new} must exceed the reference H = {H}')
    if n_extra < 1:
        raise ValueError('n_extra must be at least 1')
    return np.concatenate([base, _geom_edges(H, H_new, n_extra, ratio_out)[1:]])


def nested_upstream_edges(Lu_new, n_extra, a=1.5, Lu=10.0, nx_up=4,
                          ratio_out=1.45):
    """Upstream edges for a LONGER inlet run that keeps the reference divisions.

    Returns the reference box's `nx_up` edges unchanged with `n_extra` elements
    prepended out to -`Lu_new`.  Feed the result to
    `build_cylinder_box(xs_upstream=...)`.  The streamwise counterpart of
    `nested_lateral_edges`, and for the same reason: every node of the reference
    mesh survives, so moving the inlet changes the inlet and nothing else, and a
    solution transfers by direct copy with free stream in the new blocks.
    """
    base = _geom_edges(-Lu, -a, nx_up, 1.0/ratio_out)
    if Lu_new <= Lu:
        raise ValueError(f'Lu_new = {Lu_new} must exceed the reference Lu = {Lu}')
    if n_extra < 1:
        raise ValueError('n_extra must be at least 1')
    return np.concatenate([_geom_edges(-Lu_new, -Lu, n_extra, 1.0/ratio_out)[:-1],
                           base])


def nested_downstream_edges(Ld_new, n_extra, a=1.5, Ld=25.0, nx_dn=8,
                            ratio_out=1.45):
    """Downstream edges for a LONGER wake that keeps the reference divisions.

    The streamwise-outlet counterpart of `nested_lateral_edges` and
    `nested_upstream_edges`, and for the same reason: `_geom_edges` normalises
    its widths to span its endpoints, so raising Ld and nx_dn together would
    rescale every division between a and Ld and re-mesh the wake -- confounding
    the outlet distance with the wake resolution that sets the shedding.
    """
    base = _geom_edges(a, Ld, nx_dn, ratio_out)
    if Ld_new <= Ld:
        raise ValueError(f'Ld_new = {Ld_new} must exceed the reference Ld = {Ld}')
    if n_extra < 1:
        raise ValueError('n_extra must be at least 1')
    return np.concatenate([base, _geom_edges(Ld, Ld_new, n_extra, ratio_out)[1:]])


def chained_lateral_edges(H_new, n_per_double=2, a=1.5, H=10.0, ny_side=4,
                          ratio_out=1.45):
    """A LADDER of lateral edges in which every rung contains the previous one.

    `nested_lateral_edges` nests each rung on the REFERENCE box, which is enough
    for a single comparison but not for a ladder: H = 40 built as 2 elements
    from 10 to 20 and H = 80 built as 4 elements from 10 to 40 do not share
    their interior divisions, so consecutive rungs differ in lateral RESOLUTION
    as well as extent -- exactly the confound the nesting exists to remove
    (measured: only 78 % of H=40's nodes survive into such an H=80).

    Chaining instead doubles repeatedly, appending `n_per_double` elements each
    time, so rung k+1 is rung k plus elements strictly outside it:

        H=20   1.5 2.62 4.24 6.59 10
        H=40   ...                10 14.08 20
        H=80   ...                      20 28.16 40
        H=160  ...                            40 56.33 80

    Verified 100 % node retention from each rung to the next, which also makes
    the seed transfer between rungs an exact copy rather than an interpolation.
    Posdziech & Grundmann's protocol (J. Fluids Struct. 23:479-499) wants the
    lateral height to be the ONLY thing that varies across the set; this is what
    makes that true for a mesh built from geometric element spacing.
    """
    e = _geom_edges(a, H, ny_side, ratio_out)
    h = float(H)
    if H_new < H - 1e-12:
        raise ValueError(f'H_new = {H_new} is below the reference {H}')
    while h < H_new - 1e-12:
        h2 = min(2.0*h, H_new)
        e = np.concatenate([e, _geom_edges(h, h2, n_per_double, ratio_out)[1:]])
        h = h2
    return e

def _rect_block(xe, ye, N):
    """Elements of a rectangular block from edge arrays.  Returns X, Y, and the
    (nx, ny) shape so the caller can locate boundary elements."""
    xi = lgl_nodes(N)
    nx, ny = len(xe) - 1, len(ye) - 1
    n = N + 1
    X = np.zeros((nx*ny, n, n))
    Y = np.zeros((nx*ny, n, n))
    for i in range(nx):
        for j in range(ny):
            e = i*ny + j
            x = xe[i] + (xe[i+1] - xe[i])*(xi + 1)/2
            y = ye[j] + (ye[j+1] - ye[j])*(xi + 1)/2
            X[e], Y[e] = np.meshgrid(x, y, indexing='ij')
    return X, Y, (nx, ny)


def _square_at(theta, a):
    """Where the ray at angle `theta` meets the square of half-side `a`."""
    c, s = np.cos(theta), np.sin(theta)
    k = a/np.maximum(np.abs(c), np.abs(s))
    return k*c, k*s


def build_cylinder_box(r_cyl=0.5, a=1.5, Lu=10.0, Ld=25.0, H=10.0,
                       E_r=4, E_s=4, nx_up=4, nx_dn=8, ny_side=4, N=8,
                       stretch=1.6, ratio_out=1.45,
                       bcs=(1, 3, 6, 5), ys_side=None, xs_upstream=None,
                       xs_downstream=None):
    """O-ring on the body inside a RECTANGULAR box: curved where it must be,
    axis-aligned where the boundary conditions live.

    THE POINT OF THIS TOPOLOGY.  A pure O-grid (`build_cylinder`) puts the far
    field on a circle, and a circular outflow needs the boundary NORMAL -- which
    `obc.py` does not yet carry on a curved mesh (plan step 8).  Here the body
    keeps its curved, body-fitted ring, but the ring's outer edge is a SQUARE,
    and everything beyond it is rectangular blocks.  So every boundary carrying a
    condition is axis aligned, and the existing codes apply unchanged:

        cylinder surface  bc 1  no-slip      -- curved, u = v = 0 needs no normal
        inlet   x = -Lu   bc 3  free stream  -- flat
        outlet  x = +Ld   bc 6  Dong OBC     -- flat, so n = (1, 0) exactly
        top/bottom y = +-H bc 5 symmetry     -- flat, v = 0 and omega = 0

    The outlet being FLAT is what makes the Dong condition usable here at all:
    `obc.py` writes its rows for n = (1, 0) and checks that assumption, so a
    curved outflow arc would be refused (plan step 8).

    That removes step 8 from the critical path for G6 entirely.

    THE TOPOLOGY IS NOT A SINGLE i-j GRID, and does not need to be.  Nine blocks
    tile the box; the centre one is the O-ring rather than a rectangle, and the
    O-ring's own indexing (radial x azimuthal) has nothing to do with its
    neighbours' (x x y).  `compute_global_indices` hashes PHYSICAL COORDINATES,
    so any conforming collection of elements merges correctly whatever order the
    elements are stored in and whatever each block's internal indexing means.
    Conformity is the only requirement, and it is met by construction: the
    O-ring's outer nodes ARE the square-side nodes the surrounding blocks use.

    Azimuthal spacing is uniform in theta, so the body is resolved evenly; the
    square's sides inherit the non-uniform division where the rays land, and the
    middle blocks use that same division on their shared edges.
    """
    from .mesh import Mesh
    xi = lgl_nodes(N)
    n = N + 1
    nth = 4*E_s

    # ---- the O-ring: uniform theta on the body, rays out to the square ----
    th_e = np.linspace(0.0, 2*np.pi, nth + 1)
    t_e = np.linspace(0.0, 1.0, E_r + 1)
    Xo = np.zeros((E_r*nth, n, n))
    Yo = np.zeros((E_r*nth, n, n))
    for i in range(E_r):
        for j in range(nth):
            e = i*nth + j
            t = t_e[i] + (t_e[i+1] - t_e[i])*(xi + 1)/2
            u = t**stretch                                   # packs to the wall
            th = th_e[j] + (th_e[j+1] - th_e[j])*(xi + 1)/2
            U, T = np.meshgrid(u, th, indexing='ij')
            xin, yin = r_cyl*np.cos(T), r_cyl*np.sin(T)
            # THE OUTER EDGE IS A STRAIGHT SEGMENT, LINEARLY PARAMETERISED, and
            # getting this wrong is how the first version of this mesh came out
            # SLIT.  Evaluating _square_at at the GLL angles puts the ring's
            # outer nodes at uniform theta; the abutting rectangular block puts
            # its nodes at uniform x along the same edge.  The element CORNERS
            # coincide either way, so the picture is perfect and the area is
            # exact to 1e-15 -- and 224 interior-edge nodes fail to merge, giving
            # a mesh that is hydrodynamically a slit.  Interpolating between the
            # segment endpoints instead matches the block's parameterisation
            # exactly.  Square corners fall on element boundaries by
            # construction (4*E_s elements, corners every E_s), so each outer
            # edge lies on a single side.
            p0 = np.array(_square_at(th_e[j], a))
            p1 = np.array(_square_at(th_e[j+1], a))
            lam = ((xi + 1)/2)[None, :]
            xout = p0[0] + (p1[0] - p0[0])*lam
            yout = p0[1] + (p1[1] - p0[1])*lam
            Xo[e] = (1 - U)*xin + U*xout
            Yo[e] = (1 - U)*yin + U*yout

    # ---- the square's side divisions, where the rays land ----
    side = np.array([_square_at(t, a)[0] for t in th_e[:E_s+1]])   # top side, x
    xs_mid = np.sort(np.concatenate([side, -side[:-1]]))
    xs_mid = np.unique(np.round(xs_mid, 12))
    ys_mid = xs_mid.copy()                                  # symmetric by design

    # UPSTREAM EDGES.  Same argument as `ys_side` below: raising Lu and nx_up
    # together rescales every division between -Lu and -a, so the wide mesh
    # would not contain the narrow one and an Xu study would confound the
    # upstream extent with the upstream resolution.  `nested_upstream_edges`
    # keeps the reference divisions and prepends new elements outside them.
    if xs_upstream is None:
        xs_up = _geom_edges(-Lu, -a, nx_up, 1.0/ratio_out)
    else:
        xs_up = np.asarray(xs_upstream, dtype=float)
        if xs_up.ndim != 1 or xs_up.size < 2:
            raise ValueError('xs_upstream must be a 1-D array of at least 2 edges')
        if abs(xs_up[-1] + a) > 1e-12:
            raise ValueError(f'xs_upstream must end at -a = {-a}, got {xs_up[-1]}')
        if np.any(np.diff(xs_up) <= 0):
            raise ValueError('xs_upstream must increase strictly')
        Lu = float(-xs_up[0])           # the explicit edges define the inlet
    # DOWNSTREAM EDGES, same nesting argument as the other two directions.
    # Posdziech & Grundmann's domain protocol fixes the streamwise bounds far
    # away (L_in >= 30D, L_out >= 50D) so that variations in the measured
    # quantities depend SOLELY on the lateral height H; reaching those bounds
    # without disturbing the wake resolution needs the same nested treatment.
    if xs_downstream is None:
        xs_dn = _geom_edges(a, Ld, nx_dn, ratio_out)
    else:
        xs_dn = np.asarray(xs_downstream, dtype=float)
        if xs_dn.ndim != 1 or xs_dn.size < 2:
            raise ValueError('xs_downstream must be a 1-D array of at least 2 edges')
        if abs(xs_dn[0] - a) > 1e-12:
            raise ValueError(f'xs_downstream must start at a = {a}, got {xs_dn[0]}')
        if np.any(np.diff(xs_dn) <= 0):
            raise ValueError('xs_downstream must increase strictly')
        Ld = float(xs_dn[-1])            # the explicit edges define the outlet
    # LATERAL EDGES, AND WHY THEY CAN BE GIVEN EXPLICITLY.  `_geom_edges`
    # normalises its widths to span a0..a1 exactly, so raising H alone rescales
    # EVERY division between a and H -- the wide mesh is then not a refinement
    # of the narrow one, and a domain-size study run that way confounds the
    # domain with the lateral resolution.  Behr et al. (1995) avoided this by
    # using nested meshes that differ ONLY in how far the outermost elements
    # reach; `ys_side` is how that is done here.  Pass the narrow mesh's own
    # edges with extra ones appended (see `nested_lateral_edges`) and the two
    # meshes share every node with |y| <= the narrow half-height.
    if ys_side is None:
        # NOTE the two separate calls rather than mirroring one of them: the
        # mirror agrees only to 1 ulp, and every run recorded so far was made
        # with these exact edges.  Reproducibility is worth the duplication.
        ys_hi = _geom_edges(a, H, ny_side, ratio_out)
        ys_lo = _geom_edges(-H, -a, ny_side, 1.0/ratio_out)
    else:
        ys_hi = np.asarray(ys_side, dtype=float)
        if ys_hi.ndim != 1 or ys_hi.size < 2:
            raise ValueError('ys_side must be a 1-D array of at least 2 edges')
        if abs(ys_hi[0] - a) > 1e-12:
            raise ValueError(f'ys_side must start at the square half-width a = {a}, '
                             f'got {ys_hi[0]}')
        if np.any(np.diff(ys_hi) <= 0):
            raise ValueError('ys_side must increase strictly')
        H = float(ys_hi[-1])            # the explicit edges define the box
        ys_lo = -ys_hi[::-1]

    blocks = []
    for xe in (xs_up, xs_mid, xs_dn):
        for ye in (ys_lo, ys_mid, ys_hi):
            if xe is xs_mid and ye is ys_mid:
                continue                                     # the O-ring's hole
            blocks.append(_rect_block(xe, ye, N))

    X = np.concatenate([Xo] + [b[0] for b in blocks], axis=0)
    Y = np.concatenate([Yo] + [b[1] for b in blocks], axis=0)
    nelem = X.shape[0]

    mesh = Mesh(nelem, N)
    mesh.x0 = X.min(axis=(1, 2))
    mesh.y0 = Y.min(axis=(1, 2))
    mesh.hx = X.max(axis=(1, 2)) - mesh.x0
    mesh.hy = Y.max(axis=(1, 2)) - mesh.y0
    mesh.xnod = X[:, :, 0].copy()
    mesh.ynod = Y[:, 0, :].copy()
    attach(mesh, X, Y)

    # ---- boundary codes from geometry, not from block bookkeeping ----
    # An edge gets a code when EVERY node on it sits on that boundary.  Reading
    # it off the coordinates rather than off block indices is what keeps this
    # honest when the topology stops being a single grid.
    tol = 1e-9
    tests = ((lambda x, y: np.abs(np.hypot(x, y) - r_cyl) < 1e-8, bcs[0]),
             (lambda x, y: np.abs(x + Lu) < tol, bcs[1]),
             (lambda x, y: np.abs(x - Ld) < tol, bcs[2]),
             (lambda x, y: np.abs(np.abs(y) - H) < tol, bcs[3]))
    edges = ((np.s_[0, :], 0), (np.s_[-1, :], 1), (np.s_[:, 0], 2), (np.s_[:, -1], 3))
    for e in range(nelem):
        for sl, d in edges:
            ex, ey = X[e][sl], Y[e][sl]
            for fn, code in tests:
                if np.all(fn(ex, ey)):
                    mesh.bc[e, d] = code
                    break
    mesh.r_cyl, mesh.a_sq, mesh.box = r_cyl, a, (-Lu, Ld, -H, H)
    return mesh
