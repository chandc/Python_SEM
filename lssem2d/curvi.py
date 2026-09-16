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
    mesh.xnod = X[:, :, 0].copy()          # r-direction trace, for hashing
    mesh.ynod = Y[:, 0, :].copy()
    attach(mesh, X, Y)
    mesh.E_r, mesh.E_th = E_r, E_th
    return mesh
