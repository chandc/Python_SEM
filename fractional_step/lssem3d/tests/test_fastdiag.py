"""fastdiag: the fast-diagonalization tensor-product inverse.

The PRECONDITIONING application to the LS normal equations was measured and
REJECTED (3D_STATUS.md sec 7F): the normal operator's O(1) inter-field
couplings (continuity ties u,v,w; the vorticity DEFINITION rows tie velocity
to vorticity at full strength) defeat any field-decoupled surrogate -- 759 CG
iterations against plain Jacobi's 259 on a production stage solve, and
weakening the surrogate stiffness only made it worse (8214 at alpha = 0).

The MACHINERY stays, tested here, because it is exactly the direct solver a
projection/fractional-step stage would use (scalar Helmholtz and Poisson per
mode), where there is no coupling to defeat it.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem3d import fastdiag as FD, operator as OP, fourier as FR

L = 2.0*np.pi


def make(N=3, Ex=2, Ey=2, nz=4):
    m = build_channel(L, L, Ex, Ey, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L
    m.compute_global_indices()
    kz = FR.wavenumbers(nz, L)
    return m, diff_matrix(N), kz


def test_inverts_explicit_kronecker_surrogate():
    N, Ex, Ey, nz = 3, 2, 2, 4
    m, D, kz = make(N, Ex, Ey, nz)
    fd = FD.FastDiagPeriodic(m, D, N, nz, kz, nu=0.01, c=300.0, mask=None)
    w = lgl_weights(N)
    Mx, Ax = FD._assemble_1d(Ex, N, float(m.hx[0]), D, w)
    My, Ay = FD._assemble_1d(Ey, N, float(m.hy[0]), D, w)
    k = kz[1]
    Asur = (np.kron(Ax, np.diag(My)) + np.kron(np.diag(Mx), Ay)
            + (k*k + 1)*np.kron(np.diag(Mx), np.diag(My)))
    ngx, ngy = Ex*N, Ey*N
    rng = np.random.default_rng(3)
    x = rng.standard_normal(ngx*ngy)
    y = (Asur @ x).reshape(ngx, ngy)
    n = N + 1
    r = np.zeros((m.nelem, n, n, OP.NVAR_R, len(kz)))
    r[..., 0, 1] = y[fd.GX[:, :, None], fd.GY[:, None, :]]
    z = fd(r)
    xrec = np.zeros((ngx, ngy))
    xrec[fd.GX[:, :, None], fd.GY[:, None, :]] = z[..., 0, 1]
    err = np.abs(xrec.ravel() - x.reshape(ngx, ngy).ravel()).max()
    assert err < 1e-11*np.abs(x).max()


def test_symmetric_in_multiplicity_inner_product():
    N, Ex, Ey, nz = 3, 2, 2, 4
    m, D, kz = make(N, Ex, Ey, nz)
    fd = FD.FastDiagPeriodic(m, D, N, nz, kz, nu=0.01, c=300.0, mask=None)
    from lssem3d import solver3d as S3
    n = N + 1
    shape = (m.nelem, n, n, OP.NVAR_R, len(kz))
    mw = S3.multiplicity_weight(m, shape)
    rng = np.random.default_rng(5)
    # continuous vectors (equal copies), as CG residuals are
    a = S3.gs(m, rng.standard_normal(shape))*mw
    b = S3.gs(m, rng.standard_normal(shape))*mw
    d1 = float(np.sum(fd(a)*b*mw))
    d2 = float(np.sum(a*fd(b)*mw))
    assert abs(d1 - d2) < 1e-10*max(abs(d1), 1e-30)


def test_null_direction_zeroed():
    """The k = 0 pressure constant must map to zero, not to 1/0."""
    N, Ex, Ey, nz = 3, 2, 2, 4
    m, D, kz = make(N, Ex, Ey, nz)
    fd = FD.FastDiagPeriodic(m, D, N, nz, kz, nu=0.01, c=300.0, mask=None)
    n = N + 1
    r = np.zeros((m.nelem, n, n, OP.NVAR_R, len(kz)))
    r[..., OP.P_, 0] = 1.0                       # constant p, k_z = 0
    z = fd(r)
    assert np.all(np.isfinite(z))
