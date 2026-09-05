"""Render a wall-parallel plane of a spectral-element field SMOOTHLY.

WHY THIS EXISTS.  Feeding the raw nodal values to tricontourf is wrong twice
over.  (1) The element-local array repeats every interface node once per owning
element -- measured, 216 points carrying only 49 distinct x locations, some with
multiplicity 8 -- and Delaunay triangulation of duplicated points yields
degenerate triangles.  (2) GLL nodes cluster at element edges (3.6:1 spacing
ratio here), so even after deduplication the triangulation is full of slivers.
Together they produce the X-shaped artifacts that look like element-boundary
physics and are not.

The right thing for a spectral element is to EVALUATE THE INTERPOLANT.  The
solution is a degree-N polynomial inside each element, so it can be sampled
anywhere exactly:  interpolate in y to the requested plane, in x onto a uniform
sub-grid per element via the Lagrange basis, and in z by zero-padding the FFT
(the z direction is Fourier, so that is exact too).  The result is a uniform
image, spectrally accurate, with no triangulation anywhere.
"""
import numpy as np


def lagrange_at(nodes, xq):
    """L[q, j] = l_j(xq[q]) for the Lagrange basis on `nodes`."""
    nodes = np.asarray(nodes, float).ravel()
    xq = np.atleast_1d(np.asarray(xq, float))
    n = len(nodes)
    L = np.ones((len(xq), n))
    for j in range(n):
        for k in range(n):
            if k != j:
                L[:, j] *= (xq - nodes[k])/(nodes[j] - nodes[k])
    return L


def plane(fld, m, N, yp_target, RT=180.0, nx_per_elem=12, nzf=None, nz=None):
    """fld: (nelem, N+1, N+1, nz) physical.  Returns (x, z, image[nz', nx'])."""
    nz = nz or fld.shape[-1]
    y = yp_target/RT                                  # lower half of the channel
    # element rows/columns from the nodal coordinates
    x0 = np.array([m.xnod[e][0] for e in range(m.nelem)])
    y0 = np.array([m.ynod[e][0] for e in range(m.nelem)])
    y1 = np.array([m.ynod[e][-1] for e in range(m.nelem)])
    row = np.flatnonzero((y0 - 1e-12 <= y) & (y <= y1 + 1e-12))
    if len(row) == 0:                                  # clamp to the nearest row
        row = np.flatnonzero(np.abs(y0 - y) == np.abs(y0 - y).min())
    yel = row[0]
    band = np.flatnonzero(np.abs(y0 - y0[yel]) < 1e-12)   # every element in that row
    order = np.argsort(x0[band]); band = band[order]

    # 1. interpolate in y to the requested plane (exact: degree-N polynomial)
    Ly = lagrange_at(m.ynod[yel], np.clip(y, m.ynod[yel][0], m.ynod[yel][-1]))[0]
    slab = np.einsum('j,eijz->eiz', Ly, fld[band])        # (nex, N+1, nz)

    # 2. interpolate in x onto a uniform sub-grid inside each element
    cols, xs = [], []
    for a, e in enumerate(band):
        xe = m.xnod[e]
        xq = np.linspace(xe[0], xe[-1], nx_per_elem, endpoint=(a == len(band)-1))
        cols.append(np.einsum('qi,iz->qz', lagrange_at(xe, xq), slab[a]))
        xs.append(xq)
    img = np.concatenate(cols, axis=0)                    # (nx', nz)
    xf = np.concatenate(xs)

    # 3. refine z by zero-padding the FFT -- exact, z is a Fourier direction
    nzf = nzf or 4*nz
    F = np.fft.rfft(img, axis=1)
    P = np.zeros(img.shape[:1] + (nzf//2 + 1,), complex)
    P[:, :F.shape[1]] = F
    imgz = np.fft.irfft(P, nzf, axis=1)*(nzf/nz)
    return xf, np.arange(nzf)/nzf, imgz.T                 # image[z, x]
