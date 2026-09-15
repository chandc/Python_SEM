"""Fused @njit kernels for the 3D VVP operator.

Compiled equivalents of `operator.apply_L` and `operator.apply_LT`, working
DIRECTLY on the split-real 5-D layout `(elem, i, j, var, mode)`.

WHY THIS IS A FUSION AND NOT A TRANSLATION.  The matvec is memory-bandwidth
bound (threads tie processes, `prof3d_procs.py`), so compiling the NumPy
expression tree in place would buy nothing -- it already dispatches to BLAS.
The NumPy path makes roughly thirty passes over the state per application:

    to_complex, 14 einsums (7 fields x 2 directions), 8 row assemblies,
    the wq multiply, the row-weight multiply, to_real

These kernels make ONE.  For each (element, node, mode) they accumulate the
fourteen derivative sums and assemble all sixteen real rows in registers.

REAL ARITHMETIC ON THE SPLIT-REAL LAYOUT, deliberately.  Fields 0..6 are the
real parts and 7..13 the imaginary; rows 0..7 real, 8..15 imaginary.  Working in
real arithmetic avoids materialising a complex temporary (the `to_complex` pass)
and makes the i*k coupling explicit:

    i*k*(a + i*b)  ->  real part  -k*b,   imaginary part  +k*a

`nogil=True` IS LOAD-BEARING, not a micro-optimisation.  `lssem3d.parallel.pcg`
spreads the z-modes across a ThreadPoolExecutor and gets 6.7x from it -- which
works only because NumPy's einsum and BLAS release the GIL.  An njit kernel
holds the GIL unless told otherwise, so without this flag the fused kernels
would serialise the mode loop and hand back most of what they just won.  The
kernels take no Python objects, so releasing it is safe.

The element loop is left SERIAL (no prange).  `prof3d_procs.py` measured threads
tying processes on this operator -- it is memory-bandwidth bound -- so a second,
nested layer of threads inside a thread-parallel mode loop would oversubscribe
for no gain.  Parallelism belongs at the mode level, where the data is disjoint.

CACHE TRAP, inherited from lssem2d/kernels_numba.py: numba's on-disk cache does
NOT key on njit flags such as `fastmath`.  A cache written with fastmath=True is
silently reused when fastmath=False is asked for.  The cache directory is
therefore stamped with the flavour.
"""
import os

import numpy as np
from numba import njit

# NOT `from . import operator`: operator._bind_backend imports THIS module, so
# the reverse import at module scope is a cycle.  It survives today on import
# ordering alone -- kernels_torch.py, written the same way, did not.  Pinned to
# operator.py by test_backend_parity.

# Stamp the cache directory with the fastmath flavour -- numba will not do it.
_FASTMATH = os.environ.get('LSSEM3D_FASTMATH', '1') not in ('0', 'false', 'False')
os.environ.setdefault(
    'NUMBA_CACHE_DIR',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '__nbcache__',
                 'fastmath' if _FASTMATH else 'strict'))

NV = 7            # complex fields u v w ox oy oz p -> 14 real
NR = 8            # complex residual rows           -> 16 real


@njit(fastmath=_FASTMATH, nogil=True, boundscheck=False, cache=True)
def _kernel_L(U, D, facx, facy, kz, nu, c, wq, kap, rw, R):
    """R = diag(rw) W L0 U, in one pass.  U, R split-real 5-D."""
    nel, n = U.shape[0], U.shape[1]
    nk = U.shape[4]
    for e in range(nel):
        fx = facx[e]
        fy = facy[e]
        for k in range(nk):
            kk = kz[k]
            for i in range(n):
                for j in range(n):
                    # --- the fourteen derivative sums, accumulated in registers
                    uxr = uxi = uyr = uyi = 0.0
                    vxr = vxi = vyr = vyi = 0.0
                    wxr = wxi = wyr = wyi = 0.0
                    oxxr = oxxi = oxyr = oxyi = 0.0
                    oyxr = oyxi = oyyr = oyyi = 0.0
                    ozxr = ozxi = ozyr = ozyi = 0.0
                    pxr = pxi = pyr = pyi = 0.0
                    for a in range(n):
                        dx = D[i, a]*fx
                        dy = D[j, a]*fy
                        uxr += dx*U[e, a, j, 0, k];      uxi += dx*U[e, a, j, 7, k]
                        uyr += dy*U[e, i, a, 0, k];      uyi += dy*U[e, i, a, 7, k]
                        vxr += dx*U[e, a, j, 1, k];      vxi += dx*U[e, a, j, 8, k]
                        vyr += dy*U[e, i, a, 1, k];      vyi += dy*U[e, i, a, 8, k]
                        wxr += dx*U[e, a, j, 2, k];      wxi += dx*U[e, a, j, 9, k]
                        wyr += dy*U[e, i, a, 2, k];      wyi += dy*U[e, i, a, 9, k]
                        oxxr += dx*U[e, a, j, 3, k];     oxxi += dx*U[e, a, j, 10, k]
                        oxyr += dy*U[e, i, a, 3, k];     oxyi += dy*U[e, i, a, 10, k]
                        oyxr += dx*U[e, a, j, 4, k];     oyxi += dx*U[e, a, j, 11, k]
                        oyyr += dy*U[e, i, a, 4, k];     oyyi += dy*U[e, i, a, 11, k]
                        ozxr += dx*U[e, a, j, 5, k];     ozxi += dx*U[e, a, j, 12, k]
                        ozyr += dy*U[e, i, a, 5, k];     ozyi += dy*U[e, i, a, 12, k]
                        pxr += dx*U[e, a, j, 6, k];      pxi += dx*U[e, a, j, 13, k]
                        pyr += dy*U[e, i, a, 6, k];      pyi += dy*U[e, i, a, 13, k]
                    ur = U[e, i, j, 0, k];  ui = U[e, i, j, 7, k]
                    vr = U[e, i, j, 1, k];  vi = U[e, i, j, 8, k]
                    wr = U[e, i, j, 2, k];  wi = U[e, i, j, 9, k]
                    oxr = U[e, i, j, 3, k]; oxi = U[e, i, j, 10, k]
                    oyr = U[e, i, j, 4, k]; oyi = U[e, i, j, 11, k]
                    ozr = U[e, i, j, 5, k]; ozi = U[e, i, j, 12, k]
                    pr = U[e, i, j, 6, k];  pi_ = U[e, i, j, 13, k]
                    q = wq[e, i, j]

                    # --- the eight complex rows, as sixteen real ones.
                    # i*k*(a + i*b) -> real -k*b, imag +k*a
                    s = rw[0]*q
                    R[e, i, j, 0, k] = s*(kap*pr + uxr + vyr - kk*wi)
                    R[e, i, j, 8, k] = s*(kap*pi_ + uxi + vyi + kk*wr)
                    s = rw[1]*q
                    R[e, i, j, 1, k] = s*(wyr + kk*vi - oxr)
                    R[e, i, j, 9, k] = s*(wyi - kk*vr - oxi)
                    s = rw[2]*q
                    R[e, i, j, 2, k] = s*(-kk*ui - wxr - oyr)
                    R[e, i, j, 10, k] = s*(kk*ur - wxi - oyi)
                    s = rw[3]*q
                    R[e, i, j, 3, k] = s*(vxr - uyr - ozr)
                    R[e, i, j, 11, k] = s*(vxi - uyi - ozi)
                    s = rw[4]*q
                    R[e, i, j, 4, k] = s*(c*ur + pxr + nu*(ozyr + kk*oyi))
                    R[e, i, j, 12, k] = s*(c*ui + pxi + nu*(ozyi - kk*oyr))
                    s = rw[5]*q
                    R[e, i, j, 5, k] = s*(c*vr + pyr + nu*(-kk*oxi - ozxr))
                    R[e, i, j, 13, k] = s*(c*vi + pyi + nu*(kk*oxr - ozxi))
                    s = rw[6]*q
                    R[e, i, j, 6, k] = s*(c*wr - kk*pi_ + nu*(oyxr - oxyr))
                    R[e, i, j, 14, k] = s*(c*wi + kk*pr + nu*(oyxi - oxyi))
                    s = rw[7]*q
                    R[e, i, j, 7, k] = s*(oxxr + oyyr - kk*ozi)
                    R[e, i, j, 15, k] = s*(oxxi + oyyi + kk*ozr)


@njit(fastmath=_FASTMATH, nogil=True, boundscheck=False, cache=True)
def _kernel_LT(R, D, facx, facy, kz, nu, c, kap, C):
    """C = L0^T R, in one pass.  R split-real 16 rows, C split-real 14 fields.

    The transposes use D[a, i] (not D[i, a]) with the SAME sign as the forward
    term -- the (x, y) derivatives are not anti-self-adjoint, and assuming they
    were failed the adjoint test by ~70% in an earlier draft.  In split-real
    form the transpose corresponds to the complex CONJUGATE, so i*k -> -i*k
    while real c and nu are unchanged.
    """
    nel, n = R.shape[0], R.shape[1]
    nk = R.shape[4]
    for e in range(nel):
        fx = facx[e]
        fy = facy[e]
        for k in range(nk):
            kk = kz[k]
            for i in range(n):
                for j in range(n):
                    # transposed derivatives of each row
                    t0xr = t0xi = t0yr = t0yi = 0.0
                    t1yr = t1yi = 0.0
                    t2xr = t2xi = 0.0
                    t3xr = t3xi = t3yr = t3yi = 0.0
                    t4xr = t4xi = t4yr = t4yi = 0.0
                    t5xr = t5xi = t5yr = t5yi = 0.0
                    t6xr = t6xi = t6yr = t6yi = 0.0
                    t7xr = t7xi = t7yr = t7yi = 0.0
                    for a in range(n):
                        dx = D[a, i]*fx
                        dy = D[a, j]*fy
                        t0xr += dx*R[e, a, j, 0, k];  t0xi += dx*R[e, a, j, 8, k]
                        t0yr += dy*R[e, i, a, 0, k];  t0yi += dy*R[e, i, a, 8, k]
                        t1yr += dy*R[e, i, a, 1, k];  t1yi += dy*R[e, i, a, 9, k]
                        t2xr += dx*R[e, a, j, 2, k];  t2xi += dx*R[e, a, j, 10, k]
                        t3xr += dx*R[e, a, j, 3, k];  t3xi += dx*R[e, a, j, 11, k]
                        t3yr += dy*R[e, i, a, 3, k];  t3yi += dy*R[e, i, a, 11, k]
                        t4xr += dx*R[e, a, j, 4, k];  t4xi += dx*R[e, a, j, 12, k]
                        t4yr += dy*R[e, i, a, 4, k];  t4yi += dy*R[e, i, a, 12, k]
                        t5xr += dx*R[e, a, j, 5, k];  t5xi += dx*R[e, a, j, 13, k]
                        t5yr += dy*R[e, i, a, 5, k];  t5yi += dy*R[e, i, a, 13, k]
                        t6xr += dx*R[e, a, j, 6, k];  t6xi += dx*R[e, a, j, 14, k]
                        t6yr += dy*R[e, i, a, 6, k];  t6yi += dy*R[e, i, a, 14, k]
                        t7xr += dx*R[e, a, j, 7, k];  t7xi += dx*R[e, a, j, 15, k]
                        t7yr += dy*R[e, i, a, 7, k];  t7yi += dy*R[e, i, a, 15, k]
                    r0r = R[e, i, j, 0, k];  r0i = R[e, i, j, 8, k]
                    r1r = R[e, i, j, 1, k];  r1i = R[e, i, j, 9, k]
                    r2r = R[e, i, j, 2, k];  r2i = R[e, i, j, 10, k]
                    r3r = R[e, i, j, 3, k];  r3i = R[e, i, j, 11, k]
                    r4r = R[e, i, j, 4, k];  r4i = R[e, i, j, 12, k]
                    r5r = R[e, i, j, 5, k];  r5i = R[e, i, j, 13, k]
                    r6r = R[e, i, j, 6, k];  r6i = R[e, i, j, 14, k]
                    r7r = R[e, i, j, 7, k];  r7i = R[e, i, j, 15, k]

                    # mik = -i*k :  -i*k*(a + i*b) -> real +k*b, imag -k*a
                    C[e, i, j, 0, k] = t0xr + kk*r2i - t3yr + c*r4r
                    C[e, i, j, 7, k] = t0xi - kk*r2r - t3yi + c*r4i
                    C[e, i, j, 1, k] = t0yr - kk*r1i + t3xr + c*r5r
                    C[e, i, j, 8, k] = t0yi + kk*r1r + t3xi + c*r5i
                    C[e, i, j, 2, k] = kk*r0i + t1yr - t2xr + c*r6r
                    C[e, i, j, 9, k] = -kk*r0r + t1yi - t2xi + c*r6i
                    C[e, i, j, 3, k] = -r1r + nu*kk*r5i - nu*t6yr + t7xr
                    C[e, i, j, 10, k] = -r1i - nu*kk*r5r - nu*t6yi + t7xi
                    C[e, i, j, 4, k] = -r2r - nu*kk*r4i + nu*t6xr + t7yr
                    C[e, i, j, 11, k] = -r2i + nu*kk*r4r + nu*t6xi + t7yi
                    C[e, i, j, 5, k] = -r3r + nu*t4yr - nu*t5xr + kk*r7i
                    C[e, i, j, 12, k] = -r3i + nu*t4yi - nu*t5xi - kk*r7r
                    C[e, i, j, 6, k] = t4xr + t5yr + kk*r6i + kap*r0r
                    C[e, i, j, 13, k] = t4xi + t5yi - kk*r6r + kap*r0i


def apply_L(Ur, D, facx, facy, kz, nu, c, wq=None, kap=0.0, rw=None):
    """Signature-compatible with operator.apply_L."""
    Ur = np.ascontiguousarray(Ur, dtype=np.float64)
    nel, n = Ur.shape[0], Ur.shape[1]
    nk = Ur.shape[4]
    if wq is None:
        wq = np.ones((nel, n, n))
    rwv = np.ones(NR) if rw is None else np.asarray(rw, dtype=np.float64)
    kzv = np.ascontiguousarray(np.broadcast_to(np.asarray(kz, dtype=np.float64),
                                               (nk,)))
    R = np.empty((nel, n, n, 2*NR, nk))
    _kernel_L(Ur, np.ascontiguousarray(D, dtype=np.float64),
              np.ascontiguousarray(facx, dtype=np.float64),
              np.ascontiguousarray(facy, dtype=np.float64),
              kzv, float(nu), float(c),
              np.ascontiguousarray(wq, dtype=np.float64), float(kap), rwv, R)
    return R


def apply_LT(Rr, D, facx, facy, kz, nu, c, kap=0.0):
    """Signature-compatible with operator.apply_LT."""
    Rr = np.ascontiguousarray(Rr, dtype=np.float64)
    nel, n = Rr.shape[0], Rr.shape[1]
    nk = Rr.shape[4]
    kzv = np.ascontiguousarray(np.broadcast_to(np.asarray(kz, dtype=np.float64),
                                               (nk,)))
    C = np.empty((nel, n, n, 2*NV, nk))
    _kernel_LT(Rr, np.ascontiguousarray(D, dtype=np.float64),
               np.ascontiguousarray(facx, dtype=np.float64),
               np.ascontiguousarray(facy, dtype=np.float64),
               kzv, float(nu), float(c), float(kap), C)
    return C


def warmup(n=4, nk=2):
    """Trigger compilation on a tiny problem, so the first real call is timed
    fairly.  Compilation is ~seconds and would otherwise land inside whatever
    benchmark runs first."""
    U = np.zeros((1, n, n, 2*NV, nk))
    R = apply_L(U, np.zeros((n, n)), np.ones(1), np.ones(1), np.zeros(nk),
                1.0, 1.0, np.ones((1, n, n)), 0.0, np.ones(NR))
    apply_LT(R, np.zeros((n, n)), np.ones(1), np.ones(1), np.zeros(nk), 1.0, 1.0, 0.0)
