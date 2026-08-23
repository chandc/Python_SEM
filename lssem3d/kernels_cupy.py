"""CuPy kernels for the per-mode VVP operator -- a second, independent GPU path.

WHY A SECOND GPU BACKEND.  `kernels_torch.py` already runs this operator on the
GB10.  This module exists to be *independent of it*: a different array library,
a different kernel compiler, a different reduction implementation, validated
against the same NumPy reference.  When two ports that share no device code
agree with the reference on the whole ladder, that agreement is evidence; a
single port agreeing with itself is not (3D_STATUS.md L1).

WHY IT IS SO SHORT.  CuPy implements NEP-18, so `np.einsum`, `np.concatenate`
and friends called on a CuPy array dispatch to CuPy automatically -- which
means `deriv.py`, `fourier.py` and the whole of `pcg` already run on the device
untouched.  The ONLY thing that does not dispatch is array *creation*:
`np.empty` builds a host array no matter what its future contents are, and
assigning device data into it raises

    TypeError: Implicit conversion to a NumPy array is not allowed.

So the two output buffers below are the entire difference between the NumPy
reference and this file.  That is also why this is a transcription of the
reference algebra rather than a re-derivation: the *algebra* is not what is
being independently checked here, the *execution path* is.

DTYPE.  float64/complex128 throughout, as everywhere else in this project.
The GB10 is an AI part and its FP64 rate is ~1/41 of FP32 (measured: 0.21
against 8.61 TFLOP/s), so this backend is a CORRECTNESS platform on the Spark
and a performance platform only on FP64-capable hardware (A100: 9.7 TFLOP/s).
Do not be tempted into float32 -- `TGV_VALIDATION.md` validates sigma to seven
digits and the e-metric at the 1e-3 level.
"""
import cupy as cp

from .deriv import ddx, ddy, ddxT, ddyT
from .operator import (NVAR, NROW, U_, V_, W_, OX_, OY_, OZ_, P_)


def _to_complex(Ur):
    return Ur[..., :NVAR, :] + 1j*Ur[..., NVAR:, :]


def _to_real(Uc):
    return cp.concatenate([Uc.real, Uc.imag], axis=-2)


def _L0(U, D, facx, facy, kz, nu, c, kap=0.0):
    """8 complex residual rows from 7 complex fields -- mirrors apply_L0_complex."""
    # BATCHED DERIVATIVES.  ddx/ddy already carry arbitrary trailing axes and
    # the field axis is one of them -- so both derivatives of ALL SEVEN fields
    # are two einsum calls, not fourteen.  Identical arithmetic; 12 fewer
    # cuBLAS dispatches per application, which is what the host pays for when
    # a fast GPU sits behind a slow CPU (CUPY_BACKEND.md).
    Ux = ddx(U, D, facx)
    Uy = ddy(U, D, facy)
    ux, uy = Ux[..., U_, :], Uy[..., U_, :]
    vx, vy = Ux[..., V_, :], Uy[..., V_, :]
    wx, wy = Ux[..., W_, :], Uy[..., W_, :]
    oxx, oxy = Ux[..., OX_, :], Uy[..., OX_, :]
    oyx, oyy = Ux[..., OY_, :], Uy[..., OY_, :]
    ozx, ozy = Ux[..., OZ_, :], Uy[..., OZ_, :]
    px, py = Ux[..., P_, :], Uy[..., P_, :]
    u, v, w = U[..., U_, :], U[..., V_, :], U[..., W_, :]
    ox, oy, oz = U[..., OX_, :], U[..., OY_, :], U[..., OZ_, :]
    p = U[..., P_, :]
    ik = 1j*kz

    R = cp.empty(U.shape[:-2] + (NROW, U.shape[-1]), dtype=cp.complex128)
    R[..., 0, :] = kap*p + ux + vy + ik*w
    R[..., 1, :] = wy - ik*v - ox
    R[..., 2, :] = ik*u - wx - oy
    R[..., 3, :] = vx - uy - oz
    R[..., 4, :] = c*u + px + nu*(ozy - ik*oy)
    R[..., 5, :] = c*v + py + nu*(ik*ox - ozx)
    R[..., 6, :] = c*w + ik*p + nu*(oyx - oxy)
    R[..., 7, :] = oxx + oyy + ik*oz
    return R


def _LT(R, D, facx, facy, kz, nu, c, kap=0.0):
    """Adjoint -- mirrors apply_LT_complex, including the conjugated i*k."""
    # Batched for the same reason as _L0: eight rows, two calls.
    RxT = ddxT(R, D, facx)
    RyT = ddyT(R, D, facy)
    dxT = lambda i: RxT[..., i, :]
    dyT = lambda i: RyT[..., i, :]
    mik = -1j*kz
    r0, r1, r2, r3, r4, r5, r6, r7 = (R[..., i, :] for i in range(NROW))

    C = cp.empty(R.shape[:-2] + (NVAR, R.shape[-1]), dtype=cp.complex128)
    C[..., U_, :] = dxT(0) + mik*r2 - dyT(3) + c*r4
    C[..., V_, :] = dyT(0) - mik*r1 + dxT(3) + c*r5
    C[..., W_, :] = mik*r0 + dyT(1) - dxT(2) + c*r6
    C[..., OX_, :] = -r1 + nu*mik*r5 - nu*dyT(6) + dxT(7)
    C[..., OY_, :] = -r2 - nu*mik*r4 + nu*dxT(6) + dyT(7)
    C[..., OZ_, :] = -r3 + nu*dyT(4) - nu*dxT(5) + mik*r7
    C[..., P_, :] = dxT(4) + dyT(5) + mik*r6 + kap*r0
    return C


def apply_L(Ur, D, facx, facy, kz, nu, c, wq=None, kap=0.0, rw=None):
    """(..., 14, nmode) real -> (..., 16, nmode) real.  Weighted if wq given."""
    R = _L0(_to_complex(Ur), D, facx, facy, kz, nu, c, kap)
    if rw is not None:
        R = R*cp.asarray(rw).reshape((1,)*(R.ndim - 2) + (len(rw), 1))
    if wq is not None:
        R = R*wq[..., None, None]
    return _to_real(R)


def apply_LT(Rr, D, facx, facy, kz, nu, c, kap=0.0):
    """(..., 16, nmode) real -> (..., 14, nmode) real."""
    Rc = Rr[..., :NROW, :] + 1j*Rr[..., NROW:, :]
    return _to_real(_LT(Rc, D, facx, facy, kz, nu, c, kap))


def available():
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
