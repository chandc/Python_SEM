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

from . import device as DEV
from .operator import (NVAR, NROW, U_, V_, W_, OX_, OY_, OZ_, P_)



# --------------------------------------------------------------- fused rows
#
# WHY.  Profiled on a Colab A100, `normal_op` costs the same 3.8 ms at 0.004 M
# dof and at 6.17 M -- a 1500x range in work, identical wall clock -- so on
# that host the GPU contribution is invisible and every millisecond is Python
# issuing kernels.  96% of it sat in these row expressions: 16 formulas built
# from ~80 separate elementwise calls at ~32 us each.
#
# The algebra below is IDENTICAL to the unfused version above; it is simply
# issued as ONE kernel launch per operator instead of forty.  `cupy_parity.py`
# holds the two to 1e-16 of each other, and the validation ladder is re-run
# after any change here.
#
# Complex arithmetic in the kernel body is thrust::complex<double>, which
# supports the mixed double*complex products these formulas need.

_L0_ROWS = cp.ElementwiseKernel(
    'complex128 u, complex128 v, complex128 w, '
    'complex128 ox, complex128 oy, complex128 oz, complex128 p, '
    'complex128 ux, complex128 uy, complex128 vx, complex128 vy, '
    'complex128 wx, complex128 wy, complex128 oxx, complex128 oxy, '
    'complex128 oyx, complex128 oyy, complex128 ozx, complex128 ozy, '
    'complex128 px, complex128 py, complex128 ik, '
    'float64 fx, float64 fy, float64 nu, float64 c, float64 kap',
    'complex128 r0, complex128 r1, complex128 r2, complex128 r3, '
    'complex128 r4, complex128 r5, complex128 r6, complex128 r7',
    """
    r0 = kap*p + fx*ux + fy*vy + ik*w;
    r1 = fy*wy - ik*v - ox;
    r2 = ik*u - fx*wx - oy;
    r3 = fx*vx - fy*uy - oz;
    r4 = c*u + fx*px + nu*(fy*ozy - ik*oy);
    r5 = c*v + fy*py + nu*(ik*ox - fx*ozx);
    r6 = c*w + ik*p + nu*(fx*oyx - fy*oxy);
    r7 = fx*oxx + fy*oyy + ik*oz;
    """,
    'lssem_L0_rows')

_LT_FIELDS = cp.ElementwiseKernel(
    'complex128 r0, complex128 r1, complex128 r2, complex128 r3, '
    'complex128 r4, complex128 r5, complex128 r6, complex128 r7, '
    'complex128 x0, complex128 x2, complex128 x3, complex128 x4, '
    'complex128 x5, complex128 x6, complex128 x7, '
    'complex128 y0, complex128 y1, complex128 y3, complex128 y4, '
    'complex128 y5, complex128 y6, complex128 y7, '
    'complex128 mik, float64 fx, float64 fy, '
    'float64 nu, float64 c, float64 kap',
    'complex128 cu, complex128 cv, complex128 cw, complex128 cox, '
    'complex128 coy, complex128 coz, complex128 cp_',
    """
    cu  = fx*x0 + mik*r2 - fy*y3 + c*r4;
    cv  = fy*y0 - mik*r1 + fx*x3 + c*r5;
    cw  = mik*r0 + fy*y1 - fx*x2 + c*r6;
    cox = -r1 + nu*mik*r5 - nu*fy*y6 + fx*x7;
    coy = -r2 - nu*mik*r4 + nu*fx*x6 + fy*y7;
    coz = -r3 + nu*fy*y4 - nu*fx*x5 + mik*r7;
    cp_ = fx*x4 + fy*y5 + mik*r6 + kap*r0;
    """,
    'lssem_LT_fields')


def _dx(U, D):
    return DEV.einsum('pi,eij...->epj...', D, U)


def _dy(U, D):
    return DEV.einsum('qj,eij...->eiq...', D, U)


def _dxT(S, D):
    return DEV.einsum('pi,epj...->eij...', D, S)


def _dyT(S, D):
    return DEV.einsum('qj,eiq...->eij...', D, S)


def _bcast(fac):
    """(nelem,) -> (nelem, 1, 1, 1), to broadcast against one field slice."""
    return fac.reshape(-1, 1, 1, 1)


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
    # METRIC SCALING FOLDED INTO THE KERNEL, NOT APPLIED HERE.  ddx is
    # `einsum(...)*fac`, and that trailing multiply is a whole extra pass over
    # a complex array -- ~576 MiB of traffic per call, four calls per matvec.
    # The fused kernel below already READS every one of these values, so
    # scaling them there is free (a register multiply) and removes the pass:
    # ddx measured 1.08 ms against 0.55 for the bare contraction.
    Ux = _dx(U, D)
    Uy = _dy(U, D)
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
    _L0_ROWS(u, v, w, ox, oy, oz, p, ux, uy, vx, vy, wx, wy,
             oxx, oxy, oyx, oyy, ozx, ozy, px, py, ik,
             _bcast(facx), _bcast(facy), nu, c, kap,
             R[..., 0, :], R[..., 1, :], R[..., 2, :], R[..., 3, :],
             R[..., 4, :], R[..., 5, :], R[..., 6, :], R[..., 7, :])
    return R


def _LT(R, D, facx, facy, kz, nu, c, kap=0.0):
    """Adjoint -- mirrors apply_LT_complex, including the conjugated i*k."""
    # Batched for the same reason as _L0: eight rows, two calls.
    RxT = _dxT(R, D)          # metrics folded into _LT_FIELDS, as in _L0
    RyT = _dyT(R, D)
    dxT = lambda i: RxT[..., i, :]
    dyT = lambda i: RyT[..., i, :]
    mik = -1j*kz
    r0, r1, r2, r3, r4, r5, r6, r7 = (R[..., i, :] for i in range(NROW))

    C = cp.empty(R.shape[:-2] + (NVAR, R.shape[-1]), dtype=cp.complex128)
    _LT_FIELDS(r0, r1, r2, r3, r4, r5, r6, r7,
               dxT(0), dxT(2), dxT(3), dxT(4), dxT(5), dxT(6), dxT(7),
               dyT(0), dyT(1), dyT(3), dyT(4), dyT(5), dyT(6), dyT(7),
               mik, _bcast(facx), _bcast(facy), nu, c, kap,
               C[..., U_, :], C[..., V_, :], C[..., W_, :], C[..., OX_, :],
               C[..., OY_, :], C[..., OZ_, :], C[..., P_, :])
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
