"""PyTorch kernels for the 3D VVP operator — the CUDA path.

Third backend, after `numpy` (reference) and `numba` (fused CPU). Selected with
`LSSEM3D_BACKEND=torch` or `lssem3d.set_backend('torch')`.

WHY A THIRD BACKEND.  3D_STATUS.md §7N: Metal has no FP64, so MLX cannot run this
solver on a GPU at all; the GB10 can, at 141.8 GB/s against the Mac's best 33.6.
§7O: PyTorch beats cuPyNumeric 5.4× at 88³ and keeps a custom-CUDA-kernel path
open. GPU_PORT_PLAN.md §0: the minimal channel measures at ~35 days on this Mac,
which is what makes the port worth doing.

THE CONTRACTION SHAPE IS A MEASURED CHOICE, NOT A STYLE ONE.  §7O found torch
**faster unfused than batched** — 5.3 ms against 10.1 ms at 88³ — the opposite of
cuPyNumeric's preference. The 5-D form `pi,eijvk->epjvk` falls off a path the
4-D form stays on. So the field and mode axes are MERGED into one trailing axis
and every derivative is a 4-D contraction:

    (nel, n, n, V, K)  ->  (nel, n, n, V*K)     # a view, C-contiguous already
    einsum('pi,eijm->epjm', D, U)

which keeps the fast path *and* takes all 14 real fields in one call rather than
14. Do not "simplify" this back to the 5-D form without re-benchmarking.

NO FUSION HERE — DELIBERATELY.  `kernels_numba.py` collapses ~30 passes into one
and that is where its 7.6× came from. This module does **not**: torch has no
mechanism to express that fusion, and reaching for `torch.compile` before the
unfused version is correct would confound two changes at once. The fused CUDA
kernel is the upside case (GPU_PORT_PLAN.md §6), and `_kernel_L` in the numba
module is already written as explicit scalar arithmetic — the exact form a CUDA
kernel wants — so that path stays open.

DEVICE AND DTYPE.  float64 throughout; §7N is unambiguous that FP32 is not an
option while the normal equations square κ ≈ 1e4. The facades accept NumPy or
torch input so the parity tests can drive them, but **every host↔device copy in
the CG loop must be eliminated in Phase 2** — at ~4800 iterations per stage a
49 MB round trip per iteration is tens of GB of PCIe traffic per step.
"""
import os

import numpy as np
import torch

# NO `from . import operator` HERE.  operator._bind_backend imports THIS module,
# so importing it back at module scope is a circular import that fails the moment
# the backend is selected:
#     ImportError: cannot import name 'apply_L' from partially initialized
#                  module lssem3d.kernels_torch
# The constants are part of the format, not of operator.py's behaviour, so they
# are stated here and pinned to operator.py by test_backend_parity.
NV = 7             # complex fields u v w ox oy oz p  -> 14 real
NR = 8             # complex residual rows            -> 16 real
U_, V_, W_, OX_, OY_, OZ_, P_ = range(NV)


def device():
    """Honour LSSEM3D_DEVICE, else CUDA when present."""
    d = os.environ.get('LSSEM3D_DEVICE')
    if d:
        return torch.device(d)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _t(a, dev):
    """To a float64 tensor on `dev`, without copying a tensor already there."""
    if isinstance(a, torch.Tensor):
        return a.to(device=dev, dtype=torch.float64)
    arr = np.ascontiguousarray(a, dtype=np.float64)
    if not arr.flags.writeable:
        # torch shares memory with the numpy buffer and warns on a read-only
        # one.  Copy only in that case -- copying unconditionally would double
        # the traffic on a bandwidth-bound operator.
        arr = arr.copy()
    return torch.as_tensor(arr, device=dev)


# ------------------------------------------------------------- derivatives

def _ddx(U, D, facx):
    """d/dx of every (field, mode) at once.  U is (nel, n, n, V, K)."""
    nel, n, _, V, K = U.shape
    out = torch.einsum('pi,eijm->epjm', D, U.reshape(nel, n, n, V*K))
    return out.reshape(nel, n, n, V, K)*facx.view(-1, 1, 1, 1, 1)


def _ddy(U, D, facy):
    nel, n, _, V, K = U.shape
    out = torch.einsum('qj,eijm->eiqm', D, U.reshape(nel, n, n, V*K))
    return out.reshape(nel, n, n, V, K)*facy.view(-1, 1, 1, 1, 1)


def _ddxT(S, D, facx):
    """Adjoint of _ddx: D[a, i] rather than D[i, a], SAME sign as the forward
    term.  The (x, y) derivatives are not anti-self-adjoint — assuming they were
    failed the adjoint test by ~70% when this operator was first written."""
    nel, n, _, V, K = S.shape
    out = torch.einsum('pi,epjm->eijm', D, S.reshape(nel, n, n, V*K))
    return out.reshape(nel, n, n, V, K)*facx.view(-1, 1, 1, 1, 1)


def _ddyT(S, D, facy):
    nel, n, _, V, K = S.shape
    out = torch.einsum('qj,eiqm->eijm', D, S.reshape(nel, n, n, V*K))
    return out.reshape(nel, n, n, V, K)*facy.view(-1, 1, 1, 1, 1)


# ----------------------------------------------------------------- operator

def _apply_L(Ur, D, facx, facy, kz, nu, c, wq=None, kap=0.0, rw=None):
    """diag(rw) · W · L₀ · U.  Split-real in, split-real out.

    Fields 0..6 are real parts, 7..13 imaginary; rows 0..7 real, 8..15 imaginary.
    The i·k coupling in that representation is
        i·k·(a + i·b)  ->  real  -k·b,  imag  +k·a
    """
    dx, dy = _ddx(Ur, D, facx), _ddy(Ur, D, facy)
    R = Ur.new_empty(Ur.shape[:3] + (2*NR, Ur.shape[-1]))
    k = kz.view(1, 1, 1, -1)

    def f(i):                     # (real, imag) of field i
        return Ur[..., i, :], Ur[..., NV + i, :]

    def gx(i):
        return dx[..., i, :], dx[..., NV + i, :]

    def gy(i):
        return dy[..., i, :], dy[..., NV + i, :]

    ur, ui = f(U_);  vr, vi = f(V_);  wr, wi = f(W_)
    oxr, oxi = f(OX_); oyr, oyi = f(OY_); ozr, ozi = f(OZ_)
    pr, pi_ = f(P_)
    uxr, uxi = gx(U_); uyr, uyi = gy(U_)
    vxr, vxi = gx(V_); vyr, vyi = gy(V_)
    wxr, wxi = gx(W_); wyr, wyi = gy(W_)
    oxxr, oxxi = gx(OX_); oxyr, oxyi = gy(OX_)
    oyxr, oyxi = gx(OY_); oyyr, oyyi = gy(OY_)
    ozxr, ozxi = gx(OZ_); ozyr, ozyi = gy(OZ_)
    pxr, pxi = gx(P_);   pyr, pyi = gy(P_)

    R[..., 0, :]  = kap*pr + uxr + vyr - k*wi
    R[..., 8, :]  = kap*pi_ + uxi + vyi + k*wr
    R[..., 1, :]  = wyr + k*vi - oxr
    R[..., 9, :]  = wyi - k*vr - oxi
    R[..., 2, :]  = -k*ui - wxr - oyr
    R[..., 10, :] = k*ur - wxi - oyi
    R[..., 3, :]  = vxr - uyr - ozr
    R[..., 11, :] = vxi - uyi - ozi
    R[..., 4, :]  = c*ur + pxr + nu*(ozyr + k*oyi)
    R[..., 12, :] = c*ui + pxi + nu*(ozyi - k*oyr)
    R[..., 5, :]  = c*vr + pyr + nu*(-k*oxi - ozxr)
    R[..., 13, :] = c*vi + pyi + nu*(k*oxr - ozxi)
    R[..., 6, :]  = c*wr - k*pi_ + nu*(oyxr - oxyr)
    R[..., 14, :] = c*wi + k*pr + nu*(oyxi - oxyi)
    R[..., 7, :]  = oxxr + oyyr - k*ozi
    R[..., 15, :] = oxxi + oyyi + k*ozr

    if rw is not None:
        R = R*torch.cat([rw, rw]).view(1, 1, 1, -1, 1)
    if wq is not None:
        R = R*wq[..., None, None]
    return R


def _apply_LT(Rr, D, facx, facy, kz, nu, c, kap=0.0):
    """L₀ᵀ · R.  In split-real form the transpose is the complex CONJUGATE, so
    i·k -> -i·k while real c and nu are unchanged:  -i·k·(a+i·b) -> +k·b, -k·a."""
    tx, ty = _ddxT(Rr, D, facx), _ddyT(Rr, D, facy)
    k = kz.view(1, 1, 1, -1)
    C = Rr.new_empty(Rr.shape[:3] + (2*NV, Rr.shape[-1]))

    def r(i):
        return Rr[..., i, :], Rr[..., NR + i, :]

    def Tx(i):
        return tx[..., i, :], tx[..., NR + i, :]

    def Ty(i):
        return ty[..., i, :], ty[..., NR + i, :]

    r0r, r0i = r(0); r1r, r1i = r(1); r2r, r2i = r(2); r3r, r3i = r(3)
    r4r, r4i = r(4); r5r, r5i = r(5); r6r, r6i = r(6); r7r, r7i = r(7)
    t0xr, t0xi = Tx(0); t0yr, t0yi = Ty(0)
    t1yr, t1yi = Ty(1); t2xr, t2xi = Tx(2)
    t3xr, t3xi = Tx(3); t3yr, t3yi = Ty(3)
    t4xr, t4xi = Tx(4); t4yr, t4yi = Ty(4)
    t5xr, t5xi = Tx(5); t5yr, t5yi = Ty(5)
    t6xr, t6xi = Tx(6); t6yr, t6yi = Ty(6)
    t7xr, t7xi = Tx(7); t7yr, t7yi = Ty(7)

    C[..., U_, :]       = t0xr + k*r2i - t3yr + c*r4r
    C[..., NV + U_, :]  = t0xi - k*r2r - t3yi + c*r4i
    C[..., V_, :]       = t0yr - k*r1i + t3xr + c*r5r
    C[..., NV + V_, :]  = t0yi + k*r1r + t3xi + c*r5i
    C[..., W_, :]       = k*r0i + t1yr - t2xr + c*r6r
    C[..., NV + W_, :]  = -k*r0r + t1yi - t2xi + c*r6i
    C[..., OX_, :]      = -r1r + nu*k*r5i - nu*t6yr + t7xr
    C[..., NV + OX_, :] = -r1i - nu*k*r5r - nu*t6yi + t7xi
    C[..., OY_, :]      = -r2r - nu*k*r4i + nu*t6xr + t7yr
    C[..., NV + OY_, :] = -r2i + nu*k*r4r + nu*t6xi + t7yi
    C[..., OZ_, :]      = -r3r + nu*t4yr - nu*t5xr + k*r7i
    C[..., NV + OZ_, :] = -r3i + nu*t4yi - nu*t5xi - k*r7r
    C[..., P_, :]       = t4xr + t5yr + k*r6i + kap*r0r
    C[..., NV + P_, :]  = t4xi + t5yi - k*r6r + kap*r0i
    return C


def _kz(kz, nk, dev):
    """Wavenumbers as a length-nk tensor.

    THE `np.asarray` HERE WAS A DEVICE-TO-HOST COPY.  On a torch tensor
    `np.asarray` calls `.numpy()`, which synchronises and copies -- twice per CG
    iteration, once in apply_L and once in apply_LT.  `test_device.py` measured
    406 transfers over 200 iterations before this branch existed.  Correct
    answers, and the GPU advantage silently spent on PCIe traffic: exactly the
    failure that passes a numerical unit test.
    """
    if isinstance(kz, torch.Tensor):
        k = kz.to(device=dev, dtype=torch.float64).reshape(-1)
        return k if k.numel() == nk else k.expand(nk)
    return _t(np.broadcast_to(np.asarray(kz, dtype=float), (nk,)), dev)


# torch.compile fuses the elementwise row assembly that follows the einsums --
# the same lever kernels_cuda.py exploits by hand, but automatic and one line.
#
# MEASURED ON GB10, contended: 1.39x / 1.25x / 1.29x at 48^3 / minimal channel /
# 88^3, correct to 2.2e-16.  Independently measured at 1.36x on an A100 (full-rate
# FP64) -- the result carrying across a 4.4x difference in FP64 throttling says
# the win is launch/elementwise overhead, not arithmetic.
#
# It does NOT displace kernels_cuda: the hand-fused kernel is still 4.4-4.7x
# ahead of compiled.  This makes the `torch` backend better for hosts where the
# CUDA extension cannot be built.
#
# Off via LSSEM3D_TORCH_COMPILE=0.  First call pays compilation; shapes are fixed
# through a run, so dynamic=False avoids recompiling on every distinct nk.
# FALLBACK MUST HAPPEN AT CALL TIME.  torch.compile is LAZY -- it returns a
# wrapper and compiles on first invocation -- so guarding only the wrap catches
# nothing.  Measured: on this macOS host inductor's CPU backend cannot link
# libc++.1.dylib, and 34 tests failed with InductorError from inside the call.
#
# AND YES, THIS FALLS BACK SILENTLY, which backend.py deliberately refuses to do.
# The cases differ: `set_backend('numba')` is a user's explicit request for a
# particular IMPLEMENTATION, and quietly substituting another turns a missing
# dependency into a mysterious slowdown.  torch.compile is an internal
# optimisation with identical semantics -- eager and compiled agree to 2.2e-16 --
# so falling back costs speed on a broken toolchain instead of failing outright.
_COMPILED = {}
_COMPILE_OK = True


def _maybe_compile(fn):
    global _COMPILE_OK
    if not _COMPILE_OK or os.environ.get('LSSEM3D_TORCH_COMPILE', '1') in (
            '0', 'false', 'False'):
        return fn
    c = _COMPILED.get(fn)
    if c is None:
        try:
            c = _COMPILED[fn] = torch.compile(fn, dynamic=False)
        except Exception:
            _COMPILE_OK = False
            return fn

    def guarded(*a, **k):
        global _COMPILE_OK
        if not _COMPILE_OK:
            return fn(*a, **k)
        try:
            return c(*a, **k)
        except Exception:
            _COMPILE_OK = False         # one failure disables it for the process
            return fn(*a, **k)
    return guarded


# ------------------------------------------------------------------ facades

def apply_L(Ur, D, facx, facy, kz, nu, c, wq=None, kap=0.0, rw=None):
    """Signature-compatible with operator.apply_L.

    NumPy input is converted and the result converted back, which is the PARITY
    path, not the performance path — Phase 2 keeps tensors on device end to end.
    """
    # HONOUR THE CALLER'S DEVICE.  Forcing every input to `device()` looks
    # harmless and is not: a caller holding CPU tensors gets a cuda result back,
    # which then meets a CPU mask in normal_op --
    #     RuntimeError: Expected all tensors to be on the same device
    # Worse, in a device-resident CG loop it would silently insert a transfer per
    # call, which is the 21.9x penalty V3 measured.  The backend does not choose
    # where the data lives; the caller does.
    was_np = not isinstance(Ur, torch.Tensor)
    dev = device() if was_np else Ur.device
    U = _t(Ur, dev)
    nk = U.shape[-1]
    R = _maybe_compile(_apply_L)(U, _t(D, dev), _t(facx, dev), _t(facy, dev), _kz(kz, nk, dev),
                 float(nu), float(c),
                 None if wq is None else _t(wq, dev), float(kap),
                 None if rw is None else _t(rw, dev))
    return R.cpu().numpy() if was_np else R


def apply_LT(Rr, D, facx, facy, kz, nu, c, kap=0.0):
    """Signature-compatible with operator.apply_LT."""
    was_np = not isinstance(Rr, torch.Tensor)
    dev = device() if was_np else Rr.device
    R = _t(Rr, dev)
    nk = R.shape[-1]
    C = _maybe_compile(_apply_LT)(R, _t(D, dev), _t(facx, dev), _t(facy, dev), _kz(kz, nk, dev),
                  float(nu), float(c), float(kap))
    return C.cpu().numpy() if was_np else C
