"""Array-namespace dispatch, so the CG loop can run entirely on a GPU.

WHY THIS EXISTS.  TORCH_VERIFY_PLAN.md V3 measured the operator at 3.3x on the
GB10 -- but only when tensors STAY on the device.  Timed through the NumPy
facade, which copies host->device->host per call, the same operator costs
386.0 ms at 88^3 against 17.6 ms device-resident: a **21.9x penalty that erases
the GPU entirely**.  At ~4800 CG iterations per stage, one round trip per
iteration is tens of GB of PCIe traffic per step.

So `pcg` and everything it calls has to be agnostic about whether it holds NumPy
arrays or torch tensors.  This module is that seam, and it is deliberately thin:
the alternative -- a second copy of `pcg` -- would let the two drift, and the
reference implementation is the thing every physics result in 3D_STATUS.md was
measured with.

GATHER-SCATTER IS THE ONE GENUINELY NEW PIECE.  `lssem2d.assembly.gather_scatter`
does `QT @ (Q @ x)` with scipy sparse.  Q is a pure GATHER matrix -- every local
dof maps to exactly one global dof -- so `Q^T Q` is a segmented sum followed by a
broadcast, which is `index_add_` plus a fancy-index and needs no sparse matmul at
all.

  NON-DETERMINISM.  `index_add_` on CUDA accumulates with atomics, so the
  summation order varies between runs and results are not bit-reproducible.
  This project validates by bit-level parity, so `deterministic(True)` is
  available and the parity tests use it.

  The equivalence against scipy is asserted in the tests on a PERIODIC mesh,
  where nodes are shared 2 and 4 ways -- the configuration that caught the
  one-copy pressure-pin bug (3D_STATUS.md sec 2).  A single-element mesh would
  pass while proving nothing.
"""
import weakref

import numpy as np

try:
    import torch
except Exception:                       # torch is optional; NumPy path unaffected
    torch = None

try:
    import cupy                         # optional, exactly like torch
except Exception:
    cupy = None


def is_tensor(a):
    return torch is not None and isinstance(a, torch.Tensor)


def is_cupy(a):
    """CuPy arrays are a SECOND device namespace, independent of torch.

    They are NumPy-compatible enough that every *computational* helper below
    already works on them through NEP-18: `np.einsum` on a CuPy array
    dispatches to `cupy.einsum` with no code change.  What does NOT dispatch is
    array CREATION -- `np.zeros` makes a host array whatever `like` is -- so
    the creation helpers ask `xp(like)` instead of hard-coding `np`.  That one
    change is essentially the whole CuPy seam; see CUPY_BACKEND.md.
    """
    return cupy is not None and isinstance(a, cupy.ndarray)


def xp(a):
    """The array namespace `a` belongs to."""
    if is_tensor(a):
        return torch
    if is_cupy(a):
        return cupy
    return np


def deterministic(on=True):
    """Force reproducible reductions/scatters.  Needed for parity testing; costs
    performance, so production may prefer it off -- but decide that explicitly."""
    if torch is not None:
        torch.use_deterministic_algorithms(on, warn_only=True)


# --------------------------------------------------------------- reductions

def sum_over(a, axes):
    """Sum over `axes`, keeping the remaining shape.  numpy takes `axis`, torch
    takes `dim` -- the one API difference that actually bites here."""
    if is_tensor(a):
        return torch.sum(a, dim=tuple(axes))
    return np.sum(a, axis=tuple(axes))


def sqrt(a):
    return torch.sqrt(a) if is_tensor(a) else np.sqrt(a)


def zeros_like(a):
    return torch.zeros_like(a) if is_tensor(a) else np.zeros_like(a)


def clone(a):
    """torch has no `.copy()`; numpy has no `.clone()`."""
    return a.clone() if is_tensor(a) else a.copy()


def where(cond, a, b):
    return torch.where(cond, a, b) if is_tensor(cond) else np.where(cond, a, b)


def all_(a):
    return bool(torch.all(a)) if is_tensor(a) else bool(np.all(a))


def maximum(a, b):
    if is_tensor(a):
        return torch.clamp(a, min=b) if np.isscalar(b) else torch.maximum(a, b)
    return np.maximum(a, b)


def abs_max(a):
    return float(torch.abs(a).max()) if is_tensor(a) else float(np.abs(a).max())


# ------------------------------------------------------------------- fft
#
# The convection term is the only place complex arrays survive to the public
# API, and torch supports complex128, so the 3/2-rule dealiasing ports directly.
# numpy takes `axis=`, torch takes `dim=` -- the same asymmetry as the reductions.

def rfft(a):
    return torch.fft.rfft(a, dim=-1) if is_tensor(a) else np.fft.rfft(a, axis=-1)


def irfft(a, n):
    if is_tensor(a):
        return torch.fft.irfft(a, n=n, dim=-1)
    return np.fft.irfft(a, n=n, axis=-1)


def zeros_complex(shape, like):
    """Complex zeros matching `like`'s namespace and device."""
    if is_tensor(like):
        return torch.zeros(tuple(shape), dtype=torch.complex128,
                           device=like.device)
    return xp(like).zeros(tuple(shape), dtype=complex)


def empty_complex(shape, like):
    if is_tensor(like):
        return torch.empty(tuple(shape), dtype=torch.complex128,
                           device=like.device)
    return xp(like).empty(tuple(shape), dtype=complex)


def einsum(sub, *ops):
    """np.einsum / torch.einsum share this signature for our subscripts.

    BUT NOT THEIR TYPE PROMOTION.  numpy promotes a real operand against a
    complex one silently; torch refuses:

        RuntimeError: expected m1 and m2 to have the same dtype,
                      but got: double != c10::complex<double>

    which bites immediately in `convect`, where the real differentiation matrix
    D multiplies a complex mode array.  Promote explicitly, and lift any stray
    NumPy operand onto the same device so a mixed call cannot silently fall back
    to the host.
    """
    if not any(is_tensor(o) for o in ops):
        return np.einsum(sub, *ops)
    ref = next(o for o in ops if is_tensor(o))
    ops = tuple(o if is_tensor(o)
                else torch.as_tensor(np.ascontiguousarray(o), device=ref.device)
                for o in ops)
    if any(o.is_complex() for o in ops):
        ops = tuple(o.to(torch.complex128) for o in ops)
    return torch.einsum(sub, *ops)


# ---------------------------------------------------------- gather-scatter

# WeakKeyDictionary, NOT a dict keyed on id(mesh).  CPython reuses ids after
# garbage collection, so an id-keyed cache can hand a NEW mesh the index map of a
# DEAD one -- silently wrong gather-scatter whenever the shapes happen to be
# compatible, an exception when they are not.  A weak key is released with the
# mesh, so reuse cannot happen.
_IDX = weakref.WeakKeyDictionary()


def _index(mesh, dev):
    """Cache the flattened global-index map per (mesh, device).

    Built from `mesh.gidx`, which is (nelem, n, n) of global ids -- the same
    map `mesh.Q` encodes, so C-order flattening of (e, i, j) lines up with
    `gather_scatter`'s `U.reshape(-1, k)`.  Asserted in the tests rather than
    assumed.
    """
    per_dev = _IDX.setdefault(mesh, {})
    hit = per_dev.get(str(dev))
    if hit is None:
        flat = np.ascontiguousarray(mesh.gidx.reshape(-1)).astype(np.int64)
        hit = (torch.as_tensor(flat, device=dev), int(flat.max()) + 1)
        per_dev[str(dev)] = hit
    return hit


def gs_torch(mesh, U):
    """Q^T Q on a torch tensor: segmented sum, then broadcast back.

    U is (nelem, n, n, nvar, nmode); the (var, mode) axes ride along as one
    trailing batch, exactly as the NumPy path folds them for lssem2d.
    """
    idx, ng = _index(mesh, U.device)
    nel, n, _, nv, nk = U.shape
    flat = U.reshape(nel*n*n, nv*nk)
    g = torch.zeros(ng, nv*nk, dtype=U.dtype, device=U.device)
    g.index_add_(0, idx, flat)
    return g[idx].reshape(U.shape)


def gs_cupy(mesh, U):
    """Q^T Q on a CuPy array: segmented sum, then broadcast back.

    Same algebra as `gs_torch` -- Q is a pure gather, so Q^T Q is a scatter-add
    then a fancy-index -- with `cupyx.scatter_add` in place of `index_add_`.
    Like the torch path this uses atomics, so summation order is not
    reproducible run to run; parity is checked to a tolerance, not bitwise.
    """
    import cupyx
    idx = _index_cupy(mesh)
    nel, n, _, nv, nk = U.shape
    flat = U.reshape(nel*n*n, nv*nk)
    g = cupy.zeros((int(idx.max()) + 1, nv*nk), dtype=U.dtype)
    cupyx.scatter_add(g, idx, flat)
    return g[idx].reshape(U.shape)


_IDX_CP = weakref.WeakKeyDictionary()


def _index_cupy(mesh):
    hit = _IDX_CP.get(mesh)
    if hit is None:
        hit = cupy.asarray(
            np.ascontiguousarray(mesh.gidx.reshape(-1)).astype(np.int64))
        _IDX_CP[mesh] = hit
    return hit


def to_device(a, like):
    """Move `a` to the device/dtype of `like`, leaving NumPy alone if `like` is."""
    if is_cupy(like):
        return a if is_cupy(a) else cupy.asarray(a)
    if not is_tensor(like):
        return a
    if is_tensor(a):
        return a.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64),
                           device=like.device).to(dtype=like.dtype)
