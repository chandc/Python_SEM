"""Backend selection for the 3D VVP operator kernels.

Mirrors `lssem2d/backend.py` deliberately -- same names, same semantics, same
environment variable -- so that switching backends is one habit, not two.

  'numpy'  -- the reference implementation in operator.py (default)
  'numba'  -- fused @njit kernels in kernels_numba.py  (CPU)
  'torch'  -- PyTorch kernels in kernels_torch.py      (CUDA, unfused)
  'cuda'   -- FUSED CUDA kernels in kernels_cuda.py    (one launch, 5-7x 'torch')

Two ways to ask:

    LSSEM3D_BACKEND=numba python your_script.py    # process-wide, read at import

    import lssem3d
    lssem3d.set_backend('numba')                   # at runtime, any time

AN EXPLICIT REQUEST THAT CANNOT BE HONOURED RAISES rather than quietly falling
back to NumPy.  lssem2d states the reason and it applies unchanged here: a
silent fallback turns a missing dependency into a mysterious slowdown, which is
exactly the kind of thing that corrupts a benchmark.  Use `available()` to probe
without committing.

WHY FUSION, NOT JUST COMPILATION.  `scratch/prof3d.py` measured `normal_op` at
99.4% of a step, and `prof3d_procs.py` showed threads tie processes -- so the
matvec is MEMORY-BANDWIDTH bound, not compute bound.  Compiling the existing
NumPy expression tree in place would not help: it is already calling BLAS.  The
win has to come from making FEWER PASSES over the data, which is what the fused
kernels do -- one pass replacing ~30 (to_complex, 14 einsums, 8 row assemblies,
wq, rw, to_real).
"""
import os

VALID = ('numpy', 'numba', 'torch', 'cupy', 'cuda')

_backend = None
_listeners = []


def available(name='numba'):
    """True if `name` can actually be selected in this interpreter."""
    if name == 'numpy':
        return True
    if name == 'numba':
        try:
            import numba  # noqa: F401
            return True
        except Exception:
            return False
    if name == 'cupy':
        try:
            import cupy  # noqa: F401
            return cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    if name == 'torch':
        try:
            import torch  # noqa: F401
            return True
        except Exception:
            return False
    if name == 'cuda':
        try:
            from . import kernels_cuda as KC
            return KC.available()
        except Exception:
            return False
    return False


def _resolve(name):
    name = (name or 'numpy').strip().lower()
    if name not in VALID:
        raise ValueError(f"unknown backend {name!r}; expected one of {VALID}")
    if name != 'numpy' and not available(name):
        raise ImportError(
            f"backend {name!r} was requested but its library is not importable "
            f"in this interpreter.  Install it (uv pip install {name}) or use "
            f"the default 'numpy' backend.  lssem3d.backend.available({name!r}) "
            f"probes without raising."
        )
    return name


def get_backend():
    """Name of the active backend, resolving LSSEM3D_BACKEND on first use."""
    global _backend
    if _backend is None:
        set_backend(os.environ.get('LSSEM3D_BACKEND', 'numpy'))
    return _backend


def set_backend(name):
    """Select a backend and notify everything that binds to it."""
    global _backend
    resolved = _resolve(name)
    _backend = resolved
    for fn in _listeners:
        fn(resolved)
    return resolved


def register(fn):
    """Call `fn(backend_name)` on every switch, and once now."""
    _listeners.append(fn)
    fn(get_backend())
    return fn
