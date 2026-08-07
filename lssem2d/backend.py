"""Backend selection for the VVP operator kernels.

Two implementations of `apply_L` / `apply_LT` exist:

  'numpy'  -- the reference implementation in lssem.py (default)
  'numba'  -- fused @njit kernels in kernels_numba.py

NumPy is the default, so nothing changes unless the numba backend is asked for
explicitly.  Two ways to ask:

    LSSEM_BACKEND=numba python your_script.py      # process-wide, read at import

    import lssem2d
    lssem2d.set_backend('numba')                   # at runtime, any time

An explicit request that cannot be honoured raises rather than quietly falling
back to NumPy: a silent fallback would turn a missing dependency into a
mysterious 3x slowdown, which is exactly the kind of thing that corrupts a
benchmark.  Use `available()` if you want to probe without committing.
"""
import os

VALID = ('numpy', 'numba')

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
    return False


def _resolve(name):
    name = (name or 'numpy').strip().lower()
    if name not in VALID:
        raise ValueError(f"unknown backend {name!r}; expected one of {VALID}")
    if name == 'numba' and not available('numba'):
        raise ImportError(
            "backend 'numba' was requested but numba is not importable in this "
            "interpreter.  Install it (uv pip install numba) or use the default "
            "'numpy' backend.  lssem2d.backend.available('numba') probes without "
            "raising."
        )
    return name


def get_backend():
    """Name of the active backend, resolving LSSEM_BACKEND on first use."""
    global _backend
    if _backend is None:
        set_backend(os.environ.get('LSSEM_BACKEND', 'numpy'))
    return _backend


def set_backend(name):
    """Switch backend and return its name.  Safe to call repeatedly."""
    global _backend
    resolved = _resolve(name)
    _backend = resolved
    for fn in _listeners:
        fn(resolved)
    return resolved


def register(fn):
    """Register a callback invoked with the backend name on every switch.

    Called immediately with the current backend so registration also performs
    the initial binding.
    """
    _listeners.append(fn)
    fn(get_backend())
    return fn
