"""LSSEM 2D velocity-vorticity-pressure least-squares spectral element solver.

Kept deliberately thin: importing submodules directly (``from lssem2d.mesh
import build_channel``) works exactly as before.  The only things re-exported
here are the backend controls, so the runtime switch documented in
``lssem2d/backend.py`` is reachable as ``lssem2d.set_backend('numba')``.
"""
from .backend import available, get_backend, set_backend  # noqa: F401

__all__ = ['available', 'get_backend', 'set_backend']
