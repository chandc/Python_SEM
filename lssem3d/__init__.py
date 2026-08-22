"""3D LSSEM: 2D spectral elements in (x,y), Fourier in a periodic z.

NEW CODE ONLY.  Nothing here modifies lssem2d -- that is deliberate and load
bearing, because 3D_DEVELOPMENT_PLAN.md Stage 1 validates this module by
requiring it to reproduce lssem2d's measured results EXACTLY at k_z = 0.  A
shared-code refactor would make that comparison vacuous.

Time integration is RKW3/Crank-Nicolson (Spalart, Moser & Rogers 1991):
3-stage low-storage Runge-Kutta on the convective term, Crank-Nicolson on the
viscous/linear term.  See lssem3d.timestep.

The operator kernels have two interchangeable backends, 'numpy' (default) and
'numba'.  Select one with LSSEM3D_BACKEND in the environment or
`lssem3d.set_backend('numba')` at runtime; see lssem3d.backend.
"""
from . import fourier, timestep          # noqa: F401
from .backend import available, get_backend, set_backend  # noqa: F401

__all__ = ['fourier', 'timestep', 'available', 'get_backend', 'set_backend']
