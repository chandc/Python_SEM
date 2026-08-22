"""How far do the two backends drift apart over MANY steps?

The 2D module's NUMBA_BACKEND.md carries a measured warning: per-operator parity
to 1e-16 does NOT imply agreement on ACCUMULATED states.  On 2D Poiseuille the
two backends settled on different fixed points, 4.65e-06 vs 8.47e-06 profile
error -- a 1.8x discrepancy at the 1e-06 level.  That caveat has to be tested in
3D rather than inherited or dismissed.

This integrates the Stokes eigenmode step by step on both backends from the SAME
initial state and reports the relative state difference as it accumulates.  The
drift is a lower bound on how finely either backend can be trusted to agree.
"""
import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC); os.chdir(ROOT)
import numpy as np
import stokes3d as SD
from lssem3d import backend


def trajectory(name, s, U0, dt, nstep, tol):
    backend.set_backend(name)
    Minv = SD.make_precond(s, dt, 0.0, rowweight=True)
    U = U0.copy()
    snaps = []
    for i in range(nstep):
        U, _ = SD.step(s, U, dt, 0.0, Minv=Minv, rowweight=True,
                       tol=tol, max_iter=60000)
        snaps.append(U.copy())
    return snaps


def main(nstep=200, dt=0.0025, tol=1e-12):
    s = SD.setup(N=8)
    U0, _ = SD.initial_state(s, mode='span')
    a = trajectory('numpy', s, U0, dt, nstep, tol)
    b = trajectory('numba', s, U0, dt, nstep, tol)
    backend.set_backend('numpy')
    print(f'Stokes span mode, N=8, dt={dt}, cg tol={tol}')
    print(f'{"step":>6} {"|numpy|":>11} {"rel drift":>11}')
    for i in (0, 4, 9, 24, 49, 99, nstep-1):
        if i >= len(a):
            continue
        sc = np.abs(a[i]).max()
        print(f'{i+1:>6} {sc:11.4e} {np.abs(a[i]-b[i]).max()/sc:11.3e}')


if __name__ == '__main__':
    main()
