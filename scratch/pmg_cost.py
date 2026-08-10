"""CG cost per solve, coarse-operator weights matched vs not.

The p-MG coarse SolverState was built without w_mom/w_mass/dtau, so ls_coeffs
took its legacy branch on the coarse grid.  This measures what that cost.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from lssem2d import precond as P

RE, DT = 389.0, 0.1
m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
for e in range(m.nelem):
    if m.bc[e, 1] == 4:
        m.bc[e, 1] = 0
N = m.N
_p = S.pcg_solve

print(f"{'config':<34}{'CG (3 steps)':>14}{'wall':>9}")
for tag, kw in (('legacy (both weights None)', {}),
                ('w_mom = w_mass = 1, dt = 0.1', dict(w_mom=1.0, w_mass=1.0)),
                ('...  + dtau = 0.3', dict(w_mom=1.0, w_mass=1.0, dtau=0.3))):
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0, **kw)
    nit = [0]

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
            cgsfac=None, precond=None, **kwargs):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, N//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=1e-10, cgsfac=1e-3, precond=pre)
        nit[0] += it
        return x, it
    S.pcg_solve = pcg
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    U = np.zeros((m.nelem, N+1, N+1, 4))
    for e in range(m.nelem):
        for i in range(N+1):
            for j in range(N+1):
                yy = m.ynod[e, j]
                U[e, i, j, 0] = 3.0*yy*(1.0-yy)
    hist = [U]
    t0 = time.perf_counter()
    try:
        for s in range(3):
            U = S.step_bdf(st, hist, time=s*DT, max_newton=2, newton_tol=0.0,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=None,
                           cgsfac=1e-3, cg_max_iter=300000, line_search=True)
    finally:
        S.pcg_solve = _p
    print(f"{tag:<34}{nit[0]:>14}{time.perf_counter()-t0:>8.1f}s")
