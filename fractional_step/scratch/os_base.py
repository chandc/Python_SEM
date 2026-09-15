"""Gate test for Chan (1996) Fig. 2: is the plane-Poiseuille base flow sustained?

Before any eigenfunction goes near this, the base state must hold.  With
periodicity in x the mean pressure gradient CANNOT be carried by a periodic p,
so it has to enter as a body force f_x = 2*nu -- which is exactly the
`2.0*pr*dt` term in the 1996 F77 rhs that I had written off as case-specific.

f_known is subtracted from the residual as `su_nl -= f_known*wq`, i.e. it is the
source on the RHS of the WEIGHTED equation, so the momentum row needs
f_known[...,0] = a_flux * 2*nu.

If the weighting is wrong the parabola decays or grows and every Fig. 2 number
downstream would be meaningless.  Run with no perturbation and watch max|u|.

Mesh: Chan's -- 1 element streamwise over [0, 2pi] (periodic), 3 wall-to-wall
with the WALL elements covering 30% of the channel width each (0.6 / 0.8 / 0.6
over a total width of 2).
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S

RE = 7500.0
NU = 1.0/RE
LX = 2.0*np.pi
WIDTH = 2.0                      # y from -1 to 1
WALL_FRAC = 0.30                 # Chan: wall elements cover 30% of the width
DT = 0.1


def build(N):
    """1 x 3 elements, non-uniform in y: 0.6 / 0.8 / 0.6."""
    m = build_channel(LX, WIDTH, 1, 3, N, bcs=(0, 0, 1, 1))
    hw = WALL_FRAC*WIDTH                       # 0.6
    hm = WIDTH - 2*hw                          # 0.8
    ys = [0.0, hw, hw+hm]
    hs = [hw, hm, hw]
    for e in range(m.nelem):                   # build_channel raster: ey fastest
        m.y0[e] = ys[e]
        m.hy[e] = hs[e]
    m.setup_derived()
    m.periodic_x = LX
    m.compute_global_indices()
    return m


def base_state(m, N):
    """U = 1 - y^2 on y in [-1,1], omega = -dU/dy = 2y, p = 0."""
    n = N+1
    U = np.zeros((m.nelem, n, n, 4))
    ymid = 0.5*(m.ynod.min() + m.ynod.max())
    for e in range(m.nelem):
        for i in range(n):
            for j in range(n):
                y = m.ynod[e, j] - ymid
                U[e, i, j, 0] = 1.0 - y*y
                U[e, i, j, 3] = 2.0*y
    return U


def run(N, nsteps=200):
    m = build(N)
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0)
    _, a_flux, _ = ls_coeffs(st)

    U0 = base_state(m, N)
    n = N+1
    f = np.zeros_like(U0)
    f[..., 0] = a_flux*2.0*NU                  # mean pressure gradient as a body force

    U = U0.copy(); hist = [U]
    pin = (0, n//2, n//2)
    print(f"  N = {N}:  a_flux = {a_flux:g},  f_x (weighted) = {a_flux*2*NU:.6e}")
    print(f"  {'step':>6}{'t':>8}{'max|u|':>12}{'max|u-U0|':>13}{'rms div':>11}")
    for s in range(nsteps):
        U = S.step_bdf(st, hist, time=s*DT, max_newton=2,
                       newton_tol=0.0, newton_factor=0.0, f_known=f,
                       pin_p=pin, cgsfac=0.01, cg_tol=1e-14,
                       cg_max_iter=2000, line_search=False)
        if not np.all(np.isfinite(U)):
            print("  NON-FINITE"); return
        if s % 50 == 0 or s == nsteps-1:
            D = diff_matrix(N)
            dv = (dUdx(np.ascontiguousarray(U[..., 0]), D, m.facx) +
                  dUdy(np.ascontiguousarray(U[..., 1]), D, m.facy))
            print(f"  {s+1:>6}{(s+1)*DT:>8.1f}{np.abs(U[...,0]).max():>12.8f}"
                  f"{np.abs(U[...,0]-U0[...,0]).max():>13.3e}"
                  f"{np.sqrt((dv**2).mean()):>11.2e}")
    return U, U0, m


if __name__ == '__main__':
    print(f"Plane Poiseuille base flow, Re = {RE}, dt = {DT}")
    print(f"mesh: 1 x 3 elements, y-heights 0.6 / 0.8 / 0.6, periodic in x")
    print("exact base state: max|u| = 1.0 held indefinitely\n")
    for N in (8,):
        run(N)
