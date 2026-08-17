"""A priori: does w_mom = w_mass = 1 actually fix the pressure under-weighting?

POISEUILLE_DT_STUDY.md sec 3 gives the diagonal of L^T L as

    pressure diagonal ~ a_flux^2 * sum(P_x^2 + P_y^2)
    velocity diagonal ~ sum(a_mass^2 P^2 + a_flux^2 (P_x^2 + P_y^2))

and identifies ratio = 1 as the equal-weight point -- computable from
compute_jacobi WITHOUT running the solver.  For legacy (a_mass = fac1,
a_flux = dt) the ratio collapses to dt^2 once resolved, which is the whole
mechanism behind the 98% error at dt = 0.05.

Pinning a_flux = 1 removes dt from the NUMERATOR.  But w_mom = w_mass = 1 sets
a_mass = fac1/dt, so dt reappears in the DENOMINATOR and the ratio should still
degrade like dt^2/fac1^2.  If so, small dt at fixed weight 1 is NOT a benign
limit -- it is the same imbalance approached from the other side, and the
predicted failures at dt <= 0.5 are structural rather than incidental.

Measured as mean(diag) over pressure nodes / mean(diag) over u,v nodes, taking
the diagonal as 1/M_inv from compute_jacobi on the linearisation about the
exact Poiseuille field.
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.solver import compute_jacobi

LX, LY, NU = 10.0, 1.0, 0.01
N, EX, EY = 8, 10, 2
DTS = [5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]


def ratio(dt, w):
    m = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    pin = next((e, 0, 0) for e in range(m.nelem)
               if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.5,   # BDF2 value
                     w_mom=w, w_mass=w)
    n = N+1
    y = m.ynod[:, None, :]
    fu = np.broadcast_to(6.0*y*(1.0-y), (m.nelem, n, n)).copy()   # exact profile
    fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)              # step_bdf does this before jacobi
    M_inv = compute_jacobi(st, fu, fv, pin_p=pin)
    d = np.zeros_like(M_inv)
    ok = M_inv > 0
    d[ok] = 1.0/M_inv[ok]
    dp = d[..., 2][ok[..., 2]].mean()                # pressure block
    du = np.concatenate([d[..., 0][ok[..., 0]], d[..., 1][ok[..., 1]]]).mean()
    am, af, _ = ls_coeffs(st)
    return am, af, dp/du


print("Poiseuille control mesh, order 8, 10x2, linearised about the exact profile")
print("p-block / u-block diagonal of L^T L.  1.0 = equal weight = the optimum.\n")
print(f"{'dt':>7} | {'LEGACY  (a_flux = dt)':^34} | {'w_mom = w_mass = 1':^34}")
print(f"{'':>7} | {'a_mass':>9}{'a_flux':>9}{'p/u ratio':>16} |"
      f" {'a_mass':>9}{'a_flux':>9}{'p/u ratio':>16}")
print('-'*80)
for dt in DTS:
    am0, af0, r0 = ratio(dt, None)
    am1, af1, r1 = ratio(dt, 1.0)
    print(f"{dt:>7g} | {am0:>9.3f}{af0:>9.3f}{r0:>16.3e} |"
          f" {am1:>9.3f}{af1:>9.3f}{r1:>16.3e}")
print("\nIf the right-hand ratio also falls as dt^2, pinning the weight does not")
print("rescue small dt -- the imbalance just moves from a_flux to a_mass.")
