import numpy as np
import sys

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.bc import apply_mask
from lssem2d.solver import apply_A

N = 2
mesh = build_channel(1., 1., 2, 2, N)
mesh.y0 -= 1.
mesh.ynod -= 1.

D = diff_matrix(N)
state = SolverState(mesh, D, nu=0.1, dt=1.0, fac1=1.0)
fu = np.random.randn(mesh.nelem, N+1, N+1)
fv = np.random.randn(mesh.nelem, N+1, N+1)
state.update_linearisation(fu, fv)

U = np.random.randn(mesh.nelem, N+1, N+1, 4)
V = np.random.randn(mesh.nelem, N+1, N+1, 4)

U = apply_mask(mesh, U)
V = apply_mask(mesh, V)

AU = apply_A(state, U, fu, fv)
AV = apply_A(state, V, fu, fv)

dot1 = np.sum(AU * V)
dot2 = np.sum(U * AV)

print("dot1:", dot1)
print("dot2:", dot2)
print("diff:", abs(dot1 - dot2))
