import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L

N = 10
nu = 0.5
mesh = build_channel(L_x=2.0, L_y=2.0, E_x=2, E_y=2, N=N)
D = diff_matrix(N)
state = SolverState(mesh, D, nu=nu, dt=1.0, fac1=0.0)

X = mesh.xnod[:, :, None]
Y = mesh.ynod[:, None, :]

target_shape = (mesh.nelem, N + 1, N + 1)
U = np.zeros((mesh.nelem, N + 1, N + 1, 4))
U[..., 0] = np.broadcast_to(np.sin(X) * np.cos(Y), target_shape)
U[..., 1] = np.broadcast_to(-np.cos(X) * np.sin(Y), target_shape)
U[..., 2] = np.broadcast_to(np.cos(X) * np.cos(Y), target_shape)
U[..., 3] = np.broadcast_to(2.0 * np.sin(X) * np.sin(Y), target_shape)

fu = np.zeros(target_shape)
fv = np.zeros(target_shape)
state.update_linearisation(fu, fv)

su_num = apply_L(state, U, fu, fv)
res = su_num / mesh.wq[..., None]

print("Max error r1:", np.max(np.abs(res[..., 0])))
print("Max error r2:", np.max(np.abs(res[..., 1])))
print("Max error r3:", np.max(np.abs(res[..., 2])))
print("Max error r4:", np.max(np.abs(res[..., 3])))
