import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT

N = 2
mesh = build_channel(L_x=1.0, L_y=1.0, E_x=1, E_y=1, N=N)
D = diff_matrix(N)
state = SolverState(mesh, D, nu=1.0, dt=1.0, fac1=1.0)
target_shape = (mesh.nelem, N + 1, N + 1)
fu = np.random.randn(*target_shape)
fv = np.random.randn(*target_shape)
state.update_linearisation(fu, fv)

dofs = mesh.nelem * (N + 1) * (N + 1) * 4
shape = (mesh.nelem, N + 1, N + 1, 4)

L_matrix = np.zeros((dofs, dofs))
for j in range(dofs):
    U = np.zeros(dofs)
    U[j] = 1.0
    LU = apply_L(state, U.reshape(shape), fu, fv)
    L_matrix[:, j] = LU.flatten()

LT_matrix = L_matrix.T
diffs = []
for i in range(dofs):
    S = np.zeros(dofs)
    S[i] = 1.0
    S_reshaped = S.reshape(shape)
    
    # Weight S with wq before applying LT!
    wq = mesh.wq
    S_weighted = S_reshaped.copy()
    for k in range(4):
        S_weighted[..., k] *= wq
        
    LTS_num = apply_LT(state, S_weighted, fu, fv).flatten()
    LTS_exact = LT_matrix[:, i]
    diffs.append(np.max(np.abs(LTS_num - LTS_exact)))

print("Max diff:", max(diffs))
