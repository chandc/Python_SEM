import sys
import numpy as np
import os
from lssem2d.io import load_restart
import lssem2d.mesh as mesh
from lssem2d.lgl import diff_matrix

latest_restart = sys.argv[1]
U_history, current_time, start_step = load_restart(latest_restart)
U_final = U_history[0]

N = 6
m = mesh.build_bfs(N)
D = diff_matrix(N)

for e in range(22, 42):
    for i in range(N+1):
        x = m.xnod[e, i]
        du_deta = np.dot(D[0, :], U_final[e, i, :, 0])
        du_dy = du_deta * 4.0
        if du_dy < -1e-4:
            print(f"Negative shear at x={x:.4f}, du/dy={du_dy:.4e}")

