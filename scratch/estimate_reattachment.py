import sys
import numpy as np
import os
from scipy.interpolate import griddata

if len(sys.argv) > 1:
    latest_restart = sys.argv[1]
else:
    print("Provide restart file")
    sys.exit(1)

data = np.load(latest_restart)
U_final = data['arr_0'][0]  # U_history[0]

# get mesh info
import lssem2d.mesh as mesh
from lssem2d.lssem import LSSEM_State
m = mesh.build_bfs(6)

for e in range(m.nelem):
    # look at bottom wall elements
    if m.y0[e] == 0.0:
        u_wall = U_final[e, :, 0, 0] # j=0 is bottom wall
        x_wall = m.xnod[e, :]
        for i in range(len(x_wall)):
            if u_wall[i] < -1e-4:
                print(f"Backflow at x={x_wall[i]:.4f}, u={u_wall[i]:.4e}")

