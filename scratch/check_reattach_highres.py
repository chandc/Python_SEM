import numpy as np
from lssem2d.mesh import build_bfs
from lssem2d.lgl import diff_matrix

data = np.load('bfs_highres_010000.npz')
U_final = data['U_0']

N = 6
m = build_bfs(N, E_in_x=4, E_out_x=40, E_y=2)
D = diff_matrix(N)

print("Checking reattachment...")
wall_x = []
wall_tau = []
for e in range(m.nelem):
    if abs(m.y0[e]) < 1e-8 and m.x0[e] >= 0.0:
        for i in range(N+1):
            x = m.xnod[e, i]
            du_deta = np.dot(D[0, :], U_final[e, i, :, 0])
            du_dy = du_deta * (2.0 / m.hy[e])
            wall_x.append(x)
            wall_tau.append(du_dy)
            
idx = np.argsort(wall_x)
wall_x = np.array(wall_x)[idx]
wall_tau = np.array(wall_tau)[idx]

for x, tau in zip(wall_x, wall_tau):
    if x > 0 and x < 15:
        print(f"x={x:.4f}, tau={tau:.4e}")
