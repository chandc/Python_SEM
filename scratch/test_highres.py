import numpy as np
import matplotlib.pyplot as plt
from lssem2d.mesh import build_bfs
from lssem2d.solver import pseudo_transient_solve
from lssem2d.lssem import LSSEM_State
from lssem2d.config import Config
import time

# Increase resolution
N = 6
mesh = build_bfs(N, E_in_x=4, E_out_x=40, E_y=2)
print(f"Mesh has {mesh.nelem} elements.")

# Create state
state = LSSEM_State(mesh, N)
def custom_inlet(x, y, t):
    eta = (y - 0.5) / 0.5
    return 6.0 * eta * (1.0 - eta)

state.custom_inlet = custom_inlet

cfg = Config("bfs.toml")
cfg.max_steps = 300
cfg.cgsfac = 1e-3
cfg.dt = 0.1

print("Starting high-res solve...")
start = time.time()
U_history, _, _ = pseudo_transient_solve(mesh, N, cfg, state)
print(f"Solve took {time.time() - start:.2f}s")
U_final = U_history[0]

# Plot streamlines
x_pts, y_pts, u_pts, v_pts = [], [], [], []
for e in range(mesh.nelem):
    for i in range(N+1):
        for j in range(N+1):
            x_pts.append(mesh.xnod[e, i])
            y_pts.append(mesh.ynod[e, j])
            u_pts.append(U_final[e, i, j, 0])
            v_pts.append(U_final[e, i, j, 1])
            
x_flat, y_flat = np.array(x_pts), np.array(y_pts)
u_flat, v_flat = np.array(u_pts), np.array(v_pts)

from scipy.interpolate import griddata
xi = np.linspace(np.min(x_flat), np.max(x_flat), 300)
yi = np.linspace(np.min(y_flat), np.max(y_flat), 100)
X, Y = np.meshgrid(xi, yi)
U_interp = griddata((x_flat, y_flat), u_flat, (X, Y), method='linear', fill_value=0.0)
V_interp = griddata((x_flat, y_flat), v_flat, (X, Y), method='linear', fill_value=0.0)

fig, ax = plt.subplots(figsize=(10, 4))
ax.streamplot(X, Y, U_interp, V_interp, color='k', linewidth=0.5, density=4.0, arrowstyle='-')
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.0)
plt.fill_between([-10, 0], [0, 0], [0.5, 0.5], color='white', zorder=10)
plt.title(f'High-Res BFS Streamlines (Step 300)')
plt.savefig('/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f/bfs_highres_streamlines.png', dpi=300, bbox_inches='tight')
print("Plot saved.")
