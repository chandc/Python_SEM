import numpy as np
import matplotlib.pyplot as plt
from lssem2d.mesh import build_bfs
from lssem2d.solver import step_bdf
from lssem2d.lssem import SolverState
from lssem2d.config import Config
import time

# High-resolution mesh
N = 6
mesh = build_bfs(N, E_in_x=4, E_out_x=40, E_y=2)
print(f"Mesh has {mesh.nelem} elements.")

cfg = Config("bfs.toml")
cfg.max_steps = 200
cfg.cgsfac = 1e-3
cfg.dt = 0.1
cfg.newton_tol = 1e-4

Re = cfg.get("simulation", "Re", 389.0)
nu = 1.0 / Re

from lssem2d.lgl import diff_matrix
D = diff_matrix(N)

# Create state
state = SolverState(mesh, D, nu=nu, dt=cfg.dt)

def custom_inlet(x, y, t):
    eta = (y - 0.5) / 0.5
    return 6.0 * eta * (1.0 - eta)

print("Starting high-res solve from scratch (no restart)...")
start = time.time()
U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
U_history = [U_0]

for step in range(1, cfg.max_steps + 1):
    current_time = step * cfg.dt
    U_new = step_bdf(
        state, U_history, time=current_time,
        max_newton=5, newton_tol=cfg.newton_tol,
        newton_factor=0.1, custom_inlet=custom_inlet,
        cgsfac=cfg.cgsfac, verbose=False
    )
    diff = np.max(np.abs(U_history[0] - U_history[1])) if len(U_history) > 1 else 0.0

    if step % 10 == 0:
        print(f"Step {step}, Change: {diff:.2e}")

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
plt.title(f'High-Res BFS Streamlines (Step {cfg.max_steps})')
plt.savefig('/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f/bfs_highres_streamlines.png', dpi=300, bbox_inches='tight')
print("Plot saved.")
