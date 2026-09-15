import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

out_dir = "/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f/"

data = np.load("cavity_re1000_data.npz")
U_steady = data['U_steady']
xnod = data['xnod']
ynod = data['ynod']
u_pts = data['u_pts']
y_pts = data['y_pts']
ghia_u = data['ghia_u']
ghia_y = data['ghia_y']
Re = data['Re']

# Centerline U Profile
plt.figure(figsize=(6, 6))
plt.plot(u_pts, y_pts, '-', label="LSSEM", color='blue')
plt.plot(ghia_u, ghia_y, 'o', label=f"Ghia (Re={int(Re)})", color='red')
plt.xlabel('u-velocity')
plt.ylabel('y')
plt.title(f'Cavity Flow Centerline Velocity (Re={int(Re)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir + "cavity_u_profile.png")
plt.close()

# Centerline V Profile (y = 0.5)
# We need to gather (x, v) points at y ~ 0.5
x_pts = []
v_pts = []
nelem = xnod.shape[0]
N = xnod.shape[1] - 1
for e in range(nelem):
    for j in range(N+1):
        if abs(ynod[e, j] - 0.5) < 1e-5:
            x_pts.extend(xnod[e, :])
            v_pts.extend(U_steady[e, :, j, 1])

# Sort v_pts by x_pts
idx = np.argsort(x_pts)
x_pts = np.array(x_pts)[idx]
v_pts = np.array(v_pts)[idx]

# Ghia Re=1000 V-velocity data
ghia_v_x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
ghia_v_1000 = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.0000])

u_interp = np.interp(ghia_y, y_pts, u_pts)
err_u = np.max(np.abs(u_interp - ghia_u))
print(f"Max error against Ghia U-velocity (Re={int(Re)}): {err_u:.4f}")

# Remember that v_pts is sorted by x_pts, but x_pts might have duplicates at the exact same location.
# Let's remove duplicates for interpolation
_, unique_indices = np.unique(x_pts, return_index=True)
x_unique = x_pts[unique_indices]
v_unique = v_pts[unique_indices]
# np.interp requires strictly increasing x-coordinates.
idx_sort = np.argsort(x_unique)
v_interp = np.interp(ghia_v_x[::-1], x_unique[idx_sort], v_unique[idx_sort])[::-1]

err_v = np.max(np.abs(v_interp - ghia_v_1000))
print(f"Max error against Ghia V-velocity (Re={int(Re)}): {err_v:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(x_pts, v_pts, '-', label="LSSEM", color='blue')
plt.plot(ghia_v_x, ghia_v_1000, 'o', label=f"Ghia (Re={int(Re)})", color='red')
plt.xlabel('x')
plt.ylabel('v-velocity')
plt.title(f'Cavity Flow Centerline V-Velocity (Re={int(Re)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_dir + "cavity_v_profile.png")
plt.close()

# Streamlines
plt.figure(figsize=(7, 6))

nelem = xnod.shape[0]
N = xnod.shape[1] - 1

# xnod and ynod are shape (nelem, N+1) in 1D? NO. They are just the element node coords.
# Let's check shape.
print("xnod shape:", xnod.shape)
if len(xnod.shape) == 2:
    # 1D coordinates per element per dimension (tensor product).
    # We need to construct the 2D grid per element.
    X_full = np.zeros((nelem, N+1, N+1))
    Y_full = np.zeros((nelem, N+1, N+1))
    for e in range(nelem):
        X_full[e, :, :] = xnod[e, :][:, None]
        Y_full[e, :, :] = ynod[e, :][None, :]
    X_flat = X_full.flatten()
    Y_flat = Y_full.flatten()
else:
    # It's already (nelem, N+1, N+1)
    X_flat = xnod.flatten()
    Y_flat = ynod.flatten()

U_flat = U_steady[..., 0].flatten()
V_flat = U_steady[..., 1].flatten()
P_flat = U_steady[..., 3].flatten()

xi = np.linspace(0, 1, 150)
yi = np.linspace(0, 1, 150)
XI, YI = np.meshgrid(xi, yi)

UI = griddata((X_flat, Y_flat), U_flat, (XI, YI), method='cubic')
VI = griddata((X_flat, Y_flat), V_flat, (XI, YI), method='cubic')
PI = griddata((X_flat, Y_flat), P_flat, (XI, YI), method='cubic')

plt.contourf(XI, YI, PI, levels=50, cmap='viridis', alpha=0.5)
plt.colorbar(label='Pressure')
plt.streamplot(xi, yi, UI, VI, color='white', linewidth=1.0, density=1.5)

plt.title(f'Cavity Flow Streamlines (Re={int(Re)})')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig(out_dir + "cavity_streamlines.png")
plt.close()

print("Plots saved.")
