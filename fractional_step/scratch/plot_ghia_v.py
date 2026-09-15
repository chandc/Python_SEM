import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load simulation data
data = np.load('cavity_re1000_data.npz')
U_steady = data['U_steady']
xnod = data['xnod']
ynod = data['ynod']

# Construct full 2D coordinate arrays
nelem, nx = xnod.shape
ny = ynod.shape[1]
X = np.zeros((nelem, nx, ny))
Y = np.zeros((nelem, nx, ny))

for e in range(nelem):
    for i in range(nx):
        for j in range(ny):
            X[e, i, j] = xnod[e, i]
            Y[e, i, j] = ynod[e, j]

# Flatten coordinates and v-velocity
x_flat = X.flatten()
y_flat = Y.flatten()
v_flat = U_steady[..., 1].flatten()

# Interpolate v-velocity along the horizontal centerline (y=0.5)
xi = np.linspace(0, 1, 300)
yi = np.ones_like(xi) * 0.5
v_interp = griddata((x_flat, y_flat), v_flat, (xi, yi), method='cubic')

# Ghia (1982) benchmark data for v at y=0.5, Re=1000
ghia_x = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 
                   0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
ghia_v = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665, -0.31966, 
                   0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353, 0.29012, 0.27485, 0.0000])

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(ghia_x, ghia_v, 'o', label='Ghia et al. (1982)')
plt.plot(xi, v_interp, '-', label='MLX LSFEM Re=1000')
plt.xlabel('x coordinate')
plt.ylabel('v velocity')
plt.title('Centerline v-velocity (y=0.5) for Lid-Driven Cavity')
plt.grid(True)
plt.legend()
plt.savefig('/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f/ghia_v_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved.")
