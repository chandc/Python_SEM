import numpy as np
import matplotlib.pyplot as plt

data = np.load('cavity_re1000_data.npz')
ghia_u = data['ghia_u']
ghia_y = data['ghia_y']

u_pts = data['u_pts']
y_pts = data['y_pts']
Re = data['Re']

plt.figure(figsize=(6, 8))
plt.plot(ghia_u, ghia_y, 'o', label='Ghia et al. (1982)')
plt.plot(u_pts, y_pts, '-', label='MLX LSFEM Re=1000')
plt.xlabel('u velocity')
plt.ylabel('y coordinate')
plt.title('Centerline u-velocity (x=0.5) for Lid-Driven Cavity')
plt.grid(True)
plt.legend()
plt.savefig('/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f/ghia_comparison.png', dpi=300, bbox_inches='tight')
print("Plot saved.")
