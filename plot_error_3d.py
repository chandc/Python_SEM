import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('p_convergence_3d_data.csv')

plt.figure(figsize=(10, 6))
# Both should be identical, so we just plot one
plt.semilogy(df['p'].astype(float), df['error_numpy'].astype(float), marker='o', label='Error (L_inf)', color='red')

plt.xlabel('Polynomial Degree (p)')
plt.ylabel('Error (L_inf)')
plt.title('3D Fourier SEM: p-Convergence (Ex=10, Ey=10, Nz=16)')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.savefig('3d_fourier_error_plot.png', dpi=300, bbox_inches='tight')
