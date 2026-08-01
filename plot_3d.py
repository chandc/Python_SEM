import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('p_convergence_3d_data.csv')
plt.figure(figsize=(10, 6))
plt.plot(df['p'].astype(float), df['numpy_s'].astype(float), marker='o', label='NumPy (Accelerate)')
plt.plot(df['p'].astype(float), df['pytorch_s'].astype(float), marker='s', label='PyTorch CPU')

plt.xlabel('Polynomial Degree (p)')
plt.ylabel('Compute Time (s)')
plt.title('2.5D SEM-Fourier Performance (Ex=10, Ey=10, Nz=16)')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.savefig('3d_fourier_sweep_plot.png', dpi=300, bbox_inches='tight')
