import numpy as np
import matplotlib.pyplot as plt
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf
import time
from scipy.interpolate import griddata

def plot_cavity():
    Re = 100.0
    nu = 1.0 / Re
    N = 8
    
    mesh = build_channel(1.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    state = SolverState(mesh, D, nu=nu, dt=1e6, fac1=1.0)
    
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    U_history = [U_0]
    
    print("Solving Cavity Re=100...")
    for step in range(5):
        U_new = step_bdf(state, U_history, time=0.0, max_newton=10, newton_tol=1e-5, pin_p=True)
        diff = np.max(np.abs(U_new - U_history[0]))
        U_history = [U_new]
        if diff < 1e-5:
            break
            
    U_final = U_history[0]
    
    # 1. Plot velocity profile
    y_pts = []
    u_pts = []
    for e in range(mesh.nelem):
        for i in range(N+1):
            if abs(mesh.xnod[e, i] - 0.5) < 1e-5:
                y_pts.extend(mesh.ynod[e, :])
                u_pts.extend(U_final[e, i, :, 0])
                
    idx = np.argsort(y_pts)
    y_pts = np.array(y_pts)[idx]
    u_pts = np.array(u_pts)[idx]
    
    ghia_y = np.array([1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
    ghia_u = np.array([1.0, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058, -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0])
    
    plt.figure(figsize=(6, 5))
    plt.plot(ghia_u, ghia_y, 'o', label='Ghia et al. (1982)')
    plt.plot(u_pts, y_pts, '-', label=f'SEM N={N}')
    plt.xlabel('u-velocity')
    plt.ylabel('y')
    plt.title('Centerline u-Velocity Profile (x=0.5, Re=100)')
    plt.legend()
    plt.grid(True)
    plt.savefig('verification_plots/cavity_u_profile.png')
    plt.close()
    
    # 1b. Plot v-velocity profile at y=0.5
    x_pts_v = []
    v_pts = []
    for e in range(mesh.nelem):
        for j in range(N+1):
            if abs(mesh.ynod[e, j] - 0.5) < 1e-5:
                x_pts_v.extend(mesh.xnod[e, :])
                v_pts.extend(U_final[e, :, j, 1])
                
    idx_v = np.argsort(x_pts_v)
    x_pts_v = np.array(x_pts_v)[idx_v]
    v_pts = np.array(v_pts)[idx_v]
    
    ghia_x_v = np.array([1.0, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
    ghia_v = np.array([0.0, -0.0591, -0.0739, -0.0886, -0.1031, -0.1691, -0.2245, -0.2453, 0.0545, 0.1753, 0.1751, 0.1608, 0.1232, 0.1089, 0.1009, 0.0923, 0.0])
    
    plt.figure(figsize=(6, 5))
    plt.plot(ghia_x_v, ghia_v, 'o', label='Ghia et al. (1982)')
    plt.plot(x_pts_v, v_pts, '-', label=f'SEM N={N}')
    plt.xlabel('x')
    plt.ylabel('v-velocity')
    plt.title('Centerline v-Velocity Profile (y=0.5, Re=100)')
    plt.legend()
    plt.grid(True)
    plt.savefig('verification_plots/cavity_v_profile.png')
    plt.close()
    
    # 2. Plot streamlines
    # We will evaluate u, v on a uniform grid for streamplot
    x_1d = np.linspace(0, 1, 100)
    y_1d = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_1d, y_1d)
    U_grid = np.zeros_like(X)
    V_grid = np.zeros_like(Y)
    
    # Very slow naive interpolation (closest node or element logic)
    # Let's just use griddata to interpolate scattered points to meshgrid
    x_pts = []
    y_pts_grid = []
    u_flat = []
    v_flat = []
    
    for e in range(mesh.nelem):
        Xe, Ye = np.meshgrid(mesh.xnod[e, :], mesh.ynod[e, :], indexing='ij')
        x_pts.extend(Xe.flatten())
        y_pts_grid.extend(Ye.flatten())
        u_flat.extend(U_final[e, ..., 0].flatten())
        v_flat.extend(U_final[e, ..., 1].flatten())
        
    x_pts = np.array(x_pts)
    y_pts_grid = np.array(y_pts_grid)
    u_flat = np.array(u_flat)
    v_flat = np.array(v_flat)
    
    U_grid = griddata((x_pts, y_pts_grid), u_flat, (X, Y), method='cubic')
    V_grid = griddata((x_pts, y_pts_grid), v_flat, (X, Y), method='cubic')
    
    plt.figure(figsize=(6, 5))
    speed = np.sqrt(U_grid**2 + V_grid**2)
    plt.streamplot(X, Y, U_grid, V_grid, color=speed, cmap='viridis', density=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.colorbar(label='Speed')
    plt.title('Streamlines (Re=100)')
    plt.savefig('verification_plots/cavity_streamlines.png')
    plt.close()
    
    print("Done")

if __name__ == "__main__":
    plot_cavity()
