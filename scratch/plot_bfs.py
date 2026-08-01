import numpy as np
import matplotlib.pyplot as plt
import os
from lssem2d.mesh import build_bfs
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf
import time

def solve_and_plot_bfs():
    # Re=389 (Armaly geometry). Average inlet velocity = 2/3 (if u_max = 1)
    nu = 1.0 / 100.0
    N = 6
    
    mesh = build_bfs(N)
    D = diff_matrix(N)
    
    state = SolverState(mesh, D, nu=nu, dt=0.5, fac1=1.0)
    U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
    U_history = [U_0]
    
    def custom_inlet(x, y, t):
        # parabolic profile, max 1.0 at y=0.5. y in [0, 1]
        return 4.0 * y * (1.0 - y)
        
    print("Running BFS simulation...")
    t0 = time.time()
    for step in range(100):
        # We use large newton_tol for initial steps, but let's just use 1e-4
        U_new = step_bdf(state, U_history, time=step*0.5, max_newton=3, newton_tol=1e-3, pin_p=True, custom_inlet=custom_inlet)
        diff = np.max(np.abs(U_new - U_history[0]))
        if step % 10 == 0:
            print(f"Step {step}, Change: {diff:.2e}")
        U_history = [U_new, U_history[0]] if len(U_history) == 1 else [U_new, U_history[0]]
        if diff < 1e-5:
            print(f"Converged at step {step}")
            break
    print(f"Simulation took {time.time() - t0:.2f}s")
    
    U_final = U_history[0]
    
    # Extract flattened coordinates and data for unstructured plotting
    x_flat = mesh.xnod.ravel()
    y_flat = mesh.ynod.ravel()
    u_flat = U_final[..., 0].ravel()
    v_flat = U_final[..., 1].ravel()
    
    output_dir = "/Users/danielchan/.gemini/antigravity-ide/brain/a68b6e7f-0de8-419b-a49c-533acf66a29f"
    
    # Plot axial velocity contours
    plt.figure(figsize=(12, 4))
    # We use tricontourf since we have unstructured points
    plt.tricontourf(x_flat, y_flat, u_flat, levels=50, cmap='RdBu_r')
    plt.colorbar(label='Axial Velocity (u)')
    plt.title('BFS: Axial Velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, 'bfs_axial_velocity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot streamlines
    # For streamlines, we need a regular grid, so we interpolate
    from scipy.interpolate import griddata
    xi = np.linspace(np.min(x_flat), np.max(x_flat), 300)
    yi = np.linspace(np.min(y_flat), np.max(y_flat), 100)
    X, Y = np.meshgrid(xi, yi)
    
    U_interp = griddata((x_flat, y_flat), u_flat, (X, Y), method='linear', fill_value=0.0)
    V_interp = griddata((x_flat, y_flat), v_flat, (X, Y), method='linear', fill_value=0.0)
    
    plt.figure(figsize=(12, 4))
    # Plot velocity magnitude as background
    mag = np.sqrt(U_interp**2 + V_interp**2)
    plt.contourf(X, Y, mag, levels=50, cmap='viridis', alpha=0.5)
    
    # Add streamlines
    plt.streamplot(X, Y, U_interp, V_interp, color='k', linewidth=0.5, density=2.0)
    
    # Mask out the step region manually (x < 0, y < 0)
    # plt.fill_between([-10, 0], [-1, -1], [0, 0], color='white', zorder=10) # if needed, but mesh is only fluid
    
    plt.title('BFS: Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, 'bfs_streamlines.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    solve_and_plot_bfs()
