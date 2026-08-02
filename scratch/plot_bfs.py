import numpy as np
import matplotlib.pyplot as plt
import os
from lssem2d.mesh import build_bfs
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.solver import step_bdf
import time

def solve_and_plot_bfs():
    from lssem2d.config import Config
    cfg = Config("bfs.toml")
    Re = cfg.get("simulation", "Re", 389.0)
    nu = 1.0 / Re
    N = cfg.get("mesh", "N", 6)
    dt = cfg.get("simulation", "dt", 0.1)
    
    mesh = build_bfs(N)
    D = diff_matrix(N)
    
    state = SolverState(mesh, D, nu=nu, dt=dt, fac1=1.0)
    from lssem2d.io import save_restart, load_restart, get_latest_restart
    
    restart_prefix = "bfs_restart_"
    restart_dir = "."
    start_step = 0
    current_time = 0.0
    
    latest_restart = get_latest_restart(restart_dir, restart_prefix)
    
    if latest_restart:
        print(f"Loading latest restart file: {latest_restart}")
        U_history, current_time, start_step = load_restart(latest_restart)
    else:
        U_0 = np.zeros((mesh.nelem, N+1, N+1, 4))
        U_history = [U_0]
    
    def custom_inlet(x, y, t):
        # parabolic profile, max 1.0 at y=0.5. y in [0, 1]
        return 4.0 * y * (1.0 - y)
        
    print(f"Running BFS simulation from step {start_step}...")
    t0 = time.time()
    
    target_steps = cfg.get("simulation", "max_steps", 10000)
    save_interval = cfg.get("simulation", "save_interval", 50)
    
    max_newton = cfg.get("solver", "max_newton", 5)
    newton_tol = cfg.get("solver", "newton_tol", 1e-4)
    
    for step in range(start_step + 1, target_steps + 1):
        # We use large newton_tol for initial steps, but let's just use 1e-4
        current_time = step * dt
        U_new = step_bdf(state, U_history, time=current_time, max_newton=max_newton, newton_tol=newton_tol, custom_inlet=custom_inlet)
        diff = np.max(np.abs(U_history[0] - U_history[1]))
        print(f"Step {step}, Change: {diff:.2e}")
        
        if step % save_interval == 0:
            restart_file = f"{restart_prefix}{step:06d}.npz"
            print(f"Saving restart file: {restart_file}")
            save_restart(restart_file, U_history, current_time, step)
            
        if diff < 1e-5:
            print(f"Converged at step {step}")
            break
            
    print(f"Simulation took {time.time() - t0:.2f}s")
    
    # Measure mass conservation defect
    from lssem2d.operators import dUdx, dUdy
    U_final = U_history[0]
    u_final, v_final = U_final[..., 0], U_final[..., 1]
    u_x = dUdx(u_final, state.D, state.mesh.facx)
    v_y = dUdy(v_final, state.D, state.mesh.facy)
    div_u = u_x + v_y
    defect_max = np.max(np.abs(div_u))
    defect_l2 = np.sqrt(np.sum(div_u**2 * state.mesh.wq))
    print(f"FINAL MASS CONSERVATION DEFECT (L_inf): {defect_max:.4e}")
    print(f"FINAL MASS CONSERVATION DEFECT (L2):    {defect_l2:.4e}")
    
    restart_file = f"{restart_prefix}{step:06d}.npz"
    print(f"Saving final restart file: {restart_file}")
    save_restart(restart_file, U_history, current_time, step)
    
    U_final = U_history[0]
    
    # Extract flattened coordinates and data for unstructured plotting
    x_pts = []
    y_pts = []
    u_pts = []
    v_pts = []
    for e in range(mesh.nelem):
        for i in range(N+1):
            for j in range(N+1):
                x_pts.append(mesh.xnod[e, i])
                y_pts.append(mesh.ynod[e, j])
                u_pts.append(U_final[e, i, j, 0])
                v_pts.append(U_final[e, i, j, 1])
                
    x_flat = np.array(x_pts)
    y_flat = np.array(y_pts)
    u_flat = np.nan_to_num(np.array(u_pts), nan=0.0, posinf=1e10, neginf=-1e10)
    v_flat = np.nan_to_num(np.array(v_pts), nan=0.0, posinf=1e10, neginf=-1e10)
    
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
    
    # Add streamlines
    plt.streamplot(X, Y, U_interp, V_interp, color='k', linewidth=0.5, density=2.0)
    
    # Mask out the step region manually (x < 0, y < 0)
    plt.fill_between([-10, 0], [-1, -1], [0, 0], color='white', zorder=10)
    
    plt.title('BFS: Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(os.path.join(output_dir, 'bfs_streamlines.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    solve_and_plot_bfs()
