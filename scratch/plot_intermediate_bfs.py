import numpy as np
import matplotlib.pyplot as plt
import os
from lssem2d.mesh import build_bfs
from lssem2d.io import get_latest_restart, load_restart

def plot_intermediate():
    N = 6
    mesh = build_bfs(N)
    
    restart_prefix = "bfs_restart_"
    restart_dir = "."
    latest_restart = get_latest_restart(restart_dir, restart_prefix)
    
    if not latest_restart:
        print("No restart file found.")
        return
        
    print(f"Loading latest restart file: {latest_restart}")
    U_history, current_time, start_step = load_restart(latest_restart)
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
    
    # Plot streamlines
    from scipy.interpolate import griddata
    xi = np.linspace(np.min(x_flat), np.max(x_flat), 300)
    yi = np.linspace(np.min(y_flat), np.max(y_flat), 100)
    X, Y = np.meshgrid(xi, yi)
    
    U_interp = griddata((x_flat, y_flat), u_flat, (X, Y), method='linear', fill_value=0.0)
    V_interp = griddata((x_flat, y_flat), v_flat, (X, Y), method='linear', fill_value=0.0)
    
    plt.figure(figsize=(12, 4))
    plt.streamplot(X, Y, U_interp, V_interp, color='k', linewidth=0.5, density=2.0)
    
    plt.fill_between([-10, 0], [-1, -1], [0, 0], color='white', zorder=10)
    
    plt.title(f'BFS: Streamlines (Step {start_step})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    
    out_path = os.path.join(output_dir, 'bfs_intermediate_streamlines.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_intermediate()
