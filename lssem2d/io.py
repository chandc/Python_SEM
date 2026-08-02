import numpy as np
import os

def save_restart(filename, U_history, time, step):
    """
    Saves the solver state for restarting.
    filename: string, path to the .npz file (e.g., 'restart.npz')
    U_history: list of state arrays [U_n, U_{n-1}, ...]
    time: float, current simulation time
    step: int, current simulation step number
    """
    save_dict = {
        'time': np.array([time]),
        'step': np.array([step])
    }
    for idx, U in enumerate(U_history):
        save_dict[f'U_{idx}'] = U
        
    np.savez(filename, **save_dict)

def load_restart(filename):
    """
    Loads a previously saved solver state.
    Returns: U_history, time, step
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Restart file {filename} not found.")
        
    data = np.load(filename)
    time = float(data['time'][0])
    step = int(data['step'][0])
    
    # Reconstruct U_history list
    U_history = []
    idx = 0
    while f'U_{idx}' in data:
        U_history.append(data[f'U_{idx}'])
        idx += 1
        
    return U_history, time, step
    
import glob

def get_latest_restart(directory, prefix="restart_"):
    """
    Finds the restart file with the highest step number in the given directory.
    Assumes filenames are formatted as f"{prefix}{step:06d}.npz"
    Returns the path to the latest restart file, or None if no files are found.
    """
    pattern = os.path.join(directory, f"{prefix}*.npz")
    files = glob.glob(pattern)
    if not files:
        return None
        
    latest_file = None
    max_step = -1
    
    for f in files:
        basename = os.path.basename(f)
        try:
            # Extract step number from filename (remove prefix and .npz)
            step_str = basename[len(prefix):-4]
            step = int(step_str)
            if step > max_step:
                max_step = step
                latest_file = f
        except ValueError:
            pass
            
    return latest_file
