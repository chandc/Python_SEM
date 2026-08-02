# LSSEM Solver Improvements

This document summarizes the major improvements, architectural changes, and bug fixes made to the Python Least-Squares Spectral Element Method (LSSEM) solver.

## 1. Performance & Vectorization Upgrades

The original Python port implemented the solver logic via manual `for` loops iterating element-by-element, which was incredibly slow in Python. We applied global vectorization to eliminate these loops:

*   **Fully Batched Operators**: Replaced manual `for` loops across elements with vectorized `np.matmul` and `np.einsum` operations across the entire `(nelem, N, N)` tensor space. This allows numpy to dispatch matrix operations to highly optimized BLAS routines under the hood.
*   **Exact Analytical Preconditioner**: Replaced the slow, iterative unit-vector approach to building the Jacobi preconditioner with an exact analytical diagonal extraction (`compute_jacobi`). Preconditioner assembly time dropped to near zero.
*   **Memory Optimization (`SolverState`)**: We introduced the `SolverState` class to centrally manage and preallocate all intermediate work arrays (such as `su`, `c`, `tmp_x`, `u_x`, etc.). This completely eliminates expensive and repetitive memory allocations inside the inner implicit solver loops.

**Result**: The lid-driven cavity simulation (Re=1000), which previously required significant time to converge, now reaches steady-state convergence (600+ time steps) in approximately 60 seconds.

## 2. Physics & Boundary Condition Fixes

The solver was upgraded from a simple square domain to support complex fluid dynamics problems, exposing and fixing several critical physics bugs:

*   **Complex Geometry Support**: Successfully set up the multi-block **Backward-Facing Step (BFS)** geometry, properly handling re-entrant corners, block connectivity, and heterogeneous boundary condition assignments across the L-shaped domain.
*   **Fixed the Drifting Pressure Bug**: Discovered that the outlet boundary condition (`bc=4`) lacked the crucial `p=0` Dirichlet condition in both `apply_bc` and `apply_mask`. Without this, the global pressure field was mathematically unconstrained and allowed to drift, which previously ruined the upstream flow physics and washed out flow features.
*   **Identified Mass Conservation Defect**: By implementing a custom Gauss-Lobatto quadrature script (`check_mass.py`), we calculated the exact mass fluxes across all boundaries. We conclusively proved that the standard unpenalized LSSEM formulation inherently leaks mass at the step singularity, resulting in a ~400% mass defect that washes out the recirculation bubble. (This sets the stage for adding the $W_{cont}$ continuity penalty).

## 3. Architecture & Infrastructure

The project structure was refactored for better usability and separation of concerns:

*   **Configuration Files**: Moved all hardcoded physical (Reynolds number, timestep) and solver (Newton tolerances, PCG limits) parameters into `.toml` files (`cavity.toml` and `bfs.toml`). The core codebase is now entirely decoupled from specific simulation setups.
*   **Advanced Diagnostics**: Built robust plotting scripts (`plot_bfs.py`, `plot_intermediate_bfs.py`) capable of loading checkpoint restart files (`.npz`) on the fly, allowing for post-processing and intermediate monitoring without blocking the simulation.
*   **Plotting Artifact Fixes**: Manually masked the solid step region in our matplotlib configurations to prevent `scipy.interpolate.griddata` from interpolating fake velocities inside the solid wall. This solved the visual illusion of "flow coming out of the wall" seen in early stream plots.
