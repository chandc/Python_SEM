# Python_SEM: Matrix-Free Spectral Element Method Solver

A highly optimized, matrix-free Spectral Element Method (SEM) solver for the 2D Poisson equation, designed to run efficiently on Apple Silicon (M-series) CPUs using NumPy, MLX, and PyTorch, alongside a pure Modern Fortran benchmark.

## Features

- **Matrix-Free Tensor Contractions**: By exploiting the tensor-product structure of Gauss-Lobatto-Legendre (GLL) polynomials, the solver avoids constructing the global sparse stiffness matrix entirely. The discrete Laplacian is evaluated locally in $\mathcal{O}(p^3)$ time using highly optimized matrix multiplications.
- **Direct Stiffness Summation (DSS)**: Global $C^0$ continuity is enforced on the fly by sharing boundary residuals between nearest-neighbor elements.
- **Jacobi Preconditioned Conjugate Gradient (PCG)**: The condition number of spectral methods scales poorly with the polynomial degree $p$ $\mathcal{O}(p^4)$. We construct a global diagonal preconditioner algebraically and apply it via DSS to drastically reduce the number of iterations required to hit machine precision.
- **Multi-Backend Benchmarking**: Execute the same solver logic across:
  - **NumPy**: Delegated directly to Apple's Accelerate BLAS for near-compiled performance.
  - **Apple MLX**: Utilizes `@mx.compile` to fuse the entire PCG loop into an optimized graph.
  - **PyTorch**: Native CPU tensors (ready for GPU transition).
  - **Fortran**: A pure compiled implementation acting as the theoretical speed limit.

## Getting Started

### Prerequisites

You need Python 3.9+ and the following libraries installed:
```bash
pip install numpy mlx torch matplotlib sympy
```
You will also need `gfortran` installed on your system to compile the Fortran benchmark.

### Running the Solver

You can run the full $p$-refinement benchmark (which sweeps the polynomial degree from $p=3$ to $p=15$ on a highly oscillatory test problem) by executing:
```bash
python sem_2d.py
```
This script will automatically generate the 1D GLL matrices, compile the Fortran executable `sem_2d_f90`, run all four backends, and generate a convergence plot `p_convergence_2d_plot.png`.

## Test Problem & Benchmarks

The solver is tested against the highly oscillatory function:

$$
u(x,y) = \sin(4\pi x) \sin(4\pi y)
$$

On a fixed grid of $5 \times 15$ elements, the solver exhibits perfect exponential convergence (spectral accuracy), plunging from a large error at $p=3$ to a hard machine-precision floor of $\sim 10^{-11}$ around $p=13$.

For a deep dive into the mathematics, the Numba thread-contention issues we discovered, and how NumPy achieves Fortran-level speeds by feeding dense $(p+1)\times(p+1)$ matrix multiplications directly into the Apple Accelerate BLAS framework, please read the [SEM Benchmark Report](SEM_BENCHMARK_REPORT.md).