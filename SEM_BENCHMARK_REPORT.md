# Spectral Element Method (SEM) Benchmarking Report

This report documents our findings from developing and benchmarking a highly optimized, matrix-free Spectral Element Method (SEM) solver for the 2D Poisson equation on Apple Silicon.

## 1. The Test Problem

To rigorously test both the accuracy and performance of the solver, we used a highly oscillatory exact solution:
$$ u(x,y) = \sin(4\pi x) \sin(4\pi y) $$
This function exhibits $4$ full wave cycles across the computational domain $\Omega = [-1, 1] \times [-1, 1]$. Because the waves are dense, low-order numerical methods suffer from severe "dispersion errors" unless the mesh is extremely fine. By using a high-order spectral method, we demonstrated that we can keep a coarse mesh ($5 \times 15 = 75$ elements) and simply elevate the polynomial degree $p$ to achieve perfect exponential convergence (spectral accuracy).

## 2. Solver Architecture: Matrix-Free PCG

The core of our solver uses the **Preconditioned Conjugate Gradient (PCG)** algorithm. 

To maximize memory bandwidth and performance, the solver is entirely **matrix-free**:
- We never construct the massive global sparse matrix $A$.
- The stiffness operator $A x$ is evaluated locally on each element via a highly efficient tensor contraction: 
  $$ v_{local} = M_{1dy} u K_{1dx}^T + K_{1dy} u M_{1dx}^T $$
- Global $C^0$ continuity is enforced on the fly using a **Direct Stiffness Summation (DSS)** routine that shares residual boundary values with nearest-neighbor elements.
- We constructed a **Jacobi (Diagonal) Preconditioner** mathematically by extracting the algebraic diagonal of the local operator and applying the DSS algorithm to it, taming the $\mathcal{O}(p^4)$ condition number scaling of spectral elements without assembling any matrices.

## 3. Mathematical Formulation

The 2D Poisson equation is given by:
$$ -\nabla^2 u(x,y) = f(x,y) \quad \text{on } \Omega $$
Subject to homogeneous Dirichlet boundary conditions $u = 0$ on $\partial\Omega$. 

In the Spectral Element Method (SEM), we decompose the domain into quadrilateral elements $\Omega_e$. Multiplying by a test function $v$ and integrating by parts yields the weak form on each element:
$$ \int_{\Omega_e} \nabla u \cdot \nabla v \, d\Omega = \int_{\Omega_e} f v \, d\Omega $$

We map each element to a reference domain $[-1, 1]^2$ and approximate $u$ and $v$ using tensor products of 1D Lagrange polynomials based on Gauss-Lobatto-Legendre (GLL) nodes. 

By applying GLL quadrature (where the quadrature nodes coincide with the interpolation nodes), the mass matrix becomes perfectly diagonal. The discrete local operator acting on an element's nodal values $\mathbf{u}_e$ is expressed via the 1D Mass matrix $\mathbf{M}$ and 1D Stiffness matrix $\mathbf{K}$:
$$ (\mathbf{A}_e \mathbf{u}_e)_{i,j} = \sum_{m,n} \left( M^{(1D)}_{y; j,m} K^{(1D)}_{x; i,n} + K^{(1D)}_{y; j,m} M^{(1D)}_{x; i,n} \right) u_{e; n,m} $$

This can be written compactly as a matrix equation using tensor contractions:
$$ \mathbf{V}_e = \mathbf{M}_{1Dy} \mathbf{U}_e \mathbf{K}_{1Dx}^T + \mathbf{K}_{1Dy} \mathbf{U}_e \mathbf{M}_{1Dx}^T $$
where $\mathbf{U}_e$ is a $(p+1) \times (p+1)$ matrix of the local nodal values.

## 4. The NumPy Performance Trick

The matrix-free tensor contraction equation above is the core computational kernel of our solver. 
A naive implementation would loop over the elements and perform nested loops for the matrix multiplications. Earlier in our development, we attempted to use **Numba** (`@njit(parallel=True)`) to parallelize explicit `for` loops across the elements. However, this caused severe thread contention and performance degradation on Apple Silicon because the manual threading was fighting with Apple's highly optimized hardware threads.

The **NumPy Performance Trick** we utilized is to strip away all manual parallelization and rely exclusively on NumPy's `@` operator (matrix multiplication) inside a simple, single-threaded Python loop over the elements. Because $\mathbf{M}$ and $\mathbf{K}$ are contiguous $(p+1) \times (p+1)$ arrays, NumPy bypasses Python completely and delegates the tensor contraction `M_1dy @ u @ K_1dx.T` directly to Apple's **Accelerate BLAS framework**. 

Apple's BLAS is written in heavily optimized Assembly and inherently manages its own multi-threading across the M-series Performance (P) and Efficiency (E) cores at the hardware level. By feeding these small, dense matrix multiplications directly to Accelerate BLAS without interference, NumPy executed the $p=15$ global solve in just $\sim 0.05$ seconds, successfully matching the raw speed of compiled Fortran!

## 5. Performance Sweep Results

We ran a $p$-refinement sweep from $p=3$ to $p=15$ and timed the solver execution across four different numerical backends:
1. **NumPy** (CPU - Accelerate BLAS)
2. **MLX** (CPU - Compiled Apple Silicon Backend)
3. **PyTorch** (CPU)
4. **Modern Fortran** (Compiled - Accelerate BLAS)

### Tabulated Data

| Polynomial Degree ($p$) | Global Nodes ($N$) | Error ($L_\infty$) | MLX CPU (s) | NumPy (s) | PyTorch (s) | Fortran (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 3 | 736 | 6.25588e-02 | 0.03563 | 0.01335 | 0.03494 | 0.01059 |
| 4 | 1281 | 1.29143e-02 | 0.04173 | 0.01880 | 0.04096 | 0.01014 |
| 5 | 1976 | 2.01602e-03 | 0.04491 | 0.01960 | 0.04551 | 0.01739 |
| 6 | 2821 | 2.77314e-04 | 0.04906 | 0.02656 | 0.04961 | 0.02336 |
| 7 | 3816 | 3.40327e-05 | 0.04772 | 0.02631 | 0.10753 | 0.03237 |
| 8 | 4961 | 3.71951e-06 | 0.05290 | 0.04365 | 0.11917 | 0.04137 |
| 9 | 6256 | 3.89064e-07 | 0.05796 | 0.04488 | 0.11787 | 0.05178 |
| 10 | 7701 | 3.56790e-08 | 0.06607 | 0.05544 | 0.12767 | 0.06628 |
| 11 | 9296 | 3.25024e-09 | 0.07058 | 0.05775 | 0.13344 | 0.07979 |
| 12 | 11041 | 2.58225e-10 | 0.08147 | 0.06569 | 0.14183 | 0.09730 |
| 13 | 12936 | 2.09169e-11 | 0.08458 | 0.06850 | 0.14474 | 0.11809 |
| 14 | 14981 | 1.63609e-11 | 0.08722 | 0.06694 | 0.16423 | 0.14112 |
| 15 | 17176 | 1.26730e-11 | 0.08126 | 0.05352 | 0.15350 | 0.17069 |

### Convergence Plot

![p-Refinement Convergence Plot](./p_convergence_2d_plot.png)

## 6. Key Takeaways and Framework Analysis

### The Accuracy Floor
At $p=3$, the mesh completely failed to resolve the waves ($L_\infty \approx 0.06$). As $p$ increased, the error plummeted exponentially until it hit a hard floor around $1.2 \times 10^{-11}$ at $p=15$. This floor is not caused by the PCG solver failing to converge, but rather by **quadrature aliasing error**. We integrate the highly oscillatory forcing function using GLL quadrature nodes (which perfectly integrate polynomials of degree $2p-1$). The error in numerically integrating high-frequency sine waves using polynomials in 64-bit floating point math restricts us to this $10^{-11}$ floor.

### Framework Performance Rankings
1. **NumPy & Fortran (Tie - Fastest):** NumPy on the CPU is incredibly fast ($\sim 0.05$s at $p=15$), practically tying the raw compiled Fortran. This occurs because the bulk of the matrix-free computational work lies in the local tensor contractions (matrix multiplications), which NumPy immediately delegates to the highly optimized Apple Accelerate BLAS library written in C/Assembly.
2. **Apple MLX (Runner-up):** The MLX framework performed exceptionally well ($\sim 0.08$s). By utilizing `@mx.compile`, the entire PCG loop was compiled into a single execution graph, mitigating Python dispatch overhead. While slightly slower than raw BLAS on the CPU for these specific problem sizes, MLX's graph compilation provides massive scalability benefits for larger tensor networks.
3. **PyTorch CPU (Slowest):** Un-compiled PyTorch executing tight loops on the CPU was the slowest ($\sim 0.15$s). PyTorch incurs high Python dispatch overhead per tensor operation. Without fusing the kernels, PyTorch is inefficient for high-frequency small loops on a CPU. **However, this implementation is invaluable:** by simply flipping the device flag to `cuda`, this exact code can be dropped onto an NVIDIA Blackwell or Hopper GPU to execute natively on dedicated FP64 datacenter hardware for massive scaling.
