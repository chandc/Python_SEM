# Expanding to 3D: The "2.5D" SEM-Fourier Approach

Expanding a 2D Spectral Element Method (SEM) into a 3D solver where the third dimension ($z$) is periodic and solved in Fourier space is a classic and highly efficient computational technique. Because the Fourier basis functions (sines and cosines) are orthogonal and diagonalize the spatial derivative operator, **the 3D problem completely decouples into a series of independent 2D problems.**

Here is the mathematical formulation and architectural approach for implementation.

## 1. Mathematical Formulation (Helmholtz Decoupling)

We begin with the standard 3D Poisson equation for the physical field $u(x,y,z)$ and forcing function $f(x,y,z)$:
$$ - \nabla^2 u = - \left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} \right) u(x,y,z) = f(x,y,z) $$

Assuming the domain is periodic in the $z$-direction with length $L_z$, we can expand both $u$ and $f$ as discrete 1D Fourier series along $z$:
$$ u(x,y,z) = \sum_{k_z} \hat{u}(x,y,k_z) e^{i k_z z} $$
$$ f(x,y,z) = \sum_{k_z} \hat{f}(x,y,k_z) e^{i k_z z} $$

Where $k_z = \frac{2\pi n}{L_z}$ represents the discrete wavenumbers for mode $n$.

Taking the 1D Fourier Transform of the continuous governing equation replaces the physical spatial $z$-derivative with a simple scalar multiplier:
$$ \frac{\partial^2}{\partial z^2} \rightarrow -k_z^2 $$

This brilliantly transforms the single monolithic 3D Poisson equation into $N_z$ completely independent **2D Helmholtz equations**, one for each Fourier wavenumber $k_z$:
$$ - \left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} \right) \hat{u}(x,y,k_z) + k_z^2 \hat{u}(x,y,k_z) = \hat{f}(x,y,k_z) $$

## 2. The Algorithmic Workflow

To solve the full 3D domain at runtime, the numerical pipeline consists of three steps:

1. **Forward FFT:** Take the 3D physical forcing function array $F(x,y,z)$ and execute a 1D Fast Fourier Transform (FFT) along the $z$-axis. This yields the complex-valued array $\hat{F}(x,y,k_z)$.
2. **Solve 2D Modes (Embarrassingly Parallel):** For every discrete wavenumber $k_z$, solve its corresponding 2D Helmholtz equation. Because no information needs to be shared between different $k_z$ modes during the solve, all modes can be solved simultaneously via batched tensor operations on a GPU.
3. **Inverse FFT:** Once the Fourier coefficients $\hat{u}(x,y,k_z)$ are resolved for all modes, perform an Inverse 1D FFT along the $z$-axis to recover the final 3D physical solution field $u(x,y,z)$.

## 3. Integration into the Object-Oriented Architecture

Because of the matrix-free tensor contraction design in `sem_2d_oo.py`, extending this to the 2.5D Fourier domain requires minimal architectural disruption.

### Dimensional Scaling
The state tensors will increase by one dimension to hold the Fourier modes.
* **Old Shape:** `(p+1, p+1, E_x, E_y)`
* **New Shape:** `(p+1, p+1, E_x, E_y, N_z)`

### Modifying the Tensor Contraction (`apply_K`)
The current matrix-free operator calculates the 2D stiffness Laplacian. We simply append the new Helmholtz mass term $k_z^2 \mathbf{M} \hat{u}$ to the operation. In pseudocode:

```python
# Old 2D Poisson Operator:
v_local = (K_x @ u @ M_y.T) + (M_x @ u @ K_y.T)

# New 2.5D Helmholtz Operator:
# kz_squared is an array of shape (N_z,) broadcasted over the spatial dims
v_local = (K_x @ u @ M_y.T) + (M_x @ u @ K_y.T) + kz_squared * (M_x @ u @ M_y.T)
```

### Complex Arithmetic Support
Because Fourier coefficients are natively complex numbers, the tensor `dtype` in NumPy or PyTorch must be switched from `float64` to `complex128`. Both frameworks natively support broadcasting complex arithmetic, meaning the Conjugate Gradient (CG) algorithm implementation does not need to be rewritten.

### Preconditioner Updates
The inverse diagonal preconditioner (`inv_D`) must be updated to account for the $k_z^2$ diagonal mass components. This ensures the CG loop maintains its rapid convergence properties:

```python
inv_D[i, j, k_z] = 1.0 / (M_y[j,j] * K_x[i,i] + K_y[j,j] * M_x[i,i] + kz_squared[k_z] * M_x[i,i] * M_y[j,j])
```

By leveraging this Fourier expansion, we bypass the need to construct a massively complex 3D GLL hexahedral mesh solver, instead wrapping our existing, ultra-fast 2D SEM machinery in a highly parallelizable FFT workflow.
