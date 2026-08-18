# Expanding LSSEM VVP Navier-Stokes to 3D via Fourier Expansion

Expanding the Least-Squares Spectral Element Method (LSSEM) Velocity-Vorticity-Pressure (VVP) formulation from 2D to 3D via a Fourier series expansion is a powerful pseudo-spectral approach. 

By representing the $z$-direction as a Fourier series, we avoid building a monolithic 3D unstructured mesh. Instead, the 3D VVP system brilliantly decouples into $N_z$ independent 2D problems—one for each wavenumber $k_z$—while retaining the exact 3D physics. 

Here are the detailed equations and the step-by-step algorithm to implement this in the Python solver.

---

## 1. The 3D VVP Navier-Stokes Equations

In 3D, the VVP formulation expands from 4 variables to **7 physical variables**:
* Velocity vector: $\mathbf{u} = (u, v, w)$
* Vorticity vector: $\boldsymbol{\omega} = (\omega_x, \omega_y, \omega_z)$
* Pressure: $p$

The first-order PDE system required for the Least-Squares functional incorporates 8 equations (including a continuity constraint on vorticity for stability):

1. **Mass Continuity:** $\nabla \cdot \mathbf{u} = 0$
2. **Vorticity Definition:** $\nabla \times \mathbf{u} - \boldsymbol{\omega} = 0$
3. **Momentum (Time-Discretised):** $\frac{\alpha_0}{\Delta t} \mathbf{u} + \nabla p + \nu \nabla \times \boldsymbol{\omega} = \mathbf{f}$
4. **Vorticity Divergence Constraint:** $\nabla \cdot \boldsymbol{\omega} = 0$

*(Note: The explicit vector $\mathbf{f}$ absorbs the historical BDF time-stepping terms and the explicit non-linear convective terms $\mathbf{u} \cdot \nabla \mathbf{u}$.)*

---

## 2. Fourier Decoupling to 2.5D

We assume the domain is periodic in the $z$-direction. We expand all 7 variables into 1D Fourier series. For example, $u(x,y,z) = \sum \hat{u}(x,y,k_z) e^{i k_z z}$, where $k_z$ is the discrete wavenumber.

The spatial derivative $\frac{\partial}{\partial z}$ becomes a scalar multiplication by $i k_z$. Substituting this into our physical equations yields a complex-valued, coupled 7-equation system **for each individual wavenumber $k_z$**:

**1. Continuity:**
$$ \hat{R}_c = \partial_x \hat{u} + \partial_y \hat{v} + i k_z \hat{w} $$

**2. Vorticity Definition (3 components):**
$$ \hat{R}_{\omega x} = \partial_y \hat{w} - i k_z \hat{v} - \hat{\omega}_x $$
$$ \hat{R}_{\omega y} = i k_z \hat{u} - \partial_x \hat{w} - \hat{\omega}_y $$
$$ \hat{R}_{\omega z} = \partial_x \hat{v} - \partial_y \hat{u} - \hat{\omega}_z $$

**3. Momentum (3 components, with $c = \frac{\alpha_0}{\Delta t}$):**
$$ \hat{R}_{mx} = c \hat{u} + \partial_x \hat{p} + \nu (\partial_y \hat{\omega}_z - i k_z \hat{\omega}_y) - \hat{f}_x $$
$$ \hat{R}_{my} = c \hat{v} + \partial_y \hat{p} + \nu (i k_z \hat{\omega}_x - \partial_x \hat{\omega}_z) - \hat{f}_y $$
$$ \hat{R}_{mz} = c \hat{w} + i k_z \hat{p} + \nu (\partial_x \hat{\omega}_y - \partial_y \hat{\omega}_x) - \hat{f}_z $$

**4. Vorticity Divergence:**
$$ \hat{R}_{d\omega} = \partial_x \hat{\omega}_x + \partial_y \hat{\omega}_y + i k_z \hat{\omega}_z $$

---

## 3. The Least-Squares Functional & Normal Equations

For a specific mode $k_z$, the Least-Squares functional $J_{k_z}$ is the sum of the $L^2$-norms of the complex residuals over the 2D spatial domain $\Omega$:

$$ J_{k_z}(\hat{\mathbf{U}}) = \frac{1}{2} \int_\Omega \left( |\hat{R}_c|^2 + |\hat{R}_{\omega x}|^2 + \dots + |\hat{R}_{mz}|^2 + |\hat{R}_{d\omega}|^2 \right) d\Omega $$

Taking the variation with respect to the complex conjugate of the state vector $\delta \hat{\mathbf{U}}^*$ yields the normal equations $\mathcal{L}^* \mathcal{L} \hat{\mathbf{U}} = \mathcal{L}^* \hat{\mathbf{F}}$. 

Because of the complex coefficients, the resulting system matrix is **Hermitian** (rather than purely symmetric real). When evaluating the adjoint operator $\mathcal{L}^*$, the $z$-derivatives naturally flip sign because $(i k_z)^* = -i k_z$.

---

## 4. Algorithmic Workflow (Pseudo-Spectral Method)

Because the non-linear convective term ($\mathbf{u} \cdot \nabla \mathbf{u}$) mathematically convolves Fourier modes, it cannot be solved directly in frequency space. We must use a **Pseudo-Spectral Algorithm** to treat the non-linearity explicitly.

For every time step, the Python solver will execute the following pipeline:

### Step 1: Compute Non-Linear Forcing in Physical Space
1. Perform an Inverse 1D-FFT on the previous time step's Fourier state arrays ($\hat{u}, \hat{v}, \hat{w}$) to obtain physical variables ($u, v, w$) at all $z$-slices.
2. In physical 3D space, calculate the non-linear convective products: $N_x = u \partial_x u + v \partial_y u + w \partial_z u$, etc. *(Note: To prevent aliasing instabilities, you should evaluate this on a grid padded by the 3/2 rule, then truncate).*
3. Perform a Forward 1D-FFT on $N_x, N_y, N_z$ to bring the non-linear forcing terms back into Fourier space: $\hat{N}_x(k_z)$.
4. Assemble the complete explicit RHS vector $\hat{\mathbf{f}}$ using $\hat{N}$ and the BDF history vectors.

### Step 2: Solve the 2D Modes (Embarrassingly Parallel)
For each wavenumber $k_z \in [0, N_z/2]$ (exploiting conjugate symmetry for real physical fields):
1. Construct the 7-variable state tensor `(nelem, p+1, p+1, 7)` of type `complex128`.
2. Apply the matrix-free Hermitian operator $\mathcal{L}^* \mathcal{L}$ directly via tensor contraction.
3. Solve the normal equations using a standard complex Conjugate Gradient (CG) loop. Because the modes are decoupled, **every $k_z$ solver can be batched natively on MLX/PyTorch** into a single operation of shape `(nelem, p+1, p+1, 7, Nz)`.

### Step 3: Recover the 3D Field
Once the CG solver converges for all $k_z$, you have your new state $\hat{\mathbf{U}}^{n+1}$ in Fourier space. Execute an Inverse FFT to recover the updated 3D physical flow field, and proceed to the next time step.
