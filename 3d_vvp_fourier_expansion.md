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
3. **Momentum (Time-Discretised):** $c\, \mathbf{u} + \nabla p + \nu \nabla \times \boldsymbol{\omega} = \mathbf{f}$, with $c = 1/(\beta_k \Delta t)$ from the RKW3/Crank–Nicolson stage (§5)
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
4. Assemble the explicit RHS $\hat{\mathbf{f}}$ from $\hat{N}$ and the RKW3 stage history (§5): $\gamma_k \hat{N}^{k-1} + \zeta_k \hat{N}^{k-2}$ for convection, plus the Crank–Nicolson explicit half $\alpha_k \hat{L}^{k-1}$.

### Step 2: Solve the 2D Modes (Embarrassingly Parallel)
For each wavenumber $k_z \in [0, N_z/2]$ (exploiting conjugate symmetry for real physical fields):
1. Construct the 7-variable state tensor `(nelem, p+1, p+1, 7)` of type `complex128`.
2. Apply the matrix-free Hermitian operator $\mathcal{L}^* \mathcal{L}$ directly via tensor contraction.
3. Solve the normal equations using a standard complex Conjugate Gradient (CG) loop. Because the modes are decoupled, **every $k_z$ solver can be batched natively on MLX/PyTorch** into a single operation of shape `(nelem, p+1, p+1, 7, Nz)`.

### Step 3: Recover the 3D Field
Once the CG solver converges for all $k_z$, you have your new state $\hat{\mathbf{U}}^{n+1}$ in Fourier space. Execute an Inverse FFT to recover the updated 3D physical flow field, and proceed to the next time step.


---

## 5. Time Integration: RKW3 for Convection, Crank–Nicolson for the Viscous Term

The Fourier decoupling of §2 only survives if convection is **explicit** — a linearised-implicit treatment makes the coefficients $z$-dependent, which is a convolution in $k_z$ and recouples every mode. The integrator is therefore IMEX: a 3-stage low-storage Runge–Kutta on $\mathbf{N} = \mathbf{u}\cdot\nabla\mathbf{u}$, Crank–Nicolson on the linear operator $\mathcal{L}$ (viscous, pressure, vorticity). This is the scheme of **Spalart, Moser & Rogers (1991)**, the standard for spectral channel DNS.

For stage $k = 1,2,3$:

$$ \hat{\mathbf{U}}^{k} = \hat{\mathbf{U}}^{k-1} + \Delta t\left[\underbrace{\gamma_k \hat{\mathbf{N}}^{k-1} + \zeta_k \hat{\mathbf{N}}^{k-2}}_{\text{explicit: RK3}} + \underbrace{\alpha_k \hat{\mathcal{L}}^{k-1} + \beta_k \hat{\mathcal{L}}^{k}}_{\text{implicit: Crank–Nicolson}}\right] $$

| $k$ | $\gamma_k$ | $\zeta_k$ | $\alpha_k$ | $\beta_k$ | $1/\beta_k$ |
|---|---|---|---|---|---|
| 1 | 8/15 | 0 | 29/96 | 37/160 | 4.324 |
| 2 | 5/12 | −17/60 | −3/40 | 5/24 | 4.800 |
| 3 | 3/4 | −5/12 | 1/6 | 1/6 | **6.000** |

Consistency requires $\alpha_k + \beta_k = \gamma_k + \zeta_k$, satisfied exactly (8/15, 2/15, 1/3), and $\sum(\gamma_k + \zeta_k) = 1$. Both are asserted at import in `lssem3d/timestep.py`.

Rearranged for the per-mode solve, the momentum row carries

$$ c_k = \frac{1}{\beta_k \Delta t} $$

which is the 3D analogue of the 2D solver's $a_\text{mass} = w_\text{mass}\,\mathrm{fac}_1/\Delta t$.

### 5.1 The honest accounting

**In favour.** Adams–Bashforth 2 has **no stability interval on the imaginary axis at all** — it is unstable for pure advection at any $\Delta t$ and survives in practice only on viscous damping. RKW3 has a genuine interval, $\mathrm{CFL} \approx \sqrt{3} \approx 1.73$. For DNS on fine grids, where the convective eigenvalues are nearly imaginary, that is the decisive argument. It is also 3rd-order on convection against AB2's 2nd, needs only **2 storage registers** (classical RK4 needs 4, and at $N_z\times$ the 2D footprint registers are the binding constraint), and costs ~13% *fewer* implicit solves per unit physical time: three solves per step, but a step ~3.5× larger.

**Against, and this must not be glossed.** RKW3/CN makes $a_\text{mass}$ **worse, not better**. With $1/\beta = (4.32, 4.80, 6.00)$, the worst stage sees $c = 6/\Delta t$ against BDF2's $1.5/\Delta t$ — a factor of **4** at matched $\Delta t$ — and the 3.46× larger step does not quite recover it:

$$ \frac{6}{3.46\,\Delta t_\text{AB2}} = \frac{1.73}{\Delta t_\text{AB2}} \quad\text{vs}\quad \frac{1.50}{\Delta t_\text{AB2}} \qquad (\text{15\% worse}) $$

Given that measured 2D runs diverge at $a_\text{mass}$ = 60 and artificial compressibility holds only to $\approx 300$, this is a real charge against the scheme, not a rounding error. **Budget $\max_k 1/(\beta_k\Delta t)$ against the measured stability window — never $1.5/\Delta t$.** An implementation that assumes the BDF2 coefficient will under-estimate its own $a_\text{mass}$ by a factor of four.

**Rejected alternatives:** AB2/AB3 (smallest $\Delta t$, and AB2 has no imaginary-axis stability); classical RK4 (4 registers, 4 solves); low-storage RK4, Carpenter–Kennedy 2N (2 registers but 5 solves — the implicit solve dominates, and 5 > 3); SSP-RK3 (strong-stability-preservation buys nothing for smooth incompressible DNS).

### 5.2 Storage

Two registers. `N_prev` carries $\hat{\mathbf{N}}^{k-2}$ between stages; nothing else persists.

```python
from lssem3d.timestep import GAMMA, ZETA, ALPHA, BETA, NSTAGE

N_prev = zeros_like(U_hat)                       # register 2
for k in range(NSTAGE):
    u     = dealias_forward(U_hat, nz)           # modes -> padded physical
    N_hat = dealias_backward(convective(u), nz)  # 3/2 rule, back to modes
    rhs   = U_hat + dt*(GAMMA[k]*N_hat + ZETA[k]*N_prev + ALPHA[k]*L(U_hat))
    U_hat = solve_modes(rhs, c=1.0/(BETA[k]*dt)) # batched over ALL k_z
    N_prev = N_hat
```

`solve_modes` must stay batched across $k_z$: a Python loop over modes inside a stage multiplies away the entire benefit of the decoupling, three times per step.
