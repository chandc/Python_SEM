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

## 3. Time integration: low-storage RK3 for the convective terms

The Fourier decoupling in §2 only holds if the convective term $\mathbf{u}\cdot\nabla\mathbf{u}$ is **explicit** — a linearised-implicit treatment makes the coefficients $z$-dependent, which is a convolution in $k_z$ and couples every mode. So the time integrator must be **IMEX**: explicit for convection, implicit for the stiff viscous/pressure/vorticity part that the per-mode solve handles.

### 3.1 Choice: 3-stage low-storage RK3 (Spalart–Moser–Rogers)

Use the **RKW3 / Crank–Nicolson** scheme of Spalart, Moser & Rogers (1991), the standard integrator for exactly this class of problem (spectral channel DNS with one wall-normal and periodic directions). Per stage $k = 1,2,3$:

$$ \hat{u}^{k} = \hat{u}^{k-1} + \Delta t\left[\gamma_k \hat{N}^{k-1} + \zeta_k \hat{N}^{k-2} + \alpha_k \hat{L}^{k-1} + \beta_k \hat{L}^{k}\right] $$

where $\hat{N}$ is the (explicit) convective term and $\hat{L}$ the (implicit) linear operator.

| $k$ | $\gamma_k$ | $\zeta_k$ | $\alpha_k$ | $\beta_k$ |
|---|---|---|---|---|
| 1 | 8/15 | 0 | 29/96 | 37/160 |
| 2 | 5/12 | −17/60 | −3/40 | 5/24 |
| 3 | 3/4 | −5/12 | 1/6 | 1/6 |

Consistency requires $\alpha_k + \beta_k = \gamma_k + \zeta_k$ at every stage, which these satisfy exactly (8/15, 2/15, 1/3). **Assert this in code** — it is a one-line check that catches the most common transcription error in the table.

### 3.2 Why this one

**Storage is the binding constraint in 3D.** The state is $N_z\times$ the 2D footprint. RKW3 is a **2-register** scheme: it needs only the current field and one previous convective term $\hat{N}^{k-2}$. Classical RK4 needs four stage derivatives — at $N_z$ = 128 and 7 fields that is the difference between fitting in memory and not.

**A larger CFL limit directly relieves the `a_mass` problem.** This is the decisive argument here and is specific to this solver. Explicit convection is CFL-limited, and `a_mass = w_mass·fac1/Δt` grows as $\Delta t$ shrinks — the instability documented in `GARTLING_VALIDATION.md` and `3D_DEVELOPMENT_PLAN.md` §0.2. RKW3 has a stability limit of $\mathrm{CFL} \approx \sqrt{3} \approx 1.73$ on the imaginary axis against $\approx 0.5$ for Adams–Bashforth 2, so it permits a step **~3.5× larger** — which lowers `a_mass` by the same factor. Given that measured runs blow up at `a_mass` = 60 and AC holds only to ~300 (`3D_DEVELOPMENT_PLAN.md` §0.3), a 3.5× reduction is not a micro-optimisation; it is a meaningful part of the feasibility margin.

**The cost accounting still favours it, even at three implicit solves per step.** The implicit solve dominates everything (thousands of CG iterations per solve), so three stages per step is a real 3× penalty on the dominant kernel. Against that: the step is ~3.5× larger, so the solve count per unit physical time is *slightly lower*, the scheme is 3rd-order rather than 2nd, and `a_mass` falls by 3.5×. Net: favourable, with the accuracy and stability gains effectively free.

> **Verify this claim rather than inheriting it.** The 3.5× is the textbook imaginary-axis ratio; the realised gain depends on this operator. Measure both the achievable CFL and the per-step solve cost at `3D_DEVELOPMENT_PLAN.md` Stage 5, and if the solve count per unit time comes out worse than AB2, fall back — the scheme choice is not load-bearing for correctness.

**Rejected alternatives**

| scheme | why not |
|---|---|
| AB2 / AB3 | 1 solve per step, but CFL ≈ 0.5 → the smallest $\Delta t$ and therefore the *largest* `a_mass`, straight into the measured failure band |
| Classical RK4 | better efficiency per stage on the imaginary axis (CFL 2.83/4 vs 1.73/3), but 4 registers and 4 implicit solves; storage is the constraint |
| Low-storage RK4 (Carpenter–Kennedy 2N, 5 stages) | 2 registers and good efficiency, but 5 implicit solves per step — the dominant kernel decides, and 5 > 3 |
| SSP-RK3 | strong-stability-preserving matters for shocks, not for smooth incompressible DNS; no benefit here, same stage count |

### 3.3 Storage and pseudocode

Two registers. `N_prev` carries $\hat{N}^{k-2}$ across stages; nothing else persists.

```python
GAMMA = (8/15, 5/12, 3/4)
ZETA  = (0.0, -17/60, -5/12)
ALPHA = (29/96, -3/40, 1/6)
BETA  = (37/160, 5/24, 1/6)
assert all(abs(a + b - (g + z)) < 1e-14
           for a, b, g, z in zip(ALPHA, BETA, GAMMA, ZETA))

N_prev = zeros_like(U_hat)                  # register 2
for k in range(3):
    u = irfft(U_hat, axis=-1)               # modes -> physical
    N = convective(u)                       # dealiased, 3/2 rule in z
    N_hat = rfft(N, axis=-1)                # physical -> modes
    rhs = U_hat + dt*(GAMMA[k]*N_hat + ZETA[k]*N_prev + ALPHA[k]*L(U_hat))
    U_hat = implicit_solve(rhs, coeff=dt*BETA[k])   # per-mode, batched
    N_prev = N_hat                          # only state carried between stages
```

Each stage costs one iFFT/FFT pair and one batched implicit solve over all modes. **The implicit solve must stay batched across $k_z$** — a Python loop over modes inside an RK stage multiplies the cost of §2's decoupling by three.

### 3.4 Note on the stage-wise implicit coefficient

The implicit operator changes between stages: stage $k$ solves with $\beta_k \Delta t$, not $\Delta t$. Since `a_mass` $\propto 1/\Delta t_\text{eff}$, the *effective* `a_mass` differs per stage — with $\beta = (37/160, 5/24, 1/6)$ the stage steps are $0.231\Delta t$, $0.208\Delta t$, $0.167\Delta t$, so every stage sees an `a_mass` roughly **4–6× larger** than $1.5/\Delta t$ would suggest. Budget for that when checking against the measured stability window: the relevant quantity is $\max_k\, \mathrm{fac}_1/(\beta_k \Delta t)$, not $\mathrm{fac}_1/\Delta t$. This partially offsets the CFL gain above and must be measured, not assumed.

---

## 4. Integration into the Object-Oriented Architecture

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

> **Caveat — this holds for the symmetric Poisson/Helmholtz operator above, and not for the least-squares VVP system.** For a Hermitian positive-definite $A$, CG needs conjugated inner products ($r^H z$, not $r^T z$); NumPy's `@`/`sum` will not do that for you, and the omission is silent — it produces plausible-looking iterates that converge to the wrong thing. For the LSSEM solver, $L^{H}L$ is Hermitian rather than symmetric and the adjoint test, the Jacobi diagonal and the line-search merit all need the same treatment. `3D_DEVELOPMENT_PLAN.md` §1.2 therefore recommends **splitting real and imaginary parts into real fields** for that solver, so the existing verified real-valued CG is reused unmodified. Use `complex128` here only if the operator really is the plain Helmholtz one.

### Preconditioner Updates
The inverse diagonal preconditioner (`inv_D`) must be updated to account for the $k_z^2$ diagonal mass components. This ensures the CG loop maintains its rapid convergence properties:

```python
inv_D[i, j, k_z] = 1.0 / (M_y[j,j] * K_x[i,i] + K_y[j,j] * M_x[i,i] + kz_squared[k_z] * M_x[i,i] * M_y[j,j])
```

By leveraging this Fourier expansion, we bypass the need to construct a massively complex 3D GLL hexahedral mesh solver, instead wrapping our existing, ultra-fast 2D SEM machinery in a highly parallelizable FFT workflow.
