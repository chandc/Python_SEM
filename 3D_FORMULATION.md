# The 3D LSSEM formulation: equations solved and time-marching steps

What `lssem3d` actually solves, written out. Every equation below is transcribed
from the code, not from the derivation that preceded it — `operator.py`
`apply_L0_complex` for the rows, `timestep.py` for the coefficients, and the
drivers for the stage assembly. All eight rows were verified to reproduce
`apply_L0_complex` bit-for-bit on a random complex state.

Companion documents: [3D_STATUS.md](./3D_STATUS.md) (what has been measured),
[3D_DEVELOPMENT_PLAN.md](./3D_DEVELOPMENT_PLAN.md) (the plan and its gates).

---

## 1. Continuous problem

Incompressible Navier–Stokes on a domain periodic in $z$:

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
  = -\nabla p + \nu \nabla^2 \mathbf{u},
\qquad
\nabla\cdot\mathbf{u} = 0
$$

with $\mathbf{u} = (u, v, w)$, constant $\nu$, and density absorbed into $p$.

## 2. Velocity–vorticity–pressure (VVP) first-order form

Introduce the vorticity $\boldsymbol{\omega} = \nabla\times\mathbf{u}$ as an
independent unknown. For a divergence-free field
$\nabla^2\mathbf{u} = -\nabla\times\boldsymbol{\omega}$, so the viscous term
becomes **first order** in $\boldsymbol{\omega}$:

$$
\begin{aligned}
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u}
  &= -\nabla p - \nu\,(\nabla\times\boldsymbol{\omega}) \\[2pt]
\nabla\cdot\mathbf{u} &= 0 &&\text{continuity}\\[2pt]
\boldsymbol{\omega} - \nabla\times\mathbf{u} &= \mathbf{0}
  &&\text{vorticity definition (3 equations)}\\[2pt]
\nabla\cdot\boldsymbol{\omega} &= 0 &&\text{vorticity divergence}
\end{aligned}
$$

**Seven unknowns** — $u, v, w, \omega_x, \omega_y, \omega_z, p$ — and **eight
equations**. The system is overdetermined by one:
$\nabla\cdot\boldsymbol{\omega} = 0$ is implied by
$\boldsymbol{\omega} = \nabla\times\mathbf{u}$ and is retained as an independent
row.

> **This row must be DOWN-WEIGHTED.** An earlier version of this document said
> the functional "benefits from it"; that is refuted (3D_STATUS §7J). At weight 1
> it is the single largest source of ill-conditioning in the system: it is the
> one row 2D does not have, it involves only $\omega_x,\omega_y$ at $k_z=0$, and
> it loads their Jacobi diagonal with derivative-squared terms while
> contributing nothing to $\mathcal{A}$ for divergence-free vorticity. Measured
> cost: **10.5× the CG iterations** and a conditioning penalty growing with
> order (139× at $p$=4, 2885× at $p$=10). `operator.ROW7_WEIGHT` = $10^{-4}$. Everything is first
order, which is what permits a single $C^0$ spectral-element space for every
variable.

## 3. Fourier expansion in $z$

Every field is expanded on the periodic direction,

$$
f(x,y,z,t) \;=\; \sum_{k} \hat f_k(x,y,t)\, e^{\mathrm{i} k z},
\qquad k = \frac{2\pi m}{L_z}
$$

so $\partial/\partial z \to \mathrm{i}k$, and **the eight equations decouple into
an independent system per wavenumber $k$** — except through the convective term,
which is quadratic and therefore couples modes. That decoupling is the entire
reason for the Fourier direction: the implicit solve is embarrassingly parallel
across $k$.

Real input implies Hermitian symmetry, so only $k \ge 0$ is stored (`rfft`), with
$n_k = N_z/2 + 1$ modes. The $k = 0$ and Nyquist modes must have **real**
coefficients; that is enforced, not assumed (§7).

## 4. The eight rows, per mode

With $\partial_x,\partial_y$ the spectral-element derivatives and $c$, $\kappa_p$
defined in §5–6, the residual rows $\mathbf{R} = \mathcal{L}_0\hat{U}$ are:

$$
\begin{aligned}
R_0 &= \kappa_p\,p + \partial_x u + \partial_y v + \mathrm{i}k\,w
  &&\text{continuity } (+\text{AC})\\
R_1 &= \partial_y w - \mathrm{i}k\,v - \omega_x &&\omega_x\ \text{definition}\\
R_2 &= \mathrm{i}k\,u - \partial_x w - \omega_y &&\omega_y\ \text{definition}\\
R_3 &= \partial_x v - \partial_y u - \omega_z &&\omega_z\ \text{definition}\\
R_4 &= c\,u + \partial_x p + \nu\,(\partial_y \omega_z - \mathrm{i}k\,\omega_y)
  &&x\text{-momentum}\\
R_5 &= c\,v + \partial_y p + \nu\,(\mathrm{i}k\,\omega_x - \partial_x \omega_z)
  &&y\text{-momentum}\\
R_6 &= c\,w + \mathrm{i}k\,p + \nu\,(\partial_x \omega_y - \partial_y \omega_x)
  &&z\text{-momentum}\\
R_7 &= \partial_x \omega_x + \partial_y \omega_y + \mathrm{i}k\,\omega_z
  &&\nabla\cdot\boldsymbol{\omega}
\end{aligned}
$$

Rows $R_4$–$R_6$ are $c\,\mathbf{u} + \nabla p + \nu(\nabla\times\boldsymbol{\omega})$
component-wise; rows $R_1$–$R_3$ are
$\boldsymbol{\omega} - \nabla\times\mathbf{u}$.

**$\kappa_p$ (artificial compressibility)** is a numerical term on the continuity
row, not physics. It is **off ($\kappa_p = 0$) in the production recipe** — with
it on, the row solves $\nabla\cdot\mathbf{u} = -\kappa_p\,(p - p^{\text{prev}})$,
which costs 5–7 orders of magnitude of accuracy (3D_STATUS §7E).

### Split-real representation

$\mathcal{L}_0$ has complex coefficients (the $\mathrm{i}k$ terms), so
$\mathcal{L}_0^{*}\mathcal{L}_0$ would be **Hermitian, not symmetric**. To keep a
real symmetric operator for CG, the 7 complex fields are stored as **14 real
fields** and the 8 complex rows as 16 real rows. A complex coefficient $\alpha$
becomes the real $2\times2$ block

$$
\alpha \;\longmapsto\;
\begin{pmatrix} \operatorname{Re}\alpha & -\operatorname{Im}\alpha \\
                \operatorname{Im}\alpha & \operatorname{Re}\alpha \end{pmatrix},
$$

whose transpose corresponds to the complex **conjugate** — so in
$\mathcal{L}_0^{T}$, $\mathrm{i}k \to -\mathrm{i}k$ while real $c,\nu$ are
unchanged. (Both column norms of that block equal $|\alpha|^2$, which is why the
real and imaginary halves of a field share one diagonal entry.)

## 5. Least-squares discretisation

The functional minimised is

$$
J(\hat U) \;=\; \int_\Omega \sum_{r=0}^{7} \rho_r \,\bigl|R_r(\hat U)\bigr|^2 \, d\Omega
$$

with $\rho_r$ the **row weights** (§5.1) and the integral evaluated by GLL
quadrature with weights $W$. Its normal equations give the operator CG solves:

$$
\boxed{\;\mathcal{A} \;=\; M\,Q^{T}Q\;\mathcal{L}_0^{T}\,(\rho W)\,\mathcal{L}_0\,M\;}
$$

| factor | meaning | consequence of omitting it |
|---|---|---|
| $\mathcal{L}_0$ | the eight rows above | — |
| $W$ | GLL quadrature weights | minimises the *nodal* sum, not the integral |
| $\rho$ | row weights | see §5.1 |
| $Q^{T}Q$ | gather–scatter ($C^0$ assembly) | element-local, massively under-determined |
| $M$ | boundary mask | BCs unimposed |

$W$ and $\rho$ appear in the **forward** operator only; `apply_LT` is the
unweighted transpose, so the product is the normal operator of $J$.

### 5.1 Row weights — a choice, not a constant

`lssem2d` writes the momentum row as
$a_{\text{mass}}\,u + a_{\text{flux}}\,N(u)$ with the constraint rows at weight
1, so **$a_{\text{flux}}$ is the least-squares weight of momentum against the
constraints**. Two settings are in use:

| setting | $a_{\text{mass}}$ | $a_{\text{flux}}$ | used by |
|---|---|---|---|
| **legacy** | $1$ | $\Delta t$ | Chan (1996) Stokes decay; **the 3D production recipe** |
| $w_{\text{mom}} = 1$ | $\text{fac1}/\Delta t$ | $1$ | the 2D cavity / BFS studies |

In the code the legacy scaling is $\rho_r = 1/c^2$ on rows 4–6
(`operator.momentum_row_weights`), dividing the momentum row by $c$ so its mass
coefficient is 1 — every row then $O(1)$ in the velocity. Without it, momentum
outweighs continuity by $c^2 \approx 10^6$ and the minimiser effectively ignores
$\nabla\cdot\mathbf{u}$.

**Which weighting is right is problem-dependent** (3D_STATUS §7A.2b).

## 6. Time marching: RKW3 / Crank–Nicolson

Convection is **explicit** (3-stage low-storage Runge–Kutta), the linear operator
**implicit**. Coefficients are Spalart–Moser–Rogers (1991):

$$
\begin{aligned}
\gamma &= \left(\tfrac{8}{15},\; \tfrac{5}{12},\; \tfrac{3}{4}\right)
  &&\text{explicit, current stage}\\
\zeta  &= \left(0,\; -\tfrac{17}{60},\; -\tfrac{5}{12}\right)
  &&\text{explicit, previous stage}\\
\alpha &= \left(\tfrac{29}{96},\; -\tfrac{3}{40},\; \tfrac{1}{6}\right)
  &&\text{implicit, old state}\\
\beta  &= \left(\tfrac{37}{160},\; \tfrac{5}{24},\; \tfrac{1}{6}\right)
  &&\text{implicit, new state}
\end{aligned}
$$

subject to the consistency identity

$$
\alpha_k + \beta_k \;=\; \gamma_k + \zeta_k , \qquad k = 0,1,2
$$

asserted in exact rational arithmetic at import — mis-transcribing the table
silently costs an order and is invisible until a convergence study.

### The stage equation

Write $N(\mathbf{u}) = -(\mathbf{u}\cdot\nabla)\mathbf{u}$ for the explicit term
and $L(\hat U) = -\nabla p - \nu(\nabla\times\boldsymbol{\omega})$ for the
implicit one. Each stage $k = 0,1,2$ advances

$$
\hat U^{k} \;=\; \hat U^{k-1} + \Delta t\Bigl[\,
  \gamma_k N^{k-1} + \zeta_k N^{k-2}
  + \alpha_k L^{k-1} + \beta_k L^{k} \Bigr]
$$

Since $\zeta_0 = 0$, **the scheme is self-starting** — the first stage never
reads history, so no startup procedure is needed.

### Rearranged for the solve

$L^{k}$ is the unknown. Dividing by $\beta_k \Delta t$ puts the stage in the form
the operator of §4 expects:

$$
\underbrace{c_k\,\hat U^{k} - L(\hat U^{k})}_{\textstyle \mathcal{L}_0 \hat U^{k}}
\;=\;
\underbrace{c_k\Bigl[\hat U^{k-1}
  + \Delta t\bigl(\gamma_k N^{k-1} + \zeta_k N^{k-2} + \alpha_k L^{k-1}\bigr)\Bigr]}_{\textstyle f,\ \text{the momentum right-hand side}}
$$

$$
\boxed{\;c_k \;=\; \frac{1}{\beta_k\,\Delta t}\;}
$$

**$c_k$ is the mass coefficient in rows $R_4$–$R_6$**, the 3D analogue of the 2D
$a_{\text{mass}}$. Note

$$
\frac{1}{\beta} = (4.32,\; 4.80,\; \mathbf{6.00})
\quad\Longrightarrow\quad
\max_k c_k = \frac{6}{\Delta t},
$$

four times BDF2's $1.5/\Delta t$ at the same step — budget stability against
$6/\Delta t$, never $1.5/\Delta t$.

The right-hand side for the **constraint rows** ($R_0$–$R_3$, $R_7$) is **zero**,
except that with AC on, $R_0$ carries $\kappa_p\, p^{k-1}$.

### Order of the scheme

Second order by construction, and measured:

| configuration | order |
|---|---|
| explicit half alone (the $\gamma,\zeta$ table) | $3.025$ |
| implicit half alone (Crank–Nicolson) | $2.002$ |
| mixed, scalar model | $2.189$ |
| **full 3D PDE, Stokes decay** | $\mathbf{2.00}$ |
| **full 3D PDE, Taylor–Green (convection active)** | $\mathbf{2.00}$ |

RK3's third order lives in the convective half alone; **CN caps the mixed scheme
at 2**. A gate demanding $3.0$ overall is unachievable by any correct
implementation.

## 7. The stage solve

Each stage is one linear solve. Because the boundary conditions are
**inhomogeneous**, the system is solved as a **defect correction** — solving
$\mathcal{A}\hat U = \mathcal{L}_0^{T}Wf$ with a masked $\mathcal{A}$ is
well-posed but is a *different problem*, and produced a motionless cavity when it
was done that way.

Given the current iterate $\hat U$ and the stage right-hand side $f$:

$$
\begin{aligned}
\text{1.}\quad & r  = \mathcal{L}_0^{T}\bigl[(\rho W)\,\mathcal{L}_0\hat U - (\rho W)\,f\bigr]
  && \text{residual in the normal equations}\\
\text{2.}\quad & b  = -\,M\,Q^{T}Q\,r && \text{assemble, mask}\\
\text{3.}\quad & \text{solve } \mathcal{A}\,\delta\hat U = b && \text{preconditioned CG}\\
\text{4.}\quad & \hat U \leftarrow \hat U + \delta\hat U
\end{aligned}
$$

$f$ must carry the **same row weights** as the operator, or steps 1 and 3 refer
to different problems.

**Preconditioner.** Jacobi, $M^{-1} = 1/\operatorname{diag}(\mathcal{A})$ on free
DOFs and exactly zero on prescribed ones. The diagonal is computed in closed
form. For a row $r$ with field coefficients $a, b, c_{\text{val}}$,

$$
\frac{\partial R_r(p,q)}{\partial U_v(i,j)}
 = a\,D_{pi}\,\text{fac}_x\,\delta_{qj}
 + b\,D_{qj}\,\text{fac}_y\,\delta_{pi}
 + c_{\text{val}}\,\delta_{pi}\delta_{qj},
$$

non-zero only on the row $p=i$ or the column $q=j$, so

$$
\operatorname{diag}(\mathcal{A})_{(i,j,v)}
= \sum_r \rho_r\Bigl[\,
  W_{ij}\bigl|a D_{ii}\text{fac}_x + b D_{jj}\text{fac}_y + c_{\text{val}}\bigr|^2
  + a^2\text{fac}_x^2 \!\!\sum_{p\neq i}\! W_{pj}D_{pi}^2
  + b^2\text{fac}_y^2 \!\!\sum_{q\neq j}\! W_{iq}D_{qj}^2 \Bigr]
$$

— $O(n^2)$ per element instead of $O(n^2)$ *operator applications*, verified to
$3\times10^{-16}$ against a probing reference.

**CG tolerance** $10^{-6}$: measured to give accuracy identical to $10^{-12}$
within 1% at $\sim$40% of the iterations.

**Convergence safeguard.** CG's recursive residual drifts over thousands of
iterations, so on claimed convergence the **true** residual $b - \mathcal{A}\delta\hat U$
is computed; if it fails the test, the recursion restarts from it.

### Boundary conditions

| code | condition | prescribed |
|---|---|---|
| 1, 2, 3 | wall / lid / prescribed velocity | $u, v, w$ (3D freezes $w$ too) |
| 4 | outflow, $p = 0$ | $p$ |
| 5 | symmetry | $v, \omega_z$ |
| 0 | none / periodic seam | — |

Periodicity in $x$/$y$ is **connectivity**, not a boundary condition: setting
`mesh.periodic_x` merges the seam's global nodes, so it arrives through $Q^{T}Q$
and needs no mask. Vorticity is left free everywhere.

Two constraints that are *not* geometric:

* **A closed domain needs a pressure pin**, since constant $p$ at $k=0$ is a null
  vector. The pin must zero **every local copy** of the node — on a periodic seam
  it is shared 2–4 ways, and pinning one copy leaves the DOF free *and* makes
  $\mathcal{A}$ non-symmetric.
* **$k=0$ and Nyquist modes must be real.** Their imaginary halves are prescribed
  to zero *everywhere*, interior included: `irfft` discards those components, so
  anything the solver puts there is invisible in physical space.

## 8. The convective term

$N = -(\mathbf{u}\cdot\nabla)\mathbf{u}$, evaluated pseudo-spectrally with
**3/2-rule dealiasing in $z$**:

1. derivatives $\partial_x\mathbf{u},\ \partial_y\mathbf{u},\ \mathrm{i}k\mathbf{u}$
   in **mode space** (exact; commutes with the transform);
2. pad to $3N_z/2$ modes, inverse transform to physical $z$;
3. form $u\,\partial_x u + v\,\partial_y u + w\,\partial_z u$ pointwise;
4. forward transform, truncate back to $N_z$ modes.

Only the **products** alias, so only they need the padded grid. There is no
dealiasing in $(x,y)$ — those products are formed by collocation on the GLL grid.

**CFL.** Convection is explicit, so $\Delta t$ is limited. RKW3's imaginary-axis
stability interval is $\sqrt{3} \approx 1.732$. (Forward Euler and AB2 have *no*
imaginary-axis interval at all — an implicit scheme with explicit convection is
not automatically stable.)

## 9. One step, end to end

$$
\textbf{for } k = 0,1,2:
$$

$$
\begin{aligned}
N^{k} &\leftarrow -(\mathbf{u}\cdot\nabla)\mathbf{u}
  &&\text{dealiased in } z \quad (\S8)\\
L^{k} &\leftarrow -\bigl(\nabla p + \nu\nabla\times\boldsymbol{\omega}\bigr)
  &&\text{current state, } c=0 \quad (\S4)\\[4pt]
f_{\text{mom}} &\leftarrow c_k\bigl[\hat U + \Delta t(\gamma_k N^{k} + \zeta_k N^{k-1} + \alpha_k L^{k})\bigr]
  &&(\S6)\\
f_{\text{con}} &\leftarrow 0 \quad(\text{or } \kappa_p p \text{ for } R_0 \text{ with AC})\\[4pt]
r &\leftarrow \mathcal{L}_0^{T}\bigl[(\rho W)\mathcal{L}_0\hat U - (\rho W)f\bigr] &&(\S7)\\
b &\leftarrow -M\,Q^{T}Q\,r\\
\delta\hat U &\leftarrow \operatorname{PCG}(\mathcal{A}, b)
  &&\text{batched over all } k_z\\
\hat U &\leftarrow \hat U + \delta\hat U\\[4pt]
N^{k-1} &\leftarrow N^{k} &&\text{two registers only}
\end{aligned}
$$

The PCG advances **every Fourier mode simultaneously**, with per-mode
convergence: each mode carries its own residual and scalars, and the loop exits
when the slowest has converged.

## 10. Parameters

| symbol | meaning | production value |
|---|---|---|
| $\nu$ | kinematic viscosity | $1/Re$ |
| $\Delta t$ | time step | CFL-limited, $\mathrm{CFL} < \sqrt3$ |
| $c_k$ | $1/(\beta_k\Delta t)$, momentum mass coefficient | worst stage $6/\Delta t$ |
| $\kappa_p$ | artificial compressibility | $\mathbf{0}$ (off) |
| $\rho_r$ | row weights | **legacy**: $1/c^2$ on rows 4–6 |
| — | CG relative tolerance | $\mathbf{10^{-6}}$ |
| $N$ | GLL polynomial order per element | 6–16 |
| $N_z$ | $z$ points $\to N_z/2+1$ modes | 16–128 |

---

## Appendix: the system at $k_z = 0$

At $k = 0$ every $\mathrm{i}k$ term vanishes and the system **splits**:

$$
\underbrace{(u,\ v,\ \omega_z,\ p)}_{\text{the 2D VVP system exactly}}
\quad\oplus\quad
\underbrace{(w,\ \omega_x,\ \omega_y)}_{\text{decoupled, homogeneous}}
$$

from rows $R_0, R_3, R_4, R_5$ and $R_1, R_2, R_6, R_7$ respectively.

This is the basis of the Stage 1 / M2 gate: at $k=0$ the 3D code must reproduce
`lssem2d` exactly, which is the only test that can catch a *consistently* wrong
operator. It is equally the reason $k=0$ tests are **blind to every
$\mathrm{i}k$ term** — the transverse fields and the whole $z$-coupling need a
non-zero mode to be exercised at all.
