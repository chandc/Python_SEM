# The 3D LSSEM formulation: equations solved and time-marching steps

What `lssem3d` actually solves, written out. Every equation below is transcribed
from the code, not from the derivation that preceded it — `operator.py`
`apply_L0_complex` for the rows, `timestep.py` for the coefficients, and the
drivers for the stage assembly. Where the code and an earlier write-up
disagreed, the code won.

Companion documents: [3D_STATUS.md](./3D_STATUS.md) (what has been measured),
[3D_DEVELOPMENT_PLAN.md](./3D_DEVELOPMENT_PLAN.md) (the plan and its gates).

---

## 1. Continuous problem

Incompressible Navier–Stokes on a domain periodic in `z`:

```
∂u/∂t + (u·∇)u = −∇p + ν∇²u
∇·u = 0
```

with `u = (u, v, w)`, constant `ν`, and density absorbed into `p`.

## 2. Velocity–vorticity–pressure (VVP) first-order form

Introduce the vorticity `ω = ∇×u` as an independent unknown. For a
divergence-free field `∇²u = −∇×ω`, so the viscous term becomes a **first-order**
expression in `ω`:

```
∂u/∂t + (u·∇)u = −∇p − ν(∇×ω)
∇·u   = 0                        continuity
ω − ∇×u = 0                      vorticity definition  (3 equations)
∇·ω   = 0                        vorticity divergence
```

**Seven unknowns** — `u, v, w, ω_x, ω_y, ω_z, p` — and **eight equations**. The
system is overdetermined by one, which is deliberate: `∇·ω = 0` is implied by
`ω = ∇×u` but is retained as an independent row because the least-squares
functional benefits from it. The whole system is first order, which is what
allows a single C⁰ spectral-element space for every variable.

## 3. Fourier expansion in z

Every field is expanded on the periodic direction,

```
f(x, y, z, t) = Σ_k f̂_k(x, y, t) · e^{i k z},     k = 2πm/L_z
```

so `∂/∂z → i k`, and **the eight equations decouple into an independent system
per wavenumber `k`** — except through the convective term, which is quadratic and
therefore couples modes. That decoupling is the entire reason for the Fourier
direction: the implicit solve is embarrassingly parallel across `k`.

Real input means Hermitian symmetry, so only `k ≥ 0` is stored (`rfft`), with
`nk = N_z/2 + 1` modes. The `k = 0` and Nyquist modes must have **real**
coefficients; that is enforced, not merely assumed (see §7).

## 4. The eight rows, per mode

With `∂ₓ`, `∂_y` the spectral-element derivatives and `c`, `κ_p` defined in §5,
the residual rows `R = L₀ Û` are exactly:

| # | row | expression |
|---|---|---|
| 0 | continuity (+ AC) | `κ_p·p + ∂ₓu + ∂_y v + i k w` |
| 1 | `ω_x` definition | `∂_y w − i k v − ω_x` |
| 2 | `ω_y` definition | `i k u − ∂ₓw − ω_y` |
| 3 | `ω_z` definition | `∂ₓv − ∂_y u − ω_z` |
| 4 | `x`-momentum | `c·u + ∂ₓp + ν(∂_y ω_z − i k ω_y)` |
| 5 | `y`-momentum | `c·v + ∂_y p + ν(i k ω_x − ∂ₓω_z)` |
| 6 | `z`-momentum | `c·w + i k p + ν(∂ₓω_y − ∂_y ω_x)` |
| 7 | vorticity divergence | `∂ₓω_x + ∂_y ω_y + i k ω_z` |

Rows 4–6 are `c·u + ∇p + ν(∇×ω)` component-wise; rows 1–3 are `ω − ∇×u`.

**`κ_p` (artificial compressibility)** is a numerical term on the continuity row,
not physics. It is **off (`κ_p = 0`) in the production recipe** — with it on, the
row solves `div u = −κ_p(p − p_prev)`, which costs 5–7 orders of magnitude of
accuracy (3D_STATUS §7E). It is retained because it is a large conditioning aid
where accuracy permits (§7H).

### Split-real representation

`L₀` has complex coefficients (the `i k` terms), so `L₀ᵀL₀` would be **Hermitian,
not symmetric**. To keep a real symmetric operator for CG, the 7 complex fields
are stored as **14 real fields** (real parts, then imaginary), and the 8 complex
rows as 16 real rows. A complex coefficient `α` becomes the real 2×2 block
`[[Re α, −Im α], [Im α, Re α]]`, whose transpose corresponds to the complex
**conjugate** — so in `L₀ᵀ`, `(i k) → −(i k)` while real `c`, `ν` are unchanged.

## 5. Least-squares discretisation

The functional minimised is

```
J(Û) = ∫ Σ_r  ρ_r · |R_r(Û)|²  dΩ
```

with `ρ_r` the **row weights** (§5.1) and the integral evaluated by GLL
quadrature with weights `W`. Its normal equations give the operator CG solves:

```
A = M Qᵀ Q L₀ᵀ (ρ W) L₀ M
```

| factor | meaning | consequence of omitting it |
|---|---|---|
| `L₀` | the eight rows above | — |
| `W` | GLL quadrature weights | minimises the *nodal* sum instead of the integral |
| `ρ` | row weights | see §5.1 |
| `Qᵀ Q` | gather–scatter (C⁰ assembly) | element-local, massively under-determined |
| `M` | boundary mask | BCs unimposed |

`W` and `ρ` appear in the **forward** operator only; `apply_LT` is the unweighted
transpose, so the product is the normal operator of `J`.

### 5.1 Row weights — a choice, not a constant

`lssem2d` writes the momentum row as `a_mass·u + a_flux·N(u)` with the constraint
rows at weight 1, so **`a_flux` is the least-squares weight of momentum against
the constraints**. Two settings are in use:

| setting | `a_mass` | `a_flux` | used by |
|---|---|---|---|
| **legacy** | 1 | `dt` | Chan (1996) Stokes decay; **the 3D production recipe** |
| `w_mom` = 1 | `fac1/dt` | 1 | the 2D cavity / BFS studies |

In the code the legacy scaling is applied as `ρ = 1/c²` on rows 4–6
(`operator.momentum_row_weights`), which divides the momentum row by `c` so its
mass coefficient is 1 — every row then O(1) in the velocity. Without it,
momentum outweighs continuity by `c² ≈ 10⁶` and the minimiser effectively ignores
`div u`.

**Which weighting is right is problem-dependent** (3D_STATUS §7A.2b): legacy wins
in the channel and on the Taylor–Green/Stokes benchmarks; `w_mom` = 1 wins on the
lid-driven cavity.

## 6. Time marching: RKW3 / Crank–Nicolson

Convection is **explicit** (3-stage low-storage Runge–Kutta), the linear operator
**implicit** (Crank–Nicolson-style stage split). Coefficients are
Spalart–Moser–Rogers (1991):

```
γ = ( 8/15,   5/12,  3/4 )        explicit (convection)
ζ = ( 0,    −17/60, −5/12)        explicit, previous stage
α = (29/96,  −3/40,  1/6 )        implicit, old state
β = (37/160,  5/24,  1/6 )        implicit, new state
```

subject to the consistency identity **`α_k + β_k = γ_k + ζ_k`**, asserted in
exact rational arithmetic at import — mis-transcribing the table silently costs
an order and is invisible until a convergence study.

### The stage equation

Writing `N(u) = −(u·∇)u` (explicit) and `L(Û)` for the linear terms
`−∇p − ν(∇×ω)` (implicit), each stage `k = 0, 1, 2` advances

```
Û^k = Û^{k−1} + Δt·[ γ_k N^{k−1} + ζ_k N^{k−2} + α_k L^{k−1} + β_k L^k ]
```

`ζ₀ = 0`, so **the scheme is self-starting** — the first stage never reads
history, and no startup procedure is needed.

### Rearranged for the solve

`L^k` is the unknown. Dividing by `β_k Δt` puts the stage in the form the
operator of §4 expects:

```
c_k · Û^k − L(Û^k) = c_k · [ Û^{k−1} + Δt( γ_k N^{k−1} + ζ_k N^{k−2} + α_k L^{k−1} ) ]
                     └──────────────────── the momentum right-hand side, f ────────────┘

            c_k = 1 / (β_k · Δt)
```

**`c_k` is the mass coefficient in rows 4–6**, and it is the 3D analogue of the
2D `a_mass`. Note `1/β = (4.32, 4.80, 6.00)`: the worst stage carries **`6/Δt`**,
four times BDF2's `1.5/Δt` at the same step — budget stability against `6/Δt`,
never `1.5/Δt`.

The right-hand side for the **constraint rows** (0–3, 7) is **zero**, except that
with AC on, row 0 carries `κ_p·p^{k−1}`.

### Order of the scheme

Second order, by construction, and measured:

| configuration | order |
|---|---|
| explicit half alone (the `γ/ζ` table) | **3.025** |
| implicit half alone (Crank–Nicolson) | **2.002** |
| mixed, scalar model | 2.189 |
| **full 3D PDE, Stokes decay** | **2.00** |
| **full 3D PDE, Taylor–Green (convection active)** | **2.00** |

RK3's third order lives in the convective half alone; **CN caps the mixed scheme
at 2**. A gate demanding 3.0 overall is unachievable by any correct
implementation.

## 7. The stage solve

Each stage is one linear solve. Because the boundary conditions are
**inhomogeneous** (a moving lid, a prescribed inflow), the system is solved as a
**defect correction** rather than directly — solving `A Û = L₀ᵀ W f` with a masked
`A` is well-posed but is a different problem, and produced a motionless cavity
when it was done that way.

Given the current iterate `Û` and the stage right-hand side `f`:

```
1.  r  = L₀ᵀ [ (ρ W)·L₀ Û  −  (ρ W)·f ]          residual in the normal equations
2.  b  = −M · Qᵀ Q r                              assemble, mask
3.  solve   A δÛ = b        by preconditioned CG
4.  Û ← Û + δÛ
```

`f` must carry the **same row weights** as the operator, or steps 1 and 3 refer to
different problems.

**Preconditioner:** Jacobi, `M⁻¹ = 1/diag(A)` on free DOFs and **exactly zero** on
prescribed ones. The diagonal is computed in closed form
(`jacobi_diagonal_analytic`), verified to 3e−16 against a probing reference.

**CG tolerance:** `1e−6` — measured to give accuracy identical to `1e−12` within
1% at ~40% of the iterations.

**Convergence safeguard:** CG's recursive residual drifts over thousands of
iterations, so on claimed convergence the **true** residual `b − Aδ Û` is
computed; if it fails the test, the recursion restarts from it.

### Boundary conditions

| code | condition | prescribed |
|---|---|---|
| 1, 2, 3 | wall / lid / prescribed velocity | `u, v, w` (3D freezes `w` too) |
| 4 | outflow, `p` = 0 | `p` |
| 5 | symmetry | `v, ω_z` |
| 0 | none / periodic seam | — |

Periodicity in `x`/`y` is **connectivity**, not a boundary condition: setting
`mesh.periodic_x` merges the seam's global nodes, so it arrives through `Qᵀ Q`
and needs no mask. Vorticity is left free everywhere — the least-squares system
determines it.

Two constraints that are *not* geometric:

* **A closed domain needs a pressure pin**, since constant `p` at `k = 0` is a
  null vector. The pin must zero **every local copy** of the chosen node — on a
  periodic seam that node is shared 2–4 ways, and pinning one copy leaves the DOF
  free *and* makes `A` non-symmetric.
* **`k = 0` and Nyquist modes must be real.** Their imaginary halves are
  prescribed to zero *everywhere*, interior included: `irfft` discards those
  components, so anything the solver puts there is invisible in physical space.

## 8. The convective term

`N = −(u·∇)u`, evaluated pseudo-spectrally with **3/2-rule dealiasing in z**:

```
1.  derivatives ∂ₓu, ∂_y u, i k u        in MODE space (exact; commutes with the transform)
2.  pad to 3N_z/2 modes, inverse transform to physical z
3.  form the products u·∂ₓu + v·∂_y u + w·∂_z u   pointwise
4.  forward transform, truncate back to N_z modes
```

Only the **products** alias, so only they need the padded grid; taking the
derivatives in mode space first keeps them exact. There is no dealiasing in
`(x, y)` — those products are formed by collocation on the GLL grid.

**CFL:** convection is explicit, so `Δt` is limited. RKW3's imaginary-axis
stability interval is **√3 ≈ 1.732**. (Forward Euler and AB2 have *no* imaginary-
axis interval at all — an implicit scheme with explicit convection is not
automatically stable.)

## 9. One step, end to end

```
for each RKW3 stage k = 0, 1, 2:

    N^k  ←  −(u·∇)u             from the current state, dealiased in z   (§8)
    L^k  ←  −(∇p + ν∇×ω)        from the current state, c = 0            (§4)

    f_mom  ←  c_k·[ Û + Δt(γ_k N^k + ζ_k N^{k−1} + α_k L^k) ]           (§6)
    f_con  ←  0     (or κ_p·p for row 0 when AC is on)

    r   ←  L₀ᵀ[ (ρW)·L₀Û − (ρW)·f ]                                      (§7)
    b   ←  −M·QᵀQ r
    δÛ  ←  PCG(A, b)            batched over all k_z modes at once
    Û   ←  Û + δÛ

    N^{k−1} ← N^k               low-storage handoff: two registers only
```

The PCG in the innermost line advances **every Fourier mode simultaneously**, with
per-mode convergence — each mode carries its own residual and scalars, and the
loop exits when the slowest has converged.

## 10. Parameters

| symbol | meaning | production value |
|---|---|---|
| `ν` | kinematic viscosity | `1/Re` |
| `Δt` | time step | CFL-limited, `< √3` |
| `c_k` | `1/(β_k Δt)`, the momentum mass coefficient | worst stage `6/Δt` |
| `κ_p` | artificial compressibility | **0** (off) |
| `ρ` | row weights | **legacy**: `1/c²` on rows 4–6 |
| CG tol | relative residual | **1e−6** |
| `N` | GLL polynomial order per element | 6–16 |
| `N_z` | z points → `N_z/2 + 1` modes | 16–128 |

---

## Appendix: what each row looks like at `k_z` = 0

At `k = 0` every `i k` term vanishes and the system **splits**:

```
(u, v, ω_z, p)     the 2D VVP system exactly            ← rows 0, 3, 4, 5
(w, ω_x, ω_y)      a decoupled, homogeneous subsystem   ← rows 1, 2, 6, 7
```

This is the basis of the Stage 1 / M2 gate: at `k = 0` the 3D code must
reproduce `lssem2d` exactly, which is the only test that can catch a
*consistently* wrong operator. It is also why `k = 0` tests are blind to every
`i k` term — the transverse fields and the whole z-coupling need a non-zero mode
to be exercised at all.
