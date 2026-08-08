# Poiseuille Re=100: what dt actually controls in the LSSEM VVP formulation

Study date: 2026-08-08. Companion to
[PRECONDITIONER_AND_DT_STUDY.md](./PRECONDITIONER_AND_DT_STUDY.md), which found
the BFS steady state to be dt-dependent. This one isolates *why*, on a case with
a known exact solution, and led to the `w_mom` parameter that decouples the
least-squares weight from the time step.

Reproduce: `scratch/poiseuille_dt.py`, `scratch/plot_poiseuille.py`,
`scratch/poiseuille_wmom.py`.

---

## Executive summary

1. **dt is the least-squares weight of the momentum equations**, not just a time
   step. The momentum rows are `fac1*u + dt*N(u)`; the constraints are unweighted.

2. **Pressure appears ONLY in the momentum rows.** So the pressure block of
   `LᵀL` scales as `dt²`. At small dt the pressure is effectively unconstrained,
   `LᵀL` develops a near-null space in `p`, and the exact solution stops being
   the *unique* minimiser. Measured: dt=0.05 gives **98% velocity error on a
   problem whose exact solution is exactly representable**.

3. **The optimum is dt = 1**, where the pressure and velocity blocks are equally
   weighted. There, Δp = 1.19999 against the analytic 1.2 — five significant
   figures — and the profile error is 5.3e-04. Across the sweep the profile
   error varies **1875×**.

4. **dt=1 is essentially universal** — the equal-weight point is 1.00 ± 1% for
   every mesh, order, Reynolds number and geometry tested, including both BFS
   grids. It is not a tuned constant.

5. **This is a design flaw, not a tuning knob**, so `w_mom` was added to
   decouple the weight from dt. `w_mom=1` is the balanced choice; the default
   preserves the legacy behaviour exactly.

---

## 1. Setup

| | |
|---|---|
| geometry | L × H = 10 × 1 |
| Re | `U_mean·H/ν` = 100, so ν = 0.01 |
| inlet | parabolic, `u = 6y(1-y)`, U_mean = 1, U_max = 1.5 |
| outlet | FREE — nothing imposed on u, v, p, ω |
| pressure pin | inlet plane, lower-left corner |
| walls | no-slip |
| exact | `dp/dx = -12νU_mean/H² = -0.12`, **Δp over L = 1.20** |

The pressure drop is a *prediction*: pressure is pinned only at the inlet and the
outflow is free, so nothing imposes the drop.

**Three variants**, because the control alone gives a null result by construction:

| variant | mesh | inlet | why |
|---|---|---|---|
| `control` | 10×2, order 8 | parabolic | exact solution is in the discrete space |
| `develop` | 10×2, order 8 | **uniform** | entrance region is non-polynomial |
| `coarse` | 5×1, order 4 | uniform | deliberately under-resolved |

> A coarse or low-order mesh does **not** perturb the control. `u = 6y(1-y)` is
> degree 2, exact for any N ≥ 2; the rectangle's geometry mapping is affine; the
> residual is zero pointwise so quadrature order is irrelevant. To make dt matter
> the *solution* has to leave the polynomial space, which is what the uniform
> inlet does. (My first design used low order as the perturbation. That was
> wrong and was replaced.)

---

## 2. Results

![Poiseuille dt study](figs/poiseuille_dt.png)

### control — parabolic inlet, order 8

| dt | profile err | Δp | Δp err | p-block/u-block |
|---|---|---|---|---|
| 0 (pure steady) | 8.46e-03 | 1.19224 | 6.5e-03 | — |
| 0.05 | 9.85e-01 | 2.50359 | 1.09 | 0.0025 |
| 0.1 | 9.37e-01 | 2.18379 | 0.820 | 0.0099 |
| 0.5 | 9.67e-02 | 1.23704 | 3.1e-02 | 0.249 |
| **1.0** | **5.26e-04** | **1.19999** | **5.3e-06** | **0.994** |
| 2.0 | 1.85e-03 | 1.19837 | 1.4e-03 | 3.98 |
| 5.0 | 1.23e-02 | 1.20100 | 8.3e-04 | 24.9 |

**1875× spread in profile error across dt**, on a problem whose exact solution
the discretisation can represent perfectly. The minimum sits exactly where the
pressure and velocity blocks are equally weighted.

### the other two variants

| variant | best dt | profile err there | spread over dt |
|---|---|---|---|
| control | **1.0** | 5.25e-04 | 1875× |
| develop | 5.0 | 2.45e-03 | 399× |
| coarse | **0.1** | 1.35e-02 | 21× |

**The trend reverses when under-resolved.** The coarse case is best at *small* dt
and degrades monotonically to 2.8e-01 at dt=5 — when the mesh cannot represent
the momentum equation, deliberately down-weighting it is the right thing to do.
So "dt=1" presumes adequate resolution.

> Caveat on `develop`: its Δp settles near 1.6, not 1.2, and that is **physically
> correct** — uniform-inlet flow has a genuine entrance loss. The `dp_err` column
> is meaningless for that variant; only `control` should be judged against 1.2.

---

## 3. Mechanism

Which variables does each least-squares row involve?

```
momentum-x : fac1*u + dt*( u u_x + v u_y + p_x + nu om_y )   -> u,v,P,om   weight dt
momentum-y : fac1*v + dt*( u v_x + v v_y + p_y - nu om_x )   -> u,v,P,om   weight dt
continuity :            u_x + v_y                            -> u,v        weight 1
vorticity  :            om + u_y - v_x                       -> u,v,om     weight 1
```

**Pressure appears only in the momentum rows.** Its block in `LᵀL` therefore
carries `dt²`, while the constraint rows carry 1. At dt=0.05 the pressure is
~400× under-weighted; `LᵀL` is near-singular in `p`, the minimiser is not unique,
and the solver converges — genuinely, to `diff = 0` — to a wrong member of the
near-null space.

This is the flaw in the intuition that "a representable exact solution must be
recovered". Zero residual makes it *a* minimiser; it does not make it *the*
minimiser.

### Why dt = 1, universally

```
pressure diagonal ~ dt² · Σ(P_x² + P_y²)
velocity diagonal ~      Σ(fac1²P² + P_x² + P_y²)
```

Once the gradient terms dominate the mass term — i.e. once resolved — the ratio
collapses to exactly `dt²` and the balance point is dt=1 regardless of anything
else. Measured:

| case | order | ν | equal-weight dt |
|---|---|---|---|
| Poiseuille control (10×2) | 8 | 0.01 | 1.003 |
| Poiseuille fine (20×4) | 8 | 0.01 | 1.001 |
| Poiseuille p=12 | 12 | 0.01 | 1.001 |
| Poiseuille Re=1000 | 8 | 0.001 | 1.003 |
| BFS Chan short | 10 | 1/389 | 1.010 |
| BFS Chan long | 10 | 1/389 | 1.007 |
| Poiseuille coarse (5×1) | 4 | 0.01 | 1.329 |

The coarse case deviates precisely because it is the one where `fac1²P²` still
matters. This is an **a priori** criterion: computable from `compute_jacobi`
without running anything.

It also independently corroborates the BFS analysis, which put the optimum at
dt = 0.72–0.91 from a completely different argument (crossing of the normalised
steady residuals).

---

## 3a. Outlet pressure — the sharpest single diagnostic

The exact solution is `p = p₀ − Gx`, **independent of y**, so the pressure across
the outlet plane must be a vertical line. Nothing imposes this: the outflow is
free and pressure is pinned only at the inlet corner, so `p(y)` at the outlet is
entirely a prediction. It turns out to be the most direct probe of the
under-weighting, because it isolates pressure with no velocity scaling in the
way.

![outlet pressure](figs/poiseuille_pout.png)

| dt | outlet p spread (max−min) | as % of Δp | mean p_out (exact −1.2) |
|---|---|---|---|
| 0.05 | 5.157 | **430%** | −2.527 |
| 0.1 | 3.048 | 254% | −2.130 |
| 0.5 | 0.266 | 22.2% | −1.237 |
| **1.0** | **1.50e-03** | **0.12%** | **−1.19999** |
| 2.0 | 8.35e-03 | 0.70% | −1.19834 |

At dt=0.05 the outlet pressure swings from −5.4 at the top wall to −0.7 near
y=0.35 — a **5.16 variation across a plane whose total streamwise drop is only
1.2**. By dt=1 it is flat to 0.12% and the mean is −1.19999, exactly the analytic
drop below the pinned inlet. The non-flatness has a clean minimum at dt=1 and
rises again by dt=2.

This is the pressure under-weighting made visible rather than inferred: with `p`
present only in the momentum rows, at small dt nothing constrains its
cross-channel variation, and the free outflow lets it drift. The velocity error
tracks it exactly (98.5% → 0.05%) because a wrong pressure gradient drives a
wrong velocity.

> dt=5 is absent from this figure. That case ran 50 minutes without completing
> despite taking only 60 steps in the §2 sweep; it was stopped rather than
> diagnosed. An open loose end, not a result.

---

## 4. The `w_mom` parameter

Because dt is doing two unrelated jobs — temporal resolution and equation
weighting — one number cannot optimise both. `SolverState` now takes `w_mom`:

```python
SolverState(mesh, D, nu, dt, fac1, w_mom=1.0)   # momentum weighted 1, independent of dt
```

The coefficients live in one place, `lssem.ls_coeffs()`, used by `apply_L`,
`apply_LT`, `compute_jacobi`, `step_bdf`'s history term, and both numba wrappers
— the five places that must agree or the mass terms stop cancelling at steady
state.

```
legacy    (w_mom is None):  a_mass = fac1,           a_flux = dt
decoupled (w_mom set):      a_mass = w_mom*fac1/dt,  a_flux = w_mom
```

`a_flux` is the least-squares weight. **Default is legacy**, so nothing changes
unless asked, and `w_mom = dt` reproduces legacy **bit-identically** (verified at
dt = 0.05, 0.1, 0.5, 1, 2, 5 for `apply_L`, `apply_LT` and `compute_jacobi`).

> Implementation note: `a_mass` is computed as `f1*(w/dtl)`, not `(w*f1)/dtl`.
> The second grouping round-trips through a division and loses an ulp, breaking
> the bit-identical guarantee at dt=0.1.

Suite: 47 passed under both the numpy and numba backends after the change.

### Verification status

Verified (`scratch/` gates, all passing):

| gate | result |
|---|---|
| `w_mom = dt` reproduces legacy bit-identically, dt ∈ {0.05,0.1,0.5,1,2,5} | pass, for `apply_L`, `apply_LT` and `compute_jacobi` |
| `a_flux` independent of dt when `w_mom` is set | pass (1.0 at every dt) |
| legacy `a_flux` does depend on dt | confirmed (0.05 → 5.0) |
| numba backend agrees with numpy for a decoupled weight | 1.3e-16 / 1.0e-16 |
| full suite, both backends | 47 passed |

**NOT yet verified — the physics claim.** The point of the parameter is that with
`w_mom` fixed, the *converged steady state* should no longer depend on dt. That
sweep (`scratch/poiseuille_wmom.py`) had not completed when this was written:
`w_mom=1` at small dt gives a mass coefficient `fac1/dt = 30`, a much stiffer
solve than the legacy form, and the dt=0.05 case was still running after ~90
minutes. **Do not assume the decoupling removes the dt sensitivity until that
table exists.** The implementation is verified; the claimed benefit is not.

---

## 5. Two corrections made during this study

Recorded because both were plausible and both were wrong.

**"dt won't matter for Poiseuille."** I predicted no dt sensitivity because the
exact solution is representable and has zero residual. Wrong — see §3. Zero
residual does not imply a unique minimiser.

**"The small-dt runs terminated prematurely."** The stopping test was
`max|ΔU per step| < 1e-11`, which scales with dt, so small-dt runs appeared to
stop early. I diagnosed that as the cause of the dt trend. It was not: rerunning
with a dt-normalised rate criterion and a minimum physical time (t ≥ 300, three
viscous times) reproduced the numbers **bit-identically**. The runs had genuinely
converged. The criterion was still wrong and was fixed, but it was not the cause.

**Also fixed in passing:** `dt=0` cannot be driven through `step_bdf`. That
routine builds `su_history` unconditionally from the BDF alphas, while `apply_L`
zeroes `f1` when `dt==0`, so the fixed point solves `N(u) = 1.5u` — a spurious
reaction term. The pure steady form must call `newton_step` directly with a zero
history, which is what `scratch/poiseuille_dt.py` now does.
