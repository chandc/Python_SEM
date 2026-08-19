# Implementing Dong's convective-like energy-stable outflow condition in the 2D LSSEM

Design note, 2026-08-18. How the open boundary condition of

> S. Dong, *A Convective-like Energy-Stable Open Boundary Condition for
> Simulations of Incompressible Flows*, arXiv:1506.01320v1 (2015).
> Fetch: see `reference/README.md`.

would be added to `lssem2d`. **Nothing here is implemented yet.** This is the
design, the places it does *not* transfer cleanly, and the test ladder.

---

## 0. Read this first: it is probably not a fix for our instability

The obvious motivation is `AMASS_RESOLVED.md`, which concludes the `a_mass`
instability is an **outflow-boundary phenomenon**. Dong's paper is about outflow
instability. The inference is tempting and, on our own evidence, wrong.

Dong's backflow instability is driven by the term `½|u|²(n·u)` in the energy
balance and is gated by `Θ₀(n,u)`, which is **identically zero unless there is
backflow** (`n·u < 0`). Ours appears:

| our failure | Dong's mechanism requires |
|---|---|
| Re = 100, laminar | moderate-to-high Re (several hundred +) |
| parabolic inlet, exact solution representable, no vortices | strong vortices or backflow at the boundary |
| **with convection removed entirely** (Stokes probe, blew up at step 29) | convection — the whole term vanishes without it |

An energy mechanism carried by `½|u|²(n·u)` cannot act in an operator that has no
convection at all. **These are two different outflow pathologies.** Implement
this condition because it is a better outflow condition for high-Re flows with
vortices leaving the domain — not expecting it to move the `a_mass` threshold.

---

## 1. What the condition is, in our notation

For an outflow plane at `x = x_max` with outward normal `n = (1, 0)`, Dong's
equation (4) is a **vector** condition — in 2D, two scalar equations:

```
R_x = νD₀ ∂u/∂t − p + ν ∂u/∂x − ½[ (u²+v²) + u·u ]·Θ₀ = 0
R_y = νD₀ ∂v/∂t     + ν ∂v/∂x − ½[   0     + u·v ]·Θ₀ = 0
```

using `n·u = u` and `|u|²n = (u²+v², 0)` for this normal. The switch is

```
Θ₀(n,u) = ½(1 − tanh(u/(δU₀)))       →  1 where u < 0, 0 where u > 0
```

`D₀` is the one free parameter, and `1/D₀` plays the role of a convection
velocity: run once with `D₀` = 0, measure the mean outflow speed `U_c`, set
`D₀` = 1/`U_c`.

**ADN counting works out exactly.** `OUTFLOW_BC_STUDY.md` records that a 2D
boundary point needs **2 scalar conditions**. Dong's condition is one vector
equation = 2 scalars, so it replaces P+Z (`p = 0` *and* `∂ω/∂x = 0`) one-for-one.
No under- or over-determination.

---

## 2. Why this is *easier* in LSSEM than in Dong's own framework

Dong spends most of his §2.2–2.4 on the splitting problem: the condition couples
`u`, `p` and `∂u/∂t`, so in a velocity-correction scheme he must derive a Robin
condition for pressure at the pressure sub-step and another for velocity at the
velocity sub-step, and he notes an explicit treatment of `∂u/∂t` "does not work,
and is unstable unless `D₀` is very small".

**None of that applies here.** We do not split. The least-squares functional
accepts any residual, including a boundary one:

```
J = ∫_Ω [ w_mom²(N₁²+N₂²) + (div u)² + (ω+u_y−v_x)² ] dΩ
  + w_obc² ∫_∂Ωo [ R_x² + R_y² ] ds          ← new
```

`u` and `p` are already solved as one coupled system, so a condition coupling
them is not awkward — it is one more row. This is a genuine structural advantage
of the formulation for this particular boundary condition.

**But we also lose Dong's theorem.** His energy stability is proved for the
primitive-variable weak form. Minimising `R_x² + R_y²` in a least-squares sense
does *not* inherit that proof: least squares enforces the condition
approximately, weighted by `w_obc`, and the discrete energy argument would have
to be redone. Expect the *condition*, not the guarantee.

---

## 3. Code changes

### 3.1 The boundary residual rows

Add two rows to `apply_L`, nonzero only on outflow nodes. The existing rows are
multiplied by the **volume** weight `mesh.wq = jac·wᵢ·wⱼ`; these need the
**surface** weight along the edge:

```python
# edge at i = n-1 (x = x_max):  ds = (hy/2)·w_j
ws = 0.5*mesh.hy[e]*w[j]
```

so a new `mesh.wq_edge` (or a per-edge list) has to be built alongside `wq`.
`mesh.py` already stores `bc[e]` per edge, so the outflow edges are identifiable
without new connectivity.

Rows, with `a_obc = w_obc` the new weight and `c_b = νD₀·fac1/dt` the boundary
time-derivative coefficient:

```python
su4 = a_obc*( c_b*u - p + nu*u_x - Ex_explicit ) * ws     # only on bc==4 edges
su5 = a_obc*( c_b*v      + nu*v_x - Ey_explicit ) * ws
```

with the BDF history `−νD₀·Σ αₘ u^{n−m}/dt` folded into the right-hand side
exactly as `su_history` already does for the momentum rows (`step_bdf` builds
that; the boundary term joins it).

### 3.2 The nonlinear switch term

`E(n,u) = ½[|u|²n + (n·u)u]Θ₀(n,u)` is quadratic **and** carries a `tanh`.
Two options:

| | pro | con |
|---|---|---|
| **Explicit (lagged)**, `E(n, u*)` with `u* = 2uⁿ − uⁿ⁻¹` | matches Dong exactly; no new linearisation; `E` becomes a known RHS vector | one more explicit term, so it may impose its own step restriction |
| Linearised into `apply_L` | fully implicit, consistent with our Newton loop | must differentiate `Θ₀`; `dΘ₀/d(n·u) = −1/(2δU₀)·sech²(·)` is sharply peaked for small δ and will hurt conditioning |

**Start explicit.** It reproduces Dong's own treatment, it is a strictly smaller
change, and Stage 2 below can measure whether the implicit version is needed.

### 3.3 Boundary-condition masking — the part that changes behaviour

Currently `bc == 4` **freezes the pressure** (`p = 0` Dirichlet) in both
`SolverState.get_global_mask` and `bc.apply_mask`. Under Dong's condition:

* **`p` must be freed at the outflow.** It is now determined weakly by `R_x`.
  Both mask paths must change together — `OUTFLOW_BC_STUDY.md` §3 records that
  they once disagreed on `bc == 4` and a `p = 0` outlet consequently imposed
  *nothing*, measured `max|p|` = 4.87e−01 on a plane where `p = 0` was claimed.
  Introduce `bc == 6` for the Dong outlet rather than mutating `bc == 4`, so the
  existing P+Z results stay reproducible.
* **The `∂ω/∂x = 0` wrapper is dropped.** Every driver that patches
  `S.apply_bc` with a `bc2` closure (`gartling_run.py`, `pois_ac.py`,
  `stokes_amass_probe.py`) would skip that patch for `bc == 6`.
* **Pressure level.** With `p` no longer pinned anywhere, check whether the
  system retains a null space. `lssem3d`'s tests showed a constant-`p` null
  vector when nothing fixes the level; here `R_x` contains `−p`, so it should be
  fixed — but **assert it** rather than assume, with the same probe used in
  `lssem3d/tests/test_solver3d.py`.

### 3.4 Jacobi preconditioner

`compute_jacobi` builds the diagonal analytically and the file warns it "MUST
stay in step with" `apply_L`. The new rows contribute
`a_obc²·(c_b² + ν²·(∂ₓ)² + 1)·ws²` on outflow nodes — for `u`, `v` and `p`
respectively. Omitting this is not fatal (the preconditioner degrades, it does
not become wrong), but it will show up as an iteration-count regression on
exactly the flows this is meant to improve.

---

## 4. The new free parameter, and why it deserves suspicion

`w_obc` — the weight of the boundary rows against the volume rows — is a
**third** weighting parameter alongside `w_mom` and `w_mass`. This project's
entire history says to treat that with care:

* `GARTLING_VALIDATION.md`: `a_mass` = `w_mass·fac1/dt` has a hard stability
  threshold, and the constraint rows carry weight exactly 1.
* `AMASS_RESOLVED.md`: on a flow with a real residual, the weights decide which
  row gets sacrificed.

A boundary row integrated over a 1D edge is dimensionally unlike a volume row,
so `w_obc` = 1 is not a neutral default — it is an arbitrary one. Expect to need
a sweep, and expect the answer to depend on `Re` and on the element size at the
outlet.

Also note `c_b = νD₀·fac1/dt` is a boundary `a_mass`. With `ν` = 1/Re and
`D₀` = 1/`U_c` it is `fac1/(Re·U_c·dt)`. At Gartling's Re = 800 and `dt` = 0.05,
that is ≈ 0.04 — small, unlike the volume `a_mass` = 30 at the same `dt`. The two
time-derivative terms in the same functional would then differ by three orders of
magnitude. **Whether that is benign is exactly the kind of question this project
has repeatedly got wrong by assumption**; measure it.

---

## 5. Test ladder

Each stage has a criterion that can fail.

| stage | test | criterion |
|---|---|---|
| **0** | `D₀` = 0, `Θ₀` ≡ 0 → condition reduces to traction-free `−p + ν∂u/∂x = 0` | Poiseuille on the 12×2 N=10 channel: `Δp` = 1.44000 exact, rms `div u` ≤ 1e−08, as `ARTIFICIAL_COMPRESSIBILITY.md` §3 |
| **1** | adjoint/symmetry with the new rows | `LᵀL` symmetric to ≤1e−12; the `lssem3d` experience says add a **negative control** that the new rows actually change the operator |
| **2** | Gartling Re = 800, `D₀` = 0 vs current P+Z | reattachment within the P+Z result's own spread of Gartling's 6.10 (we measure 6.100 at N=7) |
| **3** | `w_obc` sweep at fixed everything else | reattachment vs `w_obc`; flat region identified, as `gartling_wmom_plot.py` does for `w_mom` |
| **4** | `Θ₀` term on, on a flow with genuine backflow | a case that diverges without it survives with it — **this is the whole point of the condition and needs a case that actually fails first** |
| **5** | `D₀` > 0, `D₀` = 1/`U_c` | Dong reports global quantities insensitive to `D₀` (≤2.7% on `C_d` at Re = 10000) and the effect confined to smoothness near the outlet; check both |

**Stage 4 needs a failing case to be meaningful.** Our Gartling and Armaly runs
at Re = 800 do not exhibit the backflow blow-up Dong targets — they fail for the
`a_mass` reason instead, which this condition will not fix (§0). A cylinder wake
at Re ≳ 2000, or the BFS at markedly higher Re, would be needed to exercise it.
Without such a case, Stages 4–5 test nothing, and the honest outcome is that we
have implemented a better-posed outflow condition whose headline benefit we
cannot demonstrate on any flow currently in this repo.

---

## 6. Recommendation

Worth doing as far as **Stage 3**: the traction-free/`D₀`=0 form is a
better-founded outflow condition than P+Z, it costs two rows and a mask code, and
`OUTFLOW_BC_STUDY.md` already found the free-outflow variant to be
tolerance-decided (4 of 7 outcomes invert between `cg_tol` 1e−12 and 1e−6), which
is a bad property to keep.

Stages 4–5 should wait for a flow that needs them. And none of it should be
expected to touch the `a_mass` threshold — §0.
