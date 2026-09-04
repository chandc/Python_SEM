# The time-dependent FOSLS formulation: what `c` does, and every experiment run on it

*Status 2026-09-03. Companion to `3D_FORMULATION.md` (the equations),
`3D_STATUS.md` §7A/§7J/§7Q/§7S/§7T/§7U (the primary records), and
`PMG_ALGORITHM.md` (2D preconditioner study).*

The steady FOSLS system is well understood and behaves the way the literature
says it should. **Almost everything difficult about this code comes from making
it time-dependent**, and it all enters through one number. This document
collects what that number does and every computational experiment run to
characterise it, including the ones that failed and the ones whose harnesses
were wrong.

---

## 1. How time-dependence enters: the coefficient `c`

Semi-discretising in time with RKW3/Crank–Nicolson, the implicit part of stage
`k` contributes a **mass (reaction) term** to each momentum row:

$$
c\,\mathbf{u} + \nabla p + \nu(\nabla\times\boldsymbol\omega) = \mathbf{f}^k,
\qquad c \;=\; \frac{1}{\beta_k\,\Delta t}
$$

with `BETA = (37/160, 5/24, 1/6)`, so `1/β = (4.324, 4.800, 6.000)`
(`timestep.implicit_coeff`). Convection is **explicit** and lives entirely in
the right-hand side — which is what keeps the Fourier modes decoupled, and also
why the implicit operator is Stokes-like rather than linearised Navier–Stokes.

**`c` is not a tuning parameter. It is `1/(β·Δt)`, fixed by the timestep**, and
the timestep is fixed by the explicit convective CFL. The minimal channel runs
`dt = 8e-4`, so:

| | `c` |
|---|---|
| steady / 2D cavity study (§6.9, `dt`=1) | **1** |
| a "production" value used in earlier 3D studies | 525 |
| **minimal channel, dt = 8e−4, stage 0** | **5405** |
| stage 2 (`1/β` = 6.0) | 7500 |

Everything below is a statement about where on that scale you are sitting.

## 2. What large `c` does to the operator

`momentum_row_weights` scales the momentum rows by `1/c²` — "lssem2d's legacy
scaling: momentum rows divided by c so their mass coefficient is 1, matching the
constraint rows. Squared, because the functional squares the residual."

So the functional sees

$$
\frac{1}{c^{2}}\big(c\,\mathbf u + \nabla p + \nu\nabla\times\boldsymbol\omega\big)^{2}
\;=\;\Big(\mathbf u + \frac{\nabla p}{c} + \frac{\nu}{c}\nabla\times\boldsymbol\omega\Big)^{2}.
$$

The velocity term is O(1) — that is the point of the scaling. But **pressure now
enters at O(1/c), and the pressure block of `A` at O(1/c²)** — which at c = 5405
is 3.4e−8. Pressure is the only field with no other route into the functional
(it appears solely through ∇p in the momentum rows, and through i·k_z·p in the
w-momentum row, a path that **vanishes at k_z = 0**).

**Consequence, measured repeatedly and from several directions: at large `c` the
pressure becomes a near-null direction of the operator, worst at k_z = 0.**

### 2.1 Why small `dt` does NOT make the matrix diagonally dominant

The natural expectation is the opposite: `c = 1/(β Δt)` is large for small `dt`,
the mass term dominates, so `A` should be diagonally dominant and Jacobi should
do well. **The expectation is right about the raw operator and wrong about the
weighted one, and the gap between those two is the whole design.**

Measured (`scratch/diag_dominance.py`, 2×2 N=6, k_z=5.88; `dd` = median over a
field's rows of |a_ii| / Σ_{j≠i}|a_ij|, so dd > 1 means dominant):

| c | weighting | cond(D⁻¹A) | dd_u | dd_ωx | dd_p | softest mode |
|---|---|---|---|---|---|---|
| 1 | either | 6.54e+02 | 0.79 | 0.052 | 0.751 | ω |
| 100 | 1/c² | 1.68e+03 | 0.83 | 0.051 | 0.069 | ω |
| 100 | **raw** | 5.21e+03 | **7.44** | 0.116 | 0.069 | **p 100%** |
| 5405 | 1/c² | 1.74e+03 | 0.83 | 0.051 | 0.001 | ω |
| 5405 | **raw** | **1.52e+07** | **513** | 0.002 | 0.001 | **p 100%** |
| 50000 | 1/c² | 1.74e+03 | 0.83 | 0.051 | 0.000 | ω |
| 50000 | **raw** | **1.30e+09** | **4789** | 0.000 | 0.000 | **p 100%** |

**Unweighted, the velocity rows do become strongly diagonally dominant** —
dd_u = 4789, growing like c², exactly as the mass-term argument predicts. **And
the operator is catastrophically ill-conditioned anyway**: 8,750× worse than
weighted at c=5405, and 748,000× worse at c=50000. Dominance in one block buys
nothing when it arrives by making the other blocks relatively invisible —
dd_p and dd_ωx both go to zero and the softest mode becomes 100% pressure.

Note this is measured **after** Jacobi scaling: cond(D⁻¹A) already has the
diagonal normalised. That is the point. **The 1/c² factor is a ROW (equation)
rescaling of the least-squares system, and row scaling of `L` changes
`A = LᵀρWL` in a way no diagonal preconditioner can reproduce.** Jacobi rescales
unknowns; it cannot rebalance which equations the functional weighs.

With the weighting, `A` tends to a **fixed** operator as dt → 0 — an O(1) mass
matrix on u from the momentum rows, plus the c-independent derivative-squared
terms from the constraint rows — so dd_u saturates at 0.83, dd_ωx is flat at
0.051, and cond saturates at ~1.7e3 instead of diverging. Nothing becomes
dominant, and nothing blows up either. Only pressure degrades, for the reason
in §2.

This also explains §4.4 from the other side: raising `w_mom` moves the operator
back toward the raw one, which is why it improved the iteration count on a
manufactured RHS while destroying ∇·u by 109×.

* softest eigenmode at k_z = 0 is **100% pressure**, cond(D⁻¹A) = 9.0e4 (§7S.1)
* at k_z ≠ 0 the softest mode is 99.6–100% **vorticity** instead, cond 4.1e3
* the k_z = 0 mode is the most expensive Fourier mode in the channel solve

This single fact explains the k_z-dependence of everything else in this document.

## 3. Settled results

### 3.1 Temporal order is 2.00, and artificial compressibility destroys it

RKW3/CN is second-order by construction (RK3's third order applies to the
explicit half alone; Crank–Nicolson caps the mixed scheme at 2). Stokes decay
against the analytic σ = 9.3137399, N=10, 2×4 elements, row weights on:

| `dt` | 0.01 | 0.005 | 0.0025 | 0.00125 | order |
|---|---|---|---|---|---|
| **AC off** (κ_p = 0) | 1.680e−04 | 4.189e−05 | 1.046e−05 | **2.613e−06** | **2.00, 2.00, 2.00** |
| AC on, κ_p ∝ 1/dt | 6.477e−03 | 6.063e−03 | 6.062e−03 | 6.072e−03 | 0.10, 0.00, −0.00 |

**Artificial compressibility is the obvious remedy for the pressure problem in
§2 and it cannot be used as it was.** `kap` adds κ_p·p to the continuity row,
supplying exactly the missing pressure diagonal, and in 2D it bought **27× fewer
CG iterations at a_mass = 30**. But `div u = −κ_p(p − p_prev) ≈ −κ_p·ṗ·Δt`, so
the scaling of κ_p caps the order the scheme is *permitted* to reach:

| κ_p scaling | AC error | order permitted | status |
|---|---|---|---|
| ∝ 1/dt (was production) | O(1) | zeroth | **measured 0.00** |
| fixed | O(dt) | first | measured ~0.4 |
| ∝ dt | O(dt²) | second | **untested** |
| **0 (production)** | none | second | **measured 2.00** |

`operator.py`'s own note is that AC "is consistent at a steady state (p = p_prev
makes it vanish) ... during a transient it perturbs by O(κ_p·R), so the caller
must carry κ_p·p^n on the continuity row of the right-hand side."

### 3.2 p-multigrid loses p-independence as `c` grows — in 2D **and** 3D

This was misdiagnosed twice (§7K, §7T) as a property of the 3D FOSLS
formulation. It is not. The control experiment is to run the **2D** code under
the same cold, manufactured-RHS protocol as the 3D sweeps, at matched `c`
(`scratch/p_indep_2d.py`, `scratch/c_sweep_2d.py`, 4×4 cavity, Re=1000):

| c | 1 | 10 | 50 | 200 | 525 | 2000 | 5405 |
|---|---|---|---|---|---|---|---|
| **2D ladder growth, N=8→24** | **0.85×** | 2.05× | 2.72× | 2.91× | 3.52× | 4.17× | **4.97×** |
| Jacobi/PMG ratio at N=24 | 81.6× | 19.4× | 12.6× | 9.3× | 7.9× | 7.4× | 6.4× |
| **3D, same protocol** | — | — | — | — | — | — | **5.56×** |

**p-independence exists only near c ≈ 1 and is gone by c = 10.** At the
channel's c = 5405, 2D (4.97×) and 3D (5.56×) are indistinguishable. §6.9's
celebrated 1.05× was measured at `dt = 1`, i.e. **c = 1**, with a warm
near-steady RHS — a regime the DNS never occupies.

### 3.3 The cost is intrinsic: cond(A) = cond(L)²

FOSLS forms normal equations, so the condition number is squared. Measured
cond(D⁻¹A) on a 2×2/N=6 mesh: **1.7e3–5.4e3**, rising with the mesh to the ~3e4
that produces ~2000 CG iterations per mode on the 6×18/N=8 channel. Switching to
LSQR would not help — it is mathematically equivalent to CG on the normal
equations, differing only in rounding behaviour.

## 4. Every experiment, with outcome

### 4.1 Formulation and order

| # | experiment | result |
|---|---|---|
| 1 | Stokes decay, AC off | order **2.00, 2.00, 2.00** |
| 2 | Stokes decay, AC on κ_p∝1/dt | order **0.00**, 2300× worse error |
| 3 | κ_p consistency ladder | order permitted = f(κ_p scaling), table §3.1 |
| 4 | Taylor–Green, convection active | order 2.00 |
| 5 | rotated (x,z) TG, z-convection gate | 2.00, 2.00, 2.00 |
| 6 | scalar RKW3/CN model split three ways | explicit 3.025, implicit 2.002, mixed 2.189 — CN is the limiter |
| 7 | operator = Hessian of J | 1.30e−16 |

### 4.2 Row 7 (∇·ω), the redundant row — §7S

| # | experiment | result |
|---|---|---|
| 8 | H¹ ellipticity c₂/c₁ vs w₇, all k_z, all p | **unchanged to 4 s.f.** — down-weighting costs zero coercivity |
| 9 | cond(D⁻¹A) vs w₇ at real k_z | **92× / 141× / 269×** gain at k_z = 5.88 / 11.76 / 23.53 |
| 10 | same at k_z = 0 | only 1.4× — **k_z=0 is unrepresentative, do not reason about R₇ there** |
| 11 | 14×14 point-block preconditioner at w₇=1 | recovers **2%** — confirms §7J; stiffness is intrinsic to A |
| 12 | soft-subspace count | **13% of dof**, constant with mesh — deflation is dead (2 modes buy 1.1×) |
| 13 | knee sweep w₇ ∈ [1e−5, 1] | conditioning floor reached at ~1e−3 |
| 14 | channel ladder, real step | 1e−3 vs 1e−4: **1.95× cost for 3.6× better ∇·ω** |

**Verdict: `ROW7_WEIGHT = 1e-4` stands.** The theoretical objection (R₇ supplies
H¹ coercivity for ω via the div–curl inequality) is correct and the measured
coercivity price is nil; the real cost is accuracy (§7J: TG order 2.00 → 1.72).

### 4.3 Preconditioners — §7T, §7U

| # | experiment | result |
|---|---|---|
| 15 | PMG vs Jacobi, channel p=8 | 3436 → **402 iterations (8.5×)**, but **2× worse wall** |
| 16 | `DirectCoarseE` (element-local coarse assembly) | **built, validated** identical to reference, ~100× faster build |
| 17 | exact vs Chebyshev coarse solve | **no difference** (441 vs 402 at p=8; 2–3% at p≥12) |
| 18 | Galerkin `PᵀAP` vs rediscretised coarse operator | **86/86, 168/168, 300/300** — no difference |
| 19 | 2-, 3-, 4-level ladders | 434 / 441 / 405 — indistinguishable |
| 20 | smoothing degree 1…6 | **total work flat at ~7000** — see below |
| 21 | V-cycle symmetry / definiteness | asymmetry **5.9e−16**, ⟨Mr,r⟩ > 0 — CG is valid |
| 22 | Jacobi diagonal vs true assembled diagonal | **3.1e−16** — exact, no bug |
| 23 | per-mode Chebyshev λ_max spread | **1.28×** — a shared ρ costs nothing |
| 24 | per-mode vs batched solves | **2.00× wall**, measured |
| 25 | 2D vs 3D `Chebyshev4` | identical code, same `_BETA4`, `safety`, `npow` |

**Why #20 is decisive.** Work = its × (applies/cycle + 1):

| | Jacobi | deg 1 | deg 2 | deg 3 | deg 4 | deg 6 |
|---|---|---|---|---|---|---|
| CG | 3436 | 2049 | 1138 | 814 | 602 | 402 |
| **work** | 6872 | 7417 | 7101 | 7212 | 6910 | 6721 |

Flat across a 6× range in degree. **A V-cycle whose coarse correction is weak is
a polynomial in `D⁻¹A`, and CG already builds the optimal polynomial over its
Krylov space** — so no smoothing degree can win. This one fact also explains
#17, #18 and #19: they are all the same method in disguise.

### 4.4 The momentum row weight — proposed, measured, REJECTED

| # | experiment | result |
|---|---|---|
| 26 | `w_mom` sweep, manufactured RHS | optimum **w_mom = 100**, **4.28×**, identical at c = 525/1500/5405/15000, scaling c^−2.00 |
| 27 | same, real RKW3 step, accuracy checked | **REJECTED** |

| w_mom | CG | rms ∇·u | rms ∇·ω |
|---|---|---|---|
| **1 (default)** | 7930 | **2.74e−03** | **1.20e−02** |
| 30 | 3090 (2.57×) | 9.69e−02 (**35×**) | 3.60e−01 |
| 100 | 3652 (2.17×) | 2.99e−01 (**109×**) | 1.19e+00 |

`w_mom` up-weights momentum **relative to the constraint rows**, so the
least-squares solution satisfies momentum better and continuity worse. The
iteration count improves *because the operator enforces less*. `w_mom = 1` is
correctly scaled; reading an iteration count as a figure of merit was the error.

## 5. Batching the Fourier modes — a 1.9× cost, and a systematic skew in every measurement

The Fourier modes never couple in the implicit operator (§1: convection is
explicit), so `pcg` solves all `nk` of them **as one batched array**. That is the
right data layout and the wrong stopping rule, and the stopping rule quietly
distorts every iteration count this project reports.

### 5.1 What the code actually does

* `solver3d.pcg` applies `Ap = A(p)` to the **whole** `nk`-mode array every
  iteration, with a single global `break`.
* `_dot` is **per-mode** — "sum over space and fields, keep the mode axis" — so
  α and β are per-mode and each mode converges against its own `‖b‖`. A
  converged mode's α goes to zero and it stops *updating* — but it keeps costing
  a full operator apply, because it is still in the array.
* `channel3d.step` accumulates stage counts with **`its = max(its, it)`, not a
  sum**.

So the loop runs until the **worst** mode converges, and the number it reports is
the **worst mode of the worst stage**.

### 5.2 The per-mode cost distribution (measured, correct masks)

Channel 6×18, N=8, c=5405, Jacobi, tol 1e−8, each mode solved alone:

| k | 0 | 1 | 2 | 3 | 4 | 5 | … | 15 | 16 (Nyq) |
|---|---|---|---|---|---|---|---|---|---|
| CG | **3430** | 2822 | 2379 | 1958 | 1722 | 1616 | ~1600–1730 | 1720 | **50** |

Spread 1589–3430 excluding Nyquist; mean 1798; **max/mean = 1.80**. The Nyquist
mode is cheap for a legitimate reason — its imaginary half really is unphysical,
so it is a half-size problem.

k_z = 0 is the worst, which is exactly what §2 predicts: it is the one mode where
the `i·k_z·p` path vanishes and the pressure near-null direction is fully
exposed.

### 5.3 The cost

| | mode-applies |
|---|---|
| batched: `max × nk` = 3430 × 17 | **58,310** |
| per-mode: `Σ its` | **30,574** |
| **waste** | **1.91×** |

Measured wall confirms it: one batch **117.3 s → per-mode 58.7 s, 2.00×**.

**Partial grouping does not pay.** Splitting into 2 groups is *worse* than not
splitting (0.81×) and 4 groups is a wash (1.05×); only full per-mode splitting
recovers the 2×. The modes are not unequal enough for coarse grouping to help,
and smaller batches vectorise worse.

Note the waste is **not** just the k_z=0 outlier: if k_z=0 were merely typical
(~1700) the batched cost would still be 1.66× the per-mode sum, because the
stopping rule pays `max × nk` against a distribution whose mean is 1.8× below its
max.

### 5.4 How this skews the measurements — the part that matters

This is the reason the section exists. Batching does not merely cost 1.9×; it
**changes what an iteration count means**, and four wrong conclusions in this
project trace back to it.

1. **A logged `CG=` is not total work.** It is max-over-modes *and*
   max-over-stages. Summing logged `CG` across steps to estimate cost
   undercounts by roughly **3×** (the stages) on top of hiding the mode
   structure entirely.

2. **Any preconditioner benchmarked on the batched solve is scored on ONE
   mode — the worst one.** Improvements to the other 16 are invisible. A
   preconditioner that halves the cost of every mode except k_z=0 registers
   **zero** gain.

3. **That one mode is k_z = 0, which is precisely where the pressure pathology
   of §2 lives.** So "the channel's iteration count" is a measurement of the
   pressure near-null problem, not of the typical mode. The channel's batched
   3436 *is* k_z=0's 3430; the batched PMG 402 *is* k_z=0's 388.

4. **Therefore the batched number is biased toward pressure fixes.** Anything
   that repairs the k_z=0 pressure diagonal — preconditioner-only AC (§6.1) is
   the obvious candidate — would show an outsized batched gain, while a
   remedy aimed at the vorticity modes would look worthless. Neither impression
   would be true of the *total* work.

5. **Cross-mode comparisons need per-mode data.** Every per-mode statement in
   this document had to be re-measured after the `build_mask` subset bug (§7.1),
   because the first attempt silently solved halved problems at k_z ≠ 0 and made
   k_z=0 look **65×** worse than its neighbours. It is 2× worse. The
   "10.2× available from mode grouping" that followed from those numbers is
   really **1.9×**.

### 5.5 Why the 2× may not transfer to the GPU

The 2.00× was measured on the Mac under numba with 8 threads. Batching exists
partly *because* one large array is better for vectorisation and, on a GPU, for
occupancy; 17 small solves are 17 small kernels. §7O measured the GB10 as
**bandwidth-bound**, which argues the arithmetic saving should largely carry —
but the occupancy loss is real and untested there. That 2 groups already came
out *worse* than 1 on CPU is direct evidence that per-solve overhead is not
negligible at this size.

**Recommendation: measure per-mode splitting on Spark before adopting it**, and
prefer it only for the `cuda` backend if it holds up. It is worth roughly 2×,
which is the largest verified solver saving found (§4), but it is also the kind
of gain that can evaporate on different hardware.

## 6. PARKED — revisit after the channel DNS completes

**Decision 2026-09-03: none of these is to be pursued while `run01` is running.**
The DNS is the objective; the solver work is in service of it, and every item
below is an optimisation of a run that is already producing correct physics at
an acceptable rate (~75,000 s per unit `t`, t=5.0 reachable in ~3 days). Nothing
here justifies restarting, changing settings under, or competing for the machine
with a healthy run that has survived its first bursting cycle.

Ordered by (payoff × confidence) ÷ disruption. **§6.2 is now the head of the
queue** — §6.1 was measured and demoted.

### 6.1 Warm start — MEASURED, and nearly worthless (1.08x)

`channel3d.stage` implements it completely and `minchan.run` has never passed
`warm={}`. It looked like the obvious free win: one line, cannot change the
answer (`x0` affects the path, not the fixed point), and at dt=8e-4 successive
stage solves are highly correlated.

**Measured (`scratch/warm_start_gain.py`, 6 steps, tol=1e-6, production mesh):**

| | steps | steady-state its | wall |
|---|---|---|---|
| cold (production) | 4787, 4740, 4723, 4714, 4712, 4715 | **4721** | 1020.1 s |
| warm start | 4787, 4647, 4304, 4283, 4272, 4265 | **4354** | 1010.7 s |

**1.08x on iterations, 1.01x on wall.** Take it if the line is being touched
anyway, but it is not a lever.

**Why, and it is a corollary of §3.3.** CG reduces error by about
rho = 1 - 2/sqrt(cond) per iteration; at cond ~ 3e4 that is rho ~ 0.988. Saving
367 iterations means the warm start supplied an initial residual roughly **70x
smaller** -- and 70x better starting information bought 7.8% of the work.
**When the condition number is this large the starting point barely matters;
the convergence RATE dominates.** Anything that only improves the initial guess
-- warm starts, better extrapolation of the previous stage, higher-order
predictors -- is subject to the same ceiling. Only reducing cond(M^-1 A) helps.

### 6.2 Preconditioner-only AC — matrix-free, targets the actual mechanism

κ_p·p added to the continuity row **in `jacobi_diagonal`/`M_inv` only**, zero in
the operator. Listed as untested in §7A.6.

* It attacks §2.1 exactly: pressure loses its own diagonal as dt→0 (dd_p →
  0.000), and AC's sole purpose is to supply that missing a33. `operator.py`:
  "the missing a33 pressure diagonal, which is what AC supplies, is worst exactly
  at the k_z = 0 mode" — **27× fewer CG iterations measured in 2D**.
* **The reason AC was rejected for production does not apply.** It destroyed
  temporal order (2.00 → 0.00) *because it sat in the operator and changed the
  functional*. In the preconditioner it **cannot affect the answer at all**: any
  SPD preconditioner yields the same solution.
* Cost: zero storage, zero build, no structural change.

### 6.3 Element-block Schwarz — REJECTED on architecture, not on physics

Measured **6.1× on cond(M⁻¹A)** at the channel's `c`, and it demonstrably fixes
the right thing: after element-block preconditioning the softest mode is
vorticity rather than pressure. Node blocks (14×14) buy only 1.04×, because
`∇p` is a derivative and a point block sees just its `D[i,i]` self-term.

**But it abandons the matrix-free design**, which is the objection that settles
it:

| | matrix-free (now) | element blocks |
|---|---|---|
| resident memory | **0.74 GB** | **28.3 GB** (38×) |
| build | none | 2.7 Tflop, one-time |
| apply | tensor-product contraction | batched triangular solve |
| dt | free to change | frozen (blocks depend on `c`) |

Mitigations that are real but insufficient: the implicit operator is **constant**
for the whole run (explicit convection, fixed dt), so blocks build once and
amortise over 6250 steps; and assembly needs only **1134 probes total** via the
`DirectCoarseE` trick. 28.3 GB also fits (only 0.74 of 121 GB is in use).

Still a poor trade for ~2.5× on iterations. Note also that fast diagonalisation,
the standard way to make element solves affordable, **requires separability** and
the FOSLS element block is a 14-field coupled system whose inter-field coupling
is the entire point — applying FDM field-by-field would discard the p–u coupling
that makes the block work.

### 6.4 Per-mode solves — 2.0× on CPU, likely a LOSS on this GPU

Measured 2.00× wall on the Mac (numba). But batched runs 3436 iterations
launching kernels over all 17 modes, while per-mode runs Σ its = 30,574
iterations over one mode each — **8.9× more kernel launches**. The GB10 draws
**37 W at 95% "utilization"** with SM clocks at 84% and no throttling, i.e. it is
dispatch- and bandwidth-limited, not compute-limited. Measure on Spark before
believing the CPU number transfers. See §5.

### 6.5 Lower `c` via implicit convection

The only identified route to the c ≈ 1 regime where the p-ladder is genuinely
p-independent (§3.2). `c = 1/(β·dt)` and dt is CFL-limited by the *explicit*
convective term, so this is a time-stepper change, not a preconditioner one —
much larger in scope than anything else here, and it would need its own
stability and accuracy study.

### 6.6 κ_p ∝ dt artificial compressibility

Permits second order (§3.1) and was never run. Superseded in interest by §6.2,
which gets the conditioning benefit with no accuracy exposure at all.

## 7. Measurement traps that produced wrong answers here

Recorded because each one cost hours and each produced a plausible-looking
number that survived until it was checked against something else.

1. **`BC.build_mask(mesh, nmode, nz=nz)` on a mode SUBSET.** It zeroes the whole
   imaginary half of *column 0 of the array it is handed* (and the Nyquist
   column). Passing a subset masks the **wrong** modes, so every single-mode
   k_z≠0 solve silently became a **halved problem**. Invalidated an h-sweep, a
   per-mode table, and a "12.84× from mode grouping" that is really 2.00×.
   *Build the mask once for the full nk and slice it.*
2. **A random right-hand side.** `A` is singular on its null directions, so a
   random `b` has a component outside `range(A)` and CG diverges — measured
   2.3e+22. Use `b = A·x_true`.
3. **A random element-local vector as a "residual".** Copies of a shared global
   node carry different values, so it is not a legitimate assembled residual.
   Made two correct coarse solvers both look broken (‖Az−r‖/‖r‖ = 0.70).
4. **Judging a preconditioner by stationary iteration.** `x ← x + M(b−Ax)`
   needs ρ(I−MA) < 1, which Jacobi does not satisfy on normal equations — it
   "diverged" at 1.98 while working fine inside CG. Preconditioner quality for
   CG is about the spectrum of `M⁻¹A`, not about `M` being a convergent
   splitting.
5. **Sampling eigenvectors.** 118 of 1806 directions all showed good V-cycle
   reduction while CG needed 402 iterations. Bounding ‖v − MAv‖ on eigenvectors
   does not bound ‖I − MA‖.
6. **`dt = 1e30` to mean "steady".** Makes |b| ~ 1e62 and CG's convergence test
   degenerate — every case "converged" in 1 iteration.
7. **Comparing across protocols.** §6.9's 1.05× is an *average per BDF step*
   with a warm, smooth RHS at c = 1; the 3D sweeps are cold solves of a random
   RHS at c = 5405. Four separate wrong conclusions came from treating those as
   the same measurement.
8. **`| tail` on a long run** — buffers everything until exit, so a 30-minute
   solve is indistinguishable from a hang.

## 8. One-line summary

**The steady FOSLS system is well conditioned and p-multigrid works on it. The
time-dependent system at DNS timesteps is a different operator — `c` = 5405
pushes pressure into a near-null direction, costs p-independence in 2D and 3D
alike, and leaves point Jacobi at ~2000 iterations per mode as the intrinsic
price of `cond(A) = cond(L)²`.** No preconditioner change tested so far beats
it; the untried levers are preconditioner-only AC and a larger `dt`.
