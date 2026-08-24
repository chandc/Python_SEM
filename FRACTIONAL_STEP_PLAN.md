# Fractional step as an alternative solve path

Written 2026-08-23 as a plan; **Phases 0–2 are now built and passing, and
Phase 3 is measured.** Updated 2026-08-24.

---

## RESULTS (measured — read this before the plan below)

| | fractional step | VVP LSSEM | |
|---|---|---|---|
| **TGV Re = 800, 88³** (periodic) | **3.85 s/step** | 79.4 s/step | **20.6× faster** |
| **minimal channel Re_τ = 180** (walls)³ | **6.34 s/step** | 111.6 s/step | **17.6× faster** |
| temporal order, periodic | 2.00 | 2.00 | equal |
| temporal order, walls | **2.0**² | 2.00 | equal |
| accuracy constant at equal dt, walls | 1.055e-4 | 1.045e-5 | **LSSEM ~10× better** |
| energy–enstrophy balance (Gate 3) | 5.16e-05 | 6.65e-06 | LSSEM ~8× tighter |

¹ TGV rows on the Spark GB10. ² after the fix in §3.1b — it was ~1.6 before.
³ channel rows both on the Spark GB10, with the pressure preconditioner of
§3.1c.

**The advantage holds for BOTH, ~18–21×, once the pressure Poisson is properly
preconditioned.** An earlier version of this table said the advantage was
"flow-dependent — absent for wall-bounded flow", from a channel measurement of
39.76 s/step and 12030 CG/step. **That was my preconditioner, not the flow.**
The pressure solve was using one-level additive Schwarz with no coarse grid;
with p-multigrid the same case takes **60 CG/step and 6.34 s/step** — 200×
fewer iterations. The retracted conclusion is kept in §3.1c because the way it
failed is instructive.

**Recommendation: the projection path is faster for both periodic and
wall-bounded flow.** The remaining caveat is accuracy, not speed: on the
channel, against the analytic σ, the constants differ by ~10× at equal dt, so
matching LSSEM's accuracy needs dt roughly 3.2× smaller — eating into an ~18×
advantage without erasing it.

**On TGV the constants are still UNMEASURED**, and an attempt
(`scratch/fs_constants.py`) failed its own control: temporal self-convergence
reported the least-squares path at order ~0.97 where Gate 1 verifies 2.00, so
none of its numbers can be trusted. Four variants were tried — reference 4×
and 16× finer, error in energy and in the velocity field — and the failure
survived all of them. The suspicion is too few steps at the coarse end (ten
steps at dt = 0.02) measuring startup rather than accumulation. **Until a
control reproduces 2.00 for the least-squares path, treat the ~18–21× speedups
as equal-dt, not equal-accuracy.**

---

## 0. The case, in this project's own numbers

VVP LSSEM solves the normal equations of a first-order system, and that costs
**4750 CG iterations per step** at $16\times16$, $N=8$, $N_z=128$ (48 s/step on
an A100). Four things compound:

| | LSSEM | fractional step |
|---|---|---|
| conditioning | $\kappa(L^\mathsf{T}L)=\kappa(L)^2$ | $\kappa$ of a Helmholtz / Poisson |
| unknowns | 14 real fields | 8 real fields |
| matvec | $L^\mathsf{T}WL$ — two operator applications | one, of a simpler operator |
| **multigrid** | **fails — slow modes are rough (§7K.2)** | **works — Poisson's slow modes are smooth** |

The last row is decisive and is measured, not assumed: PMG on the LS operator
gives $7.4\times$ fewer iterations for $\approx18$ matvecs of cost, a net loss,
because the modes it cannot reach are the rough ones. Poisson is the operator
multigrid was invented for.

**Estimated speedup 30–100×.** That is operator-count reasoning, not a
measurement — the point of Phase 3 below is to replace it with one. An external
sanity check: Nek5000, a projection-type SEM code with multigrid pressure, runs
TGV at this resolution in roughly a second per step.

---

## 1. Equations

Incompressible Navier–Stokes, with $\mathbf{N}(\mathbf{u}) = -(\mathbf{u}\cdot\nabla)\mathbf{u}$:

$$\frac{\partial \mathbf{u}}{\partial t} = \mathbf{N}(\mathbf{u}) - \nabla p + \nu\nabla^2\mathbf{u},
\qquad \nabla\cdot\mathbf{u} = 0 .$$

### 1.1 The time scheme is kept EXACTLY as it is

`lssem3d/timestep.py` implements Spalart–Moser–Rogers RKW3/CN, three substages
$k=0,1,2$:

$$\mathbf{U}^{k} = \mathbf{U}^{k-1} + \Delta t\Big[\underbrace{\gamma_k\mathbf{N}^{k-1} + \zeta_k\mathbf{N}^{k-2}}_{\text{explicit convection}} + \underbrace{\alpha_k\mathbf{L}^{k-1} + \beta_k\mathbf{L}^{k}}_{\text{Crank–Nicolson viscous}}\Big]$$

with $\mathbf{L} = \nu\nabla^2\mathbf{u}$ and

$$\gamma = \left(\tfrac{8}{15},\ \tfrac{5}{12},\ \tfrac{3}{4}\right),\quad
\zeta = \left(0,\ -\tfrac{17}{60},\ -\tfrac{5}{12}\right),\quad
\alpha+\beta = \gamma+\zeta,\quad
\tfrac{1}{\beta} = (4.32,\ 4.80,\ 6.00).$$

**Yes, RK3/CN carries over unchanged**, and this is the scheme's home ground:
SMR was designed for projection-based channel DNS. The projection is applied
*per substage*, and the implicit coefficient is the same one the LS path uses,

$$c_k \;=\; \frac{1}{\beta_k\,\Delta t}\;=\;\texttt{timestep.implicit\_coeff(dt, k)},$$

so the two paths can be run at identical $\Delta t$ and compared directly.
$\zeta_0 = 0$ still holds, so **checkpoint/restart stays exact** — the property
`scratch/tgv_gpu_run.py` relies on.

### 1.2 The four substeps (incremental, rotational form)

Per substage $k$, writing $\hat{\mathbf{u}}$ for the intermediate velocity and
$\phi$ for the pressure correction:

**(a) Explicit assembly.**

$$\mathbf{r} \;=\; \mathbf{u}^{k-1} + \Delta t\Big[\gamma_k\mathbf{N}^{k-1} + \zeta_k\mathbf{N}^{k-2} + \alpha_k\nu\nabla^2\mathbf{u}^{k-1}\Big] \;-\; \beta_k\Delta t\,\nabla p^{k-1}$$

**(b) Velocity Helmholtz** — three scalar solves (or one batched over
components):

$$\big(c_k I - \nu\nabla^2\big)\,\hat{\mathbf{u}} \;=\; c_k\,\mathbf{r}$$

**(c) Pressure Poisson** — one scalar solve:

$$\nabla^2\phi \;=\; c_k\,\nabla\cdot\hat{\mathbf{u}}$$

**(d) Projection.**

$$\mathbf{u}^{k} \;=\; \hat{\mathbf{u}} \;-\; \beta_k\Delta t\,\nabla\phi$$

**(e) Pressure update, rotational form.**

$$p^{k} \;=\; p^{k-1} + \phi \;-\; \nu\,\nabla\cdot\hat{\mathbf{u}}$$

The $-\nu\nabla\cdot\hat{\mathbf{u}}$ term is what makes this *rotational*
(Timmermans; Guermond & Shen). Without it the scheme is only $O(\Delta t)$ in
pressure and suffers a numerical boundary layer; with it, $O(\Delta t^2)$ in
velocity and pressure. **It is not optional** — it is the difference between
matching the LS path's verified order 2 and not.

### 1.3 Fourier reduction — every solve becomes a 2-D scalar Helmholtz

With $\partial_z \to \mathrm{i}k_z$ and $\nabla^2 \to \nabla^2_{xy} - k_z^2$,
each mode decouples exactly as it does now:

$$\Big[\big(c_k + \nu k_z^2\big)I - \nu\nabla^2_{xy}\Big]\hat{u}_{k_z} = c_k\,r_{k_z}
\qquad\text{(velocity, 3 components)}$$

$$\Big[\nabla^2_{xy} - k_z^2\Big]\phi_{k_z} = c_k\big(\nabla\cdot\hat{\mathbf{u}}\big)_{k_z}
\qquad\text{(pressure)}$$

**Four scalar 2-D Helmholtz problems per mode per substage**, against the LS
path's one coupled 14-field system.

---

## 2. Why FDM finally pays here — the key reuse

`lssem3d/fdm.py` was built today and **failed** for the VVP operator: it
inverts each field block exactly but drops the inter-field coupling, 37% of the
operator by Frobenius norm, and lost to plain Jacobi at every order.

**Those solves have no inter-field coupling at all.** Each is a scalar
Helmholtz $(\lambda I - \nabla^2_{xy})\psi = f$, which on a tensor-product
element is exactly

$$\lambda\,M\otimes M \;+\; \tfrac{f_x}{f_y}K\otimes M \;+\; \tfrac{f_y}{f_x}M\otimes K$$

— the separable form fast diagonalisation was designed for, with **nothing
dropped**. Solve $K s = \lambda M s$ once per order, and the element inverse is
transform → divide → transform back at $O(N^4)$ instead of $O(N^6)$.

So the FDM module is directly reusable, the negative result does not carry
over, and the same is true of p-multigrid: **the coarse-grid premise holds for
Poisson**, whose slow modes are smooth, where §7K measured them rough for the
LS operator.

| $\lambda$ | conditioning | expected CG iterations |
|---|---|---|
| velocity: $c_k + \nu k_z^2 \approx 1100$ | strongly mass-dominated | **~5–15** |
| pressure, $k_z \neq 0$: shift $k_z^2 > 0$ | Helmholtz, benign | **~10–30** |
| pressure, $k_z = 0$: pure Poisson | **singular** (constant null space) | **~20–60** |

Only **one mode out of 65** is singular, against the LS path where the pressure
block is singular at $k_z=0$ *and* the whole system is squared.

---

## 3. Wall boundary conditions — the hard part, stated honestly

This is where projection methods are delicate and where the LS formulation is
genuinely stronger.

### 3.1 The difficulty

Imposing no-slip on the intermediate velocity, $\hat{\mathbf{u}}|_{\Gamma}=0$,
and homogeneous Neumann on the correction, $\partial\phi/\partial n|_\Gamma = 0$,
gives a correct **normal** velocity but leaves a **tangential slip**

$$\mathbf{u}^{k}\!\cdot\mathbf{t}\Big|_{\Gamma} \;=\; -\beta_k\Delta t\,\frac{\partial\phi}{\partial t}\Big|_{\Gamma} \;\neq\; 0,$$

the classic $O(\Delta t)$ numerical boundary layer of thickness
$\sim\sqrt{\nu\Delta t}$. The LS path has no analogue: it imposes velocity
directly and never splits.

### 3.1c The channel verdict was my preconditioner, not the flow

The pressure Poisson on the Re_τ = 180 channel took **12030 CG/step**, against
780 on periodic TGV, and I concluded the projection path was ~2× slower there
and that the advantage was flow-dependent. Both were wrong, and the diagnosis
is worth keeping.

`fdm_preconditioner` is **one-level additive Schwarz** — an exact element-local
inverse plus gather-scatter, with **no coarse grid**. On this operator that
buys nothing over a plain diagonal, and both grow with element count:

| elements | Jacobi | FDM | **PMG** |
|---|---|---|---|
| 2×4 | 254 | 231 | **9** |
| 6×12 | 734 | 826 | **9** |

Growth like √elements is the signature: **Poisson's slow modes are global and
smooth**, so exactness *inside* an element cannot reach them — which is why an
exact element inverse was no better than a diagonal. A coarse grid reaches
them, and the count goes flat.

**§7K's p-MG closure does not transfer.** It closed p-MG for the VVP
least-squares operator — 7.4× fewer iterations for ~27× the cost — because
§7K.2 found *that* operator's slow modes **rough**. Poisson's are smooth, the
canonical multigrid case. The machinery §7K left behind (`p_interp`,
`coarsen_mesh`, `Chebyshev4`, `_factor_spd`) is operator-agnostic and is reused
unchanged in `lssem3d/hpmg.py`.

Channel, same machine (Spark GB10): **39.76 → 6.34 s/step**, 12030 → 60 CG/step.

### 3.1b RESOLVED, 2026-08-24: one projection per step, not per substage

**Second order at walls is recovered.** Gate 1, channel, against σ = 9.3137399:

| dt | 0.02 | 0.01 | 0.005 | 0.0025 |
|---|---|---|---|---|
| rel err | 1.049e-2 | 2.247e-3 | 5.273e-4 | **1.055e-4** |
| pairwise order | | **2.22** | **2.09** | **2.32** |

against 1.64, 1.48, 1.77 for the per-substage form.

**The cause was structural, not a coding error.** The Kim–Moin correction

$$\vec{u}^*\big|_\Gamma = \vec{u}^{n+1}\big|_\Gamma + \delta t\,\nabla\phi^{n-1}\big|_\Gamma$$

is an extrapolation in time **over a uniform step**. Applied per RKW3 substage
with $\beta_k\delta t$ — and the SMR weights $\beta = (0.2315, 0.2083,
0.1667)$ sum to **0.606, not 1** — it is scaled to the wrong interval: right in
form, wrong in magnitude. Landing at 1.6, *between* the reference's no-slip
(slope 1) and corrected (slope 2) curves, was the tell.

`project.step_kim_moin` follows Kim & Moin's own sequence: **four-stage RK for
convection only → one Crank–Nicolson viscous solve → one projection.** This
also settles the hypothesis raised in §3.1a and left unconfirmed there — that
per-substage pressure treatment is ill-founded because the weights do not sum
to a step. It was.

**The cost:** this is no longer the SMR RKW3/CN scheme the LS path uses, so the
two no longer share an integrator. `project.substage` (per-substage) is kept
for periodic work, where it measures 2.00 and the question does not arise.

### 3.1a MEASURED, 2026-08-24: the two forms bracket the problem

Both were built and run on Gate 1 (channel, no-slip, non-zero pressure):

| form | stability | order |
|---|---|---|
| incremental rotational | **unstable** — σ runs 9.316 → 9.944 over 40 steps | 2 in principle |
| pressure-free | **perfectly stable** — σ(t) flat, identical at every step count | **0.95–0.97, first** |

The instability was confirmed rather than inferred: running the same case with
the increment dropped removes it completely and reproducibly. The feedback loop
is `p^k = p^{k-1} + φ − ν∇·û` re-entering the next substage through
`−β_kΔt∇p`, which amplifies any inconsistency in the pressure boundary
condition.

**This makes the consistent condition necessary, and it is a SCHEME CHANGE, not
a boundary term.** ∂p/∂n = −ν n·(∇×ω) applies to the *full* pressure. In the
incremental form, if $p^{k-1}$ already satisfies it then φ's own condition is
nearly homogeneous — which is what we have and what is unstable. Adding a
boundary integral for φ re-imposes what is already there.

The resolution is **consistent splitting** (Guermond–Shen velocity-correction):
stop forming an increment and solve for the FULL pressure each substage,

$$\nabla^2 p^{k} = \nabla\cdot\big(\mathbf{N}^{k} + \nu\nabla^2\mathbf{u}^{k}\big),
\qquad \frac{\partial p^{k}}{\partial n}\bigg|_\Gamma = -\nu\,\mathbf{n}\cdot(\nabla\times\boldsymbol\omega)\bigg|_\Gamma$$

with the Neumann data entering as a boundary integral $\oint g\,v\,\mathrm{d}s$
on the right-hand side of the weak form. Three pieces of work, none large:
wall-edge quadrature (edge nodes, 1-D weights, the edge Jacobian $1/f_x$ or
$1/f_y$), evaluation of $\mathbf{n}\cdot(\nabla\times\boldsymbol\omega)$ —
cheap here because $\boldsymbol\omega$ is already a primary variable next door
in the VVP formulation — and restructuring `project.substage` to solve for $p$
rather than for an increment.

One hypothesis raised and **not** confirmed: that applying an incremental scheme
per RK substage is ill-founded because the SMR weights sum to
$\beta_0+\beta_1+\beta_2 = 0.606$ rather than 1. It motivated testing the
pressure-free form (which Kim–Moin–Moser lineage codes use for RKW3), and the
first-order result there is consistent with standard theory for that form and
settles nothing about the substage argument. Separating it needs the
incremental form on a single-stage integrator, where the weights are trivially
consistent.

### 3.2 Three handles, in order of preference

1. **Rotational form** (§1.2e). Reduces the pressure error to
   $O(\Delta t^2)$ and the slip to $O(\Delta t^{3/2})$ in $L^2$. Free — it is
   one extra term.
2. **Consistent pressure Neumann condition**, from the normal momentum
   equation at the wall:
   $$\frac{\partial p}{\partial n}\Big|_{\Gamma} = -\nu\,\mathbf{n}\cdot(\nabla\times\boldsymbol\omega)\Big|_{\Gamma}$$
   This removes the boundary layer rather than shrinking it, at the cost of
   evaluating $\nabla\times\boldsymbol\omega$ on the wall. **The VVP path
   already carries $\boldsymbol\omega$ as a primary variable**, so this term is
   cheaper to form here than in a primitive-variable code.
3. **Verify, do not assume.** Gate 1 below is the same Stokes-decay test the LS
   path passes at order 2.00. If the splitting has broken the order, that gate
   says so in one run.

### 3.3 Periodic cases need none of this

TGV and the CORIA benchmark are triply periodic: no walls, no slip error, and
the only boundary subtlety is the $k_z=0$ pressure null space, which is pinned
exactly as it is now (`bc.pin_dof`, every local copy of one global node).
**That is why the first three phases below are periodic** — they isolate the
splitting error from the wall treatment.

---

## 4. Linear solver

**Per mode, per field: CG with an FDM element-block preconditioner.**

- Reuse `lssem3d/fdm.py` for the element solve — now exact, since there is no
  field coupling to drop.
- Reuse `solver3d.pcg` unchanged, including `check_every` (worth 2.5×) and the
  GEMM inner product (worth 11.6×). Both are field-count agnostic.
- The $k_z=0$ pressure Poisson is the only singular solve: pin one global dof,
  and check the compatibility condition $\int_\Omega \nabla\cdot\hat{\mathbf{u}} = 0$,
  which holds automatically when the velocity BCs are consistent — and is a
  useful assertion when they are not.
- **p-multigrid becomes worth reconsidering** for the $k_z=0$ Poisson
  specifically. §7K's closure was measured on the LS operator; the reasoning
  that closed it (rough slow modes) does not apply here. The existing `PMG`
  class works as-is once handed the right operator — with the mask fix from
  today, and after a port to the device, since it is currently host-only.

---

## 5. What is reused, what is new

**Reused unchanged** — the majority:

`mesh`, `gather_scatter`, `lgl`, `deriv`, `fourier` (FFT, dealiasing),
`convect` (the convective term, identical), `timestep` (RKW3/CN coefficients),
`bc.pin_dof`, `solver3d.pcg`, `fdm`, the CuPy/torch kernels for derivatives,
and `scratch/tgv_gpu_run.py`'s checkpoint/restart driver — whose exactness
argument ($\zeta_0=0$) survives unchanged.

**New** — four modules, none large:

| module | contents |
|---|---|
| `helmholtz.py` | $(\lambda I - \nabla^2_{xy})$ operator, its Jacobi diagonal, FDM factors |
| `project.py` | the five substeps of §1.2 |
| `bc_fs.py` | intermediate-velocity BCs, the Neumann/consistent pressure condition |
| `scratch/fs_run.py` | driver, mirroring `tgv_gpu_run.py` so the two are A/B-comparable |

---

## 6. Phases and gates

Each gate is one the LS path already passes, so the two are directly
comparable — that is the point.

**Phase 0 — the scalar Helmholtz solver.** Manufactured solution
$\psi = \sin x\cos y\,\mathrm{e}^{\mathrm{i}k_z z}$; verify spectral
convergence in $N$ and exactness of the FDM element inverse. *Gate: error at
machine precision for the single-element case, spectral decay for
multi-element.*

**Phase 1 — Stokes, periodic. DONE.** Periodic order **2.00**; channel
**~1.6**, which is the O($\Delta t^{3/2}$) the literature predicts for the
tangential slip at walls, reached with a pressure-free projection plus the
Kim–Moin wall correction. The LS path holds 2.00 in *both* — wall accuracy is
the price of splitting, and a real formulation-level advantage of VVP LSSEM.
Original text: **Phase 1 — Stokes, periodic.** No convection. *Gate 1: the analytic decay
$\sigma = 9.3137399$ and second-order convergence in $\Delta t$* — the same
test `cupy_validation_ladder.py` runs. **This is the gate that decides whether
the splitting is done right**; a first-order pressure treatment shows up here
immediately.

**Phase 2 — TGV Re = 100, periodic.** Add convection. **GATE 3 PASSES,
2026-08-24**: worst deviation from the balance **5.16e-05** against a 1e-4
criterion (the LS path holds 6.65e-06 — ~8× tighter, expected, since it solves
velocity, vorticity and pressure simultaneously where this one splits). The
most informative gate available: no fitted constant, and a *joint* statement
about the convective term, the viscous term and the integrator, so it fails if
any one is wrong. `convect.convective` was reused **unchanged**, as predicted —
it reads only u, v, w at indices 0, 1, 2, exactly the fractional-step layout.
Gate 2 (rotated $(x,z)$ order) not yet run.

**Phase 3 — MEASURED, see the results table at the top.** 20.6× on TGV,
~2× slower on the minimal channel. Original text: **the A/B that settles the
speed question.** TGV Re = 800 at $88^3$,
against the LS run completing now. Same $\Delta t$, same grid, same tolerance.
*Gate: energy and enstrophy histories agree to discretisation error, and
report the measured s/step ratio.* This replaces the 30–100× estimate with a
number.

**Phase 4 — walls.** Lid-driven cavity against Ghia (the LS path reaches 0.45%
RMS at $8\times8$, order 12) and channel flow. *Gate: no slip error visible
above discretisation error; second-order convergence retained.* Only here does
§3 matter.

**Phase 5 — production.** Re = 1600 at $128^3$ against CORIA. If Phase 3
delivers even $20\times$, this becomes a **2–3 hour run** rather than 69 h, and
the resolution question ($k_{\max}\eta \approx 0.76$) can be answered by simply
running $192^3$ instead of arguing about it.

---

## 7. Risks

| risk | handle |
|---|---|
| splitting error destroys order 2 | rotational form; Gate 1 catches it in one run |
| wall slip layer | consistent Neumann BC, cheap here because $\boldsymbol\omega$ is available |
| $k_z=0$ Poisson singular | pin one dof (existing `bc.pin_dof`); assert compatibility |
| outflow BCs | **unsolved, and a genuine regression risk.** `DONG_OBC_RESULTS.md` shows the LS/Dong exit reproducing long-domain results where P+Z cannot. Projection methods have their own outflow pathologies. **Do not migrate the 2-D outflow work.** |
| two solvers to maintain | keep fractional step for periodic DNS, LSSEM where its properties are the point |

**The honest framing: these are different tools, not one replacing the other.**
For periodic DNS the LS properties — SPD system, no inf–sup constraint,
equal-order interpolation, vorticity as a primary variable — buy nothing, and
the squared condition number is pure cost. For truncated domains with outflow,
they are the whole reason the method was chosen.
