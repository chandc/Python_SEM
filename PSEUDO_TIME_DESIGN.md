# The missing pseudo-time term (δτ): design note

Status: **design only — not implemented.** Written 2026-08-09 after comparing
the original 1996 Fortran 77 source (`tj.channel.f`) against
`pmg_clean/src/lssem_baseline.f90` and against this repository's
`lssem2d/lssem.py`.

Short version: the 1996 code carries a **pseudo-time derivative** on the
momentum rows that both the modernised F90 and this Python port omit. It is a
stabilisation term with a tunable step δτ, described in Chan (1996) §"Description
of Numerical Method". Our code is its δτ → ∞ limit. This note records what the
term is, what it does and does not change, and how to add it.

---

## 1. The comparison

`L` and `Lᵀ` were compared term by term across all three codebases. **Every
term agrees except one.**

Matching, in `L`: the Newton-linearised convection
(`fu*u_x + fv*u_y + u*dfu_dx + v*dfu_dy`), the pressure gradient and the
`±ν·curl ω` coupling, the continuity row `u_x + v_y`, and the vorticity
definition `ω + u_y − v_x`. Matching in `Lᵀ`: all four columns, including the
`dt` factors on the pressure column (`dt·Dxᵀsu₁ + dt·Dyᵀsu₂`), the `±ν·dt`
factors on the vorticity column, and every convective contribution. The
constraint rows carry weight 1 in all three — confirming the row-weighting
documented in `lssem.py` matches the original.

**The difference** — a bare `u` inside the `dt` bracket, and a matching `+dt`
in the transpose coefficient:

| | momentum row of `L` | collocation term of `Lᵀ` |
|---|---|---|
| `tj.channel.f` (1996) | `fac1*u*facem + dt*( u + fu*dudx + … )*facem` [:1041] | `(dt+fac1)*su(ij,1)` [:1070] |
| `lssem_baseline.f90` | `fac1*u*facem + dt*( fu*dudx + … )*facem` [:230] | `(fac1)*su(ij,1)` [:267] |
| `lssem2d/lssem.py` | `a_mass*u + a_flux*( fu*u_x + … )` [:237] | `inv_dt*su1` with `inv_dt = a_mass/a_flux` [:274] |

The Python matches the F90 exactly. This is a **1996-original vs modernised-F90**
difference, not a porting bug.

Also present in `tj.channel.f`: `rhs` adds `2.0*pr*dt`, a body force `f_x = 2ν`
driving the plane channel. Irrelevant to the BFS, but **not** merely a test-case
detail as first written here — it is structurally required for any
streamwise-periodic channel, because a periodic pressure field cannot carry a
mean gradient. See [CHANNEL_VALIDATION.md](./CHANNEL_VALIDATION.md) §6, where
the Orr–Sommerfeld case does not run without it.

---

## 2. What the term is

Chan (1996) page 2 gives the operator with `1/δτ` on both momentum diagonals and
`u⁰/δτ` on the right-hand side, and states: *"δt is the physical time step
whereas δτ is the pseudo-time step."* The paper adds that δτ can be adjusted
"to introduce certain level of diffusion in the streamwise direction to stabilize
sharp gradient that might occur in an under-resolving grid… not exercised here."

The code confirms the pairing. `rhs` supplies `dt*fu` where `lhs` has `dt*u`
([tj.channel.f:841](file:///Users/danielchan/Downloads/tj.channel%20(1).f)), so the
contribution to the residual is `dt*(fu − u)`. Since the row has been multiplied
through by δt, matching `(δt/δτ)·u` against the code's `dt*u` gives **δτ = 1**,
hard-coded.

### It contributes nothing to the residual

At [tj.channel.f:293-302](file:///Users/danielchan/Downloads/tj.channel%20(1).f)
both `u` and `fu` are assigned from the same array `f`:

```fortran
u(ij,ne)  = f(kk+1,ne)
fu(ij,ne) = f(kk+1,ne)
```

so `fu ≡ u` whenever the residual is formed and `dt*(fu − u) = 0` **exactly**.
The term appears only in the operator that `dge` (the CG) applies. It is a
modification of the Jacobian, not of the equations being solved.

---

## 3. What it changes, and what it does not

Write the augmented operator as `L_κ = L + κ·E`, where `E` projects onto the two
momentum rows and

```
κ = a_flux / dtau          (κ = dt with dtau = 1, matching the 1996 code)
```

Each sub-iteration minimises `‖L_κ u − (f + κ u⁰)‖²` about the previous
sub-iterate `u⁰`. Its stationarity condition is

```
(L + κE)ᵀ ( L_κ u − f − κ u⁰ ) = 0
```

and at the fixed point `u = u⁰` the bracket collapses to the ordinary residual
`R = Lu − f`, leaving

```
Lᵀ R + κ Eᵀ R = 0        versus the unaugmented        Lᵀ R = 0.
```

**So the converged answer is perturbed by O(κ·R), not left exactly invariant.**
This corrects a stronger claim made in conversation ("stabilisation for free,
without changing the converged solution"). The accurate statement:

- Where the least-squares residual is driven to ~0, the two conditions coincide
  and the answer is genuinely unchanged. Poiseuille reaches `J = 5.94e-27`
  (`CG_TOLERANCE_FLOOR.md` §2), so there the term is free.
- Where `R ≠ 0` — the BFS converges to `J ≈ 3.7e-05` — the perturbation is
  proportional to the discretisation residual and therefore **vanishes under
  refinement**. It is a *consistent* stabilisation, in the same sense as SUPG:
  it changes the discrete answer at the order of the existing discretisation
  error, not at leading order.

That distinction must be measured, not assumed. See the test plan in §6.

---

## 4. Why this is NOT `w_mass`

This is the important part, and the reason the term is worth adding.

Both `w_mass` and δτ add a multiple of the identity to the momentum rows. They
differ in **what the added term is paired with on the right-hand side**:

| | added to `L` | paired with | effect on the converged steady state |
|---|---|---|---|
| `w_mass` | `a_mass·u`, `a_mass = fac1·w_mass/dt` | `u^n`, the previous **time level** | **changes it** — this is the dt-dependence in [POISEUILLE_DT_STUDY.md](./POISEUILLE_DT_STUDY.md), a 212,061× spread |
| **δτ** | `κ·u`, `κ = a_flux/dtau` | `u⁰`, the previous **sub-iterate** | perturbs by O(κ·R) only (§3) |

The entire arc of this project has been the discovery that stability and
accuracy are coupled through the momentum weighting: every attempt to damp the
iteration through `a_mass` moved the answer
([WEIGHT_VS_TIMESTEP_STUDY.md](./WEIGHT_VS_TIMESTEP_STUDY.md),
[STEADY_FORM_STUDY.md](./STEADY_FORM_STUDY.md) §4). **δτ is the knob that damps
without paying**, or at worst pays at the order of the discretisation error.
We have been substituting a non-monotone line search for a mechanism the
original algorithm had built in.

Note also that δτ survives `w_mass = 0`. The pure steady form has `a_mass = 0`
and no damping whatsoever, which is exactly the configuration that diverges
(`STEADY_FORM_STUDY.md` §2: `max|u|` 1.51 → 40.05 in one step; §8: 1.51 → 115
from an interpolated IC). δτ restores a damping term there without reintroducing
a physical time derivative.

---

## 5. Proposed implementation

### API

`SolverState.__init__(..., dtau=None)`, default `None` meaning "no pseudo-time",
so **nothing changes silently** — the same convention used for `w_mom`/`w_mass`.

Extend `ls_coeffs` to return a fourth coefficient:

```python
def ls_coeffs(state):
    ...
    a_pseudo = 0.0 if dtau is None else a_flux / float(dtau)
    return a_mass, a_flux, hist_scale, a_pseudo
```

Returning a 4-tuple breaks the six existing unpack sites, all of which use
`a, b, _ = ...`. Prefer adding a separate `ls_pseudo(state)` accessor, or return
a small named tuple, so the existing call sites stay untouched.

### Call sites

Five places consume the momentum-row coefficients and **all five must agree** —
the numba kernels are validated against the NumPy path by
`tests/test_backend_parity.py`, and `compute_jacobi` must match `apply_L` or the
preconditioner is destroyed (its own comment says so):

1. `lssem.py:_apply_L_numpy` — momentum rows become
   `((a_mass + a_pseudo)*u + a_flux*(...)) * wq`.
2. `lssem.py:_apply_LT_numpy` — the collocation term becomes
   `(a_mass + a_pseudo)/a_flux * su1'` (currently `inv_dt = a_mass/a_flux`).
   This is exactly the Fortran's `(dt+fac1)`.
3. `kernels_numba.py:_kernel_L` — same change, and the wrapper must pass the new
   coefficient through.
4. `kernels_numba.py:_kernel_LT` — same.
5. `solver.py:compute_jacobi` — `a11` and `a22` gain `a_pseudo*P`:
   `a11 = (a_mass + a_pseudo)*P + a_flux*(conv + ux*P)`.

### Keeping the residual clean

`apply_L` is used for **two** distinct purposes, and the pseudo-time term must
appear in only one of them:

- inside `apply_A` (the operator applied to the increment `dU`) — **must** carry
  the term;
- in `newton_step` at `solver.py:224`, `su_nl = apply_L(state, U, U/2, V/2) − su_history`
  — **must not**, or the residual is polluted and the fixed point moves for the
  wrong reason.

The Fortran achieves this by supplying `dt*fu` on the right-hand side with
`fu ≡ u`. The Python equivalent is to add `a_pseudo * U[..., :2] * wq` to
`su_history` for the momentum components before forming `su_nl`, so the two
cancel identically. **Verify that cancellation numerically before trusting any
result** — with `dtau` set and `U` unchanged, `su_nl` must be bit-identical to
the `dtau=None` case. This is the single most likely place to get the change
wrong.

---

## 6. Test plan

Ordered so that each step can falsify the change before the next one costs
anything.

1. **Parity.** `dtau=None` must reproduce every existing result bit-for-bit.
   Run `pytest lssem2d/tests` on both backends (48 tests).
2. **Residual cancellation.** As above: `su_nl` unchanged when `dtau` is set.
3. **Backend parity.** `test_backend_parity.py` with `dtau` set, so the numba
   kernels are checked against NumPy on the new term.
4. **Does it change the answer?** This is §3's open question. Poiseuille has an
   exactly representable solution and drives `R → 0`, so sweep
   `dtau ∈ {∞, 10, 1, 0.1}` and confirm the profile error is flat. Then repeat
   on the BFS, where `R ≠ 0`, and **measure** the perturbation rather than
   assuming it is small. Compare against the 1.1% `J` gap that separates the two
   converged states in `STEADY_FORM_STUDY.md` §8 — if δτ moves the answer by
   more than that, it is not a benign stabilisation.
5. **Does it fix what the line search fixes?** The three cases that need
   globalisation today:
   - long domain, steady form, tight tolerance — diverges at iteration 2
     (`STEADY_FORM_STUDY.md` §2);
   - short domain seeded from the long domain — diverges to `max|u| = 115` (§8);
   - short domain at `w_mom = 1.0` — stalls, needs 53 iterations with the line
     search (§5 correction).
   For each: does `dtau = 1` converge *without* `line_search=True`, and at what
   cost in iterations?
6. **Reproduce the 1996 result.** `tj.channel.f` is a plane channel with the
   `2ν` body force and δτ = 1. Running that case is the only end-to-end check
   that our reading of the term is right.

---

## 7. Risks

- **The perturbation may not be benign.** §3 shows the fixed point moves by
  O(κ·R). On the BFS, `R` is not small. Step 4 of the test plan must run before
  δτ is used for anything other than experiments.
- **Five call sites must stay consistent.** A mismatch between `apply_L` and
  `compute_jacobi` silently degrades the preconditioner rather than failing.
- **It interacts with `w_mom`/`w_mass`.** `κ = a_flux/dtau` is proportional to
  `a_flux = w_mom`, so the damping scales with the momentum weight. At
  `w_mom = 0.1` the damping is 10× weaker than at `w_mom = 1.0` for the same
  δτ. Sweeps that vary `w_mom` at fixed `dtau` are therefore varying two things
  at once — the same trap documented in `WEIGHT_VS_TIMESTEP_STUDY.md` §1.
- **`dtau` and the line search may be redundant, or may conflict.** Both damp
  the step. Test them separately before combining.

---

## 8. Open question

If δτ is the intended stabilisation, why was it dropped from
`lssem_baseline.f90`? Possibilities: deliberate simplification once the target
cases converged without it; an artifact of whichever variant of the 1996 source
was modernised; or an oversight. The F90 is internally consistent — its
transpose uses `(fac1)`, matching its own `L` — so whoever removed it removed
both halves. That is a deliberate-looking edit, not a typo.

Worth asking before assuming the 1996 form is the one to restore.
