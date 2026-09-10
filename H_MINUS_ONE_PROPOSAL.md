# Proposal: measure the momentum residual in a negative norm (dt-independent FOSLS)

Status 2026-09-08: proposal with a Stage-A measurement (momentum-weight
sweep) and a Stage-B accuracy prototype (assembled operators, direct solves)
running; results appended in §6 as they land.

## 1. The problem this addresses

Each FOSLS time step minimises

    J(u) = ‖ fac1·u − hist + dt·N(u) ‖²_{L²} + ‖∇·u‖² + ‖∇×u − ω‖²

(quadrature-weighted; N(u) = convection + ∇p + ν∇×ω). At a steady state
hist = u, so the march converges to the minimiser of

    J_∞(u) = dt² ‖N(u)‖² + ‖∇·u‖² + ‖∇×u − ω‖² .

The time step is the **weight of the momentum equation**. Since the discrete
space cannot satisfy all rows exactly (lid corners, finite N), the minimiser
is a weighted compromise and depends on dt: the Re = 1000 cavity at N = 15
reaches rms 0.003 against Ghia with dt = 1 and 0.012 with dt = 1e−2, the
difference being a node-to-node oscillation under the lid that costs J
almost nothing at weight dt² = 1e−4 (CAVITY_N15_MARCH.md; the same mechanism
gave a 1875× spread on Poiseuille, POISEUILLE_DT_STUDY.md). The 3D channel
carries the same structure with weight 1/c², c = 5405.

Ordinary Galerkin/FD time marching does not have this property: its steady
state satisfies the discrete momentum equation exactly for any dt, because
the update is a residual equation rather than a weighted minimisation.

## 2. What the theory says the weight should be

The ADN estimate for the velocity–vorticity–pressure system with no-slip
walls (Bochev & Gunzburger (3.34), q = −1; ADN_FOSLS.md §2) is

    ‖u‖₁ + ‖ω‖₀ + ‖p‖₀ ≤ C ( ‖ν∇×ω + ∇p − f‖₋₁ + ‖∇×u − ω‖₀ + ‖∇·u‖₀ ).

The momentum residual belongs in **H⁻¹**, one derivative weaker than the
constraint residuals; measured in L² it is not equivalent to any fixed norm
(BG §4.1.1, Table 1: vorticity and pressure lose an order). The legacy dt
weight is an accidental stand-in for a weaker norm that is about right when
dt ≈ h and wrong otherwise.

## 3. The proposed functional

    J₋₁(u) = ‖ R_mom ‖²₋₁,h + ‖∇·u‖² + ‖∇×u − ω‖²,
    R_mom = (fac1·u − hist)/dt + N(u)              (momentum at weight 1, i.e. divided by dt)

with the **discrete negative norm** of Bramble–Lazarov–Pasciak (Math. Comp.
66, 1997; BG (4.27)):

    ‖r‖²₋₁,h = β ‖r‖²_{L²}  +  fᵀ B f,      f = load vector of r on interior nodes (f_i = ∫ r φ_i),

where B is symmetric and spectrally equivalent to the inverse of the
discrete Dirichlet Laplacian (exact K⁻¹ in the prototype; one multigrid
V-cycle in production) and β ~ h² is the pointwise L² term. Properties:

- for smooth residuals ‖r‖₋₁,h ≈ ‖r‖₀/O(1): momentum enforced at O(1) at
  **every** dt → dt-independent steady state;
- for rough residuals of wavenumber k, weight ≈ k⁻² + β: a node-to-node
  velocity mode with rough ω has momentum residual O(k²), penalised as
  O(k²·(k⁻² + β)) ≫ its legacy weight dt²·O(k²) — the zigzag is no longer
  cheap;
- **the β term is not optional.** Tested weakly only (β on the load vector),
  the gradient load of a checkerboard pressure vanishes on interior nodes
  and the assembled matrix is singular (measured: `LinAlgError`, Stage B
  first attempt). The pointwise term on the residual *function* removes it
  — it is the equal-order inf-sup issue that the pointwise L² method never
  meets, reappearing when the momentum equation is tested weakly.

The normal operator is A₋₁ = L_momᵀ (β W + W Qₙ B Qₙᵀ W) L_mom + L_conᵀ W L_con,
symmetric positive definite; with B a V-cycle it is matrix-free at the cost
of one V-cycle per application on each velocity component.

## 4. What it would change, and what it costs

- **Accuracy:** dt-independent steady states; optimal-order ω and p (the
  BLP theorem for Stokes). For the channel this is the candidate explanation
  for the −10…−15 % p′ and wall ω′ₓ (REFERENCE_DATA_RE180.md §6).
- **Solver:** the pressure block becomes ∇ᵀ(βI + B)∇ ≈ identity-plus-small:
  the p⁴/h² pressure conditioning disappears (BLP: κ independent of h). The
  velocity block keeps its H(div)-type structure at large c; whether the
  rough divergence-free modes become *softer* under the negative norm (their
  momentum residual is now down-weighted by k⁻²) is the open solver
  question — it may strengthen the need for the vertex patch rather than
  remove it. To be measured on the assembled harness (Stage C).
- **Cost:** 2–4× per CG iteration (V-cycles on the momentum residual), plus
  a new element-block structure for the patch preconditioner: with B
  nonlocal, L_momᵀ B L_mom couples elements, so the patch matrices must be
  built from the local part (β term + a local approximation of B), a
  Schwarz-with-nonlocal-operator design question.
- **Validation:** it changes the discrete solution, so the whole ladder
  (Stokes decay, Taylor–Green, cavity, channel statistics) must be redone.

## 5. Staged plan

| stage | what | cost | decides |
|---|---|---|---|
| A | momentum-weight sweep with the existing code: steady state of the N = 15 cavity at dt = 1, 0.1, 0.03, 0.01 (legacy weighting = weight dt) | running | how much momentum weight the answer needs; the size of the dt-dependence |
| B | assembled prototype of J₋₁ (`scratch/hminus1_2d.py`), N = 8, direct solves, exact B = K_int⁻¹, β ∈ {1e−2, 1e−4}: steady states at dt = 1 and 0.1 (and 1e−2) vs legacy, vs Ghia; zigzag metric | running | does the negative norm make the steady state dt-independent and remove the zigzag; what β |
| C | solver structure: assembled A₋₁ on the 2D harness at c = 5405 — Jacobi / exact-block / vertex-patch κ, softest modes | 1 day | whether the patch preconditioner survives, what the coarse level must contain |
| D | matrix-free implementation (B = PMG V-cycle on the scalar Laplacian) in lssem2d, then the validation ladder | 1–2 weeks | adoption |

## 6. Results

### 6.1 Stage B at N = 8, 4×4 elements (direct solves; rms vs Ghia u, v; `scratch/hm1/`)

| functional | dt = 1 (Picard fixed point) | dt = 0.1 (march to steady) | dt = 0.01 (15 units on from the dt = 0.1 state) |
|---|---|---|---|
| legacy (momentum weight dt) | 0.046, 0.067 | 0.083, 0.114 | 0.045, 0.067 (rate 6e−2, far from steady) |
| H⁻¹, β = 1e−2 | 0.266, 0.362 (max\|v\| 0.45) | 0.054, 0.078 | — |
| H⁻¹, β = 1e−4 | 0.040, 0.058 | **0.025, 0.037** (rate 2e−6) | **0.023, 0.035** (rate 4e−3) |

Readings, all at a resolution (N = 8) where the best case is 0.046:

1. **The negative norm reduces the dt-dependence and improves the answer.**
   H⁻¹ with β = 1e−4 gives the best steady state at every dt, and moving
   from dt = 0.1 to 0.01 changes its rms by 7 % (0.025 → 0.023) where the
   legacy functional swings by a factor 2 (0.083 → 0.045, still moving).
2. **Non-uniqueness at this resolution.** The dt = 1 Picard iteration and
   the dt = 0.1 march minimise the same J_∞ but end in different states for
   every functional (H⁻¹ β = 1e−4: max\|v\| 0.584 vs 0.602; legacy 0.534 vs
   0.566), and for H⁻¹ β = 1e−2 the dt = 1 fixed point (max\|v\| 0.45, rms
   0.27) is reached even when started *from* the dt = 0.1 state — so at
   β = 1e−2 the discrete steady-state problem has at least two solutions
   and the dt = 1 iteration selects the wrong one. Whether this is the
   under-resolution (N = 8, 4×4 at Re = 1000) or a property of a functional
   with weak momentum control is what the N = 12 runs (§6.2) decide.
3. **β matters and its right value is not h².** β = 1e−2 (≈ h²_element/6)
   behaves badly; β = 1e−4 (≈ h²_GLL,min) behaves well. The pointwise term
   must be small enough not to reintroduce the L² weighting and large enough
   to suppress the checkerboard null space (§3).
4. The zigzag metric is meaningless at N = 8 (three intervals under the
   lid); it is reported at N = 12/15.

### 6.2 Stage A at N = 15 (legacy weighting, momentum weight = dt, patch preconditioner) and Stage B at N = 12

**Stage A** — steady states of the N = 15 cavity, rms vs Ghia (u, v),
under-lid zigzag (sign changes of Δu over 8 consecutive nodes, x = 0.5,
0.75 < y < 0.95):

| dt (= momentum weight) | 1 | 0.1 | 0.03 | 0.01 |
|---|---|---|---|---|
| rms u | **0.0095** | 0.0256 | 0.0218 | 0.0122 |
| rms v | **0.0101** | 0.0335 | 0.0288 | 0.0164 |
| zigzag | 0 / 8 | 0 / 8 | 7 / 8 | 7 / 8 |

Non-monotonic in dt: the vortex position/strength (what the 17 Ghia points
measure) shifts one way with the momentum weight and the under-lid
oscillation switches on below dt ≈ 0.03 independently of it.

**Stage B at N = 12** (assembled operators, direct solves; `scratch/hm1/N12_*`):

| functional | dt = 1 (Picard fixed point) | dt = 0.1 (march, rate 2e−4 at t = 80) |
|---|---|---|
| legacy | 0.0202, 0.0248 — zigzag 0 / 6 | 0.0462, 0.0588 — zigzag 4 / 6 |
| H⁻¹, β = 1e−4 | 0.0314, 0.0358 — zigzag 5 / 6 | 0.0149, 0.0125 — zigzag 6 / 6 |

## 7. Verdict on the proposal (2026-09-08)

1. **The negative norm does not remove the dt-dependence.** At N = 12 the
   H⁻¹ steady states at dt = 1 and dt = 0.1 differ by 2× in rms (0.031 vs
   0.015) although they minimise the same J_∞; the same happened at N = 8.
   The discrete nonlinear least-squares problem with weakly weighted
   momentum has **more than one steady solution**, and which one a march
   lands on depends on the path (dt = 1 Picard vs small-dt time marching).
   This non-uniqueness is the underlying phenomenon; the dt-weight in the
   legacy functional is only the most visible way to reach different
   minimisers.
2. **It does not cure the under-lid zigzag; it carries it at every dt.**
   The mechanism is the one the theory predicts once the sign is read
   correctly: H⁻¹ down-weights rough momentum residuals by k⁻², and a
   node-to-node velocity mode with rough vorticity is exactly a rough
   momentum residual. Only a *pointwise* momentum weight of order one has
   suppressed the zigzag in every case measured (legacy dt = 1 and 0.1 at
   N = 15; legacy dt = 1 at N = 12); β = 1e−4 is far too small to do it,
   and β = 1e−2 produced a spurious steady state (§6.1).
3. **It is not better than legacy at dt = 1.** At N = 12: 0.031 vs 0.020.
   Its best case (dt = 0.1, 0.015) is the best number at N = 12, but with
   the oscillation.
4. **What the study leaves standing.** The accuracy problem of small-dt
   marching is real and has two parts: (a) the steady state is not unique
   under weak momentum enforcement; (b) a rough mode is admitted whenever
   the pointwise momentum weight is ≪ 1. Both point to the *pointwise*
   weight, not to a weaker norm: keep the momentum row at L² weight ≈ 1 at
   every dt (w_mom = w_mass = 1, the "time-accurate" configuration, stable
   on closed domains per AMASS_RESOLVED.md), and solve the linear-algebra
   problem that this weighting creates — the (u,p) coupling at O(c) that the
   present (u,ω)‖p patch/coarse preconditioner does not handle
   (CAVITY_N15_MARCH.md §5). That is a preconditioner design task in the
   same ADN/operator-preconditioning framework used here (the norm changes,
   so the block structure changes), and it is now the recommended next
   step. The H⁻¹ route is closed for this operator; the proposal stays on
   record for its negative results, the checkerboard finding (§3) and the
   prototype (`scratch/hminus1_2d.py`), which can test any other weighting
   in an afternoon.
