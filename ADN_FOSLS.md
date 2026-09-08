# Agmon–Douglis–Nirenberg ellipticity and the FOSLS-3D solver

What ADN "stability" is, what it says about our 8-row velocity–vorticity–
pressure (VVP) operator, and what it prescribes for the preconditioner — with
the prescription **measured** on the assembled per-mode operator
(`scratch/adn_block_precond.py`, `scratch/adn_block_precond2.py`).

The short version: ADN tells you *which norm* a first-order system is
coercive in. Operator preconditioning (Mardal–Winther) then says the Riesz map
of that norm is the optimal block-diagonal preconditioner, with a condition
number bounded by the ADN constants independently of h, p and dt. For our
operator the norm changes with c = 1/(β dt): at small c the system is
H¹-coercive in every variable (where p-multigrid with Jacobi smoothing works —
§6.9 of PMG_ALGORITHM), at large c the pair (u, ω) is coercive only in
H(div)×L², a norm that pointwise smoothers cannot see. Measured: the exact
two-block preconditioner **(u,ω) | p** gives κ ≈ 3 and 9–17 CG iterations,
flat from p = 4 to 12 and from c = 1 to 5405; separating u from ω throws it
away.

## 0. How the vertex-patch preconditioner came about

The method was not derived from ADN theory; two separate bodies of theory
met, and the measurements joined them. The chain, each link measured:

1. **The symptom.** Every pointwise or per-variable preconditioner we had —
   Jacobi, node-block, per-field blocks, Jacobi-smoothed p-multigrid,
   Galerkin and direct coarse solves — needed thousands of CG iterations at
   the channel's c = 5405, and p-independence held only near c ≈ 1 (§7U of
   3D_STATUS, FOSLS_TIME_DEPENDENT §3–4). Nothing in the code explained it.
2. **ADN named the norm.** Agmon–Douglis–Nirenberg theory says which norm a
   first-order system is coercive in (§1–2). Read with c as a parameter
   (§3), our functional at large c is equivalent to
   ‖u‖²_{H(div)} + ‖ω − ∇×u‖²₀ + c⁻²|p|²₁: the pair (u, ω) is coupled by the
   curl row and controlled only in H(div), the pressure decouples.
3. **Operator preconditioning turned the norm into a prescription.**
   Mardal–Winther (§4): the optimal block preconditioner is the Riesz map
   of that norm — (u, ω) together, p apart — with κ bounded by the ADN
   constants. Measured on the assembled operator (§5.1): exact (u,ω)‖p
   gives κ ≈ 3, flat in p, h, c; splitting u from ω gives κ 1.4e4.
4. **The obstruction was identified.** After exact vorticity elimination
   the velocity block is M + K_div plus a curl-jump term, and its softest
   Jacobi mode is exactly divergence-free (§5.3): the kernel of the
   divergence, ~2/3 of the space, invisible to any node-wise correction.
   This is the classical H(div) solver problem.
5. **The H(div) literature supplied the remedy.** That literature is a
   separate lineage — Schwarz's alternating method, the domain-decomposition
   theory of Lions, Dryja–Widlund, and its H(div)/H(curl) specialisation by
   Arnold–Falk–Winther (2000) and Hiptmair (1997): pointwise smoothers cannot
   reach the divergence-free kernel, overlapping vertex-patch solves can,
   because the local kernel modes fit inside a patch. Pavarino (1994) had
   shown for spectral elements that element-sized overlap makes Schwarz
   p-independent.
6. **Measurement chose between the two standard remedies** (§8). The nodal
   Hiptmair auxiliary-space sweep failed on C⁰ GLL (the discrete kernel has
   no potential representation there); overlapping vertex patches worked,
   thin overlaps did not, and in hindsight our shelved element-block Schwarz
   (§6.3 of the parking lot, 6×) was the same method with zero overlap.
7. **Verified in the production paths** (§10–12): matrix-free 2D
   `pcg_solve`, time marching at c = 10, 2000 and with a rough field, and
   the N = 30 cavity, where fine-level patches give 19 iterations and the
   dense-patch cost model (§13) sets the order at which it pays.

Without step 2 we would have kept tuning pointwise methods; without step 5
we would have known why they fail but not what to build.

## 1. ADN ellipticity in one page

Agmon, Douglis & Nirenberg (1959, 1964) treat a system of N equations in N
unknowns, L(x,∂) U = F, whose equations may have *different orders*. Assign
integer weights sᵢ to the equations and tⱼ to the unknowns such that the
(i,j) entry of L has order ≤ sᵢ + tⱼ (entries of lower order are allowed;
entries of higher order are not). The **principal part** Lᵖ keeps in each
entry only the terms of order exactly sᵢ + tⱼ (zero if the entry is of lower
order). The system is **ADN-elliptic** if

    det Lᵖ(x, ξ) ≠ 0   for all real ξ ≠ 0,

and *uniformly* elliptic if |det Lᵖ(x,ξ)| ≥ C|ξ|^{2m}, 2m = Σ(sᵢ+tⱼ). In 2D
one also needs *proper* ellipticity (equal numbers of roots with positive and
negative imaginary part); in 3D it is automatic. A boundary operator B with
weights rₖ must satisfy the **complementing (Lopatinskii–Shapiro) condition**
with respect to Lᵖ. When all of this holds, the **ADN a-priori estimate** is

    Σⱼ ‖Uⱼ‖_{q+tⱼ}  ≤  C ( Σᵢ ‖(LU)ᵢ‖_{q−sᵢ} + Σₖ ‖(BU)ₖ‖_{q−rₖ−½,Γ} + Σⱼ ‖Uⱼ‖₀ )

for every q ≥ max(0, rₖ+1), the last term dropping when the problem has a
unique solution. Three facts drive everything below:

1. **The weights are not unique**, and different admissible (sᵢ, tⱼ) give
   different principal parts, different complementing conditions and
   different estimates for the *same* system (Bochev & Gunzburger 1998, §3.2.1
   — the VVP system has two principal parts, their (3.27) and (3.28)).
2. **Lower-order terms are invisible to ellipticity** but not to the constant
   C, and when a lower-order term carries a large coefficient (our c u) the
   estimate is uniform only if that coefficient is promoted into the principal
   part — Agranovich–Vishik *parameter-ellipticity*, in which the parameter is
   counted as a derivative: c ~ |ξ|.
3. **A least-squares functional is norm-equivalent iff it measures each
   residual in the norm the estimate names**: J(U) = Σᵢ ‖(LU)ᵢ − Fᵢ‖²_{q−sᵢ}
   ≃ Σⱼ ‖Uⱼ‖²_{q+tⱼ} (Aziz, Kellogg & Stephens 1985; Bochev–Gunzburger
   §2, (2.3)–(2.7)). The plain L² functional is norm-equivalent in a product
   of H¹ spaces only for systems admitting **sᵢ = 0, tⱼ = 1 for all i, j** —
   "fully H¹-coercive", Petrovsky type. Otherwise one uses negative norms
   (H⁻¹ methods, Bramble–Lazarov–Pasciak 1997), or mesh-weighted L² norms
   (h^{−2sᵢ}, Aziz–Kellogg–Stephens; BG §4.2), or augments the system with
   redundant equations until it becomes Petrovsky (BG §3.3).

## 2. The VVP Stokes system through ADN eyes

Rows (our numbering, `lssem3d/operator.py`): continuity ∇·u; three
vorticity definitions ∇×u − ω; three momentum c u + ν∇×ω + ∇p; and row 7,
∇·ω = 0. Unknowns (u, ω, p).

**Steady case (c = 0), 7×7 without row 7.**

- *Index choice A* — s = (0,0,0,0,0,0,0), t = (1,1,1,1,1,1,1) (everything
  first order, everything in H¹). Principal symbol rows: momentum
  iν ξ×ω̂ + iξ p̂; continuity iξ·û; curl iξ×û (the −ω̂ entry is order 0 <
  0+1, so it is *dropped*). Null vector: û = 0, p̂ = 0, ω̂ = ξ — a gradient
  vorticity is invisible to ∇×. **Not elliptic in 3D.** This is why Bochev &
  Gunzburger (and Chang, Cai–Manteuffel–McCormick) add the "seemingly
  redundant" ∇·ω = 0, our **row 7**: it contributes iξ·ω̂, kills the null
  vector, and the augmented 8×7 system is fully H¹-coercive with velocity or
  normal-velocity–pressure boundary conditions (BG (3.23), (3.33)). In 2D
  ω is scalar and the 4×4 system is already Petrovsky.
- *Index choice B* — s_mom = 1, s_div = s_curl = 0; t_u = 1, t_ω = t_p = 0
  (BG's L₂ᵖ, (3.28)). Now −ω̂ *is* principal in the curl row, ∇p and ν∇×ω are
  principal in momentum (order 1 = 1+0), and the 7×7 symbol is non-singular
  without row 7: curl row gives ω̂ = iξ×û, momentum then gives |ξ|²û −
  ξ(ξ·û) + iξp̂ = 0, continuity ξ·û = 0, hence p̂ = 0, û = 0. The estimate is
  (3.34): ‖ω‖_{q+1} + ‖p‖_{q+1} + ‖u‖_{q+2} ≤ C(‖ν∇×ω+∇p‖_q + ‖∇×u−ω‖_{q+1} +
  ‖∇·u‖_{q+1}). With q = −1 this is the H⁻¹ method, u ∈ H¹, ω, p ∈ L² only.

Which choice applies is decided by the boundary conditions through the
complementing condition: with **velocity (no-slip) boundary conditions only
choice B satisfies it** (BG p. 809, counterexample ω = −cos(nx)eⁿʸ), so the
plain L² VVP functional with no-slip walls is *not* norm-equivalent to H¹ ×
H¹ × H¹, and Table 1 of BG shows the consequence: velocity converges at the
best-approximation rate, **vorticity and pressure one order below**. Our
§7J finding (TG order 2.00 → 1.72 when row 7 is down-weighted) and the
−10 % p′, −16 % ω′ₓ at the wall in the channel comparison are the same
phenomenon; velocity statistics are unaffected, exactly as BG state ("does
not cause a catastrophic loss of accuracy, at least for the velocity").

**Row 7's weight w₇ = 10⁻⁴ in ADN terms.** Row 7 exists to give ω an H¹
estimate (choice A). Down-weighting it moves the operator from choice A
towards choice B: ω is then controlled in L² through the curl row, which is
all the time-stepping needs, and the measured facts follow — zero change in
the discrete H¹ ellipticity ratio (§7S.2) because the ω-gradient null
direction is a *continuous* deficiency that the discrete curl row already
closes, and the 139–2885× conditioning gain because at w₇ = 1 the row adds a
derivative-squared term to the ω diagonal that nothing else in A matches.

## 3. The time-dependent operator: the norm depends on c

With the momentum row weighted 1/c² (rw[4:7]) the functional per mode is

    J = ‖∇·u‖² + ‖∇×u − ω‖² + ‖u + (ν∇×ω + ∇p)/c‖² + w₇‖∇·ω‖².

Parameter-elliptic reading: the mass term u is order 0 with coefficient 1,
the derivative terms ν∇×ω/c, ∇p/c are order 1 with coefficient 1/c. The
principal symbol changes character at **c* ≈ ν |ξ|²_max ≈ ν p⁴/h²**:

- **c ≪ c*** (steady-like; the cavity, 2D §6.9, c = 1): the momentum row
  supplies O(1) control of ν∇×ω ≈ ν∇×∇×u. Eliminating ω, u is controlled by
  ‖u‖² + ‖∇·u‖² + ν²‖∇×∇×u‖²/c²: divergence *and* curl — the intersection
  H(div)∩H(curl) = H¹ on our domains. Every block is Laplacian-like, the
  functional is equivalent to H¹×H¹×H¹ (plus row 7), and this is the regime in
  which Jacobi-smoothed p-multigrid is provably optimal (Cai–Manteuffel–
  McCormick 1997). Measured: PMG p-independent at c = 1 (§7U).
- **c ≫ c*** (the channel: c = 5405, ν p⁴/h² ≈ 10²): the momentum row
  degenerates to ‖u‖². Eliminating ω from the curl row (it is an *order-0
  slave*, ω = ∇×u to the accuracy of that row) leaves u with ‖u‖² + ‖∇·u‖² —
  the **H(div) norm**, and nothing else. A solenoidal, curl-rich u with its
  matched ω is an eigenvector of A with eigenvalue ~1 while a gradient u
  sees the full p⁴/h² of ∇·; the ratio is the Jacobi condition number.
  Pressure appears only through ‖∇p‖²/c², orthogonal (Helmholtz) to the
  ν∇×ω/c term, so the p block is a scaled Neumann Laplacian and decouples.

So the large-c functional is norm-equivalent to

    |||(u,ω,p)|||² = ‖u‖²_{H(div)} + ‖ω − ∇×u‖²₀ + c⁻²|p|²₁      (up to O(ν/c))

— note the **shifted** vorticity: the curl row makes J a norm on the pair
(u, ω − ∇×u), not on u and ω separately. In the original variables the
Riesz map is therefore block-*triangular*, not block-diagonal: an H(div)
solve for u whose right-hand side has been corrected by the vorticity
residual, a mass solve for ω, and a Laplacian solve for p. Any preconditioner
that treats u and ω as independent blocks — point Jacobi, node-block Jacobi,
per-variable blocks — measures the near-null pair (u, ∇×u) with ‖∇×u‖²
where the operator measures it with ‖u‖², and pays p⁴/h² for it (§5).
This is the ADN content of the "c-regime" finding of §7U and
FOSLS_TIME_DEPENDENT §3: p-independence of Jacobi-smoothed multigrid is a
property of the H¹ regime, and the channel is not in it. It also explains
the two half-successes in the parking lot: element-block Schwarz (6.1×) is a
patch smoother, which is what H(div) multigrid needs (Arnold–Falk–Winther
2000), and it helped as much as an element patch can; the momentum-weight
increase (w_mom = 100) moves c* up by re-injecting ∇×ω control, and was
rejected because it buys that with 109× worse ∇·u — it changes the
functional, not just the preconditioner.

## 4. What ADN prescribes: operator (Riesz-map) preconditioning

Mardal & Winther (2011): if a(U,U) ≃ |||U|||² with constants C₁ ≤ C₂ then
the Riesz map B of |||·||| — block-diagonal, one block per factor of the
product norm — satisfies κ(B⁻¹A) ≤ C₂/C₁ with no dependence on h, p or the
parameters that were kept uniform in the estimate. For a *conforming* FOSLS
discretization the same constants hold discretely. Hence the test: take the
**exact** block-diagonal of A itself (the strongest block-diagonal
preconditioner there is) with the blocks the norm dictates. If κ is flat and
small, the ADN norm is right and the remaining task is to *approximate*
those blocks; if it grows, no norm-based block preconditioner can work.

## 5. Measured on the assembled operator

Setup: channel mesh 2×2 elements (h-sweep 2–4), Lx = π, N = 4…12, ν = 1/180,
one Fourier mode at a time (k_z = 0 and k_z = 5.88), pressure pinned at the
k_z = 0 mode only, rows weighted as in production (rw[4:7] = 1/c², w₇ = 10⁻⁴).
A is assembled sparsely per mode through `gidx` (the DirectCoarseE path),
masked dofs removed, κ from the CG Lanczos Ritz values at tol 10⁻¹⁰.
Blocks: u = {u,v,w}, ω = {ωₓ,ω_y,ω_z}, p, each real+imaginary.

**5.1 Which block structure is the norm?** (`adn_block_precond.py`)

| c | k_z | p | ndof | Jacobi | 7 blocks (per variable) | 3 blocks u‖ω‖p | **2 blocks (u,ω)‖p** |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 4 | 455 | 175 / 1.6e3 | 116 / 150 | 116 / 150 | **16 / 3.3** |
| 1 | 0 | 12 | 4055 | 701 / 4.4e4 | 130 / 180 | 128 / 180 | **17 / 3.5** |
| 1 | 5.88 | 12 | 8112 | 505 / 3.7e3 | 130 / 180 | 128 / 180 | **17 / 3.6** |
| 5405 | 0 | 4 | 455 | 209 / 1.6e3 | 170 / 390 | 166 / 380 | **15 / 3.3** |
| 5405 | 0 | 8 | 1807 | 620 / 1.2e4 | 426 / 3.3e3 | 424 / 3.2e3 | **14 / 3.3** |
| 5405 | 0 | 12 | 4055 | 1201 / 3.7e4 | 827 / 1.5e4 | 798 / 1.4e4 | **13 / 3.3** |
| 5405 | 5.88 | 12 | 8112 | 969 / 2.3e4 | 824 / 1.5e4 | 780 / 1.4e4 | **9 / 1.4** |

(entries: CG iterations / κ). h-sweep at p = 8, c = 5405, 2×2 → 4×4
elements: Jacobi 620 → 1244 iterations, κ 1.2e4 → 5.5e4; **2-block 14 →
13 iterations, κ 3.3 → 3.3** (k_z = 0), 9 → 9, κ 1.4 (k_z = 5.88).

So: the exact (u,ω)‖p block preconditioner is **optimal — κ ≤ 3.6, h-, p-
and c-independent**, which is the Mardal–Winther bound C₂/C₁ for the ADN
norm of §3. Separating u from ω costs everything at large c (κ 1.4e4 at
p = 12, within 2.6× of Jacobi) and nothing at c = 1 (κ 180 flat): the
u–ω coupling is subdominant in the H¹ regime and dominant in the H(div)
regime — the regime boundary predicted by the parameter-elliptic reading.
The diagonal blocks themselves are benign: κ(A_uu) ∝ p² (52 → 1.1e3),
κ(A_ωω) ∝ p (25 → 200), κ(A_pp) ∝ p⁴ (1.5e3 → 4.3e4 — a Neumann Laplacian,
which scalar p-multigrid handles).

**5.2 Is the pair norm H(div) × L²?** (`adn_block_precond2.py`,
`adn_block_precond3.py`) No — it is a norm on (u, ω − ∇×u). Riesz blocks
B_u = M + K_div (H(div)), B_ω = M_ω, B_p = A_pp, block-*diagonal*: κ =
2.5e6 at p = 8, c = 5405 (worse than Jacobi). The block-LDLᵀ form with the
exact Schur complement S_u = A_uu − A_uω A_ωω⁻¹ A_ωu reproduces the optimal
κ = 3.3 exactly (it is the same preconditioner); with S_u replaced by
M + K_div it gives κ 460 → 2.7e4 (p 4 → 12). The difference between S_u
and M + K_div is the **discrete curl-representation defect** K_curl −
Cᵀ M_ω⁻¹ C: ∇×u_h is degree p−1 and discontinuous across element faces,
so the C⁰ vorticity cannot represent it there, and min_ω ‖∇×u_h − ω_h‖²
is a face-jump penalty on ∇×u — zero at interior nodes, O(‖∇×u_h‖²) for
the roughest modes. It cannot be dropped.

**5.3 Where the difficulty finally sits.** (`adn_block_precond4.py`)
Since A_ωω ≈ M_ω is *diagonal* on GLL nodes, S_u is applicable matrix-free
(two operator applies), so the vorticity can be eliminated exactly and the
solve reduced to S_u on the 6 real velocity fields. That alone gains
nothing: Jacobi on S_u, κ 3.5e2 → 5.5e4 (p 4 → 12), the same as on A; and
Jacobi on the pure H(div) operator M + K_div is the same again, 1.8e2 →
1.4e4. The softest Jacobi-preconditioned eigenvector of S_u has
**‖∇·u‖/‖u‖ = 0 to machine precision and ‖∇×u‖/‖u‖ = O(1)**: a smooth,
exactly divergence-free velocity — the kernel of the divergence, on which
M + K_div is the identity while its diagonal is p⁴/h². This is precisely
the failure of pointwise smoothers in H(div) described by Arnold, Falk &
Winther (2000) and Hiptmair (1997): the solenoidal subspace is ~2/3 of the
space, invisible to Jacobi, and not corrected by a coarse grid built from
nodal interpolation.

## 6. What this changes

1. **The diagnosis is closed.** The channel's c = 5405 puts the operator
   in the H(div) regime; the ADN norm there couples u and ω through the
   curl row and reduces, after exact vorticity elimination, to an
   H(div)-type operator whose solenoidal kernel defeats every pointwise or
   per-variable preconditioner we tried (Jacobi, node-block, per-variable
   block, PMG with Jacobi smoothing, Galerkin/direct coarse). None of those
   could have worked; the measurements in §7U were not solver bugs.
2. **The target is now specific and its ceiling is measured:** a
   preconditioner that solves the (u,ω) block *together* to within a fixed
   factor, plus a scalar Poisson solve for p, gives κ ≈ 3, ~10–15 CG
   iterations per stage against the present ~2000–6000, independent of p,
   h and dt. That is the whole prize; nothing in the pressure block or in
   the row weights can add to it.
3. **The (u,ω) block is an H(div) problem, so use H(div) technology.**
   (Written before the 2D test of §8, which settles the ranking: the
   vertex-patch Schwarz works, the nodal Hiptmair sweep does not.)
   - *Hiptmair (hybrid) smoother* inside the existing p-multigrid: after a
     Jacobi sweep on S_u (or on the (u,ω) block), a second Jacobi sweep in
     the potential space, u = ∇×ψ, on the transformed operator
     (∇×)ᵀ S_u (∇×), which sees the solenoidal kernel as a mass-like
     operator. Cost per sweep: two extra curl applications. This is the
     3D H(div) multigrid of Hiptmair (ETNA 1997) and the "auxiliary space"
     idea of Hiptmair & Xu (2007), on our C⁰ GLL discretization instead of
     Raviart–Thomas — the theory is for RT/Nédélec, so this is a measured
     bet, not a theorem.
   - *Overlapping vertex-patch Schwarz smoother* (Arnold–Falk–Winther
     2000): the element-block Schwarz of §6.3 was the non-overlapping,
     zero-kernel-awareness version and still gained 6.1×; AFW's result is
     that vertex patches (which contain the local solenoidal modes) make
     the V-cycle optimal. On SEM this means patches of the 4 (2D) elements
     around each vertex, i.e. the same Schwarz machinery with a different
     patch — the memory objection of §6.3 remains unless done matrix-free
     by local solves.
   - *Discretization-level alternative:* represent ω in a discontinuous
     (per-element, degree p−1) space so that ∇×u_h is exactly
     representable and the curl-representation defect vanishes; then S_u
     is exactly M + K_div and the standard H(div) theory applies verbatim.
     This changes the discretization, not just the solver.
4. **Re-ranking the parking lot** (FOSLS_TIME_DEPENDENT.md §6):
   - §6.2 preconditioner-only artificial compressibility targets the
     pressure block. The pressure block is *not* the bottleneck (§5.1: the
     p-block is a Laplacian that decouples and that exact-solving it inside
     the 2-block preconditioner costs 13 iterations total). Its 2D gain
     came from a 2D system in which the (u,ω) block has no kernel of this
     kind (ω scalar). Demote.
   - §6.3 element Schwarz — keep the finding, but the lesson is that the
     *patch* (vertex, overlapping) is what matters, not the block. Promote
     the AFW/Hiptmair form.
   - §6.1 warm start (1.08×) and §6.4 per-mode solves (2.0×) are
     orthogonal and still worth their measured amounts.
   - §6.5 implicit convection to lower c: ADN says lowering c below
     c* ≈ ν p⁴/h² would restore the H¹ regime — for the channel that is
     c ~ 10², i.e. dt ~ 10⁻², 12× the CFL limit. Not reachable.
   - §6.6 κ_p ∝ dt: consistent with the p-block scaling c⁻²|p|₁²; no
     action.
5. **Accuracy corollary.** With no-slip walls the L² VVP functional is
   not norm-equivalent in H¹ for ω and p (BG Table 1: velocity optimal,
   vorticity and pressure one order lower). The −10 % p′ and −16 % ω′ₓ at
   the wall in the reference comparison, and §7J's TG order drop for ω, are
   the expected signature and are not curable by the solver. The
   textbook cure is the H⁻¹ (BLP discrete-minus-one) momentum norm, whose
   discrete form (4.27 in BG) is the same block-preconditioner machinery
   applied inside the functional — an item for after the solver.

## 7. Sources

- Agmon, Douglis & Nirenberg, Comm. Pure Appl. Math. 12 (1959) 623; 17 (1964) 35.
- Aziz, Kellogg & Stephens, "Least squares methods for elliptic systems", Math. Comp. 44 (1985) 53 — https://www.ams.org/journals/mcom/1985-44-169/
- Bochev & Gunzburger, "Finite element methods of least-squares type", SIAM Rev. 40 (1998) 789 — read in full: §2.1 (2.3)–(2.7), §3.2.1 (3.20)–(3.35), §4.1.1 Table 1, §4.2.1 (4.13)–(4.16), §4.3 (4.25)–(4.27). https://people.sc.fsu.edu/~mgunzburger/files_papers/gunzburger-ls-review.pdf
- Bochev & Gunzburger, *Least-Squares Finite Element Methods*, Springer 2009 (ADN chapter; VVP with div ω = 0).
- Cai, Manteuffel & McCormick, "First-order system least squares for the Stokes equations, with application to linear elasticity", SIAM J. Numer. Anal. 34 (1997) 1727 — https://epubs.siam.org/doi/10.1137/S003614299527299X
- Bramble, Lazarov & Pasciak, "A least-squares approach based on a discrete minus one inner product for first order systems", Math. Comp. 66 (1997) 935; Bramble & Pasciak, J. Comput. Appl. Math. 74 (1996) 155 (Stokes).
- Agranovich & Vishik, "Elliptic problems with a parameter and parabolic problems of general type", Russ. Math. Surv. 19 (1964) 53.
- Mardal & Winther, "Preconditioning discretizations of systems of partial differential equations", Numer. Linear Algebra Appl. 18 (2011) 1 — https://onlinelibrary.wiley.com/doi/abs/10.1002/nla.716
- Arnold, Falk & Winther, "Multigrid in H(div) and H(curl)", Numer. Math. 85 (2000) 197 — https://link.springer.com/article/10.1007/PL00005386
- Hiptmair, "Multigrid method for H(div) in three dimensions", ETNA 6 (1997) 133 — https://people.math.ethz.ch/~hiptmair/FILES/ETNA97.pdf
- Hiptmair & Xu, "Nodal auxiliary space preconditioning in H(curl) and H(div) spaces", SIAM J. Numer. Anal. 45 (2007) 2483.
- Scripts: `scratch/adn_block_precond{,2,3,4}.py`, logs `scratch/adn_block_precond*.log`.

## 8. 2D test (2026-09-07): the diagnosis transfers, Hiptmair fails, vertex patches work

`scratch/adn_hiptmair_2d.py`, `scratch/adn_schwarz_2d.py`,
`scratch/adn_schwarz_2d_full.py`. 2D VVP (u, v, p, ω), cavity mesh 4×4
elements, Re = 1000, fu = fv = 0, legacy weights a_mass = 1, a_flux = dt with
dt = 1.85e−4 (c = 5405), pressure pinned.

**8.1 Same structure as 3D.** Exact (u,v,ω)‖p block: 9 iterations, κ = 1.9
at N = 8, 12, 16. Splitting ω off: κ 5.5e4 → 6.3e5, within 2× of Jacobi
(9.5e4 → 1.3e6). A_ωω is diagonal to 1e−10 (GLL mass), so ω is eliminated
exactly and for free; the softest Jacobi mode of the Schur complement S_u has
‖∇·u‖/‖u‖ = 4e−3, ‖∇×u‖/‖u‖ = 100: the divergence-free kernel again.

**8.2 Nodal Hiptmair sweep: no.** Auxiliary space ψ ∈ C⁰ Q_p, C: ψ ↦
nodal average of (ψ_y, −ψ_x), D_ψ = diag(CᵀS_uC). Additive D_S⁻¹ + C D_ψ⁻¹
Cᵀ: κ 1.2e5 → 3.3e6 (worse than Jacobi). Damped symmetric multiplicative
J–H–J: κ 2.7e4 → 5.0e5, a constant 2.4× over Jacobi with the same p-growth.
Reason: the discrete kernel {u_h ∈ [Q_p]²∩C⁰ : ∇·u_h = 0 at the GLL nodes}
is not the image of any potential space on C⁰ nodal elements — the nodal
average of ∇⊥ψ_h is not divergence-free and CᵀS_uC is dominated by the
interface defect (its Jacobi damping came out 0.04–0.09). Hiptmair–Xu's
theorem needs the exact de Rham complex (Nédélec/Raviart–Thomas), which C⁰
GLL does not have. Withdrawn as the recommendation.

**8.3 Overlapping vertex-patch Schwarz (Arnold–Falk–Winther; Pavarino for
spectral elements): yes.** Patch = every dof of the (up to four) elements
sharing a mesh vertex, exact local solve, additive, on S_u at c = 5405:

| N | ndof | Jacobi | element blocks (no overlap) | **vertex patches** | + p=2 Galerkin coarse |
|---|---|---|---|---|---|
| 8 | 1922 | 1243 / 6.5e4 | 646 / 1.8e4 | **40 / 20** | 40 / 22 |
| 12 | 4418 | 2459 / 3.6e5 | 1168 / 7.9e4 | **37 / 20** | 35 / 14 |
| 16 | 7938 | 3980 / 1.2e6 | 1797 / 2.3e5 | **33 / 13** | 32 / 13 |

(iterations / κ). Iterations *decrease* with p, the Pavarino signature. The
non-overlapping element blocks replicate §6.3's ~6× and no more: the
overlap is the whole effect — a vertex patch contains the local
divergence-free modes that no element-interior block can represent. The
coarse correction is immaterial on 16 elements; the h-sweep is in §8.4.

**8.4 On the full operator (no elimination), both regimes, h-sweep.**
Vertex patches now contain u, v, p, ω; + Galerkin p = 2 coarse correction:

| c | mesh | N | Jacobi | vertex patch | vertex patch + coarse |
|---|---|---|---|---|---|
| 5405 | 4×4 | 8 | 1736 / 9.5e4 | 53 / 660 | **42 / 27** |
| 5405 | 4×4 | 12 | 3110 / 4.3e5 | 46 / 840 | **33 / 15** |
| 5405 | 4×4 | 16 | 5103 / 1.3e6 | 44 / 950 | **31 / 17** |
| 1 | 4×4 | 8–16 | 1355–2965 | 50–48 / 720–1400 | **30 / 10–19** |
| 5405 | 6×6 | 8 | 2743 / 2.1e5 | 72 / 2.0e3 | 46 / 29 |
| 5405 | 8×8 | 8 | 3616 / 3.7e5 | 90 / 4.0e3 | 53 / 45 |

Vorticity elimination is not needed: the patch on the full operator works
the same (the residual κ ≈ 700–1000 without the coarse level is a global
pressure mode, which the coarse space removes). The same preconditioner is
optimal in the H¹ regime (c = 1) — it is not tuned to the H(div) regime, it
simply contains the modes both regimes need. The one-level method degrades
with the element count, as two-level Schwarz theory says it must; with the
coarse correction the growth is mild (27 → 45 over 4× the elements).

**8.5 Overlap width is not negotiable.** Full operator, c = 5405, 4×4,
all with the p = 2 coarse correction (iterations / κ; max patch dofs):

| N | element block | element + 1 node layer | element + 2 layers | **vertex patch** |
|---|---|---|---|---|
| 8 | 655 / 2.0e4 (324) | 330 / 3.4e3 (468) | 231 / 1.2e3 (612) | **42 / 27** (1156) |
| 12 | 968 / 8.5e4 (676) | 606 / 1.4e4 (884) | 429 / 5.2e3 (1092) | **33 / 15** (2500) |
| 16 | 1285 / 2.4e5 (1156) | 912 / 3.5e4 (1428) | 577 / 1.4e4 (1700) | **31 / 17** (4356) |

Thin overlaps keep the p-growth; only the generous (element-sized) overlap
of the vertex patch is p-independent — Pavarino's condition. The 3D cost
must therefore be paid for the full four-element patch.

## 9. Net-net after the 2D test

**Recommendation: overlapping vertex-patch additive Schwarz on the full
(u, ω, p) operator, with the existing DirectCoarseE p = 2 Galerkin coarse
correction, as the PCG preconditioner (or as the PMG smoother).** Measured
in 2D at the channel's c: 31–42 CG iterations, κ 15–27, independent of p
and of the c-regime, against 1700–5100 for Jacobi. The Hiptmair
auxiliary-space route is withdrawn (§8.2): C⁰ GLL elements have no
potential representation of the discrete divergence-free kernel.

Why this is consistent with everything measured before: element-block
Schwarz (§6.3) was the same idea with zero overlap and gained the 6× that
zero overlap can give; the exact (u,ω)‖p solve (§5.1) is the κ ≈ 3 ceiling
that the patches approach to within 5–10×; and the coarse space is what
turns the one-level method's h-growth into a bounded κ.

**3D cost, production channel** (6×18 elements, periodic in x, N = 8,
17 modes): 6×19 = 114 vertex patches of (2N+1)² = 289 nodes × 14 real
dofs = 4046 dofs, i.e. 2023 complex. Per patch and mode, a complex
Hermitian Cholesky factor is 33 MB in double, 16 MB in single: 114 × 17 ×
16 MB = **31 GB in single precision** — fits Spark's 121 GB unified memory,
and since c differs per RK stage the factors are either kept for all three
stages (93 GB) or refactored per stage (≈5e12 flop, well under a second on
the GB10). Application per CG iteration ≈ 3e10 flop, the same order as one
operator apply. At ~40 iterations per stage against the present 2000–6000
this is a **40–100× reduction in solver work per stage**, from the current
~20–50 s to ~1 s, and it leaves the matrix-free operator untouched — the
memory objection of §6.3 is answered by precision and by the fact that the
factors are the only dense object.

What is *not* known yet and must be measured before touching production:
(i) the 3D per-mode operator on the assembled harness with vertex patches
(same script family, `adn_block_precond*.py`) — expected to reproduce the
2D numbers because the k_z coupling is inside the patch; (ii) whether the
p = 2 coarse correction suffices on 6×18 elements (2D h-sweep: κ 27 → 45
for 4× the elements); (iii) whether single-precision factors hold the
iteration count (they are a preconditioner, so any loss shows only as
iterations); (iv) the k_z = 0 mode with its pinned pressure.

## 10. Production-path prototype in 2D (`scratch/vertex_schwarz2d.py`)

Built the way it would go into `lssem2d/precond.py`: the global operator
stays matrix-free (`pcg_solve` unchanged); element-local blocks come from
(N+1)²·4 probes of the element operator, the same trick as DirectCoarseE;
each vertex patch's dense block is glued from those; symmetric Jacobi
equilibration then Cholesky; `__call__(r)` has PMG2's interface. Two-level
term = PMG2's own restrict / DirectCoarse(p = 2) / prolong, added.
Verified against the assembled reference to 2.6e−13.

**Lesson that cost one wrong run:** the Schwarz block is R A Rᵀ of the
*assembled* operator. Summing only the patch's own four element blocks
leaves the patch-boundary nodes under-stiffened (their outside neighbours
are missing), 16 of 25 blocks were singular and CG did not move. The ring of
neighbouring elements must contribute, restricted to the patch's dofs.

Benchmark, production `pcg_solve`, cold RHS, tol 1e−8, Re = 1000 cavity
(iterations; wall time in this numpy/scipy prototype in brackets):

| c | mesh | N | Jacobi | PMG2 ladder | **vertex patch + p=2 coarse** |
|---|---|---|---|---|---|
| 5405 | 4×4 | 8 | 1286 | 178 | **31** (0.6 s) |
| 5405 | 4×4 | 12 | 2239 | 329 | **26** (3.2 s) |
| 5405 | 4×4 | 16 | 3497 | 522 | **25** (15.6 s) |
| 5405 | 6×6 | 8 | 2089 | 286 | **35** |
| 5405 | 8×8 | 8 | 2938 | 372 | **41** |
| 1 | 4×4 | 8–16 | 1191–2644 | 60–54 | **27–26** |
| 1 | 8×8 | 8 | 2232 | 51 | **33** |

Iterations are p-independent (falling 31 → 25) in the H(div) regime where
the ladder grows 3×, and match the ladder's regime-independence at c = 1.
h-growth with the p = 2 coarse level is 31 → 41 over 4× the elements.

**Cost per iteration is the real question, not the count.** A dense patch
solve is O(n_patch²) = O((N+1)⁴) per patch against O((N+1)³) per element
for the matrix-free apply: at N = 16 in 2D the patch apply is ~1e9 flop per
iteration against ~4e6 for the operator, which is why the prototype's wall
time loses to Jacobi despite 140× fewer iterations. For the 3D channel per
mode (114 patches × 17 modes × 2023 complex dofs) the patch apply is
≈ 3e10 flop per CG iteration — about the same as the present operator apply
measured at ~9 ms on the GB10 — so ~40 iterations against 2000–6000 is a
30–60× reduction in solve time, provided the patch solves run as batched
dense triangular solves on the GPU (they will; that is the one kernel GPUs
do at peak). Factor storage 31 GB in single precision (§9). A cheaper
variant to measure next: use the patch solve only as the smoother on the
p = 4 level of the existing PMG ladder (patches 4× smaller in dofs, 16× in
flops) with Chebyshev on the fine level, and see how much of the
p-independence survives.

## 11. Time-marching test: 2D Poiseuille start-up, Re = 100 (`scratch/pois2d_vs.py`)

Periodic channel Lx = 2, Ly = 1, 4×4 elements N = 8, body force G = 12ν in
the x-momentum row (`f_known`), from rest, BDF1 then BDF2, dt = 0.1
(c = 1/dt = 10), 600 steps to T = 60, production `step_bdf`/`newton_step`
with the new `state.precond_factory` hook, CG tol 1e−10 abs / 1e−8 rel,
numpy backend on the Mac. The preconditioner is rebuilt at every step (the
2D operator carries the linearised convection). Checked against the exact
transient u(y,t) = 6y(1−y) − Σ 48/(nπ)³ sin(nπy) e^{−νn²π²t}, not only the
steady parabola. Figure `figs_fosls_vs_fs/pois2d_vs.png`.

| preconditioner | CG it/step mean / max | CG wall | build wall | total | err vs exact(T) |
|---|---|---|---|---|---|
| Jacobi | 658 / 787 | 88.5 s | 0 | 90 s | 3.2e−7 |
| PMG2 ladder (4,2)+direct | 28 / 49 | 80.3 s | 20.3 s | 102 s | 3.2e−7 |
| vertex-patch Schwarz + p=2 coarse | **23 / 27** | 320 s | 123 s | 444 s | 3.2e−7 |

All three reproduce the transient to 3e−7 and the same profile (they solve
the same system to the same tolerance, as they must). Iterations: 29× fewer
than Jacobi, flat across the whole run (27 → 20), and below the ladder's
(49 → 20). Wall time: **worse by 5×** in this prototype — a dense patch
solve (25 patches × 1156²) costs ~100× one matrix-free apply at this size,
and the per-step rebuild (probing + 25 Cholesky factorisations, 0.2 s)
is as much as Jacobi's entire step. Two things to keep in mind when
reading that: at c = 10 with ν = 0.01 this case sits in the H¹ regime
(c* ≈ ν p⁴/h² ≈ 650), where Jacobi needs only ~700 iterations and the
ladder is already good — the H(div) regime that motivated the method is
c ≫ c*; and the wall time is a numpy/scipy loop over patches, not a batched
device solve. The production question is therefore the cost model of §9–10,
not this number; but this test shows the method is correct inside the
time-stepping path and that its iteration count does not drift with the
evolving linearisation.

**Bug found on the way (fixed in the script, not in the library):**
`step_bdf` updates the history list in place; reassigning it after the call
duplicated the newest state, BDF2 saw uⁿ = uⁿ⁻¹ and the flow developed at
2/3 speed (u_c = 0.699 vs 0.923 at t = 10). The BDF1 first step being exact
gave it away.

### 11.2 The same case at c = 2000 (dt = 5e−4), smooth and rough right-hand sides

c* ≈ ν p⁴/h² ≈ 655 on this mesh, so c = 2000 is on the H(div) side. Start-up
from rest to T = 1 (2000 steps), vertex-patch factors refreshed every 10
steps; then 200 steps from a *rough* initial field (parabola + 10 % random
C⁰ velocity noise), which is what a DNS hands the solver at every step.
Figures `figs_fosls_vs_fs/pois2d_vs_c2000.png`, `…_rough.png`.

| c = 2000 | Jacobi | PMG2 ladder (4,2) | vertex patch + coarse |
|---|---|---|---|
| start-up from rest, 2000 steps: CG it/step mean / max | 650 / 672 | 54 / 94 | **24 / 26** |
| … wall (numpy prototype) | 294 s | 585 s | 1160 s |
| … err vs exact transient at T = 1 | 3.5e−7 | 6.0e−7 | 5.0e−8 |
| rough initial field, 200 steps: CG it/step mean / max | 1177 / 1422 | 150 / 174 | **33 / 36** |
| … wall (numpy prototype) | 53 s | 148 s | 162 s |
| … ms per CG iteration | 0.22 | 4.7 | 24 |

Three things this shows.

1. **A smooth right-hand side does not exercise the kernel.** Jacobi needed
   650 iterations at c = 2000 exactly as at c = 10: the laminar start-up
   residual is x-independent and smooth, and projects onto few modes. The
   cold random-RHS harness (§8) and the DNS do not have that luxury.
2. **With a rough field the ordering is the one the theory predicts.**
   Jacobi doubles (1177, max 1422), the ladder triples (150, max 174 — it
   is p-multigrid with pointwise smoothing, blind to the divergence-free
   kernel), the vertex patch rises 24 → 33 and stays flat over the 200
   steps as the noise decays: 36× fewer iterations than Jacobi, 4.5× fewer
   than the ladder, on the load that matters.
3. **On a CPU in numpy at N = 8 the dense patch cannot win wall time.** One
   patch application costs 110 Jacobi iterations here (24 ms vs 0.22 ms),
   because the operator apply on 16 elements is tiny and the 25 dense
   1156² solves are not. The iteration reduction (36×) does not cover it
   (net 3× slower). This is the cost model of §9–10, measured: the method
   pays off only where the operator apply is expensive relative to batched
   dense solves — the 3D channel on the GPU (per-mode apply ≈ patch apply
   in flops, 2000–6000 iterations to beat) — and not on a small 2D CPU
   problem. It also says the p = 4-level smoother variant (§5.1, patches 16×
   cheaper) is worth measuring before the 3D port.

Refreshing the patch factors every 10 steps instead of every step cost
nothing in iterations (24 → 24 smooth, 33 rough) and cut the build from
123 s per 600 steps to 41 s per 2000.

## 12. High order: lid-driven cavity, N = 30, 4×4 elements (`scratch/cavity32k_pmgv.py`, `scratch/pmgv2d.py`)

Re = 32000, time marching from rest, BDF, dt = 2e−3 (c = 500 > c* ≈ 405),
5 steps, CG absolute tol 1e−8, PMG preconditioners frozen on a snapshot
linearisation and refreshed every 10 steps. `PMGV` = the PMG2 ladder
30→15→7→3→2 (direct at p = 2) with the damped additive vertex-patch Schwarz
as the smoother on every level with p ≤ p_patch and Chebyshev(4) above it.

| preconditioner | CG it/step mean / max | ms per CG iteration | s per step (numpy, M3 Max) |
|---|---|---|---|
| Jacobi | 8365 / 11822 | 1.2 | 10 |
| ladder, Chebyshev all levels | 1994 / 2886 | 20 | 40 |
| ladder, patch smoother on p ≤ 7 | 2793 (step 1) | 41 | 116 |
| ladder, patch smoother on p ≤ 15 | 2596 (step 1) | 1000 | 2584 (stopped) |
| **fine-level vertex patches + p = 2 coarse** | **19 / 25** (25, 20, 18, 17, 15) | 7700 | 157 (+48 s build per refresh) |

u, v, p agree across all methods to 4e−5; ω differs by up to 0.15 at the
lid corners (where it is singular and O(10²)), which is the ill-conditioning
converting a 1e−8 residual into a visible vorticity difference — not a
preconditioner effect.

Reading: (1) **patches on coarse levels only do not work** — the
divergence-free modes that stall CG have p = 30 content and no smoother
below the fine level touches them (2886 → 2596 for a 50× cost); §5.1's
variant is closed. (2) **Fine-level patches work at N = 30 exactly as at
N = 8**: 19 iterations, falling over the steps, 100× fewer than the ladder
and 440× fewer than Jacobi. (3) The price is the patch size, (2N+1)²·4 =
14,884 dofs, 1.77 GB per dense factor, 44 GB for 25 patches, and 7.7 s per
application in numpy — 6600× a Jacobi iteration — so on this machine the
wall time is 16× worse than Jacobi even at 440× fewer iterations. The
O((2N+1)⁴) patch cost makes dense vertex patches a *moderate-order*
method (N ≲ 12–16); at N = 30 it would need a structured (tensor or
low-rank) local solver to be competitive, which is not what this prototype
has.

**Ghia comparison.** Ghia, Ghia & Shin (1982) tabulate Re ≤ 10000 only, and
5 steps from rest at Re = 32000 is not a developed flow, so no profile
comparison is possible for that run. The N = 30, 4×4 converged FOSLS
solution at Re = 1000 (dt = 1, `pmg_ghia_cavity.py`; Jacobi and ladder give
the same state) matches Ghia's Tables I/II with rms 3.2e−3 in u on x = 0.5
and 6.7e−3 in v on y = 0.5 (`figs_fosls_vs_fs/ghia_re1000_n30.png`). A
solver change cannot alter that: any preconditioner that converges to the
same tolerance reproduces the same steady state.

## 13. Memory and cost scaling of dense vertex patches

(See `LOW_MEMORY_PATCH_SOLVERS.md` for the literature on cheaper patch solves and the measured 8–12× exact reduction by static condensation.)

With F real fields per node (4 in 2D; 14 real = 7 complex per mode in 3D)
a vertex patch has n_p = (2N+1)²·F dofs. Storing one triangular factor takes
n_p²/2 entries; applying it costs n_p² flops; factorising n_p³/3.

    memory  ≈  N_vertices × N_modes × ½ [(2N+1)² F]² × bytes
    apply   ≈  N_vertices × N_modes × [(2N+1)² F]²  flops per CG iteration
    build   ≈  N_vertices × N_modes × ⅓ [(2N+1)² F]³ flops per (re)factorisation

- **Order N: (2N+1)⁴ ≈ 16 N⁴ for memory and apply, (2N+1)⁶ for the build.**
  Doubling N costs 16× and 64×. This is the term that makes dense patches a
  moderate-order method.
- **Elements: linear** (vertices ≈ elements). Doubling the mesh doubles it.
- **Fourier modes: linear** in nk = nz/2 + 1; modes never share factors.
- **Fields: F²**, so the 3D per-mode block is 12× a 2D block at equal N; the
  complex-Hermitian form (n_p/2 complex entries) halves it against the
  split-real form the prototype uses.
- **Against the operator:** the state vector is linear in (N+1)² and in
  elements, so factors/state ≈ 4 (2N+1)² F, about 5000 at N = 8 in the channel.

Exact numbers, single precision, one value of c (triangular storage;
the numpy prototype stores the full n_p² array, 2× these):

| case | N | vertices | patch dofs (real) | one factor | total |
|---|---|---|---|---|---|
| 2D cavity 4×4 | 8 | 25 | 1156 | 5 MB | 0.1 GB |
| 2D cavity 4×4 | 30 | 25 | 14,884 | 0.89 GB (double) | 22 GB (double; 44 GB as stored) |
| 3D channel 6×18, 1 mode | 8 | 114 | 4046 (2023 complex) | 16 MB | 1.9 GB |
| 3D channel, 17 modes | 8 | 114 | 4046 | 16 MB | **32 GB** |
| 3D channel, 17 modes | 12 | 114 | 8750 | 77 MB | 148 GB |
| 3D channel, 17 modes | 16 | 114 | 15,246 | 232 MB | 450 GB |

Three stage values of c multiply the channel figures by 3 if all are kept
resident (95 GB at N = 8), or cost one refactorisation (≈ 5e12 flop, under
a second on the GB10) per stage change if only one set is kept.

Reading against the hardware: on the 121 GB Spark the channel is feasible
at N = 8 (32 GB, or 95 GB fully resident), marginal at N = 12 (148 GB even
for one c), and out of reach at N = 16. Refining in h instead of p is
benign: twice the elements at N = 8 is 63 GB. Raising N is what closes the
door, and only a non-dense local solver (tensor-product or low-rank patch
factorisation, or a fixed-polynomial inner patch iteration) would reopen it.
Measured apply cost per CG iteration in the numpy prototype, 2D: 24 ms at
N = 8 (110 Jacobi iterations), 7.7 s at N = 30 (6600).
