# The failure mode is silence

*Recorded 2026-09-17, from the curvilinear port and the fractional-step DNS
campaign. Ten defects were found in one working session. **None of them raised
an exception, produced a NaN, or failed to converge.** Every one returned a
finite, plausibly scaled number that could have been written into a paper.*

This is not a list of mistakes for its own sake. The ten share a structure, the
structure is a property of *this* kind of code, and it implies a specific
discipline about what to check.

---

## 1. The ten

| # | what happened | what it returned instead of failing |
|---|---|---|
| 1 | **Rotated cavity**: bc 2 prescribes the *Cartesian* pair (u,v) = (lid, 0). On a 30°-rotated lid that has a normal component of −0.5, so with no-slip elsewhere the net boundary flux is nonzero and incompressibility is unsatisfiable anywhere in the domain. | 44–48 CG iterations vs 32 affine — reads as a 1.5× curvature penalty for the preconditioner |
| 2 | **`pcg_solve` early exit**: returns `x = 0` when \|pᵀAp\| < 1e−20, a hardcoded absolute floor. | At N = 14 the reported error was **exactly 0.000e+00** — which looked like a perfect gate pass |
| 3 | **`put()` no-op**: returns False when the source file is missing; nobody read the return value, and the resume block is `if exists(...)` with no `else`. | Statistics silently restarted at zero; discovered hours later because `nsamp` read 620 instead of 4560 |
| 4 | **`kx = 2` deformation**: the displacement carries sin(2πx), which vanishes at x = 0, **0.5**, 1. Ghia's extraction centreline was a nodal line of the deformation. | 132 nodes sat exactly where the affine mesh put them; the profile would have been read off an *undeformed* line while looking curvilinear |
| 5 | **Slit mesh**: the O-ring's outer edge placed nodes at uniform θ, the abutting block at uniform x. Corners coincided, interiors did not. | **Area exact to 2.4e−15**, min J positive, picture perfect — and 224 interior-edge nodes unmerged, so flow could not cross that interface |
| 6 | **`jacobi_add` missing term**: the tangential derivative couples each edge node to every other, so rows r ≠ j feed the diagonal at (N, j). | 5.1e−05 wrong on a deformed mesh and **exact on an affine one** (sx = 0 kills the term), so no existing test could see it. The adjoint and symmetry tests both passed |
| 7 | **R² as a discriminator**: exponential vs algebraic fits compared by R² over N spanning less than a factor of three. | 0.9955 vs 0.9869 — a difference with no discriminating power, quoted as if decisive. Settling it needed N = 18 |
| 8 | **"the steady answer is dt-independent"**: the BDF mass and history terms *do* cancel at a fixed point, but balanced weighting leaves a_flux = w_mom = √dt, so the functional keeps a dt. | Both runs converged to drift < 1e−5, **to different answers 11 % apart** |
| 9 | **"steady at step 600"**: the run hit its step cap, not its tolerance. | The word "steady" in a log line that would later be quoted as if a criterion had been met |
| 10 | **\|dU\| as a convergence criterion**: decay is geometric at r = 0.983 per step, so the change *still to come* is \|dU\|·r/(1−r) = **58×** the per-step value. | A per-step number that looks tiny sitting in front of a drift 58 times larger |

---

## 2. Why this class of code fails quietly

**A least-squares formulation cannot diverge on an ill-posed problem.** It
minimises ‖R‖, and a minimiser exists whether or not the problem has a solution.
Case 1 is the pure form of this: the constraint was unsatisfiable *everywhere in
the domain*, and the solver returned a converged answer with a plausible
iteration count. A saddle-point formulation would have produced a singular
system and stopped. **The property that makes FOSLS robust — SPD by
construction, always solvable — removes the crash as a diagnostic channel.**

**Robustness and correctness point opposite ways here.** `put()` returning False,
`pcg_solve` bailing out on a tiny denominator, the resume guarded by `if
exists(...)` — each was written so a long unattended run would survive a missing
file or a degenerate step. That is right for uptime and wrong for truth. Every
one of them is *graceful degradation*, and graceful degradation of a measurement
is a fabricated measurement.

**Structure that happens to hold conceals defects.** Case 6 was exact on affine
meshes because sx ≡ 0 there. Case 4 was invisible because the mesh *was*
deformed — just not where it was read. Case 5 had exact area because area is a
sum over elements whether or not they are joined. In each, a property that held
for an unrelated reason covered the hole.

---

## 3. What actually caught them

Every check that worked had the same shape: **a quantity computed a second,
independent way, whose correct value was known before the run.**

| check | why it worked |
|---|---|
| nodes on x = 0.5: **132 vs 4** | the deformation's analytic form says which nodes must move |
| interior-edge nodes shared: **7992 of 7992** | conformity is a definition, not a measurement |
| Jacobi diagonal vs the **probed** operator | the same number from two derivations |
| implied algebraic order **rising without bound** | a property of exponential decay, not a fit quality |
| **u_r ≡ 0, ω ≡ 2A** on Taylor–Couette | analytic constants have no shape for an error to hide in |
| annulus area vs **π(b²−a²)** | analytic |
| outflow edge weights summing to **20.0** | analytic |
| interpolation error vs solved error | isolates the space from the operator |

And the ones that did **not** catch anything, despite looking diligent:

* *"did it converge?"* — the solver's own report on its own process
* *R² of a fit* — a goodness measure, not an invariant
* *the picture looked right* — cases 4 and 5 both looked perfect
* *area* — a genuine invariant, but **insensitive to the defect**

That last one matters most. Area is exactly the kind of independent analytic
check this note recommends, and it still missed the slit mesh. So the rule is not
"have an invariant". It is:

> **Have an invariant that the bug would have been forced to break.**

---

## 4. The discipline

1. **Ask "is this quantity even possible?", not "did anything error?"** Exceptions
   fire on failed *operations*. None of the ten involved a failed operation.
2. **Count things.** Node counts, shared-edge counts, sample counts. Discrete,
   cheap, unambiguous, and four of the ten were caught this way.
3. **Prefer analytic constants to analytic functions.** A field that must be
   *constant* (u_r = 0, ω = 2A) is a sharper instrument than one with structure,
   because there is nothing for an error to be mistaken for.
4. **Compute the same number twice, by different routes.** Probe the assembled
   operator against the closed-form diagonal; compare the interpolation error to
   the solved error.
5. **Never let a report about the process stand in for a check on the answer.**
6. **Distrust graceful degradation in anything that produces a number.** A
   missing file, a degenerate denominator, a skipped resume — each should be
   loud, because the alternative is a number that means nothing and says so to
   nobody.
7. **Say which exit was taken.** Case 9 cost nothing this time only because
   somebody re-read the log.
