# The inf-sup condition in the projection code, and why it is not the divergence story

*Recorded 2026-09-16 alongside `DIVERGENCE_CONSEQUENCES.md`.  The question: does
the fractional-step spectral-element formulation violate the LBB condition, and
if so what does that cost?  The answer matters for the paper because it is one of
the least-squares formulation's few **structural** advantages — a theorem rather
than a measurement — and because it is easy to conflate with the divergence
result, which is a different deficiency entirely.*

---

## 1. Yes, and for the textbook reason

The projection code is **$P_N$–$P_N$**: velocity and pressure on the same
Gauss–Lobatto grid at the same polynomial degree ($N = 8$ here), which is what the
consistent projection's name, "$P_N$–$P_N$ consistent projection", records.

Equal-order pairings are the classical example of an inf-sup-unstable choice for
the Stokes saddle point.  The Ladyzhenskaya–Babuška–Brezzi condition asks for

$$\inf_{q_h}\ \sup_{\mathbf v_h}\ \frac{(q_h,\ \nabla\!\cdot\mathbf v_h)}{\lVert q_h\rVert\ \lVert\mathbf v_h\rVert_1}\ \ge\ \beta > 0$$

with $\beta$ independent of $h$ and $N$.  For $P_N$–$P_N$ it is not: there exist
pressure modes the discrete divergence operator barely sees, and $\beta$
degenerates.  The stable spectral-element pairing is **$P_N$–$P_{N-2}$**
(Maday–Patera), pressure two degrees lower and on the Gauss rather than the
Gauss–Lobatto points, chosen precisely to recover it.

## 2. Why the code runs anyway

**A projection method never assembles the saddle point.**  It solves a velocity
Helmholtz problem, then a pressure problem, separately.  The consistent pressure
operator

$$E \;=\; G^{\mathsf T} M^{-1} G$$

is symmetric positive **semi**-definite — coercive on the complement of its null
space — so the solve is well posed there whatever $\beta$ does.  LBB violation
therefore does not appear as an unsolvable system.  It appears as **a non-trivial
near-null space of $E$**: precisely the modes inf-sup stability would have
excluded.

The code manages this at runtime, and its own comments name the mechanism:

* `project.py` purges the constant from every preconditioned residual at
  $k_z = 0$, having found that *pinning* a pressure degree of freedom instead
  "rotates the null vector into an unknown direction and CG amplifies it to
  1e15";
* `epmg.py` inverts the coarse $E$ by eigendecomposition with null deflation —
  "eigenvalues below 1e−10·max are zeroed rather than pseudo-inverted, so the
  constant **and anything numerically null** is annihilated".

That last clause is the inf-sup deficiency being handled numerically: the code
deflates more than the constant because there is more than the constant to
deflate.

## 3. What it costs — the pressure, and only the pressure

A spurious mode is by definition nearly invisible to the velocity; that is what
makes it spurious.  So the velocity statistics are unaffected, which is what we
measure — the two formulations agree on mean profile and Reynolds stresses to
within the sampling error.

The pressure is a different matter, and the archive is explicit: the stored
fractional-step pressure is "2.5–3× the physical pressure with its peak at
$y^+\approx13$, and is neither $p$ nor $p\pm\frac12|\mathbf u|^2$ … so no
fractional-step pressure comparison is possible from the archive."  That is the
signature of an inf-sup-deficient pressure space combined with a projection
pseudo-pressure.  **Wall-pressure statistics are the quantity to distrust**, and
the one where a $P_N$–$P_{N-2}$ discretisation would be the right instrument.

## 4. The distinction that must not be blurred

These are **two different deficiencies with two different consequences**, and the
paper should keep them apart:

| | governs | mechanism | consequence here |
|---|---|---|---|
| **LBB violation** | whether the pressure is determined | equal-order $P_N$–$P_N$ leaves modes the divergence cannot see | pressure unusable; velocity unaffected |
| **weak vs strong divergence** | whether $\nabla\!\cdot\mathbf u$ is small pointwise | $G^{\mathsf T}\mathbf u = 0$ is orthogonality to *discrete gradients*, not pointwise vanishing on a $C^0$ space | pointwise divergence 1.4e−1 to 2.2e−1 against 6.6e−5 |

The consistent ($E$) path zeroes the weak divergence **identically** — that is
what "consistent" means, and $G^{\mathsf T}\mathbf u = 0$ holds to machine
precision.  Its pointwise divergence is nevertheless large.  So the divergence
result is *not* an LBB effect, and attributing it to one would be wrong and
checkable.

## 5. Why this is a genuine structural advantage of least squares

Unlike almost everything else in this comparison, this one is a theorem rather
than a measurement.  A least-squares formulation minimises a residual instead of
solving a saddle point: the system is symmetric positive definite by
construction, and **no inf-sup condition applies at all**.  Equal-order
interpolation is admissible, which is one of the three reasons Section 1 of the
paper gives for the method being attractive.

What this comparison adds is the **scope** of that advantage, which is narrower
than the usual telling suggests:

> The absence of an inf-sup condition buys a well-determined pressure and the
> freedom to use equal-order spaces.  It does not buy better velocity
> statistics — the projection method's spurious pressure modes are invisible to
> the velocity, which is precisely why they survive — and it is not the reason
> for the pointwise divergence difference.

## 5b. Has equal order been demonstrated successful? — the literature, searched 2026-09-16

The answer splits cleanly on *how the pressure is computed*, and the split is the
whole defence.

**In a coupled Galerkin setting: no.**  Equal order is inf-sup unstable and admits
spurious pressure modes.  Maday & Patera's $P_N$–$P_{N-2}$ pairing exists for that
reason, and production practice follows it: Nek5000's documentation states that
"to avoid spurious pressure modes, spatial discretisation is based on the
$P_N$–$P_{N-2}$ SEM", and its equal-order branch carries an explicit
Fischer–Mullen filter "to suppress the spurious modes of pressure and momentum at
the end of each time step".

**In a splitting / consistent pressure-Poisson setting: yes, and the literature
frames it as the point.**  Li (JCP 2020) builds a split-step finite-element
method that "completely separates the pressure updates from the solution of
velocity variables" and states that "when the pressure equation is formed
explicitly, the algorithm avoids solving a saddle-point problem; therefore, our
algorithm has more flexibility in choosing finite-element spaces" — and uses
"Lagrange finite elements of equal order for both velocity and pressure".  The
consistent-splitting SAV literature makes the same claim, circumventing the
inf-sup condition to use equal-order pairs, and a 2022 *Computational Mechanics*
paper on consistent pressure-Poisson splitting is titled, in part, "eliminating
numerical boundary layers **and inf-sup compatibility restrictions**".

**Which one is our comparison code?**  The second.  Its $E = G^{\mathsf T}M^{-1}G$
is a consistent pressure-Poisson operator formed explicitly, and the scheme
descends from Guermond–Shen consistent splitting by its own documentation.  So
the equal-order choice is established practice for what it is, not a lapse — and
the reviewer objection "you should have used $P_N$–$P_{N-2}$" answers itself:
that pairing is the remedy for a saddle point this code never forms.

**What survives of the objection**, and should be conceded in the paper: the
pressure is still determined only up to the admitted modes, so no pressure claim
can be made; and a mixed pairing would have slightly fewer pressure unknowns and
so a slightly cheaper solve.

## 6. What would test it

Two experiments, neither yet done:

1. **Compare pressure statistics properly.**  Reconstruct the physical pressure
   from the projection run (the stored field is a pseudo-pressure; $p$ follows
   from the rotational update $p = p^{k-1}+\phi-\nu\,\nabla\!\cdot\hat{\mathbf u}$)
   and compare $p'_{\rm rms}$ against the databases alongside the least-squares
   run's, which is currently 10–13 % low for reasons of its own
   (ADN_FOSLS.md §2: the $L^2$ VVP functional with no-slip walls is not
   $H^1$-norm-equivalent for $\omega$ and $p$).  **This is the experiment that
   would turn §3 from an inference into a result.**
2. **Exhibit the spurious modes.**  Assemble $E$ on a small mesh and compute its
   spectrum: the count of eigenvalues at or near zero beyond the single constant
   is the direct measure of the inf-sup deficiency, and it should grow with the
   mesh.  An afternoon on the 2D harness.
