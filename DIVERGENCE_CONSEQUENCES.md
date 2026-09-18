# What the divergence disparity does and does not affect

*Recorded 2026-09-16 for possible integration into `PAPER_DRAFT.md`.  The
question: the least-squares field carries a pointwise divergence orders of
magnitude below the projection field's — what does that buy, in measured terms?*

The temptation is to quote the ratio and let the reader infer that the larger
number is worse everywhere.  The evidence does not support that, and a referee
who knows spectral elements will say so.  What follows separates what we have
measured from what we have only argued.

---

## 1. The numbers, and what they are normalised by

| | rms&#124;∇·u&#124; / rms&#124;u&#124; | rms&#124;∇·ω&#124; / rms&#124;ω&#124; |
|---|---|---|
| fractional step (projected) | 3.6e−1 | 5.1 |
| least squares | **6.6e−5** | **8.5e−5** |

The E-path production run logs the same relative measure at 1.4e−1 rising to
2.2e−1 over $t = 3$ to 16, so the disparity is a property of the method and not
of one snapshot.  The consistent ($E$) projection does **not** reduce it: it
drives the *weak* divergence to $10^{-6}$ and removes the spurious pressure work,
while the pointwise value is unchanged from the cheap ($K$) path.

## 2. Where it demonstrably does NOT matter: the low-order statistics

Measured twice, and both are ours.

**Within the fractional-step code.**  The K and E paths differ enormously in weak
divergence — uncontrolled against $10^{-6}$ — and on matched 7,401-sample windows
the statistics differ by $u_\tau$ −0.5 %, $U^+_c$ +1.4 %, $u'$ +0.03 %, $v'$
−2.7 %, $w'$ −1.1 %, $-u'v'$ −1.1 %: all inside the finite-window sampling error.
The archive's own conclusion is that "the operator choice does not measurably
affect low-order statistics at this resolution".

**Across the two codes.**  At 24.8 turnovers of averaging the least-squares run
sits within 1.2 % of the mean of five reference databases and the fractional-step
twin within 2.9 %, on the same box and mesh — a gap that tracks the averaging
window and the minimal box, not a factor of 3,000 in pointwise divergence.

## 2b. What it is NOT: an inf-sup effect

The projection code is equal-order $P_N$–$P_N$ and therefore does violate the
LBB condition, so it is tempting to file the divergence result under that
heading.  It does not belong there.  The consistent path zeroes the **weak**
divergence identically — $G^{\mathsf T}\mathbf u = 0$ to machine precision — and
its pointwise divergence is large anyway.  LBB governs whether the *pressure* is
determined; the weak-versus-strong distinction governs whether the *divergence*
is small pointwise.  Two deficiencies, two consequences, and conflating them
would be checkable and wrong.  See `LBB_AND_THE_TWO_DEFICIENCIES.md`.

## 3. Why that is not a paradox

In a $C^0$ spectral element method the **weak** divergence is what enters the
momentum equation.  The test space is coarser than the pointwise residual, so a
large pointwise $\nabla\!\cdot\mathbf u$ can be nearly invisible to the Galerkin
formulation while being plainly visible to anything that evaluates the derivative
directly.

It follows that **neither method produces an exactly divergence-free field; they
place the residual in different places.**  The projection method satisfies the
discrete momentum equation and the weak constraint and leaves the pointwise
divergence; the least-squares method minimises the pointwise divergence and
accepts a momentum residual instead.  That framing is both more accurate and more
defensible than "ours is 5,500× better", and it is the one to use.

## 4. Where it does matter, with the evidence we have

**Vorticity — measured, with a caveat.**  Against the databases:

| | least squares | fractional step |
|---|---|---|
| $\omega'_y$ peak | −1.9 % | +13.6 % |
| $\omega'_z$ wall | −6.6 % | +4 % |
| $\omega'_x$ wall / min@5 / max@20 | −17 / −10 / −10 % | +22 / +30 / +24 % |

Least squares is closer on the first two and comparable on the third, and its
primary $\omega$ agrees with $\nabla\times\mathbf u$ to better than 0.05 % in rms,
so the field is internally consistent rather than merely close.  **The caveat is
real:** the fractional-step vorticity was obtained by differentiating a single
archived snapshot of a $C^0$ field, and our own notes call that "not a verdict".
Treat it as suggestive until recomputed over a window.

**Everything else in this class is argued, not measured.**  Any quantity that
differentiates the velocity field inherits the pointwise error: $Q$ and
$\lambda_2$ vortex identification, enstrophy budgets, *a priori* SGS model
testing.  We have not computed any of these in both fields.

## 5. Where it would matter most, and we have no data

* **Lagrangian particle tracking.**  A non-solenoidal field does not conserve
  volume, so particles accumulate or thin spuriously in regions where
  $\nabla\!\cdot\mathbf u \neq 0$.
* **Scalar transport.**  In $\partial_t c + \mathbf u\cdot\nabla c = \kappa\nabla^2 c$
  a non-solenoidal $\mathbf u$ acts as a distributed source, producing
  concentration extrema outside the initial range — a violation of a maximum
  principle that the continuous equation obeys.

These are the applications where a factor of $10^3$ would be decisive rather than
cosmetic, and we would be asserting rather than demonstrating.

## 6. The experiment that would convert the argument into a result

Advect a passive scalar in both stored fields and measure the spurious
production: the departure of $\max c$ from its initial bound, or the growth of
$\int c^2$ that the exact equation forbids.  Both fields are archived
(`results/minchan_re180_E/state_t15.95.npz`, and the least-squares run's final
checkpoint), the meshes are identical, and the transport step is a few dozen
lines on machinery both codes already have.  **A day's work**, and it would
convert the strongest claim in this comparison from a physical argument into a
measurement.

A cheaper partial version: recompute the fractional-step vorticity rms over a
window of snapshots rather than one, which removes the caveat in §4 and costs an
hour.

## 6a. MEASURED, 2026-09-17: the term incompressibility should kill

*This converts the central argument from a mechanism into a number, by a
different route than §6 proposed — the vorticity equation rather than a passive
scalar, and it needed no new solver work because both archives store fields.*

The vorticity equation is

$$\frac{D\boldsymbol\omega}{Dt} \;=\; \underbrace{\boldsymbol\omega\cdot\nabla\mathbf u}_{\text{stretching — the cascade}} \;-\; \underbrace{\boldsymbol\omega\,(\nabla\!\cdot\mathbf u)}_{\textbf{identically zero if incompressible}} \;+\; \nu\nabla^2\boldsymbol\omega$$

The second term does not exist in incompressible flow. Whatever a scheme leaves
in $\nabla\!\cdot\mathbf u$ appears there **multiplied by the vorticity**, so the
error is largest exactly where the vorticity is — in the near-wall vortices that
sustain the turbulence. And vortex stretching *is* the cascade mechanism, so a
spurious fraction of it is a dynamical statement, not a diagnostic one.

Measured on five archived snapshots of each run (`scratch/divergence_consequence.py`,
`figs/divergence_consequence.png`):

| | rms $\nabla\!\cdot\mathbf u$ | relative to $\lVert\nabla\mathbf u\rVert$ | **spurious $\omega(\nabla\!\cdot\mathbf u)$ as % of $\omega\cdot\nabla\mathbf u$** |
|---|---|---|---|
| FOSLS | 3.43e−03 ± 4.1e−04 | 1.10e−05 | **0.006 % ± 0.001 %** |
| fractional step | 6.69e+00 ± 8.8e−01 | 2.23e−02 | **12.46 % ± 1.63 %** |
| ratio | 1949× | 2024× | **2079×** |

The fractional-step value is stable at 9.8–14.4 % across five independent
instants, so it is not a sampling artefact. The separation holds at **every
height**, and the fraction *rises* toward the centreline (≈5 % near the wall to
≈40 % at $y^+>100$) because the true stretching weakens there faster than the
divergence error does.

### Two measures that do NOT discriminate, and why quoting them would mislead

* **The integrated energy leak** $-\tfrac12\int|\mathbf u|^2(\nabla\!\cdot\mathbf u)\,dV$
  comes out *comparable for both*. $\nabla\!\cdot\mathbf u$ is largely uncorrelated
  with $|\mathbf u|^2$, so the volume integral cancels. A referee shown this number
  would conclude the divergence does not matter. Only **local** measures survive.
* **Mass flux through constant-$x$ planes** is conserved to **3e−4 %** by both.
  The projection drives the *weak* divergence $G^{\mathsf T}\mathbf u=0$ to machine
  precision, so every integrated conservation statement is excellent. The failure
  is purely pointwise — which is precisely the distinction §2b draws.

### A null result worth keeping

FOSLS's stored $\omega$ differs from $\nabla\times\mathbf u$ of its own velocity by
**1.9e−06** relative. The vorticity-definition row is *minimised*, not enforced —
exactly like continuity — yet it comes out satisfied six digits tighter than the
divergence. So the structures FOSLS shows in the core are not a least-squares
artefact, which was the main way this comparison could have flattered it.

### What still is not measured

The passive-scalar experiment of §6 remains the cleanest demonstration for a
reader who does not think in vorticity, and is still a day's work. This section
does not replace it; it removes the need for it to carry the argument alone.

## 7. What to claim in the paper

> At equal cost, the least-squares formulation delivers a velocity field whose
> pointwise divergence is three orders of magnitude smaller.  This does not
> change the low-order statistics — on matched 24.8-turnover windows the two
> agree to ~2 % on every mean and Reynolds-stress quantity, consistent with the
> weak divergence being what the momentum equation sees, and both codes conserve
> mass through constant-$x$ planes to 3e−4 %.  What it does change is the
> **vorticity dynamics**: the term $\boldsymbol\omega(\nabla\!\cdot\mathbf u)$,
> which incompressibility removes exactly, reaches **12.5 % ± 1.6 % of the true
> vortex stretching** in the projection field against **0.006 % ± 0.001 %** in
> the least-squares field — a factor of 2000, stable over five independent
> snapshots and present at every height.

Narrow, supported, and it does not invite the obvious objection.  It is also now
a *measurement* rather than a mechanism, which §6 was written to achieve.

**Do not quote the integrated energy leak or any global conservation statement
as evidence** — both are insensitive by construction (§6a), and offering one
would hand a referee the counter-argument.  The broader claims about particle
tracking and scalar transport remain *expected consequences* with the mechanism
stated, or need §6 first.
