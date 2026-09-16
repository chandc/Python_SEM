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

## 7. What to claim in the paper

> At equal cost, the least-squares formulation delivers a velocity field whose
> pointwise divergence is three orders of magnitude smaller.  This does not
> change the low-order statistics — we measure no significant difference in the
> mean profile or the Reynolds stresses, consistent with the weak divergence
> being what the momentum equation sees — but it is inherited directly by every
> quantity computed from derivatives of the velocity field, for which we show the
> vorticity statistics.

Narrow, supported, and it does not invite the obvious objection.  The broader
claims about particle tracking and scalar transport should be offered as
*expected consequences* with the mechanism stated, or backed by §6 first.
