# Time-marching least-squares spectral elements for unsteady incompressible flow: the weighting that makes them accurate and the preconditioner that makes them affordable

*Draft. Sections 1–6 are written from completed work; Sections 7–10 are outlined
and carry the work still in progress. Plan: `PAPER_PLAN_PRACTICAL.md`. Every
number quoted here is traceable to `ZIGZAG_CURE_RESEARCH.md`,
`BALANCED_CONDENSED_PLAN.md` or the scripts named in the text.*

---

## Abstract (provisional)

A least-squares finite element discretisation turns any first-order system into a
symmetric positive definite algebraic problem, with no inf-sup condition to
satisfy and a built-in error estimator. For unsteady incompressible flow it is
nevertheless little used, for two reasons that are usually stated separately:
the method is held to be inaccurate, and it is held to be expensive. We show
that both are consequences of a single parameter, the mass coefficient
$c=\mathrm{fac}_1/\Delta t$ that the implicit step imposes on the momentum rows,
and that each has a remedy.

For accuracy: the fixed point of the time-stepping map is *not* the minimiser of
any steady least-squares functional, and the weight with which the momentum
equation enters it is the product of the two row weights, $ma$. Under the
conventional scaling that product is $\Delta t$, so the momentum equation leaves
the problem as the step is refined and what remains is the constraint block
alone. Refining the time step then makes the answer worse. We prove that the
requirement $ma=O(1)$, together with time consistency, determines the weighting
uniquely as $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$, and that no
reformulation can evade the requirement because the ratio it constrains is
invariant under row and column scaling. The transient error separates as
$C_2\Delta t^2+\Phi\lVert R_h\rVert$ with $C_2$ independent of the weighting and
$\Phi$ measured at $0.19$ against $0.05$; the two terms cross at
$\Delta t^\ast=\sqrt{\Phi\lVert R_h\rVert/C_2}$, a criterion we compute on coarse
meshes and then verify by prediction on a fine one.

For cost: the same $c$, now as a zeroth-order term in the operator, couples
velocity and vorticity above a crossover $c^\ast\approx\nu p^4/h^2$ and renders
the divergence-free directions invisible to pointwise relaxation, which accounts
for the observed failure of point Jacobi, block Jacobi, $p$-multigrid and
low-order-refined algebraic multigrid alike. An overlapping vertex-patch Schwarz
preconditioner with a coarse space, made affordable by an exact static
condensation, gives iteration counts that are flat in polynomial order where
those methods grow by two orders of magnitude.

We close with a direct numerical simulation of turbulent channel flow, and with
the condition under which the accuracy result does *not* apply: a stage with
explicit convection is a projection, its right-hand side is not solenoidal, and
there the conventional weighting is the correct one.

---

## 1. Introduction

*(to be written last; the argument is the abstract's, at length. Points to
make: why LSFEM is attractive and why it is absent from unsteady production
work; that the two objections have never been connected; that the connection is
the mass coefficient; and that the paper is organised as a ladder from an
algebraic statement to a DNS.)*

## 2. The weighted least-squares time step

Let the incompressible Navier–Stokes equations be written as the first-order
velocity–vorticity–pressure system, and let one implicit step from the history
$\{U^{n-m}\}$ to $U=(\mathbf u,\omega,p)$ minimise

$$
J(U)=\Big\lVert\,\underbrace{\tfrac{w_{\rm mass}}{\Delta t}\Big(\mathrm{fac}_1\mathbf u-\sum_m\alpha_m\mathbf u^{n-m}\Big)}_{\text{mass and history}}+\underbrace{w_{\rm mom}\,\mathcal N(U)}_{\text{momentum}}\Big\rVert_W^2
+\big\lVert C U\big\rVert_W^2 ,
$$

$$
\mathcal N(U)=(\mathbf u\cdot\nabla)\mathbf u+\nabla p+\nu\nabla\times\omega-\mathbf f,
\qquad
CU=\big(\nabla\!\cdot\mathbf u,\;\omega-\nabla\times\mathbf u\big),
$$

with $W$ the quadrature weights and $\mathrm{fac}_1=\sum_m\alpha_m$ the BDF
consistency constant. Write the momentum row as
$a_{\rm mass}\mathbf u+a_{\rm flux}\mathcal N(U)-\mathrm{hist}\sum_m\alpha_m\mathbf u^{n-m}$,
so that

$$
a_{\rm mass}=\frac{w_{\rm mass}\mathrm{fac}_1}{\Delta t},\qquad
a_{\rm flux}=w_{\rm mom},\qquad
\mathrm{hist}=\frac{w_{\rm mass}}{\Delta t},
$$

and abbreviate $m=a_{\rm mass}$, $a=a_{\rm flux}$. Two combinations of the
weights carry all the structure:

$$
c=\frac{m}{a}=\frac{w_{\rm mass}\mathrm{fac}_1}{w_{\rm mom}\Delta t}
\qquad\text{and}\qquad
ma=\frac{w_{\rm mass}w_{\rm mom}\mathrm{fac}_1}{\Delta t}.
$$

The first is the mass coefficient the linear algebra sees; the second, as
Section 3 shows, is the weight with which the momentum equation enters the
solution the scheme actually converges to. Three choices appear in the
literature and in practice:

| name | $w_{\rm mom}=w_{\rm mass}$ | $a_{\rm mass}$ | $a_{\rm flux}$ | $c$ | $ma$ |
|---|---|---|---|---|---|
| conventional | $\Delta t$ | $\mathrm{fac}_1$ | $\Delta t$ | $\mathrm{fac}_1/\Delta t$ | $\mathrm{fac}_1\Delta t$ |
| **balanced** | $\sqrt{\Delta t}$ | $\mathrm{fac}_1/\sqrt{\Delta t}$ | $\sqrt{\Delta t}$ | $\mathrm{fac}_1/\Delta t$ | $\mathrm{fac}_1$ |
| unit | $1$ | $\mathrm{fac}_1/\Delta t$ | $1$ | $\mathrm{fac}_1/\Delta t$ | $\mathrm{fac}_1/\Delta t$ |

All three have the same $c$: they pose the same linear algebra and admit the
same preconditioners. They differ only in $ma$, and therefore only in what the
scheme converges to.

The conventional choice is the natural one and is what a code writes if it
simply forms the BDF residual and squares it; it weights the momentum equation
by $\Delta t^2$ against the constraints in the functional. It is also the
setting in which the method was validated in most of the literature, because at
the moderate steps used for steady-state marching the distinction below does not
appear.

## 3. The fixed point

**Proposition 1 (the fixed point of the step map).** *Let $U^\ast$ satisfy
$U^\ast=U^n=U^{n-1}=\cdots$. Then for every BDF order the mass and history terms
cancel identically, the momentum residual collapses to
$a\,(\mathcal N(U^\ast))$, and stationarity of $J$ gives, after division by $a$,*

$$
\Big[\;m\,\Pi_u^{T}WA\;+\;a\,A^{T}WA\;+\;\tfrac1a\,C^{T}WC\;\Big]U^\ast
=m\,\Pi_u^{T}Wf+a\,A^{T}Wf ,
\tag{3.1}
$$

*where $A$ is the linearisation of $\mathcal N$ and $\Pi_u$ selects the velocity
components.*

*Proof.* The cancellation is the BDF consistency condition: the mass term is
$a_{\rm mass}\mathbf u^\ast=\mathrm{fac}_1\,\mathrm{hist}\,\mathbf u^\ast$ and the
history term is $\mathrm{hist}\sum_m\alpha_m\mathbf u^\ast=\mathrm{fac}_1\,\mathrm{hist}\,\mathbf u^\ast$.
The remainder is the stationarity condition of $J$ with the momentum residual
replaced by $a\mathcal N(U^\ast)$. $\square$

Three consequences, and the second is the mechanism of the whole paper.

**(a) Every step is symmetric positive definite; the fixed point is not.** The
step itself solves a normal-equation system, which is SPD by construction. The
operator in (3.1) contains the cross term $m\,\Pi_u^TWA$, which is not symmetric;
nor does it arise from any steady least-squares functional, since
$a^2\lVert AU-f\rVert^2+\lVert CU\rVert^2$ would give
$[a^2A^TWA+C^TWC]U=a^2A^TWf$ with no cross term at all.

**(b) The size of the non-symmetric part is exactly $ma$.** In the velocity row
block the constraint contribution enters at $1/a$ and the momentum contribution
at $m$, so their ratio is $(ma)^{-1}$, and

$$
\frac{\lVert K-K^{T}\rVert}{\lVert K\rVert}\;=\;O(ma).
$$

Under the conventional weighting $ma=\mathrm{fac}_1\Delta t$ and the fixed-point
operator becomes *symmetric* as $\Delta t\to0$. That symmetry is the disease:
what survives the limit is the constraint block alone, and the fixed point
becomes any field the constraints admit, selected at $O(\Delta t^2)$. Under the
balanced weighting $ma=\mathrm{fac}_1$ and the cross term and the constraints
remain at the same order for every step.

**(c) The requirement is scaling-invariant, and fixes the weighting uniquely.**

**Proposition 2 (uniqueness).** *(i) The ratio of the constraint contribution to
the momentum contribution within the velocity row block of (3.1) equals
$(ma)^{-1}$ and is unchanged by any nonsingular diagonal row or column scaling.
(ii) Hence the momentum equation is represented in the limit $\Delta t\to0$ if
and only if $ma=O(1)$, that is $w_{\rm mass}w_{\rm mom}=O(\Delta t)$. (iii) The
scheme integrates the effective step
$\Delta t_{\rm eff}=\Delta t\,w_{\rm mom}/w_{\rm mass}$, so time consistency
requires $w_{\rm mom}=w_{\rm mass}$. Together, (ii) and (iii) give*

$$
\boxed{\,w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}\,}
$$

*up to a constant factor, and no other weighting satisfies both.*

Part (i) is what makes the statement a theorem rather than a preference: a
diagonal scaling multiplies both contributions in a row equally, so no
rescaling, preconditioning or reformulation of the same discrete system can
restore the momentum equation once the weights have removed it. The conventional
choice has $ma=\mathrm{fac}_1\Delta t\to0$ and loses momentum; the unit choice
has $ma=\mathrm{fac}_1/\Delta t\to\infty$ and loses the constraints instead,
which is the behaviour we observe for it in practice (Section 6.3).

The analysis is independent of the time discretisation's order: only
$\mathrm{fac}_1$ changes, and it appears in $ma$. We confirm this numerically for
BDF1 and BDF2 in Section 4, where the measured symmetry defect is $1.00$ and
$1.50$ times a common value.

### 3.1 Relation to the weighted least-squares literature

The estimate of Bochev and Gunzburger [BG98, (3.34)] for the
velocity–vorticity–pressure system,

$$
\lVert\omega\rVert_{q+1}+\lVert p\rVert_{q+1}+\lVert\mathbf u\rVert_{q+2}
\le C\big(\lVert\nu\,\mathrm{curl}\,\omega+\mathrm{grad}\,p\rVert_{q}
+\lVert\mathrm{curl}\,\mathbf u-\omega\rVert_{q+1}
+\lVert\mathrm{div}\,\mathbf u\rVert_{q+1}\big),
$$

already places the momentum residual and the constraint residuals in norms
differing by one Sobolev order, and their weighted-$L^2$ class exists precisely
to emulate that difference on a finite element space, replacing norms "by
$L^2$-norms weighted by the respective equivalence constants". The three
purposes they enumerate for such weights — replacing $H^1$ norms, replacing
boundary norms, and handling solution singularities — are all
*parameter-independent*. The present result is the missing fourth: when the
system carries a large zeroth-order coefficient, as every implicit time step
does with $c=\mathrm{fac}_1/\Delta t$, the correct relative weight depends on
that coefficient. The same review places time-dependent problems outside its
scope.

## 4. The mechanism in closed form

To exhibit (3.1) without the complications of a production code we take one
Fourier mode of the two-dimensional velocity–vorticity–pressure Stokes system,
reduced to a two-point boundary-value problem in $x$ — which is exactly the
per-mode structure of the three-dimensional solver of Section 9 — and discretise
it with the same $C^0$ spectral elements, collocating residuals at the
Gauss–Lobatto nodes and summing with the Gauss–Lobatto weights. The problem is
linear, so the fixed point is a single dense solve rather than a marched
trajectory. (`scratch/model1d_fixedpoint.py`.)

**Figure 1** shows the three measurements.

*Panel (b), the mechanism.* The symmetry defect is $ma$: a straight line of slope
one in $\Delta t$ for the conventional weighting, falling from $2.0\times10^{-3}$
to $2.0\times10^{-7}$ over four decades, against a constant $2.0\times10^{-2}$
for the balanced one. At BDF2 both are multiplied by $\mathrm{fac}_1=3/2$, as
Proposition 2 requires.

*Panel (a), the consequence.* With the exact solution lying in the discrete
space — a polynomial datum — both weightings return it for every step, to
machine precision. With a transcendental datum the discrete rows cannot be
satisfied simultaneously, and the conventional fixed point settles three to
twenty-five times further from the truth than the balanced one, the ratio
growing with resolution. This is the first appearance of a theme that recurs
throughout: **the defect is proportional to what the discrete space cannot
represent, and vanishes when it can.**

Panel (a) also carries a point worth isolating. In the *resolved* case, where
the weighting ought to be irrelevant, the conventional fixed point is
nonetheless polluted by its own conditioning as the step falls, reaching
$1.2\times10^{-5}$ at $\Delta t=10^{-6}$ — worse there than the *unresolved*
balanced error. A vanishing residual protects the answer only while the
arithmetic holds.

*Panel (c), well-posedness.* The fixed-point operator has no pressure rows in
the naive limit, since $\Pi_u^T$ selects velocity rows and $C$ does not involve
pressure; pressure is carried only by $aA^TWA$. Scaling velocity and vorticity
rows by $a$ and pressure rows by $1/a$ removes that degeneracy, and the resulting
condition number is **flat in $\Delta t$ for the balanced weighting** —
$2.2\times10^{7}$ from $\Delta t=10^{-4}$ to $10^{-6}$ — while the conventional
operator under the same scaling continues to grow like $1/\Delta t$, reaching
$1.7\times10^{13}$. That flatness is what a balanced norm means. We verify the
scaled limit is nonsingular across element count, polynomial order, wavenumber
and viscosity; we have not proved it, and Appendix A records an energy argument
that yields only a sufficient condition which almost no discretisation satisfies
and whose viscosity dependence runs the wrong way.

A final measurement from the same model, which anticipates Section 9: the field
the conventional weighting damages most is the pressure, by three orders relative
to velocity and four relative to the balanced weighting. Pressure enters the
functional only through $\nabla p$ in the momentum rows, so weighting those rows
down removes its only route into the problem.

## 5. How much accuracy it costs

Section 4 is a fixed-point statement. A practitioner integrates for a finite
time, so we now measure the transient error directly, with a manufactured
solution chosen so that the irreducible spatial residual $\lVert R_h\rVert$ can
be varied by orders of magnitude while everything else is held fixed
(`scratch/mms2d_temporal.py`).

The solution is a time-modulated cellular flow on the unit square, generated from
the streamfunction $\psi=\cos t\,\sin^2\pi x\,\sin^2\pi y$, so that
$\mathbf u=(\psi_y,-\psi_x)$ is divergence-free by construction and vanishes on
all four walls for every $t$. It is transcendental, hence outside the polynomial
space at every order. The convective term is carried in the manufactured forcing
and Newton is iterated out, so the test exercises the full nonlinear path. The
forcing must carry $a_{\rm flux}$, being the source of the *weighted* momentum
row; supplying the physical forcing instead under-forces the problem by the
weight and changes the answer silently.

**Figure 2** shows relative error in the velocity at the final time against the
step, for four polynomial orders and both weightings. The structure is:

1. **While the temporal term dominates, the two weightings are indistinguishable.**
   At $N=12$ they agree to five significant figures — $3.4117\times10^{-6}$
   against $3.4118\times10^{-6}$ — with observed order $1.98$. The weighting is
   not a temporal-order effect, and no conventional order study can detect it.
2. **Each curve then flattens at a floor proportional to $\lVert R_h\rVert$**,
   with constants $\Phi=0.19$ (conventional) and $0.04$–$0.06$ (balanced) that
   are stable across three decades of $\lVert R_h\rVert$.

So the transient error separates as

$$
\lVert U_h(T)-U(T)\rVert\;\approx\;C_2\Delta t^2+\Phi\,\lVert R_h\rVert ,
\tag{5.1}
$$

with $C_2$ common to both weightings. The practical statement is not a loss of
order but a floor: **at a given mesh there is an accuracy the conventional
weighting cannot reach at any time step.**

### 5.1 A criterion, and a test of it

The two terms of (5.1) cross at

$$
\Delta t^\ast=\sqrt{\Phi\lVert R_h\rVert/C_2},
\tag{5.2}
$$

below which the weighting decides the answer. Computed from the two coarsest
meshes, (5.2) accounts for the whole of Figure 2 after the fact, and — the point
of the exercise — it *predicts* where the separation must appear on a mesh where
none is visible. At $N=12$ it places the crossover at
$2.6\times10^{-4}$, a factor of three below the smallest step in Figure 2.
Continuing that row:

| $\Delta t$ | conventional | balanced | ratio |
|---|---|---|---|
| $7.81\times10^{-4}$ | 3.991e−9 | 3.974e−9 | 1.00 |
| $3.91\times10^{-4}$ | 1.079e−9 | 9.965e−10 | 1.08 |
| $1.95\times10^{-4}$ | 8.522e−10 | 2.548e−10 | 3.34 |
| $9.77\times10^{-5}$ | **2.397e−9** | 8.249e−11 | **29.1** |

The balanced error falls monotonically at orders $2.00$, $1.97$, $1.63$. The
conventional error reaches a minimum straddling the predicted $\Delta t^\ast$
and then **grows**, by a factor $2.8$ for the last halving. There is an optimal
time step for the conventional weighting, and refining past it is harmful.

Because the conventional fixed-point operator's conditioning degrades like
$1/\Delta t^2$ (Section 4), an ill-conditioned solve is a competing explanation
for that rise and has to be excluded. Sweeping the solver tolerance that
actually binds gives identical answers to five digits with identical iteration
counts at $10^{-12}$ and $10^{-16}$, so the solves are converged and the rise
belongs to the discrete problem. At a loose tolerance of $10^{-8}$, by contrast,
the conventional formulation is $700\times$ more sensitive than the balanced one:
it also demands a much tighter solve to reach its own worse answer.

### 5.2 A warning about verification

Point 1 above has an uncomfortable corollary. A manufactured solution that the
discrete space represents exactly, or nearly so, produces identical results
under both weightings at every step. A code verified that way passes while
carrying the defect, and will exhibit it only in production, where the flow is at
the resolution limit by definition. Our own start-up Poiseuille test, whose
parabolic solution is essentially representable, returns order $2.04$ for both
weightings and is silent on the matter.

## 6. What it does to standard benchmarks

### 6.1 Lid-driven cavity at $Re=1000$

Marched to steady state on a $4\times4$ mesh of order 15, the conventional
weighting produces a steady state that *depends on the time step used to reach
it*, non-monotonically: root-mean-square distance to the Ghia reference on the
two centrelines is $0.0095$, $0.0256$, $0.0218$ and $0.0122$ at steps of
$1$, $0.1$, $0.03$ and $0.01$. The balanced weighting gives $0.0090$ at
$\Delta t=0.1$ and $0.0061$ at every step from $10^{-2}$ to $10^{-4}$ — flat over
three decades, and closer to the reference than the conventional weighting
achieves at any step. The $v$ error at the smaller steps is below the value of
the converged $N=30$ reference solution itself.

Below a momentum weight of about $10^{-3}$ the conventional solution also
develops a node-to-node oscillation beneath the lid, alternating at seven or
eight of eight consecutive intervals. It is not a solver artefact: it is
reproduced by an independent implementation that shares no code with the solver,
and by the original Fortran program from which the method was ported. The
balanced weighting removes the alternation and reduces the amplitude three- to
fourfold at the same step and mesh.

### 6.2 Orr–Sommerfeld growth rate at $Re=7500$

A case with a known answer, and the sharpest of the three. The least-stable mode
at $\alpha=1$ has growth rate $\sigma=0.00223497$. At $\Delta t=0.1$ both
weightings reproduce it to within $0.25\%$ at order 10 and above, including the
characteristic under-prediction at order 8 that the original reference reports.
Below that step they separate: at $\Delta t=0.02$ and $0.01$ the conventional
weighting is $3$–$6\%$ in error with a local growth rate that drifts from $-10\%$
to $+4\%$ *within a single run*, while the balanced weighting holds
$0.02$–$0.08\%$. Tightening the linear solver makes the conventional result
*worse*, from $3.3\%$ to $6.4\%$, confirming that the error is in the discrete
problem and not in its solution.

One practical caveat emerged here and belongs with the criterion of Section 5.1:
over $10^4$ steps a per-step relative solver reduction of $10^{-2}$ accumulates
to $0.5\%$ in the growth rate. Long integrations need a tolerance chosen for the
step count, not for the step.

### 6.3 Where the claim stops: boundary singularities

The lid-driven cavity has a velocity discontinuity at two corners, and there the
spatial residual is irreducible in a stronger sense. A Richardson
self-convergence study of the transient, from a smooth state and with Newton and
the linear solver converged, gives an observed order of about $0.3$ for *both*
weightings on the singular problem. Regularising the lid to
$u=16x^2(1-x)^2$ restores second order for the balanced weighting on the first
step-halving, while the conventional weighting remains below first order at
$0.6$–$0.7$.

The interpretation follows (5.1): with a singular corner the second term is large
and step-independent, so neither weighting can converge in $\Delta t$ past it.
The level at which this happens, roughly $5\times10^{-4}$ in relative velocity
error, is an order of magnitude below the spatial error of the same computation,
so it is a resolution problem that the time step exposes rather than a defect the
time step causes. Mesh grading at the corner is the remedy; a smaller step is
not.

### 6.4 The unit weighting

For completeness, the third row of Section 2's table. Setting
$w_{\rm mom}=w_{\rm mass}=1$ gives $ma=\mathrm{fac}_1/\Delta t\to\infty$, which by
Proposition 2 loses the *constraints* rather than the momentum. In practice the
solver stalls: iteration counts rise by more than an order of magnitude and the
divergence degrades by a comparable factor. It is the mirror image of the
conventional failure, and it confirms that $ma=O(1)$ is a two-sided requirement.

---

## 7. The same parameter in the operator *(to be written)*

Diagnosis: the crossover $c^\ast\approx\nu p^4/h^2$; below it the pair
$(\mathbf u,\omega)$ decouples and standard relaxation works, above it the
divergence-free kernel is invisible to anything pointwise. Accounts for the
measured failure of point Jacobi, block Jacobi, $p$-multigrid with a Chebyshev
smoother, and low-order-refined AMG. Exact block preconditioning experiments
giving $\kappa\approx3$ flat in $p$, $h$ and $c$ as the target.

## 8. An affordable patch preconditioner *(to be written)*

Overlapping vertex patches plus a $p=2$ coarse space; ring assembly; exact static
condensation and its $8$–$13\times$ memory reduction. Iteration counts: 2D
$19$–$21$ flat for $N=5$–$20$ against $435$–$4010$; 3D $36$–$38$ for $N=4$–$10$
against $1158$–$4802$; production channel $72$ against $4675$ with the step
identical to every logged digit. Memory model and its $N^4$ scaling.

## 9. Implementation and cost *(to be written)*

Factor sharing by mask pattern, verified rather than assumed; explicit inverses
applied as batched matrix products in place of batched triangular solves; graph
capture. Apply $160\,\mathrm{ms}\to8.3\,\mathrm{ms}$; build $40\,\mathrm{min}\to30\,\mathrm{s}$;
channel step $64\,\mathrm{min}\to25\,\mathrm{s}$ on a GB10. Costs on three
machines.

## 10. Direct numerical simulation, and a necessary exception *(to be written)*

**The exception first.** With convection treated explicitly by RKW3, each
implicit stage is a Stokes *projection* whose right-hand side is not solenoidal.
There the constraint rows must dominate for the stage to return a
divergence-free field, and the conventional weighting is the correct choice: the
balanced weighting drives the divergence from $8\times10^{-4}$ to $3.3\times10^{-1}$
in a single step and $5.7\times10^{-1}$ in ten, with the bulk physics unchanged
to four digits. Consistently, the mechanism of Sections 3–6 does not fire there:
the production fields carry no mesh-scale mode, with node-alternation statistics
matching a fractional-step solution on the same mesh and less energy in the top
modes. The criterion of Section 5.1 applies to a step whose *fixed point* is the
intended solution, which means implicit convection; a projection stage has no
fixed point to degenerate toward.

**The simulation.** Minimal-channel DNS at $Re_\tau=180$, statistics against five
reference databases. *(Long run pending.)*

## Appendix A. An energy argument and why it is not enough *(to be written)*

The three row blocks of the scaled limit give $\omega=\nabla\times\mathbf u$
exactly; an energy identity closed by a discrete Poincaré inequality and an
inverse inequality then gives the trivial solution under
$\mathrm{fac}_1\nu C_P^2C_{\rm inv}^2<2$. Both constants are computable exactly
for the Gauss–Lobatto space, and doing so shows the condition is satisfied only
by the coarsest discretisations — violated by three orders elsewhere while the
operator remains nonsingular — and that its viscosity dependence is backwards:
the limit degenerates as $\nu\to0$, the regime the condition declares safe. The
argument is recorded because it is instructive about what a correct proof must
avoid, not because it establishes the result.

## References *(partial)*

- [BG98] P. B. Bochev and M. D. Gunzburger, *Finite element methods of
  least-squares type*, SIAM Review **40**(4) (1998) 789–837.
- P. B. Bochev and M. D. Gunzburger, *Least-Squares Finite Element Methods*,
  Applied Mathematical Sciences 166, Springer (2009).
- J. H. Adler, I. Lashuk, S. P. MacLachlan and L. T. Zikatanov, *Discrete energy
  laws for the first-order system least-squares finite-element approach*, LNCS
  10665 (2018).
- J. H. Adler, S. P. MacLachlan and N. Madden, *First-order system least squares
  finite-elements for singularly perturbed reaction-diffusion equations*,
  arXiv:1909.08598.
- T. Führer and M. Karkulik, *New a priori analysis of first-order system
  least-squares finite element methods for parabolic problems*, arXiv:1805.04147.
- D. N. Arnold, R. S. Falk and R. Winther, *Multigrid in H(div) and H(curl)*,
  Numer. Math. **85** (2000).
- L. F. Pavarino, *Additive Schwarz methods for the p-version finite element
  method*, Numer. Math. **66** (1994).
- U. Ghia, K. N. Ghia and C. T. Shin, *High-Re solutions for incompressible flow
  using the Navier–Stokes equations and a multigrid method*, JCP **48** (1982).
