# Time-marching least-squares spectral elements for unsteady incompressible flow: the weighting that makes them accurate and the preconditioner that makes them affordable

*Draft. Sections 2–9 are written from completed work; Section 1, Section 10 and
Appendix A remain outlined, and Section 10 awaits the long channel run. Plan: `PAPER_PLAN_PRACTICAL.md`. Every
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
condensation, gives iteration counts flat in polynomial order and in mesh size
across three geometries where those methods grow by two orders of magnitude, and
an advantage that *grows* as the step is refined: from 36-fold to 309-fold fewer
iterations as $c$ rises from 15 to 1500 on a backward-facing step. Batched to a
form that a GPU can execute, it turns a channel-flow time step of 60 seconds into
25.

We close with a direct numerical simulation of turbulent channel flow, and with
the condition under which the accuracy result does *not* apply: a stage with
explicit convection is a projection, its right-hand side is not solenoidal, and
there the conventional weighting is the correct one.

---

## 1. Introduction

A least-squares finite element method takes a first-order system
$\mathcal L U=F$ and minimises the residual in a discrete norm. Three properties
follow whatever the underlying physics: the algebraic problem is symmetric and
positive definite, so the conjugate gradient method applies and no saddle point
has to be navigated; the velocity and pressure spaces need satisfy no inf-sup
compatibility condition, so equal-order interpolation is admissible; and the
value of the functional is a sharp, computable a-posteriori error estimator.
Written in velocity–vorticity–pressure form the incompressible Navier–Stokes
equations are such a system, and the theory covering it is mature. Against a
projection or fractional-step method, which must choose pressure boundary
conditions that the continuous problem does not supply and which fixes a
splitting error at the outset, the attractions are substantial.

Unsteady production computation nevertheless does not use it. Two objections are
usually offered, and they are usually offered separately. The first is
accuracy: least-squares solutions are held to be over-constrained, to lose
accuracy relative to a Galerkin discretisation of the same order, and — in
reports that are harder to dismiss — to *degrade* when the time step is refined,
which is the opposite of what any time-marching scheme should do. The second is
cost: the operator is a squared one, its condition number scales like the square
of a second-order operator's, and the standard preconditioners of high-order
finite element practice have been reported to work in some settings and to stall
completely in others, with no stated criterion separating the two.

This paper's claim is that the two objections are the same number seen twice,
and that both have remedies.

Write one implicit step as the minimisation of
$\lVert w_{\rm mass}\Delta t^{-1}(\mathrm{fac}_1\mathbf u-\sum_m\alpha_m
\mathbf u^{n-m})+w_{\rm mom}\mathcal N(U)\rVert^2+\lVert CU\rVert^2$, with
$\mathcal N$ the momentum residual, $C$ the divergence and curl constraints, and
$\mathrm{fac}_1$ the backward-difference consistency constant. The mass
coefficient the step imposes on the momentum rows is
$c=\mathrm{fac}_1/\Delta t$, and it appears in two places.

**In the fixed point.** At a fixed point of the step map the mass and history
terms cancel identically, for every backward-difference order, because
$\mathrm{fac}_1=\sum_m\alpha_m$ is exactly that statement. What the iteration
converges to is therefore *not* the minimiser of any steady least-squares
functional: an extra cross term survives, and the weight with which the momentum
equation enters the fixed-point operator is the product $ma$ of the two row
weights. Under the scaling used throughout the literature that product is
$\Delta t$. The momentum equation therefore leaves the problem as the step is
refined, the constraint block alone survives, and the computed steady state
depends on $\Delta t$ and drifts away from the correct answer as $\Delta t$
falls. We prove (Section 3) that $ma=O(1)$ is necessary, that the ratio it
constrains is invariant under every diagonal row and column scaling so that no
reformulation evades it, and that time consistency independently forces the two
weights to be equal — so that $w_{\rm mom}=w_{\rm mass}=\sqrt{\Delta t}$ is the
unique admissible choice. Sections 4 to 6 quantify what the conventional choice
costs, on a one-dimensional model where everything is computable in closed form,
on a transient manufactured solution where the error separates cleanly into a
$\Delta t^2$ term and a floor, and on two standard benchmarks.

**In the operator.** The same $c$ is the zeroth-order coefficient of the
algebraic system each step must solve. Read as a parameter-dependent elliptic
system, the operator changes character at $c^\ast\approx\nu p^4/h^2$: below it
every field is controlled in $H^1$ and standard relaxation is optimal; above it
the momentum row degenerates, velocity and vorticity are coupled by the curl row
and controlled only in $H(\mathrm{div})$, and the discretely divergence-free
subspace — two thirds of the space — becomes invisible to any pointwise or
per-variable correction. A production channel simulation sits two orders above
$c^\ast$ and a steady cavity computation sits below it, which is why the
literature contains both reports. Section 7 establishes this by exact block
preconditioning, and Sections 8 and 9 supply the remedy the $H(\mathrm{div})$
literature prescribes — overlapping vertex-patch Schwarz with a low-order coarse
space — in a form whose memory and arithmetic are affordable at spectral order.

The paper is organised as a ladder, and each rung is a complete statement resting
on the one below it: an algebraic identity that needs no inequality (Section 3),
a model problem in which every constant is computed (Section 4), a manufactured
solution that measures the error model and then verifies it by prediction
(Section 5), standard benchmarks with published answers (Section 6), a solver
diagnosis and its remedy on three geometries (Sections 7 and 8), an
implementation that runs at the machine's floor (Section 9), and a direct
numerical simulation of turbulent channel flow (Section 10).

Two boundaries of the claim are stated where they arise rather than in a
concluding caveat. Near a boundary singularity both weightings converge only
sublinearly in $\Delta t$, at a level an order below the spatial error
(Section 6.3). And when convection is treated explicitly, each implicit stage is
a Stokes *projection* whose right-hand side is not solenoidal; that stage has no
fixed point to degenerate toward, the constraint rows must dominate for the
projection to do its work, and the conventional weighting is there the correct
choice (Section 10). The criterion of this paper applies to a step whose fixed
point is the intended solution, which is to say to implicit convection.

One practical consequence deserves stating at the outset, because it explains how
a defect of this size has persisted. With the solution well resolved in space,
the two weightings agree to five significant digits. The error they differ in is
proportional to the spatial residual, so it is invisible on exactly the test a
careful developer runs: **a code verified against a well-resolved manufactured
solution passes this defect silently**, and meets it later as a mesh-scale
oscillation or an unexplained time-step sensitivity in a production run.

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

## 7. The same parameter in the operator

Sections 3–6 read $c=\mathrm{fac}_1/\Delta t$ as a property of the *fixed point*.
The same number is also a property of the *operator that each step must solve*,
and it is there that it accounts for the second half of the method's reputation.

Note first that $c$ is unchanged by the weighting. Multiplying the momentum row
by $w_{\rm mom}$ scales $a_{\rm mass}$ and $a_{\rm flux}$ together, so all three
members of the family in Section 2 share
$c=a_{\rm mass}/a_{\rm flux}=\mathrm{fac}_1/\Delta t$ and differ only in the
scale of that row relative to the constraints. The consequence is practical: a
preconditioner built and tuned under the conventional weighting transfers to the
balanced one unchanged, which we confirm below. The fix of Section 5 costs
nothing in the solver, and the solver of Sections 8–9 is not a workaround for the
weighting.

### 7.1 Two regimes, and the crossover between them

Divide the functional by $a_{\rm flux}^2$ so that the momentum row reads
$\mathbf u+(\nu\nabla\times\omega+\nabla p)/c$:

$$
J \;=\; \Big\lVert\,\mathbf u+\tfrac1c\big(\nu\nabla\times\omega+\nabla p\big)\Big\rVert^2
\;+\;\lVert\nabla\!\cdot\mathbf u\rVert^2\;+\;\lVert\nabla\times\mathbf u-\omega\rVert^2 .
$$

Read as a parameter-dependent system in the sense of Agmon, Douglis and
Nirenberg, the mass term is of order zero with coefficient one and the flux
terms are of order one with coefficient $1/c$. The principal part therefore
changes character where the two balance on the finest representable scale,
$\lvert\xi\rvert_{\max}\sim p^2/h$, that is at

$$
c^\ast \;\approx\; \nu\,\lvert\xi\rvert_{\max}^2 \;\approx\; \nu\,p^4/h^2 .
$$

For $c\ll c^\ast$ the momentum row supplies $O(1)$ control of
$\nu\nabla\times\omega\approx\nu\nabla\times\nabla\times\mathbf u$. Eliminating
$\omega$, the velocity is controlled in divergence *and* curl, whose
intersection is $H^1$ on our domains; every block is Laplacian-like and the
functional is $H^1$-elliptic in each variable. This is the regime in which
Jacobi-smoothed $p$-multigrid for least-squares systems is provably optimal, and
in which we measure it to be $p$-independent.

For $c\gg c^\ast$ the momentum row degenerates to $\lVert\mathbf u\rVert^2$. The
curl row makes $\omega$ an order-zero slave of $\nabla\times\mathbf u$, and
eliminating it leaves the velocity controlled by
$\lVert\mathbf u\rVert^2+\lVert\nabla\!\cdot\mathbf u\rVert^2$ alone — the
$H(\mathrm{div})$ norm and nothing more. The functional is then norm-equivalent
to

$$
\lvert\!\lvert\!\lvert(\mathbf u,\omega,p)\rvert\!\rvert\!\rvert^2
=\lVert\mathbf u\rVert^2_{H(\mathrm{div})}
+\lVert\omega-\nabla\times\mathbf u\rVert_0^2
+c^{-2}\lvert p\rvert_1^2
\qquad\text{up to }O(\nu/c),
$$

with two features that decide everything that follows. The vorticity appears
*shifted*: the functional is a norm on the pair $(\mathbf u,\omega-\nabla\times
\mathbf u)$, not on $\mathbf u$ and $\omega$ separately. And the pressure
decouples, entering only through a scaled Neumann Laplacian.

A production channel simulation at $Re_\tau=180$ with $\Delta t=8\times10^{-4}$
has $c=5405$ against $c^\ast\approx10^2$: it is two orders into the second
regime. A steady cavity computation at $\Delta t=O(1)$ is in the first. The two
literatures that report success and failure for the same preconditioners are
reporting from opposite sides of $c^\ast$.

### 7.2 The norm is the right one, and it is not separable

Operator preconditioning in the sense of Mardal and Winther turns a norm
equivalence into a prescription: the Riesz map of the equivalent norm is the
optimal block preconditioner, with a condition number bounded by the equivalence
constants and independent of $h$, $p$ and the parameters held uniform. This is
testable directly. Take the **exact** block diagonal of the assembled operator —
the strongest block preconditioner with a given block structure — and vary the
blocks. If the norm of §7.1 is right, the grouping it dictates gives a flat, small
$\kappa$, and the groupings it forbids do not.

Assembled per Fourier mode on a channel mesh, $\nu=1/180$, iterations and
$\kappa$ from the conjugate-gradient Lanczos values:

| $c$ | $p$ | dofs | Jacobi | 7 blocks, one per variable | 3 blocks $\mathbf u\lVert\omega\lVert p$ | 2 blocks $(\mathbf u,\omega)\lVert p$ |
|---|---|---|---|---|---|---|
| 1 | 4 | 455 | 175 / 1.6e3 | 116 / 150 | 116 / 150 | **16 / 3.3** |
| 1 | 12 | 4055 | 701 / 4.4e4 | 130 / 180 | 128 / 180 | **17 / 3.5** |
| 5405 | 4 | 455 | 209 / 1.6e3 | 170 / 390 | 166 / 380 | **15 / 3.3** |
| 5405 | 8 | 1807 | 620 / 1.2e4 | 426 / 3.3e3 | 424 / 3.2e3 | **14 / 3.3** |
| 5405 | 12 | 4055 | 1201 / 3.7e4 | 827 / 1.5e4 | 798 / 1.4e4 | **13 / 3.3** |

Refining $h$ at $p=8$, $c=5405$ takes Jacobi from 620 to 1244 iterations and
$\kappa$ from $1.2\times10^4$ to $5.5\times10^4$, while the two-block value goes
from 14 to 13 iterations at $\kappa=3.3$ unchanged.

So $\kappa\le3.6$, flat in $h$, $p$ and $c$, is available, and the whole
difficulty is the $(\mathbf u,\omega)$ coupling: separating the two costs a
factor of $4\times10^3$ in $\kappa$ at $c=5405$ and nothing at all at $c=1$,
which is the regime boundary of §7.1 appearing in the numbers. The diagonal
blocks are individually benign — $\kappa(A_{uu})\propto p^2$,
$\kappa(A_{\omega\omega})\propto p$, $\kappa(A_{pp})\propto p^4$ for what is a
Neumann Laplacian.

The shift matters too. Building the Riesz blocks literally as
$H(\mathrm{div})\times L^2\times H^1$, that is $M+K_{\rm div}$ for velocity and
a mass matrix for vorticity, gives $\kappa=2.5\times10^6$, worse than Jacobi.
Writing the same preconditioner in block-$LDL^{\mathsf T}$ form with the exact
Schur complement $S_u=A_{uu}-A_{u\omega}A_{\omega\omega}^{-1}A_{\omega u}$
recovers $\kappa=3.3$ exactly, and replacing $S_u$ by $M+K_{\rm div}$ loses it
again ($\kappa$ 460 to $2.7\times10^4$ over $p=4$ to 12). The difference is a
discrete curl-representation defect: $\nabla\times\mathbf u_h$ is of degree $p-1$
and discontinuous across faces, the $C^0$ vorticity space cannot represent it
there, and the minimisation over $\omega_h$ leaves a face-jump penalty on
$\nabla\times\mathbf u_h$. It is a property of the discretisation, not of the
continuous norm, and it cannot be dropped.

### 7.3 Why every pointwise method failed, and had to

Because $A_{\omega\omega}$ is diagonal on Gauss–Lobatto nodes, $S_u$ is exactly
and matrix-freely applicable, so the question can be put to the velocity block
alone. It gains nothing: Jacobi on $S_u$ runs $\kappa$ from $3.5\times10^2$ to
$5.5\times10^4$ over $p=4$ to 12, the same as on the full operator, and Jacobi on
the pure $H(\mathrm{div})$ operator $M+K_{\rm div}$ is the same again. The softest
Jacobi-preconditioned eigenvector of $S_u$ has
$\lVert\nabla\!\cdot\mathbf u\rVert/\lVert\mathbf u\rVert=0$ to machine precision
with $\lVert\nabla\times\mathbf u\rVert/\lVert\mathbf u\rVert=O(1)$: a smooth,
exactly divergence-free velocity, on which $M+K_{\rm div}$ acts as the identity
while its diagonal is $p^4/h^2$.

That is the classical obstruction of $H(\mathrm{div})$ solvers. The solenoidal
subspace is roughly two thirds of the space, no node-wise correction can see it,
and a coarse space built by nodal interpolation does not correct it either. It
accounts in one statement for four independent failures in our own history —
point Jacobi, node-block Jacobi, per-variable blocks, and $p$-multigrid with
Jacobi or Chebyshev smoothing and either a Galerkin or a direct coarse solve —
each of which needed thousands of iterations per stage at the channel's $c$, and
each of which we had previously suspected of being an implementation fault.

Two remedies that work for high-order Poisson problems fail here for the same
reason, and both are worth recording because both are natural things to try.
A nodal Hiptmair auxiliary-space sweep fails because on $C^0$ Gauss–Lobatto
elements the discrete divergence-free subspace has no potential representation.
Low-order-refined preconditioning fails in the second regime specifically:

| $c$ | $N$ | Jacobi | exact $A_{\rm LOR}^{-1}$ solve | AMG$(A_{\rm LOR})$ on $A_{\rm SEM}$ | AMG$(A_{\rm LOR})$ on $A_{\rm LOR}$ |
|---|---|---|---|---|---|
| 1 | 8 | 1355 / 1.0e5 | 219 / 630 | 324 / 1.4e3 | 60 / 47 |
| 1 | 16 | 2965 / 9.1e5 | 180 / 370 | 422 / 2.5e3 | 91 / 130 |
| 5405 | 8 | 1736 / 9.5e4 | **1392 / 4.7e4** | 1433 / 5.1e4 | 73 / 73 |
| 5405 | 16 | 5103 / 1.3e6 | **4833 / 6.5e5** | 5199 / 7.8e5 | 112 / 280 |

At $c=1$ the low-order operator is spectrally equivalent and algebraic multigrid
on it is a serviceable preconditioner. At $c=5405$ the **exact** low-order solve
is almost as bad as Jacobi, so no multigrid, however good, can rescue the
approach; and the last column shows that the multigrid is not at fault, since it
solves its own matrix in 73 to 280 iterations. The bilinear discretisation on the
Gauss–Lobatto sub-cells and the spectral element have different discretely
divergence-free kernels, and it is precisely those modes that dominate at large
$c$. On $C^0$ nodal elements the kernel has no faithful representation in any
space but its own.

What remains is the remedy the $H(\mathrm{div})$ literature prescribes: solve
local problems large enough to contain the local kernel modes.

## 8. An affordable patch preconditioner

### 8.1 The method

Let $\mathcal P_v$ be the overlapping patch of all elements sharing an interior
vertex $v$, with the unknowns on its closure. The preconditioner is additive
Schwarz on these patches with a low-order coarse level,

$$
M^{-1}=P_c\,A_c^{-1}P_c^{\mathsf T}+\sum_v R_v^{\mathsf T}\big(R_vAR_v^{\mathsf T}\big)^{-1}R_v ,
$$

where $R_v$ restricts to the patch and $P_c$ is the natural injection from the
same discretisation at $p=2$. Three details are not optional in practice. Patch
matrices are assembled from the full ring of elements touching the vertex, not
from the patch interior alone, or the local kernel modes are cut by the
artificial boundary. Each local matrix is symmetrically equilibrated before
factorisation, since the row scales of a least-squares operator span several
orders. And the coarse level carries the pressure null-space treatment of the
fine problem. Element-sized overlap is what makes Schwarz $p$-independent for
spectral elements, and the vertex star is the standard relaxation for
divergence-free kernels; neither idea is new here. What is new is the pairing
with the regime diagnosis of §7 — which says when the expense is necessary — and
with the condensation of §8.2, which is what makes it affordable at spectral
order.

### 8.2 Static condensation is exact, and pays for itself

A dense patch factor costs $\tfrac12 n_P^2$ with $n_P=(2N+1)^dF$ for $F$ coupled
fields: 32 GB on the three-dimensional channel at $N=8$, 44 GB for a
two-dimensional cavity at $N=30$. Condensing element interiors removes the
$(2N+1)^{2d}$ constant. Each element's interior block is factored once and shared
by the $2^d$ patches containing it; each patch keeps only a dense Schur
complement on its edge unknowns. The result is algebraically identical, not an
approximation, and we verify that iteration counts and $\kappa$ agree exactly:

| $N$ | dofs | dense it / $\kappa$ | stored | condensed it / $\kappa$ | stored | memory ratio |
|---|---|---|---|---|---|---|
| 8 | 4099 | 42 / 27 | 7.67e6 | **42 / 27** | 1.00e6 | 7.7× |
| 12 | 9219 | 33 / 15 | 3.66e7 | **33 / 15** | 3.47e6 | 10.6× |
| 16 | 16387 | 31 / 17 | 1.12e8 | **31 / 17** | 9.35e6 | 12.0× |
| 24 | 36867 | 31 / 25 | 5.52e8 | **31 / 25** | 4.24e7 | 13.0× |

The ratio grows with $N$ because the interiors scale as $(N-1)^{2d}$ per element
while the Schur complements scale as the square of the edge count, and the apply
is two to four times faster because the work moves into the smaller shared
interior solves. On the three-dimensional channel this is the difference between
32 GB and 4 GB; at $N=24$ in two dimensions, between 4.4 GB and 0.34 GB.

### 8.3 Iteration counts

**Two dimensions, production path.** Lid-driven cavity at $Re=1000$, $4\times4$
elements, marched from rest, preconditioner built once and reused:

| $N$ | dofs | Jacobi it/step | patch it/step | stored | ratio |
|---|---|---|---|---|---|
| 5 | 1,764 | 435 | **21** | 2.9 MB | 21× |
| 10 | 6,724 | 1493 | **19** | 21 MB | 80× |
| 15 | 14,884 | 2709 | **20** | 80 MB | 138× |
| 20 | 26,244 | 4010 | **19** | 221 MB | 213× |

**Three dimensions.** Channel mesh, $c=5405$, balanced weighting:

| mesh | Jacobi | patch + coarse | patches only |
|---|---|---|---|
| $4\times4$, $N=4$ | 1158 | **36** | 64 |
| $4\times4$, $N=6$ | 2306 | **35** | 63 |
| $4\times4$, $N=8$ | 3593 | **36** | 64 |
| $4\times4$, $N=10$ | 4802 | **38** | 64 |

Flat at 19–21 in two dimensions over $N=5$ to 20, and at 35–38 in three over
$N=4$ to 10, while Jacobi grows roughly linearly in $N$ in both. The one-level
variant sits at a flat 64, so the coarse term is worth $1.8\times$ on this mesh
and more on larger ones. Under the conventional weighting the patch counts are
identical (17 against 17 on the small rig) while Jacobi's change by $-10\%$ to
$+100\%$ depending on $c$, which is the claim of §7 that the preconditioner
depends on $c$ and not on the weighting.

**Production.** On the minimal channel itself, restarting a running simulation
and sharing one preconditioner across the three Runge–Kutta stage values of $c$:
worst-stage 77 iterations against Jacobi's 4675, a factor of 61, with the step
reproducing the Jacobi step in every logged digit, including the friction
velocity, the divergence norm, the energy and the dissipation.

### 8.4 A third geometry, and the crossover

The cavity is closed and Dirichlet; the channel is periodic with walls. Neither
tests an inflow–outflow problem, a re-entrant corner, or a graded mesh, and in
three dimensions our implementation shares factors between elements, which a
non-uniform mesh forbids. Gartling's backward-facing step at $Re=800$ supplies
all three. Marched from the converged steady field, conjugate gradients to a
relative $10^{-8}$:

| | grid, order | Jacobi it | patch it | ratio | wall |
|---|---|---|---|---|---|
| order | 11×4, $N=5$ | 5906 | **48.2** | 123× | 3.1× |
| | 11×4, $N=6$ | 9290 | **52.8** | 176× | 4.2× |
| | 11×4, $N=7$ | 11367 | **55.7** | 204× | 3.9× |
| mesh | 11×4, $N=6$ | 9290 | **52.8** | 176× | 4.2× |
| | graded, $N=6$ | 9822 | **53.5** | 184× | 4.1× |
| | 18×4, $N=6$ | 10323 | **53.3** | 194× | 4.0× |

Flat in $p$ and flat in $h$, including on the graded grid, on boundary
conditions neither earlier case covers; the free-outlet variant, with the
outflow plane left unknown and the pressure pinned at a corner, gives 142×.

The sharpest test of §7 is the third sweep, holding the mesh fixed and varying
only $\Delta t$, hence only $c$:

| $\Delta t$ | $c$ | Jacobi it (worst) | patch it | ratio | wall speed-up |
|---|---|---|---|---|---|
| $10^{-1}$ | 15 | 3883 (3917) | 108.8 | 36× | 0.9× |
| $10^{-2}$ | 150 | 9393 (10512) | 53.8 | 175× | 3.8× |
| $10^{-3}$ | 1500 | 13687 (25647) | **44.2** | 309× | 6.7× |

The patch preconditioner gets *better* as the step is refined, from 109 to 44
iterations, while Jacobi degrades from 3883 to 13687 with its worst solve
reaching 25647. This is the regime argument stated as a measurement: increasing
$c$ moves the operator further into the $H(\mathrm{div})$ regime, where a
pointwise method is blind to the kernel and a patch method is not. It also fixes
the honest boundary of the claim. At $\Delta t=10^{-1}$, with $c$ near $c^\ast$
for this grid, the patch preconditioner is slightly *slower* in wall time than
Jacobi. It is a solver for time steps small enough that the physics, or the
explicit convection, requires them — which is to say for unsteady simulation.

### 8.5 Where it stops being the right choice

Storage is $O(N^{2d})$ per element interior and $O(N^{2(d-1)})$ per patch face,
so memory grows like $N^4$ in three dimensions even after condensation: 1.0 GB
at $N=10$ on a $4\times4$ mesh, an estimated 52 GB for the production channel at
$N=12$ in double precision, 13 GB in single. Build cost grows like $(N-1)^6$.
Refining in $h$ instead is linear in both. Beyond roughly $N=12$, the routes that
remove the $(N-1)^{2d}$ interior cost altogether — sparse patch bases built from
fast diagonalisation, or inexact local solves by patch-local multigrid — become
necessary rather than optional; we have not needed them.

## 9. Implementation and cost

The algorithm of Section 8 is arithmetic that batches almost perfectly and, in a
naive implementation, does not. The prototype's apply at $N=20$ in two dimensions
costs $2.5\times10^8$ floating-point operations, about 5 ms at the rate the
machine sustains, and measured 329 ms: a factor of 60 lost to a Python loop over
patches with per-call library overhead. In three dimensions the same gap put the
channel step at 64 minutes against Jacobi's 60 seconds, with the iteration count
already 61 times better. The iteration counts of §8.3 were therefore a real
result attached to an unusable implementation, and closing that gap is most of
the engineering in this work.

Four changes close it, in order of what each was worth.

**Uniform shapes and shared factors.** Blocks are padded to a small number of
uniform sizes so that every local solve in a sweep is one batched call. On the
channel, the masks produce three element types and five patch types per Fourier
mode, and eleven at the zero mode where the pressure is pinned. Factors are
shared between blocks with identical mask patterns — and the sharing is verified
equal to round-off before it is used, not assumed from the mesh topology. Build
cost fell from 40 minutes to 2.3 minutes, storage from 18 GB to 1.4 GB, and the
apply from 16 s to 0.21 s per iteration.

**Explicit inverses applied as matrix products.** A profile of the resulting
device code showed 62% of the time inside batched triangular solves, executed as
roughly 3200 separate blocked kernels per apply. Since the local matrices are
equilibrated and factored once and applied thousands of times, we store the
explicit inverse of each and apply it as a batched matrix–matrix product. The
storage is the same, the result agrees with the triangular-solve form to
$3\times10^{-15}$, and the apply halves.

**The coarse level.** At production size the dense coarse factor is 5.3 GB, whose
traffic dominates the apply on a bandwidth-limited device. Two remedies are
available and the better one is machine-dependent: applying its explicit inverse
as one batched product, or keeping the coarse solve as a sparse factorisation on
the host.

**Graph capture.** With shapes fixed, the whole apply is captured once and
replayed, removing launch overhead from what is by then a millisecond-scale
kernel sequence.

Measured on three machines, all in double precision, all reproducing the Jacobi
solution to every logged digit:

| machine | apply per iteration | step, patch | step, Jacobi |
|---|---|---|---|
| CPU, 16 cores | 0.21 s (batched solves) | 70 s | ≈80 s |
| GB10 | 113 ms (matrix products) | **25.4 s** | 60 s |
| A100 | 71 ms, 8.3 ms after graph capture | 11.5 s, ≈2 s projected | 22.3 s |

Each row's step uses 71 to 72 iterations per stage. The GB10 apply is
arithmetic-bound in double precision, and there the patch method is 2.4 times
faster per step than the running Jacobi simulation it replaced. The A100 apply of
8.3 ms is at that device's bandwidth floor, where reading the coarse factor alone
accounts for 4.3 ms; its measured 11.5 s step predates both the matrix-product
apply and graph capture, and the projection is $213\times9.9\,\mathrm{ms}$ of
applies plus operator work. The
useful summary is that a solve which cost 4675 iterations and 60 seconds per step
now costs 72 iterations and 25 seconds on the same device, with the remaining
factor of ten visible in the profile rather than hypothetical.

Three practical notes for anyone repeating this. The preconditioner may be
shared across the three Runge–Kutta stage values of $c$ and rebuilt rarely; on
the channel one build at the middle stage serves all three with no change in
iteration count, which is what makes the build cost irrelevant in production.
Fused kernels tuned for one device are not portable: our fused path is 20 times
*slower* than the generic one on a different accelerator. And a compiled backend
that silently falls back to interpretation when a shape cache overflows will hide
all of this; ours did, until the cache limit was raised.

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

**The step is set by the physics, not by the scheme.** A direct simulation must
resolve the fastest motion in the flow, and it is worth checking which of the two
candidate limits actually binds. Measuring the dissipation from the velocity
gradients of the computed field gives a minimum Kolmogorov time
$\tau_\eta=\sqrt{\nu/\varepsilon}$ of $5.2\times10^{-3}$ at the wall, while the
convective CFL of the same field is 1.11 against the RKW3 limit of $\sqrt3$. So
resolving $\tau_\eta$ to a tenth requires $\Delta t\le5.2\times10^{-4}$ and
stability permits $1.24\times10^{-3}$: the physical requirement is the tighter of
the two by a factor of 2.4. The production step $\Delta t=8\times10^{-4}$ sits
between them, resolving $\tau_\eta$ by a factor of 6.5 at $\Delta t^+=0.14$.

This matters for Section 7 rather than for Section 5. A direct simulation is
*forced* to a small time step by the flow it is resolving, a small step means a
large $c$, and large $c$ is exactly the regime in which the divergence-free
directions are invisible to pointwise relaxation. The step cannot be enlarged to
suit the solver, so the solver has to be built for the step — which is the same
statement as the backward-facing step's $c$-sweep in Section 8.4, read from the
other end.

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

- S. Agmon, A. Douglis and L. Nirenberg, *Estimates near the boundary for
  solutions of elliptic partial differential equations satisfying general
  boundary conditions*, CPAM **12** (1959) 623-727; **17** (1964) 35-92.
- K.-A. Mardal and R. Winther, *Preconditioning discretizations of systems of
  partial differential equations*, Numer. Linear Algebra Appl. **18** (2011) 1-40.
- Z. Cai, T. A. Manteuffel and S. F. McCormick, *First-order system least squares
  for the Stokes equations, with application to linear elasticity*, SINUM **34**
  (1997) 1727-1741.
- R. Hiptmair, *Multigrid method for H(div) in three dimensions*, ETNA **6**
  (1997) 133-152.
- R. Hiptmair and J. Xu, *Nodal auxiliary space preconditioning in H(curl) and
  H(div) spaces*, SINUM **45** (2007) 2483-2509.
- P. E. Farrell, M. G. Knepley, L. Mitchell and F. Wechsung, *PCPATCH: software
  for the topological construction of multigrid relaxation methods*, ACM TOMS
  **47** (2021) 25.
- W. Couzy and M. O. Deville, *A fast Schur complement method for the spectral
  element discretization of the incompressible Navier-Stokes equations*, JCP
  **116** (1995) 135-142.
- P. D. Brubeck and P. E. Farrell, *A scalable and robust vertex-star relaxation
  for high-order FEM*, SISC **44** (2022) A2991-A3017.
- W. Pazner, T. Kolev and C. R. Dohrmann, *Low-order preconditioning for the
  high-order finite element de Rham complex*, SISC **45** (2023) A675-A702.
- D. K. Gartling, *A test problem for outflow boundary conditions - flow over a
  backward-facing step*, Int. J. Numer. Methods Fluids **11** (1990) 953-967.
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
