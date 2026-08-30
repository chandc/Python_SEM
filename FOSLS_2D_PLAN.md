# Testing McCormick's FOSLS claims on `lssem2d` — review and plan

Review of `Steve McCormick's  VVP Least Squares.md`, and a staged plan to test its
three claims against the 2D solver.

---

## 1. What the document claims, and what it omits

The summary is a fair statement of the FOSLS framework (Cai–Manteuffel–McCormick).
Three claims are load-bearing:

| # | claim | testable here? |
|---|---|---|
| **C1** | The functional is SPD and **$H^1$-norm-equivalent**, hence uniformly elliptic | yes — measurable |
| **C2** | Boundary conditions belong **in the functional**, weakly, to keep coercivity | yes — we currently do the opposite |
| **C3** | AMG then gives **$h$-independent** convergence | yes — and we already have contrary evidence |

**Four things the document omits that decide whether any of this transfers.**

**Norm equivalence is a statement about *scaling*, not just about form.** The theory
requires constants $0 < c_1 \le c_2$ with

$$
c_1 \,\|\mathbf{Q}\|_{1,\Omega}^2 \;\le\; \mathcal{F}(\mathbf{Q};\mathbf{0}) \;\le\; c_2 \,\|\mathbf{Q}\|_{1,\Omega}^2 ,
$$

and everything downstream — ellipticity, AMG's $h$-independence, the error bound —
follows from $c_2/c_1$ being $O(1)$. **Arbitrary row weights destroy it.** Our
functional is
$\mathcal{F} = \sum_r \rho_r \|R_r\|^2_{0,\Omega}$ with $\rho_r$ chosen for
*time-stepping* reasons, not for norm equivalence (§5.1). That is the first thing
to measure and the document never raises it.

**The 3D VVP system needs the redundant $\nabla\cdot\boldsymbol{\omega} = 0$ row
for ellipticity — and we down-weight it by $10^{-4}$.** §7J measured that
down-weighting as worth **14.6×** in Jacobi iterations at production mesh size.
If FOSLS theory is right, we have bought Jacobi conditioning by *breaking the
very ellipticity AMG would need*. **In 2D this tension does not exist** — $\omega$
is a scalar, $\nabla\cdot\boldsymbol{\omega}$ is not a thing — which is precisely
why 2D is the clean testbed and why testing here first is the right call.

**"Elliptic" is not our operator.** FOSLS $h$-independence is proved for the
*steady* elliptic system. Ours carries a mass term $a_{\text{mass}}\,\mathbf{u}$
from the implicit time step, making it reaction-dominated at production
$\Delta t$ ($a_{\text{mass}} \sim 10^3$). That *helps* conditioning — and it also
means an $h$-independence result obtained at large $a_{\text{mass}}$ proves
nothing about the elliptic limit. Any test must sweep $a_{\text{mass}}$.

**AMG needs a *sparse* matrix, and the SEM operator is not one.** The document
says AMG applies "naturally" to the FOSLS stiffness matrix. For a *low-order* FOSLS
discretisation it does. For a **spectral element** discretisation it does not: the
element block is dense ($324\times324$ at $N=8$), which is exactly what AMG cannot
coarsen. The fix is low-order refined preconditioning (F2) — standard practice for
high-order methods, and the single most important thing the document omits for our
setting.

---

## 2. The central prediction — which our data currently contradicts

C3 says multigrid convergence should be **bounded away from 1 independently of
$h$**. §7K measured p-multigrid on this operator:

| $N$ | Jacobi its | PMG its | ratio | wall |
|---|---|---|---|---|
| 8 | 755 | 102 | 7.4× | 0.28× |
| 12 | 1183 | 159 | 7.4× | 0.43× |
| 16 | 1580 | 217 | **7.3×** | 0.48× |

**The ratio is pinned at 7.3–7.4× while both counts grow.** That is a
constant-factor preconditioner, *not* a mesh-independent one — textbook multigrid
would hold PMG's column flat. And the 3D $h$-scan showed the same: 1118 → 2297
iterations over a 9× element increase.

So either FOSLS theory does not apply to our functional, or our multigrid is the
wrong kind. **Three candidate explanations, and the plan is built to separate
them:**

| | hypothesis | discriminator |
|---|---|---|
| **H1** | the functional is **not norm-equivalent** (row weights break it) | measure $c_2/c_1$ directly |
| **H2** | **p-MG ≠ AMG** — algebraic coarsening succeeds where polynomial coarsening failed | run AMG on the assembled $A$ |
| **H3** | **strong BC masking** destroys the coercivity FOSLS gets from weak BCs | compare masked vs functional BCs |

---

## 3. Plan

### F0 — Assemble $A$ (½ day, low risk, unlocks everything)

`lssem2d` is matrix-free; AMG is not. Build
$A = M Q^{T} Q\, L_0^{T} (\rho W) L_0 M$ as a `scipy.sparse` matrix by
element-wise assembly, **not** by probing (probing is $O(n^2)$ matvecs and was
how `spectrum.py` got expensive).

**Gate:** $\|A x - \texttt{apply\_LT}(\texttt{apply\_L}(x))\|/\|Ax\| < 10^{-12}$
on random continuous $x$ — the assembled matrix must *be* the matrix-free
operator, not a lookalike. Also assert symmetry to $10^{-14}$ and positive
definiteness of the free block.

*This gate matters more than it looks: §2.1 of 3D_STATUS records four missing
factors in $A$ that survived a full suite of symmetry and convergence tests.*

### F1 — Measure norm equivalence (½ day) — tests **H1**

Compute the generalised eigenvalues of

$$
\mathcal{F}(\mathbf{Q}) = \mathbf{Q}^{T} A\, \mathbf{Q}
\quad\text{against}\quad
\|\mathbf{Q}\|_{1}^2 = \mathbf{Q}^{T} H \,\mathbf{Q},
\qquad H = \text{block-diag}(K + M),
$$

with $K$ the SEM stiffness and $M$ the mass matrix. Then
$c_1 = \lambda_{\min}$, $c_2 = \lambda_{\max}$, and **$c_2/c_1$ is the FOSLS
ellipticity constant** — the quantity the whole theory rests on.

Sweep: `a_flux`/`a_mass` weightings (legacy vs `w_mom=1`), $a_{\text{mass}}$ from
the steady limit to production, and $N \in \{4, 6, 8\}$.

**Gate:** if $c_2/c_1$ is $O(1)$ and $h$-independent, the functional *is*
norm-equivalent and H1 is refuted — then C3 should hold and any failure is the
solver's. If $c_2/c_1$ grows with $h$ or $N$, **H1 is confirmed and no
preconditioner can deliver $h$-independence** until the weights are fixed.

*This single measurement decides whether the rest of the plan is about the
preconditioner or about the formulation.* Run it first.

### F2 — AMG on a **low-order refined** operator (1 day) — tests **H2**

**Not on the SEM matrix.** The high-order element block is *dense* — $324\times324$
at $N=8$ — and AMG's whole premise is exploiting sparsity and strong connections
in the matrix graph. In a dense block every DOF is strongly connected to every
other, so there is nothing to coarsen and the setup cost is quadratic in the
block size:

| $N$ | SEM block | LOR nnz/elem | sparser |
|---|---|---|---|
| 4 | $100^2 = 10{,}000$ | 900 | 11× |
| **8** | $\mathbf{324^2 = 104{,}976}$ | **2,916** | **36×** |
| 16 | $1156^2 = 1{,}336{,}336$ | 10,404 | **128×** |

The standard remedy is **low-order refined (LOR) preconditioning**, going back to
Orszag (1980) and used by Nek5000, MFEM and libParanumal: build the $Q_1$ FEM
operator on the *same GLL nodes*, treating each GLL cell as a bilinear quad. That
matrix $A_{\text{LOR}}$ is sparse (9-point stencil), shares the DOF set with
$A_{\text{SEM}}$, and — for an $H^1$-elliptic operator — is **spectrally
equivalent with a constant independent of $N$** ($\approx \pi^2/4$ for the
Laplacian). AMG then coarsens $A_{\text{LOR}}$, and the result preconditions
$A_{\text{SEM}}$.

**So there are three matrices, and only the middle one is ever handed to AMG:**

| | what | assembled? | used for |
|---|---|---|---|
| $A_{\text{SEM}}$ | the actual operator | F0, small cases only | verification, spectra (F1) |
| $A_{\text{LOR}}$ | $Q_1$ on GLL nodes | **yes, sparse** | **AMG input** |
| coarse levels | AMG's own hierarchy | by pyamg | the V-cycle |

Production stays **matrix-free** for $A_{\text{SEM}}$; only the preconditioner is
assembled.

**F1 is a precondition, not merely an input.** The LOR–SEM equivalence is an
$H^1$ result. It extends to a FOSLS system *if and only if* the functional is
$H^1$-norm-equivalent — which is exactly what F1 measures. **If F1 fails, F2's
whole approach loses its basis**, not just its expected performance. The chain is
C1 $\Rightarrow$ LOR-AMG viable $\Rightarrow$ C3.

Then, in order:

1. **Smoothed aggregation, block size 4** (`pyamg.smoothed_aggregation_solver`),
   since $A_{\text{LOR}}$ is a coupled 4-field system and scalar AMG has no notion
   of that.
2. **Supply the near-null space $B$ explicitly** rather than letting pyamg guess
   constants. §7J found the softest 3D modes were $(\omega_x,\omega_y)$-dominated;
   extract the 2D analogue with `eigsh` and hand it over.
3. **Expect anisotropy trouble.** GLL nodes cluster at element edges with spacing
   $O(1/N^2)$, so the LOR mesh is severely stretched there — a known difficulty
   for AMG coarsening, and the most likely reason a first attempt underperforms.

**Gate — stated in advance:** iterations **flat within 20%** across a 4×
$h$-refinement at fixed $N$. A constant-factor reduction is the result §7K already
rejected and must be recorded as such, not reported as success.

### F3 — Weak boundary conditions (1 day) — tests **H3**

Currently `bc.apply_mask` imposes BCs **strongly** by zeroing rows. FOSLS puts
them in the functional:

$$
\mathcal{F} = \|\mathcal{L}_0\mathbf{Q}-\mathbf{F}\|_{0,\Omega}^2
            \;+\; \beta\,\|\mathcal{B}\mathbf{Q}-\mathbf{g}\|_{0,\partial\Omega}^2 .
$$

Add a boundary term with weight $\beta$ and compare against masking on the
**cavity** (essential BCs, singular corners) and the **BFS** (outflow — where
`obc.py` already exists and OUTFLOW_BC_STUDY.md records a real bug).

**Gate:** unchanged accuracy on Kovasznay and Poiseuille — both have exact
solutions — plus the ellipticity constant from F1 *improving*, which is the
mechanism the theory claims. Weak BCs that change the answer are a bug, not a
trade-off.

*Genuine upside here beyond conditioning:* weak outflow BCs are exactly what
`OUTFLOW_BC_STUDY.md` wants, and the FOSLS functional gives a principled way to
impose them without over-specifying characteristics.

### F4 — Decide (½ day)

| F1 result | F2 result | conclusion |
|---|---|---|
| $c_2/c_1$ $O(1)$ | AMG flat | **FOSLS works; port to 3D** — and re-examine `ROW7_WEIGHT` |
| $c_2/c_1$ $O(1)$ | AMG not flat | our AMG setup is wrong (near-null space?), not the theory |
| $c_2/c_1$ grows | — | **the weights break norm equivalence** — fix the functional before any preconditioner |

**Total ≈ 3 days.** F1 alone (1 day including F0) may settle it, and should be run
before committing to F2–F3.

---

## 4. What could sink it, and one thing that would be worth knowing anyway

**The row weights are load-bearing and were measured, not guessed.** §7F, §7J and
WEIGHT_VS_TIMESTEP_STUDY.md establish that the legacy weighting is required for
accuracy — Poiseuille is 1875× worse under the alternative. If F1 shows those
weights break norm equivalence, we have a genuine conflict between *accuracy* and
*ellipticity*, not a bug to fix. That would be the most interesting outcome and
the plan should not be written to avoid it.

**Convection makes the operator non-symmetric in the underlying PDE**, though the
normal equations remain SPD. FOSLS theory for convection-dominated problems is
weaker, and our production cases run at $Re$ up to 1000. Test at low $Re$ first,
then sweep.

**AMG setup cost may exceed the win.** §7K's PMG was rejected on wall time despite
7.4× fewer iterations. AMG's setup is *more* expensive than PMG's. The gate in F2
is deliberately about $h$-independence rather than wall time, because a
constant-factor win is already known not to pay — but if F2 passes, wall time
becomes the next question, not a settled one.

**Regardless of the outcome, F0 and F1 are worth having.** An assembled $A$ enables
direct solves for small cases, exact spectra, and a check on every future
preconditioner claim; the ellipticity constant is the invariant that four wrong
preconditioner conclusions in §7I–§7K were reached without.
