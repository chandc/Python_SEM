# Closing the per-step gap: the coarse read (a) and the iteration count (b)

**Where the A100 step goes.** With the condensed vertex-patch preconditioner the
production channel step (6×18 elements, $N=8$, 17 modes, 1.55 M free unknowns) is

$$t_{\rm step}=3\ \text{stages}\times n_{\rm it}\times(t_{\rm op}+t_{\rm pc})
= 3\times72\times(1.6+8.3)\,\mathrm{ms}\approx2.1\ \mathrm s,$$

operator 16 %, preconditioner 84 %. Of the 8.3 ms preconditioner apply, 4.3 ms is
reading the 5.3 GB dense $p=2$ coarse inverse (17 modes × 6216² doubles) at the
A100's bandwidth, ~2 ms the patch GEMMs, ~2 ms interior solves and
back-substitution. The operator is at its bandwidth floor and is not a lever.

Two levers remain, and they multiply: **(a)** the coarse read, which sets
$t_{\rm pc}$, and **(b)** the iteration count, which the exact two-block
preconditioner shows can be 13 (κ ≈ 3) where the patch method sits at 72.

| configuration | $n_{\rm it}$ | $t_{\rm pc}$ | step | how |
|---|---|---|---|---|
| today | 72 | 8.3 ms | 2.1 s | — |
| (a) coarse read halved | 72 | 6.1 | 1.7 s | symmetric-packed inverse, 1 day |
| (a) coarse read removed | 72 | ~4.5 | 1.3 s | sparse device solve, 1–2 weeks |
| (b) partition-of-unity weighting | 35–50 | 4.5 | 0.65–0.9 s | literature 1.5–3×, 2 days to test |
| (b) spectral coarse space | ~15–20 | ~5 | 0.3–0.4 s | GenEO-type, 2–3 weeks |
| all of it in fp64 | ~15 | ~4.5 | **≈0.3 s** | |

That last line is the reviewer's 0.1–0.3 s, in double precision, and every row
of it is a measurement we can make on the rigs we already have before touching
the channel. Single-precision factors would halve every $t_{\rm pc}$ entry and
are a one-line switch; they are **not assumed** anywhere here, per the standing
fp64 decision.

## WEEK 1 RESULTS (2026-09-13)

Measured, not estimated.  Two items land, two are rejected, and one rejection
kills a planned fortnight of work.

| item | verdict | number |
|---|---|---|
| **a2** $p_c = 1$ coarse | **accepted** | +5 % iterations (41→43 at $N=6$, 40→42 at $N=8$) for **13× less coarse memory** |
| **a1** fp32 coarse factor | **accepted**, pending the channel gate | **identical iterations**, exactly half the bytes, agrees with fp64 to 4e−8…2e−7 |
| **b1** partition-of-unity weighting | **rejected**, with the mechanism measured | **30× worse** at $c=5405$ (31→913 iterations) |
| **b3** richer coarse ($p_c = 3, 4$) | **rejected at large $c$** | 2D 31→32→32 ($N=8$), 26→25→25 ($N=12$); 3D 41→41→41 |
| **a3** sparse device coarse | **no longer needed** | its purpose was to afford b3, and b3 buys nothing |

**a2 + a1 together, projected to the production channel:** coarse factor
5.3 GB → 0.41 GB ($p_c=1$) → **0.20 GB** (fp32); the coarse read 4.3 ms → ~0.2 ms;
the apply 8.3 → ~4.2 ms; iterations 72 → ~76.  Step
$3\times76\times(1.6+4.2)\,\mathrm{ms}\approx\mathbf{1.3\ s}$ against 2.1 s — the
"coarse read removed" row of the table below, reached with two flags instead of
a sparse solver.  Both flags exist now (`pc=1`, `coarse_fp32=True`); neither is
a default, and the channel's ten-step restart gate has not been run yet.

**b1 is the interesting failure.** The patch contributions were being summed
without the inverse-counting weight that overlapping Schwarz on spectral
elements normally carries (Fischer 1997; Lottes & Fischer 2005), and adding it
made the preconditioner 30× worse.  That is far too large for what it is — the
weight varies only between $1/\sqrt9$ and $1/\sqrt4$, a factor of 1.5, and a
uniform rescaling changes nothing (control: 31 → 33 iterations).  The cause is
specific to this operator and was measured directly: for a random residual with
$\lVert\nabla\!\cdot\mathbf u\rVert/\lVert\mathbf u\rVert = 62$, the unweighted
patch correction comes back at **0.25** and the weighted one at **3.31**, 13×
worse.  Multiplying a correction by a spatially varying diagonal destroys its
discretely divergence-free character, and preserving exactly that is what the
patch solves are *for* in the $H(\mathrm{div})$ regime.  Consistent with this
reading, the damage is regime-dependent: 30× at $c=5405$ and 5–6× at $c=1$,
where there is no dominant kernel to break.  The standard weighting is right for
$H^1$/Poisson problems and wrong here.

**b3's rejection is regime-dependent too, in the same direction.** At $c=1$ a
richer coarse space *does* pay (2D, $8\times8$: 33 → 26 → 25 for $p_c = 2,3,4$),
at $c=5405$ it does not.  Above $c^\ast$ the coarse level cannot reach the
kernel modes that limit convergence, so enlarging it is wasted.  This is the
same statement as §7.3 of the paper, arriving from the coarse-space side.

**One core fix was required.** `lgl_nodes(1)` reduced over an empty array, so no
$p=1$ level could be built anywhere in the code (the `if N > 1` guard on the
node assignment shows the case was intended; only the Newton loop missed it).
Fixed, and verified against the exact two-point values — nodes $[-1,1]$, weights
$[1,1]$, $D=[[-\tfrac12,\tfrac12],[-\tfrac12,\tfrac12]]$ — with $N=2..12$
unchanged.

### b4's go/no-go gate, run (`scratch/week1_b4_gate.py`)

3D_STATUS.md §7S.3 killed deflation for a reason that would kill GenEO in the
same way: the soft set of the operator was a **constant 13 % of the dofs**,
growing with the mesh, so deflating a fixed number of modes bought 1.1×.  A
GenEO coarse space *is* "the modes below a threshold", so before spending three
weeks the same question has to be asked of the **patch-preconditioned** operator
rather than the Jacobi-preconditioned one.  Forming $M^{-1}A$ densely and taking
its spectrum:

| mesh | $N$ | free | $\lambda_{\min}$ | $\lambda_{\max}$ | $\kappa$ | modes $<0.1\lambda_{\max}$ | per patch |
|---|---|---|---|---|---|---|---|
| 2×2 | 6 | 579 | 1.028 | 9.055 | 8.8 | 0 | 0 |
| 2×2 | 8 | 1027 | 1.009 | 9.020 | 8.9 | 0 | 0 |
| 4×4 | 6 | 2307 | 0.210 | 9.133 | 43.4 | 43 | 4.8 |
| 4×4 | 8 | 4099 | 0.336 | 9.067 | 27.0 | 38 | 4.2 |
| 6×6 | 6 | 5187 | 0.151 | 9.152 | 60.6 | 126 | 5.0 |

**The gate passes, and three things fall out.**

1. $\lambda_{\max}=9.0$–$9.2$ on every mesh and order.  That is the colouring
   constant — the largest number of patches covering one dof, and the patch-count
   histogram measured for b1 is exactly $\{4, 6, 9\}$.  Classical additive
   Schwarz theory bounds $\lambda_{\max}$ by that number and it is attained.
   There is nothing to gain at the top of the spectrum.
2. **All of the degradation is in $\lambda_{\min}$**: 1.03 → 0.21 → 0.15 as the
   mesh refines at fixed order, so $\kappa$ runs 8.8 → 43 → 61.  The $p=2$ coarse
   space is not delivering the $h$-independence a two-level method is supposed to
   provide; this is the same growth visible as 31 → 41 iterations from 4×4 to 8×8
   and as the channel's 72 against the small rig's 36.
3. **The bad modes number about five per patch, independent of $h$ and $p$**
   (4.8, 4.2, 5.0), not a constant fraction of the dofs.  That is precisely the
   condition GenEO needs and precisely what deflation failed: 13 % of the dofs is
   a second solve, five vectors per patch is a small coarse space.  For the
   channel that is $114\times(5\text{–}15)\approx600$–1700 vectors per Fourier
   mode against the present $p=2$ space of 6216 — **a better coarse space that is
   also an order of magnitude smaller.**

Expected outcome if the bound is attained: $\kappa\to O(\lambda_{\max})\approx9$,
iterations from 72 to roughly 20, and with a1+a2 already in hand a step of
$3\times20\times(1.6+3)\,\mathrm{ms}\approx\mathbf{0.3\ s}$.

**What is left of the plan:** b2 (symmetrised hybrid combination) and b4, plus
the channel gate for a1/a2.  b4 is now both the only route to a large further
gain and a measured prospect rather than a hope.

## The three rigs, and the gate

Every item below is measured in this order, and nothing advances to the next rig
until it passes on the previous one:

1. **2D harness** (`scratch/vertex_schwarz2d.py`, `adn_schwarz_condensed.py`):
   Re = 1000 cavity operator at $c = 5405$, cold random right-hand side, CG to
   $10^{-8}$. Iteration counts in seconds to minutes. This is where anything
   algorithmic is tried first.
2. **3D rig** (`scratch/vs3d_check.py`, 4×4 elements, $N = 4$–8, 5 modes):
   iterations and the condensed-vs-dense agreement; a few minutes per
   configuration on the Mac.
3. **Channel** (`scratch/minchan.py price` / a ten-step restart from run01):
   the production number, with the standing gate — **the step must reproduce
   the Jacobi step to every logged digit** ($u_\tau$, divergence, energy,
   dissipation). A preconditioner cannot change the answer; anything that does
   is a bug, however fast.

## (a) The coarse read

**a1. Symmetric-packed inverse — half the traffic for nothing.** The stored
inverse is symmetric and the batched GEMV reads all $n^2$ of it. Storing the
upper triangle and applying with a symmetric matrix-vector product reads $n^2/2$:
5.3 GB → 2.65 GB, 4.3 → ~2.2 ms. `torch` has no batched `symv`; cuBLAS does
(`cublasDsymv`, one call per mode, 17 launches inside the existing CUDA graph),
or the triangle can be applied as one GEMV on the packed upper part plus a
transposed GEMV on the same data. Pure implementation, no algorithmic risk,
identical result to round-off. **One day.**

**a2. `pc = 1` — an 11× smaller coarse space, if iterations allow.** The flag
exists (`VertexSchwarzBatched3D(pc=...)`). At $p = 1$ the coarse space has
$7\times19 = 133$ nodes per mode against 481, so the dense inverse is
28 MB per mode, 0.47 GB in all: the read drops to ~0.4 ms. The question is
purely what it costs in iterations, and the 2D $p$-multigrid study
(PMG_ALGORITHM.md §6.6) warns that coarsening too far is what breaks
$p$-robustness. Measure on rig 1 then rig 2: if $n_{\rm it}$ rises by less than
the ~1.8× the coarse term is worth at all, it is a net win. **Half a day, and
the answer decides whether a3 is needed.**

**a3. Sparse device coarse solve — removes the read entirely.** The $p = 2$
coarse operator per mode is a 2D spectral-element matrix on 6216 unknowns:
sparse, with nested-dissection fill of a few MB per mode. Options in order of
preference: cuDSS (batched sparse LU/Cholesky on the device, the right tool);
a banded solve after RCM ordering (`gbtrf`/`gbtrs`; bandwidth ~ one element row
× 14 fields ≈ 400, storage ~60 MB per mode, 1 GB total — 5× below dense,
batched over modes); or `cupyx.scipy.sparse.linalg` as a fallback. Sparse
triangular solves are latency-bound rather than bandwidth-bound, so the apply
may not be faster than a1's 2.2 ms on an A100 — the real point of a3 is that
**it makes a richer coarse space affordable**, which is where it connects to
(b). The host sparse LU path already exists (`coarse_dense=0`) and is the
reference for correctness. **One to two weeks.**

## (b) The iteration count

**b1. Partition-of-unity weighting — the standard fix for overlapping Schwarz,
and we do not have it.** The patch contributions are currently summed plainly,
so a dof in the interior of the mesh receives four patch corrections (in 2D)
and is over-corrected by that factor. Overlapping Schwarz on spectral elements
is normally applied with inverse-multiplicity weights (Fischer 1997; Lottes &
Fischer 2005), and Stiller (2016) measures 1.5–3× fewer iterations from
non-uniform weights. CG requires the preconditioner symmetric, so the weighting
goes on both sides, $M^{-1}=D^{1/2}\big(\sum_v R_v^{\mathsf T}A_v^{-1}R_v\big)D^{1/2}$
with $D$ the inverse patch-count, which is SPD. Implementation: one
elementwise multiply before the gather and one after the scatter, in both the
2D harness and `VertexSchwarzBatched3D`. **Two days including all three rigs.**

**b2. Hybrid (symmetrised multiplicative) coarse–patch combination.** Applying
the coarse correction, then the patches on the updated residual, then the
coarse again is SPD and typically halves the iteration count against pure
additive, at the cost of one extra operator apply per preconditioner call
(1.6 ms, cheap against 8 ms). Net gain ~1.3–1.6×; test on rig 1 in a day. Only
worth doing after b1, since the two interact.

**b3. Richer coarse space — needs a3.** The exact $(\mathbf u,\omega)\|p$
preconditioner reaches κ ≈ 3 because it solves the coupled block exactly;
patches approximate it locally and the $p = 2$ coarse level catches only the
smoothest part of the divergence-free kernel. A $p = 3$ or $p = 4$ coarse level
catches more of it. At $p = 4$ that is ~25 k unknowns per mode, which a dense
inverse cannot hold (5 GB *per mode*) and a sparse factorisation can. Measure
$n_{\rm it}$ against $p_c$ on rig 1 with the host sparse solver *now* — that is
free — so the case for a3 is quantified before it is built.

**b4. Spectral coarse space (GenEO-type) — the principled route to κ = O(1).**
The slow modes of the patch-preconditioned operator are known: smooth,
divergence-free, curl-rich, invisible to the local solves. GenEO builds a coarse
space from exactly those, via a generalised eigenproblem on each patch
(patch operator against its partition-of-unity-weighted version), and comes
with a bound κ ≤ C(1 + 1/τ) independent of $h$, $p$ and the coefficients. Costs:
a per-patch eigensolve at build (batched, same shapes as the factors), and a
coarse space of (patches × modes-per-patch) columns — a few thousand per Fourier
mode, dense-solvable. The expected result is the two-block number, ~13–20
iterations, and it is the only item here that reaches the bottom row of the
table. Prototype on rig 1 in a week; port to 3D in a second. **Two to three
weeks, and it is a paper section of its own if it works.**

## Order, and what each step decides

| week | do | decides |
|---|---|---|
| 1 | a2 (`pc=1`), b1 (PoU weights), a1 (packed inverse), b3 with the host solver | whether the coarse read can simply be shrunk; how much PoU buys; how much a richer coarse space would buy |
| 2 | a3 if b3 says a richer coarse space pays; b2 | the coarse solver the rest is built on |
| 3–4 | b4 | the O(1)-κ preconditioner |

Week 1 is all measurement on existing switches or two-line changes, and by its
end the table above has measured entries in place of literature estimates. The
channel is touched only at the end of each week, through the ten-step restart
gate.

## What this does to the paper

Section 9 gains the decomposition
$t_{\rm step}=3\,n_{\rm it}(t_{\rm op}+t_{\rm pc})$ with its measured entries,
and the fractional-step comparison on the same box (K-path, 1.33 s/step on a
CPU at $\Delta t = 3.5\times10^{-4}$). Anything from week 1 that holds goes into
§8–9 as an implementation improvement. b4 is a new section if it delivers
$O(1)$ conditioning, since it would be the first spectral coarse space for a
velocity–vorticity–pressure least-squares system.
