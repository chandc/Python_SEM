# Progress Report: Divergence Control and Operator Consistency

Date: 2026-08-27.  Companions: `TGV_VALIDATION.md`, `SCHEME_COMPARISON.md`,
`KIM_MOIN_REVIEW.md`.  Code: worktree `sem_fs_wt`, branch `fractional-step`.

---

## 1. Where we stand

| Milestone | Result | Status |
|---|---|---|
| TGV Re=800, 160³ RK3-CN | peak $\varepsilon$ within 0.12% of Gourianov DNS; mean error 0.22%; energy balance $\in[0.9988, 1.0076]$ for the whole run | **validated DNS** |
| Resolution criterion | $k_{max}\eta \ge 1.5$ priced the run a priori; prediction confirmed | validated |
| Minimal-channel trip | shaped low-$k_x$ solenoidal noise trips transition; white noise decays $\times 4800$ by $t=0.38$ | recipe established |
| Channel statistics (K-path) | $\kappa = 0.407$ (KMM: 0.40), $v',w'$ within 2%, burst periods $t^+\!\approx\!139/450$ match literature | ~2 h from done |
| Channel statistics (E-path) | same window on the consistent operator, Spark GPU | ~2.5 days |
| Consistent operator $E$ | built, gated (symmetry exact, PSD, weak div $10^{-6}$), E-multigrid at 60–130 it | production-capable |
| CPU stack | GEMM + mode freezing + mode pool: 4.6 → 1.27 s/step (3.6×) | in production |

## 2. The problem, stated precisely

The incompressible system requires every velocity update to end on the
divergence-free manifold.  A projection method computes an intermediate
velocity $\hat u$ and corrects it with a pressure-like scalar $\phi$:

$$u^{n+1} = \hat u - \Delta t\,\nabla\phi, \qquad \nabla^2\phi = \frac{\nabla\cdot\hat u}{\Delta t}.$$

In the continuum this is exact: $\nabla\cdot u^{n+1} = 0$ identically.
Discretely it is exact **only if the three operators are compatible**:
with $D$ the discrete divergence, $G$ the discrete gradient, and $L$ the
discrete Laplacian used in the Poisson solve, the corrected divergence is

$$D u^{n+1} = D\hat u - \Delta t\, D G \phi = D\hat u - \Delta t\, L\phi + \Delta t\,(L - DG)\,\phi .$$

The solve cancels the first two terms; **the residue is $(L - DG)\phi$** — zero
only when $L = DG$ exactly.

- **Staggered finite differences (Kim & Moin 1985)**: $L = DG$ by construction
  of the mesh.  This is why the reference scheme has its long clean history.
- **Our collocated C0 SEM**: the natural Poisson operator is the weak
  (Galerkin) Laplacian $K$, while $D, G$ are the strong elementwise operators,
  and $K \ne DG$.  The residue lives at the wavenumbers where they disagree —
  the high-$k$ interface modes.

Everything measured this week traces to that residue: the constant
"divergence floor," the Kim–Moin stage-pressure instability, the 22%
divergence growth in the tripping channel.

## 3. Two divergences — measure both, conflate neither

**Weak (enforced) divergence** — what the scheme's own constraint tests:

$$ (D_w u)_q = \sum_e \int_e \nabla q \cdot u \, dV \quad\text{(assembled)} .$$

**Strong (pointwise) divergence** — elementwise $\partial_x u + \partial_y v + i k_z w$ at nodes.

They diagnose different things:

| | weak div | strong div |
|---|---|---|
| meaning | is the solver doing its job | can the grid represent a solenoidal field here |
| K-path, turbulent channel | uncontrolled (grew 5e-4 → 0.22) | ~0.15 plateau |
| E-path, same flow | **1.2e-6, flat** | ~0.14 (unchanged) |
| fixed by | operator consistency | **resolution only** |

## 4. The paths to consistency, with trade-offs

### 4a. K-path (classical weak Poisson) — cheap, residue uncontrolled

Solve $K\phi = -M\,D\hat u/\Delta t$ (weak form), correct with strong $G$.
Per the identity above, each step leaves $O((K^{-1}\!-\!(DG)^{-1})$-mismatch)
divergence.  Mitigations that make it usable in practice:

- **Frequent projection**: RKW3 substages project 3× per step at amplitude
  $\beta_k\Delta t \approx \Delta t/5$; each injection is small and partially
  cleaned by the next projection.  Measured: divergence floor stays $O(10^{-3\ldots-1})$,
  bounded but flow-dependent.
- **Skew-symmetric convection**, $H = \tfrac12[\nabla\cdot(uu) + u\cdot\nabla u]$:
  conserves energy for **any** $u$, divergence-free or not — so the energy
  consequence of the residue is neutralised even though the residue remains.
  (The advective form blew up at $t=9.3$; skew survived identically-configured runs.)

Cost: pressure CG ≈ 12–15 iterations with p-multigrid.  **Use when**: periodic
or mildly-walled flows, statistics runs where §5's A/B shows no bias.

### 4b. E-path (consistent P$_N$–P$_N$ projection) — weak div exact

Build the Poisson operator as the exact composition the update uses:

$$E \;=\; G^{T} M^{-1} G, \qquad E\phi = \frac{1}{\Delta t} G^{T}\hat u_{\mathcal V}, \qquad u^{n+1} = \hat u - \Delta t\, M^{-1} G\,\phi,$$

with $G$ the assembled weak gradient (`gs(wq·strong grad)`, then the velocity
mask), $M$ the assembled diagonal mass, and $\hat u_{\mathcal V}$ the
velocity-space projection of $\hat u$ (essential: with wall values present the
raw $G^T\hat u$ has a component outside $\mathrm{range}(E)$ and CG diverges).
Then $G^T u^{n+1} = 0$ **identically**: the residue term vanishes because the
same $G$ appears in operator, RHS, and update.  $E$ is symmetric PSD by
construction; its null space is the pure constant at $k_z = 0$ (do **not** pin
a dof — pinning rotates the null vector; purge the constant per iteration).

Measured: weak div $1.2\times10^{-6}$ flat through developed turbulence;
zero spurious-energy feedback.  Cost: the price is conditioning — CG needs
60–130 iterations even with the dedicated E-multigrid (K-multigrid: 465;
Jacobi: 878), ≈ 2–5× a K-path step.  **Use when**: correctness is at stake —
wall pressure conditions, energy-sensitive long runs, and as the referee for
K-path results.

### 4c. Compatible discretisations — exact by construction, a rebuild

- **Staggered FD (Kim–Moin)**: $L = DG$ on the mesh; pointwise divergence
  zero to solver tolerance.  A different code, not an upgrade to this one.
- **P$_N$–P$_{N-2}$ SEM**: pressure two orders lower on interior Gauss points;
  the inf-sup-stable pair removes the spurious modes and the Uzawa operator
  is consistent.  The classical "right answer" for SEM (Maday–Patera);
  substantial rebuild of masks, quadrature, and transfer operators.

**Use when**: a next-generation solver is on the table, not as an increment.

### 4d. Resolution — the only lever for the strong divergence

The E-path proves the point: with the weak divergence at $10^{-6}$, the
strong divergence still reads 0.14, because it measures unrepresentable
scales, not solver error.  It obeys the same criterion as every other
truncation quantity:

$$k_{max}\,\eta \gtrsim 1.5 \;\Rightarrow\; \text{strong div} \to O(\text{tol});$$

TGV at $160^3$ (criterion satisfied): balance 1.000 throughout.  Minimal
channel (marginal by design): 0.14 during turbulence.  **Use when**: the
pointwise number itself matters — budgets, dissipation-range spectra.

### 4e. What does NOT work (measured, for the record)

- **Post-hoc pressure terms** (incremental in the CN solve): energy already
  skimmed during the convective sweep; velocity changed 1.3e-5, balance unchanged.
- **Rotational update** $p \mathrel{+}= \phi - \nu\nabla\cdot\hat u$ on the
  K-path with one projection/step: injects the floor into $p$ each step —
  anti-dissipative, blow-up at step 184.
- **Naive strong-composite** $D\!\cdot\!G$ without $M^{-1}$ and adjoint pairing:
  asymmetric (2.8e-2), NaN by step 2.
- **Tightening CG tolerance** against an operator-mismatch floor: no effect
  (measured 1e-6 → 1e-7: floor unchanged).

## 5. Decision map

| situation | path |
|---|---|
| periodic box, resolved ($k_{max}\eta \ge 1.5$) | K-path + skew; balance meter as guard |
| walls, statistics, cost-sensitive | K-path + skew, **pending** the E/K A/B verdict (~2.5 days) |
| walls, energy-critical or pressure-BC-sensitive | E-path (E-multigrid, unpinned mask, velocity-space RHS) |
| pointwise divergence must be small | refine to $k_{max}\eta \ge 1.5$ (only §4d moves this number) |
| clean-slate solver | P$_N$–P$_{N-2}$ or staggered — compatibility by construction |

The live experiment: K-path and E-path statistics twins on the same tripped
channel.  If their profiles and Reynolds stresses agree, the K-path's 2–5×
cost advantage is safe for production statistics; if they differ, §4b is the
price of correctness and this document's map changes accordingly.

---

## 6. Closing findings (2026-08-28)

**Kim-Moin RK4-CN, final disposition.**  The E-consistent variant (stage force
$M^{-1}Gp$ with $Ep = G^T N$, E-projection, dual real-mode kernel purge)
eliminates the energy defect completely — balance 1.0003 at full dt against
1.24 for the faithful original — but still fails at t = 5.12 in TGV at 88^3,
the same sheet-sharpening window as every stage-pressure variant.  Conclusion:
the scheme's robustness through sharp transients is a property of its native
staggered grid's POINTWISE divergence-free fields, which no weak-sense
projection supplies.  RK3-CN survives the identical grid/dt by projecting 3x
per step at 1/5 amplitude.  Continuations if revived: x-y over-integration
dealiasing, or resolved grids (k_max eta >= 1.5) where the offending content
is absent.  Solver infrastructure built along the way (E-multigrid, kernel
purges, dead-lane guards, mode pool) is validated and retained.

**Channel operator A/B (interim, E-run 2/3 complete).**  u_tau means agree to
0.3% (1.0125 K vs 1.0091 E, matched windows), trajectories decorrelated as
they must, streak structure visually indistinguishable at identical rendering.
No evidence so far that the K-path divergence blemish biases low-order
statistics.  Final verdict with the completed E window.

**CPU performance stack** (Mac M3 Max, measured): tensordot/GEMM derivatives
1.63x; mode-adaptive freezing 2.04x cumulative; mode-space multiprocessing
(4 workers, pipes, shared memory) 3.62x cumulative — channel statistics step
4.6 -> 1.27 s.  Five concurrency/singularity defects found and guarded, each
documented in code: spawn re-import (missing __main__ guard), event-handshake
race (-> pipes), worker-local dead thresholds (-> global scale), roundoff-lane
amplification (-> calibrated dead-lane guard + zero-on-divergence), and E's
singularity at BOTH real Fourier modes on periodic domains (kernel dim 4,
purged exactly).

## 7. The operator A/B verdict (2026-08-29)

Matched windows t = 3..15.95, 7401 samples each (E-run stopped at an
E-solve stagnation plateau; statistics banked continuously):
u_tau -0.5%, U+c +1.4%, u' +0.03%, v' -2.7%, w' -1.1%, -u'v' -1.1% --
every difference within the ~1-3% finite-window sampling error.
**The divergence treatment does not measurably affect low-order channel
statistics at this resolution.**  Decision map confirmed: K-path for
production statistics (2-5x cheaper); E-path where correctness of the
enforced divergence / pressure energetics is itself the requirement.
Open item: E-multigrid stagnation on some turbulent states (grinds near
the CG cap without diverging) -- next preconditioner target.

## 8. K vs E, closed form (2026-08-29)

This section collects the algebra behind the A/B verdict of §7 in one
place: what each operator provably controls, why E is exact where K is
not, and what E's exactness costs.

### 8a. Notation

Let $\psi_i$ be the C0 SEM velocity basis, $q_j$ the (identical,
P$_N$–P$_N$) pressure basis, $M$ the lumped (diagonal) GLL mass matrix.
Define the **weak gradient** and **weak divergence**

$$ (G\phi)_i = \int_\Omega \psi_i\, \nabla\phi \; dV, \qquad
   (D u)_j \;=\; -\int_\Omega \nabla q_j \cdot u \; dV
   \;\;\Rightarrow\;\; D = -G^{T}, $$

and the two candidate Poisson operators

$$ K_{jk} = \int_\Omega \nabla q_j \cdot \nabla q_k \, dV
   \qquad\text{(weak Laplacian)}, \qquad
   E = G^{T} M^{-1} G \qquad\text{(consistent operator)}. $$

On a staggered grid $K = G^T M^{-1} G$ identically and there is nothing
to choose.  On collocated C0 SEM the quadrature and continuity
constraints make $K \ne G^T M^{-1} G$; the difference is supported on
the element-interface (high-$k$) modes.

### 8b. E is exact by construction

The E-path projection uses the *same* $G$ and $M$ in the solve and in
the velocity update:

$$ E\,\phi = \frac{1}{\Delta t}\, G^{T}\hat u, \qquad
   u^{n+1} = \hat u - \Delta t\, M^{-1} G \phi . $$

Apply $G^T$ to the update:

$$ G^{T} u^{n+1}
   = G^{T}\hat u - \Delta t\, \underbrace{G^{T} M^{-1} G}_{E}\,\phi
   = G^{T}\hat u - G^{T}\hat u = 0 . $$

The residue of §2, $(L - DG)\phi$, vanishes **identically** — not
because the operators happen to agree, but because the update is the
$M$-orthogonal projection of $\hat u$ onto $\ker(G^T)$: E is the Schur
complement of that projection.  Equivalently, $u^{n+1}$ is the closest
field to $\hat u$ in the $M$-norm satisfying every weak-divergence
constraint.  Measured: weak divergence 1.2e-6 (solver tolerance), flat
for the whole run.

With K, the update residue is $(K - E)\phi \ne 0$: the projection is
approximate, and the weak divergence floats at the level set by how much
$\phi$'s interface modes excite $K - E$ (~1e-3 in the channel).

### 8c. What E cannot fix: the strong divergence

Exactness above is exactness of $G^T u = 0$ — a set of
$n_p$ integral constraints, one per pressure dof.  Pointwise
$\nabla\!\cdot\!u = 0$ at every GLL node is a *different and larger* set
of constraints, and the collocated C0 velocity space simply does not
contain a subspace satisfying them while representing the flow: the
polynomial interpolant of a solenoidal field is not itself solenoidal,
with the mismatch concentrated at element interfaces and sharp features.
Hence the measured split:

| | weak $\|G^T u\|$ | strong $\|\nabla\!\cdot\!u\|_\infty$ |
|---|---|---|
| K-path channel | ~1e-3 | ~0.15 |
| E-path channel | **1.2e-6** | ~0.14 |
| BFS corner cell | — | O(1) (singularity) |

The strong divergence is an interpolation/resolution error: it decays
spectrally with $N$ (TGV 88³ → 160³) and is *invariant* to the choice of
K vs E.  Only compatible (staggered / P$_N$–P$_{N-2}$) discretisations
control it by construction — which is also the honest summary of the
Kim–Moin negative result (§ KIM_MOIN_REVIEW): its robustness through
sharp transients is a property of the staggered mesh, not of the RK4-CN
time discretisation.

### 8d. The price of E: kernel and conditioning

**Kernel.** $M$ is SPD, so $\ker E = \ker G$: fields with
$\int \psi_i \nabla\phi = 0$ for all $i$.  On a periodic mesh this
kernel is 4-dimensional — the constant plus the three checkerboard
modes — and it appears at *both* real Fourier lanes ($k_z = 0$ and
Nyquist).  The checkerboard is exactly the classical collocated-grid
pressure mode: K, being an assembled elliptic operator, gives it a
positive eigenvalue and suppresses it silently; E is honest about its
existence and demands an explicit purge (computed null basis, projected
out of RHS and iterates every solve).  Walls reduce the kernel to the
constant alone; an outflow-Dirichlet boundary removes it entirely
(measured on BFS: high-mode pressure fraction 2e-5, no purge needed).

**Conditioning and cost.**  Each E application is two derivative sweeps
plus a mass inversion (~2-3x a K application), and $\kappa(E) \gg
\kappa(K)$ because $E$'s spectrum inherits the near-null checkerboard
family shifted only slightly off zero by mesh non-uniformity.  Measured
on a stiff turbulent-channel state: Jacobi-CG 878 iterations for E; the
deg-2 coarse E-multigrid stalls; a deg-6 coarse space restores
convergence in 60 iterations.  This is the open engineering item — a
sparse-assembled, factorise-once coarse level to turn the 15x iteration
reduction into wall-time.

### 8e. Decision rule (final form)

$$ \textbf{K: } \; \min_\phi \;\text{cost}, \;\; G^T u \approx 0
   \qquad\qquad
   \textbf{E: } \; u^{n+1} = \Pi_{M}\,\hat u, \;\; G^T u = 0 . $$

- **Production statistics** (means, Reynolds stresses, spectra): K.
  The A/B of §7 shows every low-order statistic agrees within sampling
  noise; the weak-divergence residue K leaves behind does not couple to
  the momentum budget at measurable level.
- **Cumulative-conservation problems** — passive scalar transport,
  long-horizon mass budgets, pressure-work energetics — where a 1e-3
  divergence residue integrates into secular drift: E (or a compatible
  discretisation if pointwise conservation is required).
- **Verification**: E remains the instrument that turns "the divergence
  is fine" from an assumption into a measurement.
