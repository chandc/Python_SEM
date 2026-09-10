# Cheaper vertex-patch solves: literature and a measured first step

The vertex-patch Schwarz preconditioner (ADN_FOSLS.md §8–12) is p-, h- and
c-independent in iteration count but its dense patch factors cost
½[(2N+1)²F]² per patch: 32 GB and ~3e10 flop per CG iteration on the 3D
channel at N = 8, 44 GB and 7.7 s per iteration at N = 30 in 2D
(ADN_FOSLS.md §13). This document records what the literature offers to
bring that down, ranked by how much it removes and whether the result is
exact, and one option measured.

## 1. What is expensive, precisely

Per patch and mode: storage n_p²/2, apply n_p², factor n_p³/3, with
n_p = (2N+1)²F. Three separate multipliers: (2N+1)⁴ in order, F² in the
number of coupled fields, and the patch count × modes. Any remedy attacks
one of the three.

## 2. Options from the literature

| # | approach | attacks | exact? | memory per patch | needs | reference |
|---|---|---|---|---|---|---|
| A | **Static condensation of element interiors** (substructuring): factor each element's interior block once, share it across the 4 patches that contain the element, keep only the dense Schur complement on the patch's edge dofs | (2N+1)⁴ constant | **yes, identical iterations** | Σ_e ((N−1)²F)²/2 + Σ_P (12NF)²/2 ≈ 1/8–1/20 of dense | nothing new; element blocks already probed | Couzy & Deville, JCP 116 (1995) "A fast Schur complement method for the SE discretization of the incompressible NS equations"; Karniadakis & Sherwin, *Spectral/hp Element Methods*, ch. 4; Huismann–Stiller–Fröhlich, "Fast static condensation for the Helmholtz equation in a SE discretization" (2016) |
| B | **Vorticity elimination before patching**: A_ωω is diagonal on GLL, so S_u = A_uu − A_uω A_ωω⁻¹ A_ωu is exact and matrix-free; patch on u (and p separately by PMG) | F² (14 → 6 real fields, 5.4×) | yes (measured: κ unchanged, ADN_FOSLS §5.2) | (5.4×) less than dense on (u,ω,p) | p-block solver (scalar Poisson, PMG) | ADN_FOSLS.md §5.2, §8.3 (patches on S_u: 40 → 33 iterations) |
| C | **Fast-diagonalisation / sparse patch bases** (Brubeck–Farrell): a tensor-product basis that diagonalises the interior blocks of each cell makes the vertex-star patch matrix as sparse as a low-order stencil; sparse Cholesky with the static-condensation pattern gives time and space optimal in p; non-separable operators handled by a separable surrogate or an auxiliary sparse operator with incomplete Cholesky | (2N+1)⁴ → O(p^d) | exact only for separable operators; otherwise a spectrally-equivalent surrogate | O((N+1)²F) per patch (2D) | affine tensor-product cells (we have them); a separable surrogate of the FOSLS element operator, or its Kronecker-sum structure exploited directly | Brubeck & Farrell, SISC 44 (2022) A2991, arXiv:2107.14758; Brubeck & Farrell, SISC 46 (2024) A1549, arXiv:2211.14284 (de Rham complex, Pavarino/AFW/Hiptmair patches, code on Zenodo 7358044) |
| D | **Inexact local solves by nested p-multigrid inside each patch** ("multigrid within multigrid"): no patch factors at all, O(p^{d+1}) per local iteration | all three | no (inexact; must be a fixed linear operator to keep CG symmetric) | none beyond the operator | patch-local operator applies (we have them: element blocks are the operator) | arXiv:2510.17785 (2025), "Local solvers for high-order patch smoothers via p-multigrid" |
| E | **Overlapping Schwarz with FDM local solves on tensor-product surrogates** (Fischer/Lottes; Nek5000 pressure solver): local problems from 1D FE/SE discretizations, solved by fast diagonalisation, O(N³) per patch in 2D; weighting by the inverse counting matrix essential; nonuniform weights (Stiller) cut iterations 1.5–3× | (2N+1)⁴ → N³ | surrogate (Poisson/Helmholtz-type local operator) | O(N²) per patch | a separable local operator — the FOSLS block is not; would need a surrogate whose kernel matches | Fischer, JCP 133 (1997) 84; Lottes & Fischer, J. Sci. Comput. 24 (2005) 45; Stiller, J. Sci. Comput. (2016), arXiv:1512.02390 |
| F | **Low-order-refined (LOR) preconditioning + AMG**: assemble the sparse Q₁ discretization on the GLL sub-grid, precondition the high-order operator with AMG on it; spectrally equivalent for H¹, and for H(curl)/H(div) with histopolation bases; GPU-ready | all three, O(dofs) | **tested (§3.3): equivalent at c = 1, NOT at c = 5405 — the Q₁ and GLL divergence-free kernels differ; ruled out** | O(dofs) sparse | an AMG that handles the H(div) kernel (hypre ADS/AMS) — which again presumes RT/Nédélec structure | Pazner, Kolev & Dohrmann, SISC 45 (2023) A675, arXiv:2203.02465; Pazner, Kolev & Camier, IJHPCA (2023) GPU LOR |
| G | **Compressed factors** (block low-rank / HSS / H-matrix on the dense patch factor) | (2N+1)⁴ → ~n log n | approximate (controlled tolerance) | 3–10× below dense | a library (MUMPS BLR, STRUMPACK) | Amestoy et al., SISC 37 (2015) BLR; Ghysels et al. STRUMPACK |
| H | **Discretisation change** to de Rham-conforming spaces (RT/Nédélec for (u, ω)): then Hiptmair–Xu auxiliary-space and AMS/ADS work with sparse matrices, and C, F become theorems | root cause | — | O(dofs) | a new FOSLS discretisation and re-validation | Hiptmair & Xu, SINUM 45 (2007) 2483; Arnold–Falk–Winther, Numer. Math. 85 (2000) 197 |

Also relevant: Farrell, Knepley, Mitchell & Wechsung, "PCPATCH", ACM TOMS 47
(2021) — the vertex-star ("star") patch as the standard relaxation for
divergence-free kernels in augmented-Lagrangian Stokes/Navier–Stokes
solvers, which is the same object as our vertex patch; their high-order
cost discussion motivated C.

## 3. Option A measured: static condensation is exact and cuts memory 8–12× (`scratch/adn_schwarz_condensed.py`)

2D, c = 5405, 4×4 elements, full operator, both with the p = 2 coarse term
(iterations / κ identical by construction — confirmed):

| N | ndof | dense: it / κ | stored entries | ms per apply | condensed: it / κ | stored entries | ms per apply | memory ratio |
|---|---|---|---|---|---|---|---|---|
| 8 | 4099 | 42 / 27 | 7.67e6 | 17 | **42 / 27** | 1.00e6 | 8 | **7.7×** |
| 12 | 9219 | 33 / 15 | 3.66e7 | 110 | **33 / 15** | 3.47e6 | 30 | **10.6×** |
| 16 | 16387 | 31 / 17 | 1.12e8 | 539 | **31 / 17** | 9.35e6 | 124 | **12.0×** |
| 24 | 36867 | 31 / 25 | 5.52e8 | 3086 | **31 / 25** | 4.24e7 | 1174 | **13.0×** |

The ratio grows with N (the interior blocks scale as (N−1)⁴ per element,
the Schur complements as (12N)², and each element's interior factor is
shared by its four patches) and the apply is 2–4× faster because most of
the work moves into the smaller per-element interior solves. At N = 24 the dense
factors would be 4.4 GB in double against 0.34 GB condensed.

**3D channel at N = 8 with condensation** (114 patches, 108 elements, 17
modes, complex Hermitian): interior blocks 343 complex per element → 0.9 GB;
patch Schur complements on 93 edge nodes × 7 = 651 complex → 3.3 GB; total
**≈ 4 GB against 32 GB**, one value of c, single precision. Flops per CG
iteration ≈ 1.7e10 against 3e10 (the interior–edge coupling products do not
shrink as much as the storage). Build cost falls from Σ n_p³/3 ≈ 5e12 to
≈ 6e11 per value of c.

**2D cavity at N = 30:** interior 3364 dofs per element (16 elements,
45 MB each in double) plus 25 Schur complements of ≈ 1430 dofs (8 MB each):
≈ 0.9 GB against 22 GB triangular / 44 GB as stored — the 4×4 N = 30 case
becomes an ordinary-memory problem, though still (N−1)⁴ in the interiors.

## 4. Ranking

1. **Do A now.** Exact, no theory risk, 8–20× memory, 2–4× faster apply,
   built entirely from the element blocks the code already probes. It is
   the difference between 32 GB and 4 GB on the channel, and it is what the
   SEM community has done since Couzy & Deville.
2. **B is a free 5× on top in the dense form** but does not combine
   multiplicatively with A: after ω elimination the Schur complement S_u
   couples an element's interior u-dofs to its neighbours through shared ω
   nodes, so the interior blocks are no longer element-local. Use A on the
   full (u, ω, p) block; keep B as the alternative if a u-only patch is ever
   preferred (e.g. a p-multigrid handles p separately).
3. **C is the O(p^d) route** and the only one that removes the (N−1)⁴ of the
   interiors. It needs either a separable surrogate of the FOSLS element
   operator (the operator is a sum of Kronecker products of 1D matrices on
   affine cells, so a surrogate is constructible) or Brubeck–Farrell's
   auxiliary sparse operator with incomplete Cholesky. Research-grade port;
   worth it only if N > 12 is a goal for the channel.
4. **D removes factors entirely** and is the natural GPU form (all
   matrix-free); the cost is inexactness and the requirement that the local
   solver be a fixed polynomial so CG stays symmetric. Measurable on the 2D
   harness in a day: replace the patch Cholesky by k V-cycles of the patch
   p-multigrid with Chebyshev smoothing and see what k restores the
   iteration count.
5. **E, F, G** are established for Poisson/H¹ problems; for our C⁰ FOSLS
   operator each rests on an equivalence that is not proven and, for E and
   F, on an H(div)-aware coarse or AMG component that presumes de Rham
   spaces. Try only after C or D.
6. **H** is the principled long-term answer (sparse, O(dofs), theorems), at
   the price of a new discretisation.

## 5. Immediate next measurements

1. N = 30 row of the condensation table.
2. Condensation in the 2D production path (`vertex_schwarz2d.py`): replace
   the dense per-patch factor by shared interior factors + edge Schur
   complements; re-run `vs2d_bench.py` and the Poiseuille cases for wall
   time.
3. Option D on the assembled harness: patch-local p-multigrid as the local
   solver, iterations vs number of inner cycles.
4. Then the 3D port (VERTEX_SCHWARZ_IMPLEMENTATION.md §4) with A built in
   from step 1, which changes its memory line from 31 GB to ≈ 4 GB.

### 3.1 N-sweep in the production path: Jacobi vs condensed vertex patch (`scratch/nsweep_ghia.py`)

Ghia Re = 1000 cavity, 4×4 elements, time marching from rest, BDF, dt = 1e−3,
10 steps through `step_bdf`, CG absolute tol 1e−8. Condensed patches +
p = 2 coarse built once at step 1 on a frozen snapshot, reused for the 10
steps. 8 jobs, 6 concurrent, 2 threads each, numpy, M3 Max. Memory: stored
factor bytes (interior factors + interior–edge couplings + edge Schur
complements) and process peak RSS. Figure `figs_fosls_vs_fs/nsweep_ghia.png`.

| N | dofs | Jacobi it/step | ms/it | wall/step | RSS | condensed it/step | ms/it | CG/step | build (once) | wall/step | stored | RSS | it ratio | wall ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 1,764 | 435 | 0.15 | 0.06 s | 88 MB | **21** | 2.8 | 0.06 s | 0.1 s | 0.07 s | 2.9 MB | 103 MB | 21× | 0.9× |
| 10 | 6,724 | 1493 | 0.29 | 0.44 s | 139 MB | **19** | 18.7 | 0.35 s | 1.0 s | 0.45 s | 21 MB | 223 MB | 80× | 1.0× |
| 15 | 14,884 | 2709 | 0.37 | 1.0 s | 208 MB | **20** | 84 | 1.65 s | 8.2 s | 2.5 s | 80 MB | 570 MB | 138× | 0.4× |
| 20 | 26,244 | 4010 | 0.66 | 2.7 s | 460 MB | **19** | 329 | 6.2 s | 42 s | 10.5 s | 221 MB | 1466 MB | 213× | 0.26× |

(wall/step for the condensed column includes the one-time build spread
over the 10 steps; the CG/step column is the steady-state cost. Dense
patches at N = 20 would hold 4.5 GB; condensed holds 221 MB, 20×.)

Reading. Iterations are flat at 19–21 for every N while Jacobi grows
∝ N (435 → 4010): the p-independence carries into the time-marching
production path unchanged. Memory is modest at every N (221 MB at N = 20;
the 1.5 GB RSS is dominated by the transient element blocks and the coarse
solver, not the factors). Wall time is the honest part: on a CPU in numpy
the condensed patch matches Jacobi up to N = 10 and loses beyond it
(0.4× at N = 15, 0.26× at N = 20), because the patch apply grows ∝ N^3.4
(2.8 → 329 ms) while a Jacobi iteration grows ∝ N (0.15 → 0.66 ms): 213×
fewer iterations do not cover a 500× costlier iteration. The solutions of
the two methods differ by up to 8.6e−3 in u at N ≥ 15, which is the
absolute 1e−8 residual tolerance acting on a system with κ ≳ 1e5 (impulsive
lid corners), the same effect seen in the Re = 32000 run, not a
preconditioner effect.

Net-net from this sweep: for a small 2D operator on a CPU the method wins
on iterations, is indifferent on memory, and does not win on time above
N ≈ 10. The case for it remains where the operator apply is expensive and
the local solves batch well — the 3D channel per mode on the GPU — and
that case is still an estimate (ADN_FOSLS.md §13, VERTEX_SCHWARZ_IMPLEMENTATION.md
§3b) until step 4 of the port is run.

**Amortised reading (build excluded — it is paid once per linearisation
change, i.e. never in 3D):**

| N | Jacobi solve/step | condensed CG/step | Jacobi / condensed |
|---|---|---|---|
| 5 | 0.06 s | 0.06 s | 1.0× |
| 10 | 0.44 s | 0.35 s | **1.26×** |
| 15 | 1.0 s | 1.65 s | 0.61× |
| 20 | 2.7 s | 6.2 s | 0.44× |

**Flop bound of the prototype's apply.** At N = 20 one condensed apply is
≈ 2.5e8 flop (16 interior solves of 1444 dofs, the interior–edge products,
25 Schur solves of ≈ 960 dofs) — ≈ 5 ms at the ~50 GFLOP/s two threads
deliver — against 329 ms measured: the Python loop over 25 patches × 4
elements with per-call scipy overhead runs ~60× off the bound. A compiled
or batched implementation of the same algorithm would put the condensed
step near 19 × (5 + 0.7) ms ≈ 0.1 s against Jacobi's 2.7 s at N = 20
(~25×), and the build would shrink in proportion. The CPU losses at
N ≥ 15 are therefore an implementation ceiling of the prototype, not an
algorithmic one; the GPU channel estimate rests on the same arithmetic.

### 3.3 Option F tested and ruled out (`scratch/lor_fosls_2d.py`)

The Q₁ FOSLS operator on the GLL sub-mesh was built from the same four
residual rows as the SEM operator (bilinear cells between consecutive GLL
nodes, 2×2 Gauss quadrature, same dof numbering and mask; 30 nonzeros per
row against ~700 for the SEM operator, built in < 1 s). Two questions:
is A_LOR spectrally equivalent to A_SEM, and does AMG on A_LOR precondition
A_SEM. CG to 1e−10 on the free system, κ from Ritz values:

| c | N | Jacobi | **exact A_LOR⁻¹ solve** (ceiling of any LOR method) | SA-AMG(A_LOR) → A_SEM | SA-AMG(A_LOR) → A_LOR itself |
|---|---|---|---|---|---|
| 5405 | 8 | 1736 / 9.5e4 | **1392 / 4.7e4** | 1433 / 5.1e4 | 73 / 73 |
| 5405 | 12 | 3110 / 4.3e5 | **2911 / 2.2e5** | 3170 / 2.5e5 | 96 / 140 |
| 5405 | 16 | 5103 / 1.3e6 | **4833 / 6.5e5** | 5199 / 7.8e5 | 112 / 280 |
| 1 | 8 | 1355 / 1.0e5 | 219 / 630 | 324 / 1.4e3 | 60 / 47 |
| 1 | 12 | 2155 / 3.5e5 | 206 / 490 | 381 / 2.0e3 | 79 / 85 |
| 1 | 16 | 2965 / 9.1e5 | 180 / 370 | 422 / 2.5e3 | 91 / 130 |

(iterations / κ). Rayleigh-quotient ratio xᵀA_SEM x / xᵀA_LOR x: 1.00 on a
smooth field, 2.3 on a random one, at both c.

Reading. At c = 1 the LOR operator is spectrally equivalent to the SEM
operator (κ of the exact LOR-preconditioned system 370–630, flat or
falling in N) and AMG on it is a usable if unremarkable preconditioner
(324–422 iterations against 1355–2965 for Jacobi; the ladder does better).
At c = 5405 the **exact** LOR solve is almost as bad as Jacobi (κ 4.7e4 →
6.5e5, growing with N like Jacobi's), so no AMG, however good, can make a
LOR-based preconditioner work there: the equivalence itself fails. The
Q₁ discretisation on the sub-cells and the GLL spectral element have
different discretely divergence-free kernels, and it is exactly those
modes that dominate the large-c operator (ADN_FOSLS.md §5.3). AMG's own
quality on the LOR matrix is fine (κ 73–280), which confirms that the
failure is in the equivalence, not in AMG. This is the LOR analogue of the
Hiptmair result (§8.2 of ADN_FOSLS): on C⁰ nodal elements the kernel has
no representation in any other space than its own. Option F is closed; the
route to sparse, O(dofs) preconditioning of this operator runs through a
change of discretisation to de Rham-conforming spaces (option H), where
Pazner–Kolev–Dohrmann's equivalence theorem applies.
