# Vertex-patch Schwarz preconditioner for FOSLS-3D: findings and implementation plan

Companion to `ADN_FOSLS.md` (theory and measurements). This document is the
engineering record: what was established, the 2D reference implementation
that exists, and the step-by-step port to `lssem3d` with the checks that gate
each step.

## 1. Findings (2026-09-07)

1. **The solver problem is the (u, ω) block at large c.** ADN analysis and
   measurement agree: at the channel's c = 1/(β dt) = 5405 the FOSLS
   functional is a norm on the pair (u, ω − ∇×u), and after vorticity
   elimination the velocity block is an H(div)-type operator whose
   divergence-free kernel is invisible to any pointwise or per-variable
   preconditioner. Exact (u,ω)‖p block solve: κ ≈ 3, 9–17 iterations,
   independent of p, h, c — the ceiling (`ADN_FOSLS.md` §5).
2. **Overlapping vertex-patch additive Schwarz reaches that kernel.** Patch =
   all dofs of every element sharing a mesh vertex; exact local solve;
   additive; plus a p = 2 Galerkin/rediscretised coarse correction.
   2D, production `pcg_solve`, c = 5405: 31 → 26 → 25 CG iterations at
   N = 8 → 12 → 16 (Jacobi 1286 → 3497; PMG ladder 178 → 522). At c = 1:
   27–26. h-sweep 4×4 → 8×8 elements at N = 8: 31 → 41 (§10 of ADN_FOSLS).
3. **Overlap must be element-sized.** Element blocks (zero overlap): 655
   iterations; element + 1 node layer: 330; + 2 layers: 231; full vertex
   patch: 42 (all with coarse, N = 8). Pavarino's condition for spectral
   elements. This is also why §6.3's element Schwarz stopped at 6×.
4. **The nodal Hiptmair / auxiliary-space sweep does not work on C⁰ GLL**
   (additive worse than Jacobi, multiplicative 2.4×): the discrete
   divergence-free kernel has no potential representation on nodal elements.
5. **The patch block must be R A Rᵀ of the assembled operator.** Summing
   only the patch's own elements leaves patch-boundary nodes without their
   outside neighbours' stiffness: 16/25 blocks singular, CG stalled at
   residual 0.997. The ring of neighbouring elements contributes, restricted
   to patch dofs.
6. **Equilibrate before Cholesky.** Pressure rows carry a_flux² ~ dt² ~ 1e−8
   against O(1) mass rows; unscaled dense Cholesky fails (potrf info > 0).
   Symmetric Jacobi scaling D^{−1/2} K D^{−1/2} fixes it; LU fallback kept.
7. **Cost per iteration, not count, decides the win.** Dense patch solve is
   O(n_patch²) per patch; for the channel per mode ≈ 3e10 flop per CG
   iteration, the same order as the present matrix-free apply (~9 ms on the
   GB10). With ~40 iterations against 2000–6000 the expected reduction in
   solve time per stage is 30–60×, if the patch solves run as batched dense
   triangular solves on the GPU. In the numpy 2D prototype (25 sequential
   scipy calls per iteration) wall time loses to Jacobi; that is an
   implementation artefact, not the algorithm.
8. **The factors never change during the run.** The FOSLS operator is
   Stokes-like with explicit convection: it depends on (mesh, ν, k_z, c) and
   c takes three values (one per RKW3 stage). All patch factors can be built
   once at start-up and reused for the entire DNS.

## 2. The algorithm

Per Fourier mode k (modes never couple), with A_k the split-real normal
operator on (u, v, w, ωₓ, ω_y, ω_z, p) × (re, im) = 14 fields per node:

    M_k r  =  Σ_v  R_vᵀ K_{v,k}⁻¹ R_v r   +   P (A_{c,k})⁻¹ Pᵀ r

- R_v: restriction to the free dofs of patch v = {all nodes of every element
  having vertex v as a corner}; in 2D-mesh × Fourier that is (2N+1)² nodes
  × 14 = 4046 dofs at N = 8 (2 elements at domain-boundary vertices, 4
  elsewhere; periodic x merges the boundary vertices into interior ones
  through `gidx`).
- K_{v,k} = R_v A_k R_vᵀ, assembled from the element-local blocks of every
  element touching a patch dof (patch elements + ring), masked dofs removed.
- P, A_c: the existing p = 2 level of `PMG` (`_restrict`, `DirectCoarseE`,
  `_prolong`), used additively. Alternative: make the patch solve the
  smoother of a two-level `PMG` V-cycle (multiplicative); both are symmetric.

Properties CG needs: M_k is symmetric positive definite (sum of SPD terms),
and the same weighting/mask conventions as `S3.pcg` (multiplicity-weighted
inner product, mask on prescribed dofs).

## 3. 2D reference implementation (exists, verified)

`scratch/vertex_schwarz2d.py`, class `VertexSchwarz2D(state, fu, fv, pin_p,
coarse)`, same interface as `PMG2`; benchmark `scratch/vs2d_bench.py`.

| piece | how | check |
|---|---|---|
| element-local blocks | (N+1)²·4 probes of `apply_LT(apply_L(·))`, unit at the same local (i,j,var) in every element at once → (nelem, nde, nde) | `fosls_assemble.gate` reproduces `apply_A` to round-off |
| global dof map | `g = gidx[...,None]*4 + var`, free = mask > 0.5 at any copy | — |
| patches | corner nodes → element lists; dofs = unique(g[patch elems]) ∩ free | 25 patches on 4×4, 1156 dofs at N = 8 |
| ring assembly | for every element touching any patch node, add its block restricted to patch dofs | reference: `A[patch,patch]` of the assembled matrix, rel diff 2.6e−13 |
| factor | symmetric Jacobi scaling then `cho_factor`; `lu_factor` fallback counted | 0 fallbacks after scaling |
| apply | r·mw → bincount to global → per-patch solve → scatter-add → expand `zg[g]` → ×mask; + coarse term | iteration counts in §1 |
| coarse | `PMG2(pc=2, coarse_solver='direct')`, use `_prolong(_coarse_solve(_restrict(r)))` | — |

## 3b. Memory and cost scaling (from ADN_FOSLS.md §13)

With F real fields per node (4 in 2D; 14 real = 7 complex per mode in 3D)
a vertex patch has n_p = (2N+1)²·F dofs. Storing one triangular factor takes
n_p²/2 entries; applying it costs n_p² flops; factorising n_p³/3.

    memory  ≈  N_vertices × N_modes × ½ [(2N+1)² F]² × bytes
    apply   ≈  N_vertices × N_modes × [(2N+1)² F]²  flops per CG iteration
    build   ≈  N_vertices × N_modes × ⅓ [(2N+1)² F]³ flops per (re)factorisation

- **Order N: (2N+1)⁴ ≈ 16 N⁴ for memory and apply, (2N+1)⁶ for the build.**
  Doubling N costs 16× and 64×. This is the term that makes dense patches a
  moderate-order method.
- **Elements: linear** (vertices ≈ elements). Doubling the mesh doubles it.
- **Fourier modes: linear** in nk = nz/2 + 1; modes never share factors.
- **Fields: F²**, so the 3D per-mode block is 12× a 2D block at equal N; the
  complex-Hermitian form (n_p/2 complex entries) halves it against the
  split-real form the prototype uses.
- **Against the operator:** the state vector is linear in (N+1)² and in
  elements, so factors/state ≈ 4 (2N+1)² F, about 5000 at N = 8 in the channel.

Exact numbers, single precision, one value of c (triangular storage;
the numpy prototype stores the full n_p² array, 2× these):

| case | N | vertices | patch dofs (real) | one factor | total |
|---|---|---|---|---|---|
| 2D cavity 4×4 | 8 | 25 | 1156 | 5 MB | 0.1 GB |
| 2D cavity 4×4 | 30 | 25 | 14,884 | 0.89 GB (double) | 22 GB (double; 44 GB as stored) |
| 3D channel 6×18, 1 mode | 8 | 114 | 4046 (2023 complex) | 16 MB | 1.9 GB |
| 3D channel, 17 modes | 8 | 114 | 4046 | 16 MB | **32 GB** |
| 3D channel, 17 modes | 12 | 114 | 8750 | 77 MB | 148 GB |
| 3D channel, 17 modes | 16 | 114 | 15,246 | 232 MB | 450 GB |

Three stage values of c multiply the channel figures by 3 if all are kept
resident (95 GB at N = 8), or cost one refactorisation (≈ 5e12 flop, under
a second on the GB10) per stage change if only one set is kept.

Reading against the hardware: on the 121 GB Spark the channel is feasible
at N = 8 (32 GB, or 95 GB fully resident), marginal at N = 12 (148 GB even
for one c), and out of reach at N = 16. Refining in h instead of p is
benign: twice the elements at N = 8 is 63 GB. Raising N is what closes the
door, and only a non-dense local solver (tensor-product or low-rank patch
factorisation, or a fixed-polynomial inner patch iteration) would reopen it.
Measured apply cost per CG iteration in the numpy prototype, 2D: 24 ms at
N = 8 (110 Jacobi iterations), 7.7 s at N = 30 (6600).

## 4. 3D port, step by step

Each step has a gate; do not proceed on a failed gate. Run everything on the
Mac harness first (`OMP_NUM_THREADS` set, `python -u … > log`, never
`| tail`). Spark only at step 6, and never while run01 is alive.

### Step 0 — confirm in 3D on the assembled harness (½ day)
Extend `scratch/adn_block_precond.py` (it already assembles A_k per mode,
2×2 → 4×4 elements, N 4–12, c = 1 and 5405, k_z = 0 and 5.88) with the
vertex-patch Schwarz from `adn_schwarz_2d.py` on the full 14-field operator,
with and without the Galerkin p = 2 coarse term.
*Gate:* iterations flat in p (≤ 1.3× from N = 4 to 12) at c = 5405 for both
k_z, and the k_z = 0 mode with pinned pressure behaves like the others.
Expected: 30–45 iterations, κ 15–30, as in 2D.

### Step 1 — `VertexSchwarz` in `lssem3d/precond.py` (2 days)
Signature mirroring the others: `VertexSchwarz(level, coarse=None)` with
`level` a `_Level` (mesh, mask, `A_un`, `mw`, shape (nelem, n, n, 14, nk)).

1. **Element-local blocks for all modes at once:** copy the probe loop of
   `DirectCoarseE.__init__` verbatim — `nloc = n·n·14` probes through
   `level.A_un`, giving `Aloc (nelem, nk, nloc, nloc)`. Local index
   `(i, j, f) → (i·n + j)·14 + f`, matching `shape.reshape(nelem, nloc, nk)`.
2. **Patch topology (host, once):** corner node ids `gidx[e, {0,N}, {0,N}]`;
   `corners[node] → [elements]`; `node_elems[node] → [elements]` for the
   ring; per patch: `dofs` = sorted unique global dofs (`gidx·14 + f`) of the
   patch elements, restricted to free dofs of the *fine mask* (free = mask
   > 0.5 at any copy, built from the caller's mask exactly as `_Level`
   does — never rebuild the mask on a mode subset); ring = union of
   `node_elems` over the patch's nodes.
3. **Patch matrices:** for each patch and mode, `K = Σ_{e∈ring}
   scatter(Aloc[e, k])` restricted to patch dofs, symmetrised. Pad to
   `n_max` (4046 at N = 8) with unit diagonal so all patches batch into one
   tensor `(npatch, nk, n_max, n_max)`; domain-boundary vertex patches (2
   elements) are the padded ones.
4. **Equilibrate and factor:** `s = 1/sqrt(diag K)`, `Ks = s K s`,
   `torch.linalg.cholesky(Ks)` batched (or `cupy`/`numpy` per backend);
   keep `s` `(npatch, nk, n_max)`. Precision: build in float64, store
   factors in float32 (step 4 tests whether that costs iterations).
5. **Apply `__call__(r)`:** `r` is split-real local `(nelem, n, n, 14, nk)`
   with assembled values on every copy (that is what `S3.pcg` hands a
   preconditioner). (a) `rg = bincount(gd, r·mwl)` per mode — exactly
   `DirectCoarseE.__call__`'s gather; (b) gather patch vectors with a
   precomputed index tensor `(npatch, n_max)` (padding indices point at a
   dummy zero slot); (c) `x = s · cholesky_solve(s · r_v)` batched;
   (d) scatter-add into `zg` (`index_add_`); (e) expand `zg[gd]`, multiply
   by `level.mask`; (f) if `coarse` is set, add it.
6. **Coarse term:** build `PMG(mesh, …, orders=(N, 2), direct_coarse='element',
   mask=mask)` and use `pmg._prolong(pmg.coarse(pmg._restrict(r, 0)), 0)`;
   or, multiplicative variant, subclass `PMG` with `self.smooth[0] =
   VertexSchwarz(self.levels[0])` and `orders=(N, 2)`.

*Gates:* (i) action identical to the assembled-reference Schwarz on the
2×2 harness (rel diff < 1e−10 in float64); (ii) symmetry: `⟨r, M s⟩_mw =
⟨s, M r⟩_mw` to round-off for random r, s; (iii) `p_independence.py`
protocol (3×3 elements, N 8–20, k_z = 0 and 5.88, cold RHS): iterations flat
in p and ≤ 50; (iv) `pmg_symmetry.py`-style check that CG's Ritz values are
all positive.

### Step 2 — device batching (1 day)
Backends: `numpy` (reference), `numba` (host), `cuda` via torch (production
on Spark, `LSSEM3D_BACKEND=cuda`), `cupy`. Implement apply with the `DEV`
helpers (`to_device`, `zeros_like`, `cat`) and backend-native batched
`cholesky_solve`, `index_select`, `index_add_`. Keep the host path as the
oracle.
*Gate:* device result equals host result to float32 round-off on the
production channel mesh for one mode set; time per apply printed.

### Step 3 — memory and precision on the production mesh (½ day)
Mesh 6×18, N = 8, nz = 32 → nk = 17, 114 patches: factors
114 × 17 × 4046²/2 × 4 B = **31 GB fp32**, ×3 stages = 93 GB, or refactor
per stage (≈ 5e12 flop, < 1 s). Since the operator never changes during the
run, prefer keeping all three if memory allows; otherwise cache one and
refactor when c changes (three times per step, or reorder so all substeps
of one stage… no: stages alternate, so it is three refactorisations per
step ≈ 2 s — still small against the present 60–150 s per step).
Complex Hermitian storage halves this; leave for later.
*Gate:* fp32 factors give the same iteration count as fp64 (±2) on the
harness; peak memory measured with `nvidia-smi`/`torch.cuda.max_memory_allocated`.

### Step 4 — channel stage solve, off-line (½ day)
`scratch/pmg_pcg_channel.py` protocol: warm turbulent state from a run01
checkpoint, one RKW3 stage at each c, all 17 modes batched, tol as
production, on Spark once run01 is finished. Compare Jacobi vs PMG ladder vs
VertexSchwarz(+coarse): iterations (remember `its = max over modes`, so
also record the per-mode distribution), wall per stage, solution difference
vs Jacobi (must be below the solve tolerance).
*Gate:* ≤ 60 iterations per stage; solution identical to Jacobi within
tolerance; wall per stage below the current by ≥ 10× (the 30–60× estimate
is the target, 10× is the go/no-go).

### Step 5 — integrate (½ day)
`scratch/minchan.py::_precond` / `channel3d.make_precond`: build the
preconditioner once per c (three objects, or one with refactor), on the
host, move factors to device (`to_device`), pass as `M_inv` callable to
`S3.pcg`. Warm start (§6.1) and per-mode/grouped solves (§6.4) are
orthogonal and can be layered afterwards. Leave `ROW7_WEIGHT`, row weights
and the operator untouched — the preconditioner is a pure solver change and
must reproduce the run01 physics (TG/Stokes-decay validation ladder
unchanged to tolerance).

### Step 6 — production
Restart the channel from the last run01 checkpoint with the new
preconditioner for ~100 steps; compare `u_tau`, `rms_w`, `div` and the
statistics accumulator increments against the Jacobi continuation over the
same window. Only then adopt.

## 5. Variants to measure after the baseline works (ranked)

**Update 2026-09-08:** static condensation of element interiors is exact and cuts patch memory 8–12× (measured, `LOW_MEMORY_PATCH_SOLVERS.md` §3); build it into step 1 of the port. The 'patch smoother on the p = 4 level only' variant below was tested at N = 30 and does not work (ADN_FOSLS.md §12).


1. **Patch smoother on the p = 4 level only** (Chebyshev on the fine level,
   patches 16× cheaper): how much p-independence survives? This is the
   cheapest configuration if it holds.
2. **Complex Hermitian arithmetic:** halves factor memory and flops.
3. **Multiplicative two-level (V-cycle with patch smoother) vs additive:**
   usually 1.5–2× fewer iterations for the same cost.
4. **Boundary patches:** wall vertices have 2-element patches; check whether
   they are the ones limiting κ (softest Ritz vector localisation).
5. **Coarse level richer than p = 2** if the 6×18 mesh shows h-growth
   (2D: 31 → 41 over 4× elements).

## 6. Pitfalls checklist (each cost time already)

- Patch block from patch elements only → singular. Use the ring.
- No equilibration → Cholesky fails on pressure rows (a_flux² ~ dt²).
- `BC.build_mask` on a mode subset masks the wrong modes; build for full nk,
  slice.
- Pressure pinned at k = 0 only (`BC.pin_dof(m, mask, OP.P_, 0)`); coarse
  levels must reproduce the caller's convention (see `PMG.__init__`).
- `S3.pcg` residuals carry assembled values on every copy; gather with
  `mw`-weights and `bincount`, exactly as `DirectCoarseE`.
- `its = max over modes` in `channel3d.step`; batched iteration count is the
  slowest mode's (§7S.5).
- Periodic x: vertices on the periodic boundary are interior after
  `compute_global_indices`; do not special-case them.
- Long runs: `python -u … > log`, never `| tail`.
- Do not touch Spark while run01 runs; `pgrep -f` matches your own ssh
  command line — use `ps | grep -v "bash -c"`.

## 7. Acceptance

The preconditioner is adopted when, on the production channel: (a) CG
iterations per stage ≤ 60 and flat when N is raised 8 → 12 on a 3×3 test
mesh; (b) wall time per step ≤ 1/10 of the Jacobi baseline; (c) the
validation ladder (Stokes decay, TG) is unchanged to tolerance; (d) 100
steps of channel continuation match the Jacobi continuation in `u_tau`,
`rms_w`, `div` to the solve tolerance.

## 8. Files

- Theory and measurements: `ADN_FOSLS.md` (§5 3D assembled tests, §8 2D
  tests, §10 2D production-path prototype).
- 2D reference implementation: `scratch/vertex_schwarz2d.py`,
  `scratch/vs2d_bench.py`, logs `scratch/vs2d_bench.log`.
- 2D assembled tests: `scratch/adn_hiptmair_2d.py`, `adn_schwarz_2d.py`,
  `adn_schwarz_2d_full.py`, `adn_schwarz_2d_thin.py`.
- 3D assembled harness to extend for step 0: `scratch/adn_block_precond.py`
  (assembly, `blockdiag_solver`, `pcg_ritz`), `adn_block_precond2–4.py`.
- Production pieces to reuse: `lssem3d/precond.py` (`_Level`,
  `DirectCoarseE` probes/gather, `PMG` transfers), `lssem3d/solver3d.py`
  (`pcg`, `gs`, `jacobi_diagonal_analytic`), `lssem3d/device.py`.
