# The CuPy backend: a second, independent GPU path

Status 2026-08-22: **operator parity 2.6e−16, 14× over host NumPy, running in
Docker on the DGX Spark.** Branch `cupy-backend`, deliberately separate from
the PyTorch/CUDA port so the two can be compared rather than merged by
accident.

## Why a second GPU backend at all

`kernels_torch.py` already runs this operator on the GB10. This one exists to
be *independent of it*: different array library, different kernel compiler,
different reduction implementation, same NumPy reference. Lesson L1 of
`3D_STATUS.md` is that tests comparing an operator to itself cannot find a
consistently wrong operator — **two ports that share no device code agreeing
with the reference is evidence; one port agreeing with itself is not.**

## What the port actually required

Far less than expected, and the reason is worth recording. **CuPy implements
NEP-18**, so `np.einsum`, `np.concatenate` and friends *called on a CuPy array
dispatch to CuPy automatically*. `deriv.py`, `fourier.py`, and the entire `pcg`
loop therefore run on the device untouched.

The only thing that does not dispatch is array **creation**: `np.empty` builds
a host array regardless of what will be written into it, and assigning device
data into it raises `TypeError: Implicit conversion to a NumPy array is not
allowed`. So the whole port is:

| file | change |
|---|---|
| `device.py` | `is_cupy()`, `xp()` recognises CuPy, creation helpers ask `xp(like)` instead of hard-coding `np`, plus `gs_cupy` (scatter-add gather-scatter, mirroring `gs_torch`) |
| `solver3d.py` | **+2 lines** — a `DEV.is_cupy(U)` branch in `gs` |
| `operator.py` | **+2 lines** — an `elif name == 'cupy'` in `_bind_backend` |
| `backend.py` | `'cupy'` in `VALID` + an `available()` probe |
| `kernels_cupy.py` | **new** — the two output buffers are the entire difference from the NumPy reference |

Every edit to a shared file is additive, so the merge with the torch port is a
review rather than a conflict resolution.

## Measured on the DGX Spark (GB10), in Docker

```
normal_op parity  max rel err = 2.613e-16   (PASS)
matvec  numpy(host)  32.0 ms   cupy(GB10)  2.3 ms   speedup 13.97x   (0.53 M dof)
```

Suite: **167 passed, 36 skipped**.

`test_deriv.py::test_trailing_axes_are_independent` used to fail here, and was
recorded as an ARM/NumPy environment difference. It was actually a **wrong
test**: it asserted an *absolute* 1e-14 on derivative values of magnitude
~1e2, i.e. a demand for 1e-16 relative — bit-exact agreement between a batched
contraction and a per-slice one. That held only because the Mac's BLAS
happened to associate the batched case identically; aarch64 associates it
differently and missed by one ulp (1.4e-14 absolute, **1.3e-16 relative**).
The property under test is real; the threshold was not, and is now relative.

## THE FP64 WARNING — read before quoting any speedup

The GB10 is an AI part. Measured on it with CuPy:

| | GB10 (DGX Spark) | A100 80GB |
|---|---|---|
| FP64 GEMM | **0.21 TFLOP/s** | ~9.7 TFLOP/s (**46×**) |
| FP32 GEMM | 8.61 TFLOP/s | — |
| memory bandwidth (fp64 triad) | **112 GB/s** | ~1900 GB/s (**17×**) |

FP64 runs at ~**1/41** of FP32 here, and the measured bandwidth is *below* the
M3 Max's ~400 GB/s. Our matvec is bandwidth-bound and float64 throughout, so:

* **the Spark is a CORRECTNESS platform for this work, not a performance one** —
  which is exactly what it was used for above;
* the 14× is against *host NumPy on the same Grace cores*, **not** against the
  Mac's numba path, and must not be quoted as the latter;
* real throughput numbers must come from FP64-capable hardware (A100).

Do not be tempted into float32: `TGV_VALIDATION.md` validates σ to seven digits
and the *e*-metric at the 1e−3 level, and CG runs to tol 1e−6.

## Environment

`docker/Dockerfile.cupy` — Ubuntu 24.04, venv, `cupy-cuda13x[ctk]`, **no torch**.
The `[ctk]` extra is not optional: CuPy JIT-compiles its elementwise kernels at
first use and needs the CUDA *headers* at run time. Without it the image builds
and imports cleanly, then dies on the first arithmetic operation — a failure
that only appears once real work starts.

```bash
docker build -f docker/Dockerfile.cupy -t lssem-cupy:latest .
docker run --rm --gpus all -v "$PWD":/work -w /work lssem-cupy:latest \
       python scratch/cupy_parity.py
```

## Phase 2: whole-solve parity and a physics stage — PASSED

`scratch/cupy_ladder.py`, in the container on the GB10:

```
A. WHOLE-SOLVE PARITY (device-resident CG)
   numpy :   542 iters   err vs planted solution 9.54e-07
   cupy  :   544 iters   err vs planted solution 8.57e-07
   cupy vs numpy       : 3.25e-07
   cupy vs cupy (rerun): 9.04e-07   <- the atomics floor, measured
   relative residual   : numpy 6.69e-11   cupy 6.26e-11
   verdict: PASS -- backend difference is 0.4x the self-spread   (1.6x wall)

B. PHYSICS STAGE (convection + dealiased FFT + solve), host vs device
   numpy: E = 30.9998230587  Omega = 93.00164866  (441 CG)
   cupy : E = 30.9998230587  Omega = 93.00164866  (441 CG)
   rel diff: E 0.00e+00   Omega 1.53e-16   PASS
```

**A** is the entire preconditioned CG — hundreds of iterations, every dot
product, the gather-scatter, the convergence test — running device-resident.
**B** is a real TGV stage: convection, the 3/2-dealiased FFT, the
defect-corrected right-hand side and the solve, compared in the two quantities
this project actually validates on.

### Getting the acceptance criterion right took two wrong tries

Worth recording, because both wrong versions looked reasonable:

1. **Absolute threshold** (`< 1e-8`) — wrong. At this conditioning *both*
   solves sit ~1e−6 from the planted solution at `tol` = 1e−10, so this asks
   the two backends to agree far more closely than either agrees with the
   truth.
2. **Fraction of the solver error** (`< 10%`) — also wrong, and it failed.
   `gs_cupy` accumulates with atomics, so **CuPy does not reproduce itself**
   run to run; the criterion demanded determinism the method does not have.
3. **Self-calibrated** — right. Solve twice on the device: *that spread is the
   reduction noise*. The port is correct if cupy-vs-numpy is the same size as
   cupy-vs-cupy, and if both solutions independently meet the residual
   tolerance they were asked for. Measured: the backend difference is **0.4×
   the self-spread** — CuPy differs from NumPy *less than it differs from
   itself* — with residuals 6.7e−11 and 6.3e−11 against a requested 1e−10.

This is L5 ("know your floor before calling something an error") in a new
costume, and it is the reason a bitwise-parity habit has to be dropped
deliberately rather than by accident when a backend goes non-deterministic.

## Phase 3: the VALIDATION LADDER, re-run on the CuPy path — ALL THREE PASS

`scratch/cupy_validation_ladder.py`, device-resident throughout, in the
container on the GB10. Each gate is checked against something NumPy cannot
influence — an analytic rate, the design order, or a parameter-free identity —
and, where a NumPy run is on record, against that record too.

### Gate 1 — Stokes decay against the analytic σ = 9.3137399

| `dt` | CuPy σ | rel err | NumPy record (`stokes_afterfix.log`) |
|---|---|---|---|
| 0.01 | 9.3153041 | 1.679e−04 | 9.3153041, 1.680e−04 |
| 0.005 | 9.3141300 | 4.188e−05 | 9.3141300, 4.189e−05 |
| 0.0025 | 9.3138373 | 1.045e−05 | 9.3138373, 1.046e−05 |

**σ reproduces the recorded NumPy values to all eight printed digits**, with
convergence order **2.00**.

### Gate 2 — rotated (x,z) TG, z-convection active

| `dt` | CuPy L2 err | NumPy record (§7E.1) | ratio |
|---|---|---|---|
| 0.02 | 5.7239e−07 | 5.724e−07 | **1.000** |
| 0.01 | 1.4308e−07 | 1.431e−07 | **1.000** |
| 0.005 | 3.5769e−08 | 3.577e−08 | **1.000** |

Order **2.00, 2.00** — the design order, on the path that exercises `w ∂/∂z`,
the `i k_z` terms and the 3/2-dealiased mode convolution inside the stage loop.

### Gate 3 — the parameter-free balance

`E(0)` = 31.006277 and `Ω(0)` = 93.018830 against the analytic
$(2\pi)^3/8$ and $3(2\pi)^3/8$; the balance
$-dE/dt = 2\nu\Omega$ holds over ten steps with a worst deviation of
**6.65e−06**.

**What this establishes.** The CuPy path is not merely bit-comparable to NumPy
on one operator application — it reproduces the project's physics results, to
the recorded digits, on a different array library, a different kernel compiler
and a different device. Two ports that share no device code now agree with the
reference; that is the evidence L1 asks for, and neither port alone provides
it.

## Phase 4: the dispatch problem, and why CUDA graphs are not the answer here

### The symptom: a benchmark that does not scale

On a Colab A100 (FP64 17.16 TFLOP/s, 1356 GB/s — a full card, not a MIG
slice), `normal_op` measured **11.45 ms at every size from 0.53 M to 6.17 M
dof**. A 12× range in work with no change in wall clock is not a performance
result; it is a **dispatch-bound** loop. The host cannot issue the ~200
Python-level array operations behind one matvec faster than the GPU retires
them, so the A100 idles.

Quantified: at 6.17 M dof the operator moves ≈2.5 GB (inferred from the GB10's
22.39 ms at its measured 112 GB/s), which at 1356 GB/s is **~1.85 ms of real
work** — so **~84% of the wall clock was the host talking.** No faster GPU
fixes that.

### CUDA graphs: implemented, and blocked by cuBLAS

Capture is the textbook fix — record one CG iteration, replay it, and
per-launch host cost largely vanishes. It is implemented in
`lssem3d/cupy_graph.py`, including the two rules that make capture work
(fixed buffers, warm the memory pool first) and a batched convergence test,
since capture forbids the host synchronisation that a per-iteration residual
check requires.

It does not run:

```
NotImplementedError: calling cuBLAS API during stream capture is
currently unsupported
```

The einsum contractions route through `gemmStridedBatchedEx`, so the matvec —
the bulk of the launches — cannot be captured at all. **PyTorch can capture
cuBLAS**, so graphs remain available on the torch port; that is a concrete
advantage for it on dispatch-bound hosts, and worth knowing when choosing
which backend runs production on Colab. The module is kept: it is correct and
becomes useful the day CuPy lifts the restriction.

### So the dispatch cost was attacked directly — 3.25×

Profiled at a size where GPU work is negligible, so the wall clock *is* the
host cost (`scratch/cupy_dispatch_profile.py`):

| | before | after |
|---|---|---|
| `normal_op` total | 3.536 ms | **1.088 ms** |
| ├ `apply_L` | 1.162 | 0.484 |
| ├ `apply_LT` | 1.050 | 0.503 |
| └ gather-scatter | **1.726** | **0.085** |

Two causes, and the first was mine:

1. **A host synchronisation per matvec.** `gs_cupy` computed the global-dof
   count as `int(idx.max())` on every call — a device reduction *and* a
   device→host sync, once per matvec, stalling the pipeline. It is a property
   of the mesh, so it is now cached with the index. **1.726 → 0.085 ms**, and
   this fix has no trade-off on any device.
2. **Fourteen einsum calls where two suffice.** `_L0` took derivatives
   field-by-field, but `ddx`/`ddy` already carry arbitrary trailing axes and
   the field axis is one of them; `_LT` did the same for eight rows.
   Identical arithmetic, **24 fewer cuBLAS dispatches** per matvec.

Parity is unchanged at **2.613e−16**, and gates 1 and 3 still pass
(σ = 9.3141300 / 9.3138373 at order 2.00; balance 6.65e−06).

### Phase 4b: fusing the row assembly — 6.2× in total

Cell 7b on the A100 settled where the rest sat: `normal_op` cost **3.825 ms at
0.004 M dof and 3.789 ms at 6.17 M** — a *1500×* range in work, identical wall
clock, so the GPU contribution was entirely invisible and 96% of the host cost
was in `apply_L`/`apply_LT`. Those are 16 row formulas built from ~80 separate
elementwise calls, at ~32 µs each on that vCPU.

They are now **two `ElementwiseKernel` launches** — one computing all eight
forward rows, one all seven adjoint fields. Identical algebra, issued once
instead of forty times:

| dispatch floor (Spark, tiny size) | |
|---|---|
| original | 3.536 ms |
| + gather-scatter sync removed | 1.088 ms |
| **+ fused row assembly** | **0.570 ms** |
| | **6.2× total** |

Per part: `apply_L` 1.162 → 0.244, `apply_LT` 1.050 → 0.220, gather-scatter
1.726 → 0.085 ms. Parity is unchanged at **2.613e−16**, and gates 1 and 3
still pass (σ = 9.3141300 / 9.3138373 at order 2.00, balance 6.65e−06).

Scaled to the A100 host (~3.5× slower at issuing than the Spark's Grace
cores), this should put `normal_op` near **2 ms against ~1.85 ms of real GPU
work** — i.e. the loop finally becomes GPU-bound, which is where host
optimisation stops paying and hardware starts mattering again.

### Measured on the A100: GPU-bound at last

| | 0.53 M | 1.23 M | 3.43 M | **6.17 M** |
|---|---|---|---|---|
| first run | 11.64 | 11.50 | 11.58 | **11.45** |
| after the sync + batching fixes | 4.10 | 4.00 | 3.99 | **4.01** |
| **after fusion** | 2.40 | 2.32 | 2.32 | **2.93** |

The last row is the result: after two rounds of identical timings across a
12× (and in the profile, 1500×) range in work, **the largest case finally
separates from the floor**. Host issue cost is ~2.32 ms; the GPU needs ~2.93 ms
at 6.17 M dof. They have crossed over, so the loop is **GPU-bound** — which is
the right place to stop optimising the host: the remaining time is genuine
memory traffic, and a further 20% is all that overlap could buy.

**3.9× end to end on the A100** (11.45 → 2.93 ms), and against the Mac's numba
path (~25 ms per CG iteration) roughly **8×**:

| | per CG iteration | 88³ Re = 800 | Re = 1600 @ 128³ (CORIA) |
|---|---|---|---|
| Mac + numba | ~25 ms | 50 h | 15 days |
| **Colab A100** | **~3.2 ms** | **~6.4 h** | **~1.9 days** |

That last column is the one that matters: the CORIA benchmark comparison
(`TGV_VALIDATION.md` §9), shelved as a post-M6 undertaking at 15 days, is now
a two-day run. **The bottleneck has moved from throughput to the missing
checkpoint/restart driver**, which is what Colab's session limits require.

### Phase 5: the inner product was two-thirds of the solve

The Re = 1600 128³ case priced at **174 s/step — 248 h**, against a projected
46 h. Four rounds of diagnosis, each wrong, before the measurement that
settled it. Recorded in order, because the errors are the useful part:

| checked | result |
|---|---|
| wrong GPU? | genuine A100-SXM4-40GB ❌ |
| memory pressure? | 8.7 of 40 GiB ❌ |
| bad conditioning? | 4850 its/step, `capped = 0`, only 1.3× the 88³ run ❌ |
| CG's per-iteration host sync | real, but only 5.4 ms of 36 |
| **the inner product** | **9.41 ms against a 0.56 ms bound — 17× off** ✅ |

`_dot` keeps only the mode axis and sums the other four, so ~19 M inputs
produce **65 outputs** — a 290,304:1 reduction. CuPy gives that roughly one
block per output: 65 blocks on a **108-SM** A100, most of the card idle. At
two calls per iteration it was **two thirds of the entire solve**:

| | ms |
|---|---|
| matvec | 8.5 |
| 2 × `_dot` | **18.8** |
| vector ops | 1.7 |
| sum | 29.0 |
| *measured, by differencing solves at two iteration counts* | *29.68* |

Written as `(1 × M) @ (M × nk)` it is a GEMM, and cuBLAS fills the card:
**0.81 ms, 11.6×**. The control matters more than the fix — fusing the triple
product into a `ReductionKernel`, removing **both** 144 MiB temporaries, gave
10.17 ms, *no better*. So the cost was never memory traffic. It was the
reduction's **shape**, which is invisible to every bandwidth calculation.

Iteration **29.7 → 11.8 ms (2.5×)**; the case goes 212 → ~85 h. Verified:
167 tests, parity 2.613e−16, and Stokes σ **bit-identical** to the pre-change
run (9.3141300 / 9.3138373, order 2.00).

**Two lessons worth more than the speedup.** First, `normal_op` measured 8.45 ms
against a 8.9 ms prediction — the matvec was *exactly* right the whole time.
Benchmarking it in isolation and assuming the rest of the iteration was
negligible is what hid a 17× anomaly through four wrong projections. Second,
the differencing measurement — run the solver at two iteration counts and
subtract, so everything per-iteration survives and everything per-solve
cancels — took two minutes to write and would have pointed straight here.
**Instrument before predicting.**

Confirmed on the A100: **30.59 → 12.53 ms per iteration**, 148.7 → 59.4 s per
step, and the case 211.9 → **84.6 h** (20 → 8 sessions).

**Does torch pay the same cost? No — measured, not assumed.** The starved
shape belongs to the problem rather than to CuPy, so the torch path looked
like it should. On the *same* A100:

| | ms |
|---|---|
| CuPy `sum(axis=0..3)` | 9.41 |
| **torch `sum(dim=0..3)`** | **0.78** |
| torch, as a GEMM | 0.79 (no gain) |

**Torch's reduction kernels handle this shape; CuPy's do not** — a 12×
library difference on identical hardware and identical arrays. So the GEMM
form is applied on the CuPy path *only*, and there is nothing to fix in the
PyTorch port. `scratch/torch_dot_variants.py` re-checks it if CuPy ever
improves.

Worth carrying into the cupy-vs-torch question generally: the two are not
interchangeable at the kernel level. A shape CuPy handles badly may cost 12×,
and the only way to know is to time it.

Still open: the derivative einsums measure 4.61 ms against a 0.96 ms bound
(5× off, and now the dominant term in a 12.53 ms iteration) — worth perhaps
another 1.4×.

### The per-iteration host sync — `check_every`

The first real finding of the 174 s/step investigation, worth 5.4 ms of 36.
CG tested convergence every iteration:

```python
rn = DEV.sqrt(_dot(r, r, mw))
if DEV.all_(rn < target):        # returns a Python bool
```

`DEV.all_` forces a host synchronisation — the CPU drains the GPU queue,
waits, reads one bit, re-issues everything. Invisible to a matvec benchmark,
which is part of why `normal_op` timings extrapolated so badly, and it also
explains 100% reported GPU utilisation while doing far less work than the card
can: the queue keeps emptying.

`pcg` now takes **`check_every`** and skips both the residual reduction and
the sync it feeds in between. **Default 1**, so every CPU path stays
bit-identical; `scratch/tgv_gpu_run.py` defaults to **10** on GPU. On the
Spark: 2.44 → 0.96 ms per iteration, **2.5×**, while the iteration count moved
894 → 900 — six extra iterations out of nine hundred, 0.7%, which is the whole
price.

### CuPy vs PyTorch on the same A100: a tie

Settled by measurement, in separate processes (two GPU allocators in one
process interfere, and library init would be charged to whichever ran second).
`scratch/backend_shootout.py`, 18.87 M dof:

| | CuPy 14.0.1 | torch 2.11.0 |
|---|---|---|
| `normal_op` | 9.04 ms | 9.99 ms |
| **CG iteration** (differenced) | **12.04 ms** | **12.72 ms** |
| `to_complex` | 0.47 | 0.47 |
| gather-scatter | 0.61 | 0.65 |
| elementwise `U*mask` | 0.34 | 0.33 |
| `_dot` as the solver calls it | 0.69 | 0.68 |
| **raw reduction** `sum(axis/dim=0..3)` | **9.30** | **0.68** |

**6% apart on the real solver, and every primitive matches within noise.** The
one genuine library difference is the raw reduction, where CuPy is **13.7×**
worse — and it disappears entirely once written as a GEMM: `_dot` is 0.69 vs
0.68 ms. CuPy's weakness here is real but exactly one line of code wide.

#### Net-net: which is faster?  torch — via `torch.compile`

**On raw primitives, neither.** Every primitive above matches within noise.
The one real difference — CuPy's raw reduction at 13.7× torch's — is removable
in a single line, after which `_dot` is 0.69 vs 0.68 ms.

**But `torch.compile` settles it, and not in CuPy's favour.**
`scratch/torch_compile_experiment.py` wraps the inner `_apply_L`/`_apply_LT`
in `torch.compile` — no other change to `kernels_torch.py`, a module that
deliberately contains no fusion:

| | matvec | CG iteration |
|---|---|---|
| torch eager | 9.60 | 12.85 |
| **torch compiled** | **6.07** | **9.42** |
| CuPy, hand-written Phase 6 | 6.40 | 9.62 |

**Inductor found everything Phase 6 found, and about 5% more** — in **20 s of
compile time**, against roughly a day of hand-written `ElementwiseKernel`
work. Correct to 2.26e-16 against eager. Complex128, which was the expected
blocker, was not a problem at all.

That inverts the prediction recorded here before the experiment ran (this
document said compile failure was the likely outcome) and it makes most of
Phase 6 redundant *on the torch path*.

**In the production driver the two are a dead heat.** The isolated benchmark
is not the number to plan with, so both were re-measured through
`scratch/tgv_gpu_run.py --price` on the same case, same session:

| | ms/iteration | s/step | Re = 1600 128³ |
|---|---|---|---|
| **torch + `torch.compile`** | **10.10** | 48.0 | **68.4 h** |
| CuPy, hand-written Phase 6 | 10.29 | 48.8 | 69.6 h |

**1.9% apart — noise.** Both are ~0.6 ms slower here than in the isolated
shootout, consistently, because the driver applies the Jacobi preconditioner
that the shootout passed as `M_inv=None`.

Two things that gap teaches. `torch.compile` wraps only `_apply_L`/`_apply_LT`,
so the convective term with its FFTs, the RHS assembly and gather-scatter run
uncompiled — **compile the hot function and you win the microbenchmark; the
production number depends on what fraction of the step that function actually
is.** And a 1.36× on an isolated CG iteration became 1.02× end to end, which
is the difference between a benchmark result and a run you can plan two days
around.

The honest conclusion:

**On speed, they tie — so choose on effort, and there torch wins outright.**
68.4 vs 69.6 h is not a reason to prefer either. But CuPy reached its figure
through roughly a day of hand-written `ElementwiseKernel` fusion, metric
folding and LᵀL fusion, all of which must now be maintained; torch reached the
same figure with **one line and 20 s of compile**, and improves with the
compiler rather than with our attention. CuPy's advantage —
`ElementwiseKernel` making hand-fusion easy — turns out to be an advantage at
solving a problem torch does not have.

Caveats worth keeping: compile time is per-process, so a chained-session run
pays it once per session (negligible against 65 h); a shape change triggers
recompilation, which is fine for fixed-size runs; and a compiler regression
across versions is a real risk that a hand-written kernel does not carry.

CuPy remains worth keeping as the **reference and parity backend** — it runs
NumPy code unchanged via `__array_function__`, which is exactly what made the
2.613e-16 parity check cheap to build.

So: **pick torch for speed, and keep CuPy for parity.** The earlier advice in
this document to "pick on ergonomics, not speed" was written before this
experiment and is superseded.

| | lever it gives you |
|---|---|
| CuPy | `ElementwiseKernel` / `RawKernel` — writing a custom fused kernel is a few lines, which is what Phases 4b and 6 used. Drop-in NumPy semantics via `__array_function__`, so most code runs unchanged. |
| torch | `torch.compile`, which could plausibly perform the Phase 6 fusions automatically — **untested here**, and the obvious next experiment for that port. Plus the whole ecosystem. |

One caveat if you choose CuPy: **its reductions have a real weak spot.** A
290,304:1 reduction to 65 outputs cost 17× its bandwidth bound, and nothing
about the code looked wrong. Time your reductions rather than assuming.

Two further results worth keeping:

- **Fusion buys only 6% here.** The CuPy path has hand-written fused
  `ElementwiseKernel`s; `kernels_torch.py` is *deliberately* unfused. At
  18.87 M dof that is worth almost nothing, because fusion mainly saves
  dispatch and the loop is bandwidth-bound. Consistent, not disappointing:
  the same fusion was worth **6.2×** when the loop *was* dispatch-bound.
- **The 4-D contraction merge is worth nothing at this size — on either
  backend** (1.01×). `kernels_torch.py` records merging the field and mode
  axes for ~1.9× at 88³ and warns against reverting it without
  re-benchmarking; this *is* the re-benchmark, and on torch 2.11 at 18.87 M
  dof the two forms are identical. That note may be stale or
  version-dependent. A 1.91× *did* appear in a tiny smoke case — which is a
  dispatch artifact, and a good reminder that a micro-benchmark at the wrong
  size will confidently tell you the wrong thing.

### Phase 6: folding the metric scaling into the kernels

`ddx` is `einsum(...)*fac`, and that trailing multiply is a whole extra pass
over a complex array — ~576 MiB of traffic per call, four calls per matvec.
It measured 1.08 ms against 0.55 for the bare contraction, so roughly half of
every derivative call was the scaling, not the derivative.

The fused kernels already read every one of those values, so the `fx`/`fy`
multiply moves inside them, where it is a register operation and free. The
CuPy path now calls the bare contraction and passes the metrics through.

Two more wasted passes came from calling *correct* pieces in sequence.
`normal_op` calls `apply_L` then `apply_LT` back to back — and `apply_L` ends
with `_to_real(R)` while `apply_LT` opens by converting straight back, a whole
real→complex→real round trip per matvec. `apply_LTL` does both while staying
complex in between. And the row and quadrature weights were two further full
passes over an 8-row complex array *after* the kernel had already written it;
they are now kernel parameters, declared as scalars so an unweighted call
passes `1.0` and reads nothing rather than streaming an array of ones.

| | normal_op | CG iteration |
|---|---|---|
| batched + fused rows | 9.04 | 12.04 |
| + metric folding | 8.01 | 10.68 |
| **+ LᵀL fusion, weights in-kernel** | **6.40** | **9.62** |

Parity **2.613e-16 throughout**, 167 tests, and gate 1's σ bit-identical to
the pre-change run.

### Where it ended, and why to stop

**248 h → ~65 h, ~3.8×**; 20 chained Colab sessions down to about 6.

| | ms/iteration | Re = 1600 128³ |
|---|---|---|
| first A100 run | — | 248 h |
| CG convergence test every 10 | 30.6 | 212 h |
| **inner product as a GEMM** | **12.53** | **84.6 h** |
| metric folding | 10.68 | 72 h |
| LᵀL fusion + weights | **9.62** | **~65 h** |

What remains is diffuse: `to_complex` 2.1× off bandwidth, gather-scatter
1.8×, `_dot` 2.1×, derivatives ~2.5×, the matvec 2.9×. **No single anomaly is
left** — every large win today was a 12–17× outlier, and no comparable one is
visible. Elementwise ops sitting at exactly **1.0×** is the evidence that the
card is being driven properly, rather than that measurement has stopped
working.

The one structural idea left is abandoning split-real packing and carrying
complex through the whole operator, killing the remaining conversions. That is
a format change across the operator and the BC/mask machinery — and the first
change that would genuinely threaten the 2.613e-16 that has held through
everything above.

### One honest caveat on the batching fix

Batching trades dispatches for **strided field views**. That is a clear win on
a fast GPU behind a slow host (Colab) and may be a *loss* on a bandwidth-bound
device like the GB10, which is work-bound at production size and gains little
from fewer launches. The Spark A/B that would settle it is **contaminated** —
the GPU was at 95% running the parallel session's `minchan` job when it was
taken — so it is deliberately not quoted here. If a clean re-measure shows a
regression, the batching should become conditional rather than unconditional.

## Preconditioning: what is and is not worth building

Three avenues were measured rather than argued about. Two are closed; the
third is open, but only above a certain polynomial order.

**A better preconditioner at N = 8: no.** At 16×16 N=8 Nz=128 the current
scalar-diagonal Jacobi takes 920 stage-0 iterations against ≥40000
unpreconditioned — worth at least **43.5×**. It is already doing the heavy
lifting at this order.

**Row weights: no — the gain is bought with continuity.** `mom ×10, vort ×0.1`
takes CG from 920 to 290 iterations, 3.2×, and passes the entire validation
ladder with *bit-identical* σ and balance. It is still rejected: a direct
divergence check (the L2 norm of row 0 of L0, which **is** div u) shows
2.572e-06 → **1.473e-04, 57× worse**. Energy, enstrophy and the parameter-free
balance are all blind to it. `ROW7_WEIGHT = 1e-4` is confirmed optimal and is
worth **21×** at production scale (w7 = 1.0 costs 19430 iterations against 920).

**Above N ≈ 12: yes, and this is the one that matters.** A single-N headroom
measurement cannot see it — what counts is the *slope*. Point-Jacobi removes
metric and element-size variation but does not touch the N-dependence of the
spectral-element condition number; fast diagonalisation, element block-Jacobi
and overlapping Schwarz exist precisely to give near-N-independent convergence.

Sweeping N at **fixed dof** (mesh coarsened as N rises, total held within ±13%
of ~1.5 M) isolates conditioning from size:

| N | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|---|
| Jacobi its | 320 | 360 | 510 | 650 | 840 | 950 | **1260** |
| Jacobi benefit vs none | 86.8× | 70.2× | 58.5× | 53.4× | 71.4× | 63.2× | **47.6×** |

**Iterations scale as N^1.01 — 3.9× over N = 4 → 16, at constant dof.** (The
fixed-*mesh* sweep gives N^1.80, but dof grew 11.6× over the same range, so
roughly half that exponent was size.) The second row is the corroboration:
Jacobi's *relative* benefit erodes as N rises, which is what a preconditioner
that ignores N-dependence does.

Assuming an ideal N-independent preconditioner holds iterations at the N = 4
level and costs ~1 extra matvec per application:

| N | ideal gain | net after cost |
|---|---|---|
| 8 | 1.6× | **0.80× — a loss** |
| 10 | 2.0× | 1.02× |
| 12 | 2.6× | 1.31× |
| 16 | 3.9× | **1.97×** |

**Crossover at N ≈ 10–12** — independently matching the project's existing
observation that PMG excels above N ≈ 12. Both assumptions are optimistic, so
the true crossover is likely N ≈ 12–14.

So **FDM is not a speedup for current runs; it is what makes high-order runs
affordable.** Since spectral accuracy is the entire reason to use LSSEM, that
is an enabler rather than an optimisation — and it is a better argument for
building it than any factor measurable at N = 8.

On the GPU, FDM is also the right *shape*: batched small dense multiplies over
elements and Fourier modes, no coarse level, and therefore none of the
dispatch-floor penalty that costs PMG its advantage here (the 2D study measured
p-MG cutting iterations 9.9× for ~14 matvecs, a net 1.25×, and coarse levels do
not get proportionally cheaper on a machine with a per-matvec host floor).

### FDM was built, and it does not work

The open question was never whether the conditioning justifies it but whether
FDM *applies*. It applies **exactly** — every field-diagonal block is a sum of
tensor products to machine precision (`scratch/fdm_structure.py`), so fast
diagonalisation inverts each one exactly. It still loses, measured on the A100
with a working symmetry gate (`scratch/fdm_bench.py`):

| N | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| Jacobi | 80 | 140 | 230 | 360 | 530 | 760 |
| FDM | 170 | 270 | **2050** | **3250** | 1330 | 1960 |

**Jacobi N^1.80, FDM N^2.19** — worse at every order by 2–9×, and a worse
slope. Multiplicity-weighted (symmetric) Schwarz helps modestly and changes
nothing structural.

The erratic pattern is itself the diagnostic. A preconditioner capturing the
right structure degrades smoothly; quality swinging 5× between adjacent orders
means something essential is missing. FDM inverts each field's **spatial**
operator exactly while dropping the field coupling — 37% of the operator by
Frobenius norm — and the coupling between velocity, vorticity and pressure *is*
the difficulty of the VVP system. Point-Jacobi's diagonal comes from the full
assembled operator, so its magnitudes reflect the coupling even with no
off-diagonal structure. **FDM is exact about the part that was never the
problem and blind to the part that is.**

**Worth keeping regardless of the negative result:**

- Every field-diagonal block is separable to machine precision — reusable if
  anything ever needs an exact element solve.
- **Continuity.** CG's vectors live in discontinuous element storage but are
  constrained to the continuous subspace, since `normal_op` ends with
  gather-scatter. An element-local preconditioner maps continuous input to
  **discontinuous** output, and the iterates then leave the subspace the
  operator is defined on. Adding the gather-scatter — making it additive
  Schwarz — took this from 40000 iterations to 170. It applies to *any*
  element-local preconditioner built here.
- **The legacy 1/c² momentum weighting is load-bearing**: it cancels the c²
  from `c*u` and leaves a clean unit mass term, which is what makes the
  momentum blocks separable at all.
- Symmetry must be checked **in the space and inner product CG uses** —
  continuous probes, multiplicity-weighted. Random probes under a plain sum
  report ~1e-2 for an operator symmetric to 1e-17, and a gate that cries wolf
  every run is worse than no gate.

**If revisited, the evidence points the other way**: a 14×14 **nodal field
block** — exact about the coupling, crude about space — is the opposite trade
from FDM. Note that `3D_STATUS.md` §7K already records block-Jacobi as
measured with *null effect*, and §7L already rejects fastdiag for the
inter-field couplings — this session rediscovered that result rather than
finding it. **Read §7K/§7L before touching preconditioning here.**

### PMG re-audited with skeptical eyes — the implementation is sound

Prompted by "should we consider 3-level PMG?" — which the code already
supports (`orders=(8, 4, 2)`). Five things were checked and all pass:

| checked | result |
|---|---|
| `np.linalg.pinv` truncating the coarse operator | **No** — κ 5e6–3.4e8, full rank every mode |
| Chebyshev spectral bound λ_max | **Adequate** — converged 3.247 vs 20-iteration 2.984, ratio 1.088 inside the 1.3 safety factor |
| Chebyshev4 recurrence | matches Lottes' 4th-kind form |
| V-cycle structure | correct; restriction is the adjoint, multiplicity-weighted |
| `DirectCoarse` | genuinely exact, built in the global continuous basis |

**So PMG's pinned 7.4× is not a bug.** The reason is structural, and §7K.2 had
it: **the slow modes are ROUGH.** Multigrid's premise is that slow modes are
smooth and therefore coarse-representable. When they are rough, the coarse grid
cannot touch them and the smoother is already doing all it can — its bound is
verified adequate above. **Adding a third level makes cheaper the end that was
never the bottleneck**, which is a stronger argument against it than the
wall-clock one.

It also explains why plain Jacobi + CG is hard to beat here: **CG's own
polynomial adapts to the actual spectrum**, which is the right response to
rough slow modes, and it costs one matvec.

Three non-defect limitations, worth knowing before anyone invests:
`DirectCoarse` setup is O(dof) gather-scatter calls (fine at 700 dofs/mode,
unusable at production scale); there is **no GPU path** — `pinv`, the basis
construction and the setup loops are host-only; and intermediate levels use
rediscretised operators while the coarsest uses Galerkin.

## Running long jobs: `scratch/tgv_gpu_run.py`

Colab has no batch queue and every VM eventually dies, so a long run is a
**chain of sessions**: each resumes the last checkpoint, works until its
`--budget` is nearly spent, checkpoints, and exits saying how much compute
remains. Re-running the launch cell continues; it never restarts from t = 0.

**Restarts are exact, for a specific reason.** RKW3's `ZETA[0] = 0`, so the
convective history `N_prev` is multiplied by zero at the top of every step and
carries **no information across step boundaries**. A checkpoint therefore
needs only the state and the time — no history, no stage index. `--selftest`
proves this rather than asserting it, comparing 6 straight steps against
3 + checkpoint + 3:

| backend | max abs difference |
|---|---|
| numpy (deterministic) | **0.000e+00 — bit-exact** |
| CuPy | 8.7e-11 relative |

The CuPy figure is that backend's scatter-add atomics, which the run has
anyway, restart or not — not a restart defect.

Checkpoints are written to a temp name and renamed, so a file is complete or
absent, never half-written; one generation of history is kept so a crash
mid-write cannot destroy the only copy; and a configuration fingerprint is
stored and checked, so resuming into a mismatched `--outdir` is a clear error
rather than a silently wrong run. Diagnostics append to CSV across sessions.

`--price` reports **iterations *and* ms/iteration**, per stage, plus GPU pool
and device memory. Reporting only their product is what let a 46 h projection
become a measured 248 h without anyone noticing which factor was wrong: too
many iterations is a conditioning problem, a slow iteration is a memory or
dispatch one, and they have opposite remedies. The driver also **names the GPU
and warns on a card without a fast FP64 path** — every timing here is FP64 and
bandwidth-bound, A100/H100/V100 run it at ~1/2 the FP32 rate while T4 and L4
run it at 1/32 to 1/64, and Colab does not always give you the card you asked
for.

`--outdir` for `--price` should be a scratch directory: pricing advances the
state a few steps without recording them.

## Diagnostic tools written for this work

Each exists because a prediction failed, and each is reusable:

| script | answers |
|---|---|
| `cupy_parity.py` | does the CuPy operator match numpy (2.613e-16) |
| `cupy_validation_ladder.py` | gates 1–3: Stokes σ, rotated TG, energy balance |
| `cupy_dispatch_profile.py` | how much of a matvec is host dispatch |
| `cupy_matvec_profile.py` | where a CG iteration's GPU time goes, **with the bandwidth bound beside each component**, and the true per-iteration cost by **differencing two solves at different `max_iter`** — everything per-iteration survives the difference, everything per-solve cancels |
| `cupy_dot_variants.py` | five ways to compute the per-mode inner product |
| `torch_dot_variants.py` | whether torch pays the same reduction cost (it does not) |
| `backend_shootout.py` | CuPy vs torch, real solver and primitives, one GPU |

**The differencing measurement is the one to reach for first.** It required no
model of the code, took two minutes to write, and would have pointed straight
at the inner product — instead of which it arrived after four wrong
predictions built on a matvec benchmark that was, the whole time, correct.

## Next

1. **Deterministic mode** for CuPy, mirroring `DEV.deterministic()` on the
   torch path — a sort-based or sparse-matmul gather-scatter would restore
   bitwise reproducibility for parity work, at a cost only paid in testing.
2. ~~Benchmark on an A100~~ — **done**; see the phases above. The production
   figure is a **9.62 ms CG iteration at 18.87 M dof**, and the Re = 1600 128³
   case at **~65 h**. The 1.6× whole-solve figure earlier in this document is
   GB10 FP64 against Grace-core NumPy and is *not* a production number.
3. Merge coordination with the torch/CUDA port. Two findings belong to that
   session rather than this one:
   - **There is nothing to fix in `_dot` on the torch path.** The GEMM form
     buys it 0.79 vs 0.78 ms — nothing. Torch's reductions already handle the
     shape.
   - **`kernels_torch.py`'s 4-D contraction merge does not reproduce.** It
     documents ~1.9× at 88³ and warns against reverting without
     re-benchmarking; at 18.87 M dof on torch 2.11 the 4-D and 5-D forms are
     **1.01×** — identical — on *both* backends. That note may be stale or
     version-dependent. (A 1.91× did appear in a tiny smoke case, which was a
     dispatch artifact — a good reminder that a micro-benchmark at the wrong
     size will confidently tell you the wrong thing.)
4. ~~`torch.compile` on `kernels_torch.py`~~ — **done, and it wins**: 12.85 →
   **9.42 ms** per iteration, past the hand-written CuPy path's 9.62, for 20 s
   of compile time and no code change — **but only 1.02× in the production
   driver** (68.4 vs 69.6 h), because the compiled region is a smaller share of
   a full RK stage than of an isolated CG iteration. `--backend torch` is now
   wired into `scratch/tgv_gpu_run.py` and verified bit-exact on restart. The
   remaining follow-ups are `mode="max-autotune"` (untried) and extending
   `torch.compile` to the convective path, where the FFT-adjacent elementwise
   work is exactly what inductor fuses well and is where torch's uncompiled
   remainder sits.
5. The split-real format. Dropping it and carrying complex through the whole
   operator would remove the remaining `to_complex`/`to_real` traffic, worth
   perhaps 1.15× — and would be the first change to genuinely threaten the
   2.613e-16 parity that has held throughout.
