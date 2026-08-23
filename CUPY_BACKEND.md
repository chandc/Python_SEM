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

Suite: 166 passed, 36 skipped, 1 failed — and that failure
(`test_deriv.py::test_trailing_axes_are_independent`) **reproduces on the
unpatched baseline in the same container**, so it is an ARM/NumPy-version
environment difference, not a regression from this work.

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

**Backend choice is therefore not a performance decision.** Pick on
ergonomics: torch for ecosystem and `torch.compile`; CuPy for drop-in NumPy
semantics and easy custom kernels.

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

## Next

1. **Deterministic mode** for CuPy, mirroring `DEV.deterministic()` on the
   torch path — a sort-based or sparse-matmul gather-scatter would restore
   bitwise reproducibility for parity work, at a cost only paid in testing.
2. Benchmark on an **A100** (Colab), where FP64 is 46× the GB10's — that is
   where a production number can honestly be quoted. The 1.6× whole-solve
   figure above is GB10 FP64 against Grace-core NumPy and is *not* a
   production number.
3. Merge coordination with the torch/CUDA port: the shared-file surface is the
   four small hunks listed above, all additive.
