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

## Next

1. **Deterministic mode** for CuPy, mirroring `DEV.deterministic()` on the
   torch path — a sort-based or sparse-matmul gather-scatter would restore
   bitwise reproducibility for parity work, at a cost only paid in testing.
2. Re-run the **validation ladder** (Stokes σ, rotated-TG order 2.00, TGV
   balance) end to end on the CuPy path. Phase 2 above proves one stage; the
   ladder proves the physics.
3. Benchmark on an **A100** (Colab), where FP64 is 46× the GB10's — that is
   where a production number can honestly be quoted. The 1.6× whole-solve
   figure above is GB10 FP64 against Grace-core NumPy and is *not* a
   production number.
