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

## Next

1. Device-resident **whole-solve** parity (`pcg` end to end), the analogue of
   the torch port's Phase 2 — the shim work above should already permit it.
2. Re-run the **validation ladder** (Stokes σ, rotated-TG order 2.00, TGV
   balance) on the CuPy path. A backend is not trusted until it re-passes the
   ladder.
3. Benchmark on an **A100** (Colab), where FP64 is 46× the GB10's — that is
   where a production number can honestly be quoted.
