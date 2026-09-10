# FP64 on the Apple GPU: what exists, what is battle-tested, what fits our need

**Question (2026-09-10).** The batched vertex-patch preconditioner
(BALANCED_CONDENSED_PLAN.md step 4.5) is now a set of dense fp64 triangular
solves and matrix products. On the Mac the Apple GPU has no double precision.
Is there an FP64 emulator for Metal that is battle-tested and fast enough to
run those solves on the GPU?

**Answer in one paragraph.** No. Apple GPUs have no FP64 hardware, Metal
exposes none, and every software route to IEEE double on the Apple GPU is
either unfinished (metal-float64, archived in the planning stage), unanswered
(PyTorch MPS and MLX feature requests, open since 2022 and 2025), or a
research technique without a Metal implementation (Ozaki-scheme GEMM). The
HPC projects that looked hardest at this (AMReX, Folding@home/OpenMM) ended
at the same place: FP64 stays on the CPU. That is also the right answer for
us on the Mac, and it is a better answer than it sounds, because the M3 Max
CPU's matrix unit runs our apply in double at 300–330 GFLOP/s — the same
order as an emulated-FP64 GPU would reach even if one existed. The production
fp64 engine is the GB10, which has native FP64 plus cuBLAS's Ozaki-scheme
emulation on its tensor cores.

---

## 1. The hardware and API facts

| item | state | source |
|---|---|---|
| Apple GPU FP64 | none, all generations through M4; FP32 and FP16/INT8 only | [Apple vs. Oranges, arXiv 2502.05317](https://arxiv.org/html/2502.05317v1); [metal-float64 README](https://github.com/philipturner/metal-float64) |
| Metal Shading Language `double` | not a type; no extension | same; [RWT thread](https://www.realworldtech.com/forum/?threadid=217891&curpostid=218072) |
| PyTorch MPS float64 | unsupported ("Cannot convert a MPS Tensor to float64 dtype"); emulation request #83264 open since 2022, "needs research", no maintainer response | [pytorch#83264](https://github.com/pytorch/pytorch/issues/83264); [Apple dev forum](https://developer.apple.com/forums/thread/797778) |
| MLX float64 | CPU only; `linalg` (cholesky, inv, eigh) fp32 only; GPU emulation request #1905 open since Feb 2025, no response | [mlx#1905](https://github.com/ml-explore/mlx/issues/1905); [mlx#1893](https://github.com/ml-explore/mlx/issues/1893) |
| OpenCL on Apple silicon | present but deprecated; no `cl_khr_fp64` on the Apple GPU; Folding@home runs its FP64 on the CPU | [Folding forum 2026 audit](https://forum.foldingathome.org/viewtopic.php?t=43450) |
| Vulkan/MoltenVK, WebGPU | no `shaderFloat64` on Apple GPUs; WebGPU has no f64 | [gpuweb#2805](https://github.com/gpuweb/gpuweb/issues/2805) |
| M3 Max CPU (this machine) | Accelerate DGEMM 406 GFLOP/s measured; `cho_solve` 137–150 GFLOP/s; NEON 352 / AMX 700 GFLOP/s peak measured on M1 Max | measured here (§4); [amx-benchmarks](https://github.com/philipturner/amx-benchmarks/blob/main/README.md) |

## 2. The software routes, ranked by maturity

### 2.1 Full IEEE emulation in Metal — metal-float64 (not usable)

Philip Turner's [metal-float64](https://github.com/philipturner/metal-float64)
is the only serious attempt at IEEE binary64 on the Apple GPU. State of the
project: **archived read-only on 2024-08-11, README says "still in the
planning stage – this is not a finished library"**. Implemented: add, mul,
fma; division, sqrt and transcendentals were roadmap items. Every operation
is a function call, not inlined. Its own throughput estimate is 1/32–1/64 of
FP32, i.e. 200–450 GFLOP/s on an M3 Max GPU (14 TFLOPS FP32) at best — the
CPU's AMX already delivers that. The author's earlier estimate in the PyTorch
issue was ~24 GFLOPS for genuine FP64 on an M1 Max, with a proposed 35-bit
"FP64 range, FP32 precision" compromise format instead. Nothing in the
library touches linear algebra.

### 2.2 Double-double / float-float arithmetic (mature technique, no library for our operation)

The "double-single" representation (a value as an unevaluated sum of two
fp32, ~48 significant bits, fp32 exponent range) goes back to Dekker (1971)
and Bailey's DSFUN90; it is the technique behind double precision in
WebGL/OpenGL fractal renderers and in the 2024 Vulkan visualisation paper
([arXiv 2408.09699](https://arxiv.org/pdf/2408.09699)), and ArrayFire
discussed it as a fallback ([arrayfire#1886](https://github.com/arrayfire/arrayfire/issues/1886)).
It is battle-tested **as an arithmetic**; what does not exist is a
double-double BLAS for Metal. An add costs ~10 fp32 ops, a multiply ~8 with
FMA, a triangular solve or GEMM in double-double would have to be written as
custom Metal/MLX kernels by us, with a throughput ceiling of roughly 1/10 of
FP32 (≈1.4 TFLOP/s on an M3 Max in the ideal case, realistically a fraction
of that), no denormals, and a 48-bit mantissa that is not IEEE double. Also,
what we need is a *solve* not a GEMM: a factor-and-solve in double-double
needs the renormalisation branches tensor-style hardware handles badly
(the point of [arXiv 2607.06881](https://arxiv.org/abs/2607.06881) for
NVIDIA). Feasible as a research project; not a drop-in, not battle-tested for
this use.

### 2.3 Ozaki scheme: FP64 GEMM from low-precision GEMM (production on NVIDIA, nothing on Metal)

The Ozaki scheme splits fp64 operands into slices, multiplies the slices with
INT8/FP8/FP16 tensor-core GEMMs and recombines them exactly. It is **shipped
in cuBLAS 12.9+** (`cublasSetEmulationStrategy`, fixed-point emulation with
dynamic mantissa control that falls back to native FP64 when the required
bits exceed the budget) and is the basis of the "150 TFLOPS emulated FP64
matrix" figure for Blackwell; the US DOE Genesis mission leans on it
([HPCwire, Feb 2026](https://www.hpcwire.com/2026/02/17/genesis-mission-will-lean-heavily-on-ozaki-scheme-for-fp64-capability/);
[NVIDIA blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/);
[cuBLAS 12.9 docs](https://docs.nvidia.com/cuda/archive/12.9.2/cublas/index.html);
[arXiv 2508.00441](https://arxiv.org/abs/2508.00441) for the FP8 variant).
This is the battle-tested FP64 emulator — **on NVIDIA**. On Apple silicon
there is no implementation, MLX/MPS expose no INT8 GEMM with INT32
accumulation to build one on, and it only covers GEMM, so our apply would
first have to be rewritten in GEMM form (§3). Relevant to us because the GB10
is Blackwell: if its native FP64 rate turns out limiting, cuBLAS emulation is
the switch to flip, not something we build.

### 2.4 Keep FP64 on the CPU (what every HPC code on Apple silicon does)

[AMReX](https://github.com/AMReX-Codes/amrex/issues/5151) evaluated Metal,
MoltenVK and SYCL for Apple GPUs and closed the issue "wontfix": FP32-only GPU,
no single-source path. Folding@home/OpenMM keep FP64 on the CPU. Apple's own
answer for doubles is the CPU: Accelerate dispatches DGEMM to the AMX
coprocessor (M1–M3) or SME (M4+), and on this M3 Max that is 406 GFLOP/s
double-precision GEMM, measured, from NumPy.

## 3. What our apply actually needs, measured on this Mac (M3 Max, fp64)

The batched preconditioner apply per CG iteration on the channel mesh is
17 modes × [Schur solve 1302² with ~100 right-hand sides + two interior
solves 686² with 108 + two 448×686 products], ≈1.2e10 flop.

| form of the apply | where | precision | time per apply | rate |
|---|---|---|---|---|
| batched `cholesky_solve` (current) | CPU, torch/Accelerate | fp64 | 171 ms (patch part) | ≈70 GFLOP/s |
| same shapes as plain `cho_solve` | CPU, scipy/Accelerate | fp64 | 81 ms for the Schur part | 137 GFLOP/s |
| **GEMM form: precomputed inverses × right-hand sides** | CPU, numpy/Accelerate AMX | fp64 | **30 ms** | 301 GFLOP/s (torch bmm: 332) |
| GEMM form | Apple GPU via MPS | fp32 | 3 ms | 1834 GFLOP/s |
| batched `cholesky_solve` | Apple GPU via MPS | fp32 | 164 ms | 68 GFLOP/s (MPS has no batched trsm) |

Two conclusions follow.

1. **On the Mac, fp64, the win is the GEMM form on the CPU, not the GPU.**
   Storing the inverses S⁻¹ and K_II⁻¹ (dense, same size as the triangular
   factors) and applying them as matrix products puts the apply on the AMX
   at 300+ GFLOP/s: about 30 ms against 171 ms now, in double, with no new
   dependency. For a preconditioner, applying an explicitly computed inverse
   is legitimate (the inverse is formed once from the fp64 Cholesky factor;
   its rounding error is 1e-13 relative, far inside what CG tolerates).
   Sharing means the inverses are ≈1.4 GB, the same as the factors.
2. **The Apple GPU would only help in fp32.** MPS `bmm` in fp32 runs the
   GEMM-form apply in 3 ms, 10× the CPU's fp64 rate, but MPS has no
   batched triangular solve worth using (164 ms) and no fp64 at all.

## 4. The precision question, stated precisely

The requirement is fp64 accuracy of the *solution*. In a preconditioned CG
the solution accuracy is set by the residual, the operator and the inner
products, all of which stay fp64. The preconditioner can be any SPD
approximation of A⁻¹; applying it in lower precision changes the number of
iterations slightly and nothing else, provided the CG recurrence is made
flexible (FCG, or a periodic true-residual restart, which `pcg` already does).
This is the standard mixed-precision arrangement: LAPACK's `dsgesv`,
PETSc/hypre fp32 preconditioners under fp64 Krylov solvers, MAGMA's
mixed-precision iterative refinement
([Haidar et al., PMC7735315](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7735315/)),
Carson & Higham's analysis. So the honest statement is: **keeping fp64 in
the solver is the requirement; fp64 in the preconditioner apply is a
choice.** If that choice is relaxed for the preconditioner only, the Apple
GPU becomes usable today through MPS fp32 GEMM (3 ms per apply); the
solution stays fp64 to the last digit. If it is not relaxed, the Mac path is
§3 item 1 and the GPU path is the GB10.

## 5. Recommendation

1. **Mac, fp64:** convert the batched apply to GEMM form with precomputed
   inverses on Accelerate (numpy/torch CPU). Expected 5× on the apply,
   ~30 ms per CG iteration, no new dependency, exact same preconditioner.
   Half a day.
2. **Production, fp64:** the GB10. Native FP64 first; measure the batched
   `cholesky_solve`/GEMM rates there; if the native rate is the limit, enable
   cuBLAS FP64 emulation (Ozaki, `CUBLAS_EMULATION_STRATEGY=performant`) for
   the GEMM-form apply — that is the one battle-tested FP64 emulator that
   exists, and it is already installed with CUDA ≥ 12.9.
3. **Do not build a Metal FP64 emulator.** Nothing exists to adopt, and the
   ceiling of anything we could write (double-double kernels, ≤ ~1 TFLOP/s
   ideal, non-IEEE) is close to what the CPU's matrix unit gives us for free
   in real IEEE double.
4. **Optional, your call:** an fp32 preconditioner apply on the Apple GPU
   under the fp64 solver (§4). It is the only way the Apple GPU contributes
   to this solve; it does not change fp64 solution accuracy; it needs a
   one-line device/dtype switch plus the flexible-CG guard.

Sources: [metal-float64](https://github.com/philipturner/metal-float64) ·
[metal-benchmarks](https://github.com/philipturner/metal-benchmarks/blob/main/README.md) ·
[amx-benchmarks](https://github.com/philipturner/amx-benchmarks/blob/main/README.md) ·
[pytorch#83264](https://github.com/pytorch/pytorch/issues/83264) ·
[mlx#1905](https://github.com/ml-explore/mlx/issues/1905) ·
[mlx#1893](https://github.com/ml-explore/mlx/issues/1893) ·
[AMReX#5151](https://github.com/AMReX-Codes/amrex/issues/5151) ·
[Folding@home 2026 hardware audit](https://forum.foldingathome.org/viewtopic.php?t=43450) ·
[Apple vs. Oranges (arXiv 2502.05317)](https://arxiv.org/html/2502.05317v1) ·
[Vulkan double-precision visualisation (arXiv 2408.09699)](https://arxiv.org/pdf/2408.09699) ·
[Ozaki FP8 DGEMM (arXiv 2508.00441)](https://arxiv.org/abs/2508.00441) ·
[Multiple-double on tensor cores (arXiv 2607.06881)](https://arxiv.org/abs/2607.06881) ·
[cuBLAS 12.9 emulation](https://docs.nvidia.com/cuda/archive/12.9.2/cublas/index.html) ·
[NVIDIA blog: FP emulation in cuBLAS](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/) ·
[HPCwire: Genesis mission and the Ozaki scheme](https://www.hpcwire.com/2026/02/17/genesis-mission-will-lean-heavily-on-ozaki-scheme-for-fp64-capability/) ·
[Mixed-precision iterative refinement (PMC7735315)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7735315/) ·
[gpuweb#2805](https://github.com/gpuweb/gpuweb/issues/2805) ·
[arrayfire#1886](https://github.com/arrayfire/arrayfire/issues/1886)
