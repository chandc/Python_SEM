# The fractional-step (projection) solver, vendored for a like-for-like comparison

This directory is a **source-only copy** of the sibling code base that was used
to produce the fractional-step reference runs quoted throughout the parent
repository — `results/minchan_re180_K` (weak-Laplacian projection) and
`results/minchan_re180_E` (consistent $P_N$–$P_N$ projection).  It is here so
that the comparison in Section 10 of the paper can be run **on the same machine,
mesh, box, time step and statistics machinery** as the least-squares code,
rather than quoting numbers measured on different hardware months apart.

It is a separate solver that happens to share ancestry with the parent
repository: both descend from the same spectral-element core, both define the
minimal channel identically (`RE_TAU=180`, `DELTA=1`, `LX=pi`, `LZ=0.34*pi`,
`FX=1`, `N=8`, `6x18` elements, `nz=32`), and the statistics collector here,
`scratch/fs_minchan_stats.py`, is the direct ancestor of the parent's
`scratch/minchan_stats.py`.  The two therefore produce `stats` archives with an
identical schema, which is what lets one analysis script score both.

## What is and is not here

**Included:** every `.py` (377 files), every `.md` (42), the TOML configs and the
Dockerfile — 3.7 MB.

**Excluded:** the run archives and figures that made up the other 286 MB of the
original tree (`_fs800*`, `_km800`, `figs/`, `verification_plots/`,
`scratch/_minchan_*`, `scratch/tgv_*`), plus `reference/` and `grids/`, which are
already in the parent repository.  The restart states live with the data, not
here: the statistically stationary E-path field is `chk_latest.npz` in the
original `scratch/_minchan_stat_E/`, and the parent repository keeps the archived
window in `results/minchan_re180_E/`.

**The `.md` files duplicate names in the parent** (`3D_STATUS.md`,
`CHANNEL_VALIDATION.md`, …) but are *different documents* describing *this*
solver.  They are kept because they are the provenance of these numbers.

## Selecting the consistent (E) path

The projection operator is chosen by a flag in the setup dictionary,
`s['consistent_p']`, read in `lssem3d/project.py`:

* **unset** — the weak-Laplacian path (K).  The pressure Poisson operator is the
  assembled Laplacian; cheaper by 2–5×, and the weak divergence is uncontrolled.
* **set** — `project_consistent()`, the $P_N$–$P_N$ consistent projection with
  $E = G^{\mathsf T}M^{-1}G$, inverting the same operators the velocity update
  uses, so the **weak** divergence cancels identically ($\sim10^{-6}$) instead of
  only approximately.

The statistics driver exposes it as a command-line switch:

```
python scratch/fs_minchan_stats.py --consistent --restart FILE \
       [--backend cupy|torch] [--dt 3.5e-4] [--tend 28] [--outdir DIR]
```

**Note what the consistent path does and does not fix.**  It controls the *weak*
divergence and removes the spurious pressure work.  The *pointwise* divergence
is unchanged: the E-run's own log carries `div = 1.4e-01` at $t=3$ rising to
`2.2e-01` at $t=16$, against $9.1\times10^{-4}$ for the least-squares run on the
same mesh.  That is structural — a projection method enforces
$\int q\,\nabla\!\cdot\!\mathbf u = 0$, and the choice of discrete $E$ changes how
exactly *that* holds, not whether the pointwise divergence is small.

## Profiling (2026-09-15)

`colab/fs_profile.py` in the parent repository profiles this solver without
reimplementing its setup: it wraps `CV.convective`, `HH.solve` and
`project_consistent` on their module objects and then runs
`scratch/fs_minchan_stats.py` unchanged through `runpy`, so what is timed is the
production path.  Every phase boundary synchronises the device, since an
unsynchronised timer on an asynchronous backend measures launch overhead.

**Host-side result** (numpy, 3 steps, E path, production mesh):

| phase | s/step | share | iterations/call |
|---|---|---|---|
| convective | 0.21 | 0.2 % | — |
| velocity Helmholtz | 0.67 | 0.8 % | 6 |
| **pressure (consistent E)** | **85.0** | **99.0 %** | **82** |

The projection path is one solve.  The velocity Helmholtz is already excellent
at 6 iterations under the FDM preconditioner, and the explicit terms are free.
The 82 + 6 iterations per substage also reconcile exactly with the `CG=250-300`
logged by the E production run, which sums both solves over three substages.

That leaves a single optimisation target, and `lssem3d/epmg.py` explains why it
is hard: $E = G^{\mathsf T}M^{-1}G$ is a mass-weighted Schur complement, not the
assembled Laplacian, and a $K$-based V-cycle preconditions it at **465**
iterations against ~15 for $K$ on its own system.  Building the V-cycle on $E$ at
every level brought that to 82.  `colab/fs_profile_a100.ipynb` repeats the
profile on an A100 and sweeps the pressure tolerance to separate a
tolerance-bound solve from a stagnating V-cycle.

### The A100 result, and why it is not a statement about the formulation

| | s/step | pressure it | h per eddy turnover |
|---|---|---|---|
| E path (consistent) | 19.96 | 88 | 15.84 |
| K path (weak Laplacian) | 3.21 | 20 | 2.55 |
| least squares, same GPU | 1.62 | 84 | 0.56 |

Two measurements explain the gap, and neither concerns projection versus
least squares.

**The V-cycle converges; it is merely weak.** 60, 89 and 120 iterations at
tolerance $10^{-3}$, $10^{-4}$, $10^{-5}$ — thirty per decade, a per-iteration
reduction of 0.926.  Geometric convergence, not the stagnation the E-run's stall
had suggested.  A healthy multigrid takes 1–3 iterations per decade, and the K
path already reaches 20 on its own operator, so ~4.5× is available here.

**The solve is launch-bound, measured rather than inferred.**  The V-cycle holds
the same operator $E$ at three polynomial orders, so timing it at each separates
work from overhead — work scales with degrees of freedom, launches do not:

| order | dofs/mode | ms |
|---|---|---|
| 8 | 17,496 | 2.371 |
| 4 | 5,400 | 2.404 |
| 2 | 1,944 | 2.384 |

**Flat to 1.4 % across a ninefold range.**  The order-2 level does one ninth of
the work in the same time.  Against a generous 33 MB of traffic the A100's
bandwidth floor for the fine apply is 0.021 ms, so 2.37 ms is 111× above it and
the flatness says that factor is dispatch.  It also explains the otherwise
absurd observation that the A100 is 2.6× *slower* than the GB10 on identical
iteration counts: dispatch cost is a property of the host, and the GB10's
tightly coupled CPU issues kernels faster.

*(An earlier version of the probe compared `cupyx.profiler.benchmark`'s CPU and
GPU times and reported "device-bound".  That test cannot discriminate — CUDA
event intervals include the idle gaps between kernels — and the conclusion drawn
from it was withdrawn.)*

**The fix is smaller than the least-squares one was.**  `ConsistentPMG._v` is a
fixed, branch-free kernel sequence with no host reads: recursion of fixed depth,
Chebyshev of fixed degree, a dense coarse solve.  That is the ideal CUDA-graph
target, and capturing *the V-cycle alone* takes 72.6 of the ~75 ms per CG
iteration — so the CG loop itself, whose convergence test forces a host
synchronisation, need not be captured at all.  `cupy_graph.py` had to batch CG
iterations to work around exactly that; this does not.

Projected, at the measured iteration counts:

| | step | h per turnover |
|---|---|---|
| today | 19.96 s | 15.84 |
| V-cycle captured, 10× | 2.15 s | 1.71 |
| V-cycle captured, 20× | 1.16 s | 0.92 |
| and the iteration count fixed too | — | 0.31 |

Against the least-squares path's 0.56, the consistent projection could plausibly
end up **cheaper**.  Publishing the raw column above would report how much
optimisation each code has had.

### What the optimisation recovered (2026-09-16)

| | s/step | h per eddy turnover |
|---|---|---|
| E path, cupy, as vendored | 19.96 | 15.84 |
| E path, torch | 11.22 | 8.90 |
| **E path, torch + captured V-cycle** | **1.91** | **1.52** |
| K path, cupy | 3.21 | 2.55 |
| least squares, same GPU | 1.62 | 0.56 |

**10.5× on the consistent path**, in two independent pieces.

*The backend port, 1.78×.*  Moving to torch changed nothing about the algorithm
— 88 pressure iterations per substage and 10 velocity iterations, identical to
cupy — but cost fell uniformly across three phases doing unrelated work
(convective 1.81×, helmholtz 1.68×, pressure 1.78×).  A uniform factor across
unrelated work is per-launch overhead, which is the flatness measurement
confirmed a second way.

*The captured V-cycle, 5.9× more.*  The apply went 39.6 → 7.97 ms, and the step
10.87 → 1.91 s.  One V-cycle issues roughly 3,000 kernel launches; replay hands
the recorded graph to the device as one.  The iteration count is unchanged (88 →
87) because nothing about the algorithm changed — the same kernels in the same
order, agreeing with the uncaptured V-cycle to 2–6 × 10⁻¹⁶ relative, which is
round-off from cuBLAS choosing its algorithm differently under capture.

**Why torch and not cupy:** cupy refuses to record cuBLAS in a stream capture,
and every derivative here is an einsum, so there is no capturable subset.  That
is what the port was for.

### What this does to the comparison

Before this work the consistent projection looked 28× more expensive per eddy
turnover than the least-squares path.  It is now **2.7×**, and the remaining gap
is one number: 88 pressure iterations against the 20 the K path achieves on its
own operator.  A V-cycle on $E$ as good as the K path's on $K$ would put the
consistent projection at ≈0.42 h per turnover — *cheaper* than least squares.

So the honest statement for the paper is that **cost does not separate these
methods**; what separates them is the pointwise divergence (9.1e−4 against
1.4e−1 to 2.2e−1) and the vorticity accuracy, at comparable expense once both
codes have had comparable attention.

## Why the comparison is worth running

Costs from the E-run's own log (GB10, CuPy, $\Delta t = 3.5\times10^{-4}$):
12.95 turnovers in 78.6 h, i.e. **6.07 h per eddy turnover** at 7.65 s/step with
250–300 CG iterations in the pressure solve.  The least-squares code on the same
machine and mesh cost 8.82 h per turnover, and on an A100 with the condensed
vertex-patch preconditioner it now costs **0.56 h per turnover**.

So the cost gap between the two formulations is 1.45× on the one machine where
both have run — not the 3.4× quoted in the parent's
`FOSLS_VS_FRACTIONAL_STEP.md`, which predates the preconditioner work — and the
comparison on matched hardware may well reverse it.  That is the measurement this
directory exists to make possible.
