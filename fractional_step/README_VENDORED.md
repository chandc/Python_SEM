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
