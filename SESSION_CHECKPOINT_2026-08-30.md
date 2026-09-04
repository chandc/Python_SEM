# Session checkpoint — 2026-08-30 (pre-OS-upgrade)

Safe to reboot: no jobs running on Mac or Spark, git worktree clean,
all commits pushed to origin/fractional-step (HEAD cd4880f).

## Completed and closed
- **TGV Re=800 160^3** (Spark): peak dissipation 0.12% vs Gourianov. TGV_VALIDATION.md.
- **Kim-Moin RK4-CN**: closed as negative result (staggered-grid property). KIM_MOIN_REVIEW.md.
- **Minimal channel Re_tau=180 K-path** (Mac): 8,571-sample statistics, kappa=0.408, B=5.69,
  archived results/minchan_re180_K/.
- **E-path + operator A/B**: E-run stalled t=15.95 (E-solve conditioning), archived
  results/minchan_re180_E/. Matched-window A/B: all stats within sampling noise.
  Verdict + closed-form math: DIVERGENCE_AND_CONSISTENCY.md sections 7-8.
  **K-path = production, E-path = correctness instrument.**
- **BFS ladder**: outflow OBC v1 validated (Gates 1-3); corner-mask fix
  (gs(1-mask)>0.5); Gate 3 re-validated at N=9 (x_r fit 8.108 vs 8.11/8.145);
  **Tier 1b passed**: ER=2.0 Re=600 x_r/h = 10.2 +/- 0.1 vs Erturk 10.05 (+1.5%).
  BFS_VALIDATION_LADDER.md sections 6-8. Archives: results/bfs_armaly_re389/,
  bfs_er194_re600/ (OBC stress record), bfs_er200_re600/ (fit figure).
- **CPU stack**: GEMM derivs + mode freezing + mode pool = 3.6x (1.27 s/step channel).

## Open items (priority order)
1. **E-PMG sparse coarse level** — the E-solve fix. Proven: deg-6 coarse cuts
   878 -> 60 iterations; CuPy 14.2 on Spark (lssem-cupy:latest) has working
   splu (0.14s factor, 47ms/solve host round-trip, structured 20k dof);
   bicgstab/cg native GPU with rtol kwarg (not tol); spilu-per-iteration ruled out.
   Plan: assemble coarse E as CSR per kz at setup, splu factor once,
   batched multi-RHS coarse solves per V-cycle (or densify inverse -> GPU GEMM).
   Files: lssem3d/epmg.py (_EDirect, _ELvl). Then rerun E-path channel on Spark.
2. **Tier 2a Barkley null test** — BFS Re~600, L_z=6.9S, seeded 3D perturbation
   must DECAY (Re_c~748). Driver: scratch/gate_bfs_armaly.py (add --lz, seed).
3. **Dong OBC v2** (Theta-stabilizer) — only needed if exit backflow appears.
4. KM continuations (documented, unscheduled): x-y over-integration dealiasing.

## Machine/env notes
- Worktree: /Users/danielchan/sem_fs_wt (branch fractional-step,
  remote https://github.com/chandc/Python_SEM.git).
- Python: sem_demo/.venv (uv-managed; use `uv pip install --python <venv>`).
- Spark: ssh spark-b85b, docker image lssem-cupy:latest (CuPy 14.2.0, scipy 1.18.1).
- Reattachment sidecar: scratch/bfs_crossings_sidecar.py chk.npz out.csv ER N [Lout].
- Monitors/sidecars all stopped; nothing to resurrect after reboot.
