# Re = 1000 cavity, N = 15, time marching: what the long runs showed (2026-09-08)

`scratch/ghia_n15_run.py` (restartable driver: `--resume`, checkpoints every
`--ckpt` steps, both BDF history levels and cumulative records saved),
`scratch/plot_ghia_n15_ck.py` (checkpoints vs Ghia). Condensed vertex-patch
Schwarz + p = 2 coarse, rebuilt every 100 steps; 4×4 elements, N = 15.

## Runs

| run | dt | steps | t_end | CG it/step | s/step | rms vs Ghia at end (u, v) |
|---|---|---|---|---|---|---|
| `scratch/ghia_n15` | 1e−3 | 5000 | 5.0 | 12 (max 26) | 1.0 | 0.174, 0.236 (start-up transient, not steady) |
| `scratch/ghia_n15_dt1e-2` | 1e−2 | 9000 (resumed at 4789 with relative CG tol) | 90 | 10 (max 27), 7–8 at the end | 1.2 | 0.0122, 0.0164 (frozen since t = 50; max\|dU\|/dt = 6e−6 at t = 90, decaying as e^{−0.1t}) |
| reference: dt = 1 pseudo-time solve, N = 30 | 1 | 164 | — | — | — | 0.0032, 0.0067 |

Figures: `figs_fosls_vs_fs/ghia_n15_dt1e-3_checkpoints.png`,
`ghia_n15_dt1e-2_checkpoints.png`.

## Findings

1. **Solver:** 7–16 CG iterations per step throughout both runs, falling as
   the flow settles; ~1 s per step at N = 15 in numpy with 8 threads; 50
   rebuilds of 7.5 s each over 5000 steps. Restart validated (resumed final
   state equals the uninterrupted one to 1.6e−10).
2. **Time to steady state:** the rms distance to Ghia decays roughly as
   e^{−0.1t}; from rest, 40–60 time units are needed, i.e. 4000–6000 steps
   at dt = 1e−2 or 40,000–60,000 at dt = 1e−3. For a steady target the
   dt = 1 pseudo-time solve (164 steps) remains the right tool.
3. **A false "steady state":** with an absolute CG tolerance (1e−8), once
   the right-hand-side norm falls below it `pcg_solve` returns zero without
   iterating, dU = 0 exactly and any |dU|-based stop fires. Happened at
   step 4789 (true rate 1.1e−3). Use a relative tolerance (`--cgsfac`) for
   long marches; the run was resumed that way.
4. **The dt = 1e−2 march plateaus 4× further from Ghia than the dt = 1
   solve** (rms 0.012 / 0.017 vs 0.003 / 0.007; max centreline difference
   0.056 in u). The cause is a node-to-node oscillation in u under the lid
   (0.75 < y < 0.95 on x = 0.5): mean |Δu| ≈ 0.04, sign alternating at 7–8
   of 8 consecutive nodes, present from the first checkpoint at both dt.
   It is not the preconditioner: 20 steps restarted with Jacobi give the
   same profile and rms to four digits, and a 500-step Jacobi run from rest
   reproduces it. With the legacy weighting a_flux = dt the momentum row
   enters the functional at weight dt² (1e−4 at dt = 1e−2, 1e−6 at
   dt = 1e−3) against the constraint rows at 1, so a discretely
   divergence-free, curl-consistent oscillation is barely penalised — the
   dt-as-weight mechanism of POISEUILLE_DT_STUDY.md, now seen in a closed
   cavity.
5. **The time-accurate weighting (w_mom = w_mass = 1) is not a drop-in
   remedy here.** At dt = 1e−3 with Jacobi it blew up (max|u| = 3.6 by
   t = 0.1, 35,000 iterations per step); with the vertex patch at dt = 1e−3
   and at dt = 1e−2 it did not complete a single step in 8–18 minutes.
   That weighting changes the operator's block structure (momentum, and
   with it the u–p coupling, at weight 1 against constraints at 1) and the
   (u,ω)‖p patch/coarse preconditioner built for the legacy weighting no
   longer fits it. Both runs were stopped. The zigzag is therefore an open
   formulation item, not a solver one: candidates are a continuity-row
   weight (`w_con`), artificial compressibility on the continuity row
   (`dtau_p`, ARTIFICIAL_COMPRESSIBILITY.md), or the H⁻¹ momentum norm of
   ADN_FOSLS.md §6.5, each needing its own preconditioner analysis.

## Addendum: the momentum-weight curve at N = 15 (Stage A of H_MINUS_ONE_PROPOSAL.md)

| dt (= momentum weight) | 1 | 0.1 | 0.03 | 0.01 |
|---|---|---|---|---|
| rms u vs Ghia | **0.0095** | 0.0256 | 0.0218 | 0.0122 |
| rms v vs Ghia | **0.0101** | 0.0335 | 0.0288 | 0.0164 |
| zigzag under the lid | none | none | yes | yes |
| run | `scratch/ghia_n15_dt1` (223 steps) | `…dt0.1` (1521) | `…dt0.03` (4000) | `…dt1e-2` (9000) |

The N = 15 dt = 1 anchor corrects an earlier reading in this document that
used the N = 30 value (0.0032): at N = 15 the dt = 1 steady state is
0.0095, so the dt = 0.01 march (0.0122) is 30 % further from Ghia, not 4×.
The dependence is non-monotonic, and the oscillation is a separate effect
that switches on below dt ≈ 0.03. Verdict on the negative-norm remedy in
H_MINUS_ONE_PROPOSAL.md §7.

## Code review of the under-lid oscillation (2026-09-08)

Question: is the node-to-node oscillation in u under the lid at small dt an
implementation error? Everything new in these runs relative to the validated
cavity results was tested; no error was found. The evidence:

| check | result |
|---|---|
| **Independent implementation.** `scratch/hminus1_2d.py` re-implements the time step from scratch — assembled operator by element probes of `apply_L` (gate: equals `fosls_assemble` to 4e−16), own BDF loop, dense direct solve, no preconditioner, no `pcg_solve`, no `step_bdf`/`newton_step` | reproduces the oscillation with the legacy weighting at N = 12, dt = 0.1 (4 of 6 sign changes) |
| **Preconditioner independence.** 20 and 500 steps with Jacobi instead of the vertex patch | same profile, same rms to four digits, same oscillation |
| **Newton convergence.** 200 steps with 3 Newton sub-iterations per step from the dt = 1e−2 state: sub-iterations 2–3 take 1 CG iteration (residual already zero) | state unchanged to 1.6e−5, oscillation intact → a fixed point of the discrete nonlinear system, not an unconverged iterate |
| **Not an interface artefact.** Sampled on x = 0.30, 0.45, 0.49, 0.50 (element interface), 0.51, 0.55, 0.70 | present at every x with the same amplitude (mean \|Δu\| ≈ 0.039, 7 of 8 sign changes); U is C⁰ across elements (jump 0.0) |
| **Residual structure.** Steady residual rows of the dt = 1e−2 state vs the dt = 1 state: momentum per unit weight 0.54 vs 0.17, div 3.5e−3 vs 0.13, vorticity 4.4e−3 vs 0.10 | exactly the weighted compromise: constraints 30× better, momentum 3× worse; momentum residual 9× larger on element-edge nodes than interior at both dt |
| **Boundary conditions.** `apply_bc`: lid u = 1 constant in time, no ramp; corner regularisation unchanged | — |
| **Driver.** History list managed by `step_bdf` in place (the earlier double-update bug was in `pois2d_vs.py` only and is fixed); restart reproduces the uninterrupted run to 1.6e−10; relative CG tolerance for long marches | — |
| **Library change.** The only edit to `lssem2d` in this work is the optional `state.precond_factory` hook in `newton_step` (Jacobi default unchanged) | `lssem2d/tests` re-run, see below |
| **Prior evidence.** WEIGHT_VS_TIMESTEP_STUDY.md §5 (`cavity_dt.py`, N = 8): rms vs Ghia 0.066 / 0.083 / 0.068 / 0.046 / 0.039 / 0.234 at dt = 0.05 / 0.1 / 0.5 / 1 / 2 / 5 — the same non-monotonic dt-dependence measured here at N = 15 (0.0122 / 0.0218 / 0.0256 / 0.0095 at dt = 0.01 / 0.03 / 0.1 / 1); profiles were not inspected then. ARTIFICIAL_COMPRESSIBILITY.md §5.3 recorded a spurious spatially-oscillatory fixed point of the steady form that the functional could not distinguish from the correct one | the phenomenon was in the code's behaviour before; it was not looked for under the lid |

Verdict: the oscillation is a genuine steady solution of the discrete
least-squares problem at momentum weights ≪ 1 — a discretely
divergence-free, curl-consistent mode that the momentum rows are too weak
to suppress — reproduced by two independent implementations and stable
under converged Newton. It is a formulation property (§4–5 above and
H_MINUS_ONE_PROPOSAL.md §7), not a bug in the solver, the preconditioner,
the driver or the boundary conditions.

## Fortran cross-check (2026-09-09): the original code reproduces the oscillation

`F90_SEM/pmg_clean/SEM_2D_PMG_CLEAN` (the original Fortran LSSEM cavity
driver: legacy weighting, BDF, p-multigrid) run on the same 4×4 order-15
cavity (grid written in its format by `scratch/mesh_cavity_f90.py`, lid
code 2 on the north face), Re = 1000, dt = 0.01, 1000 steps to t = 10, CG
relative tolerance 1e−6 (namelist `cavity_n15/in_dt1e-2.nml`, field
`cavity_n15/r_dt1e-2.dat`, read with `scratch/fsol.py`, elements matched by
corner coordinates). Compared with the Python checkpoint at the same t and
dt (`scratch/f90_cavity_compare.py`; figure
`figs_fosls_vs_fs/f90_vs_python_cavity_dt1e-2_t10.png`):

| | Fortran | Python |
|---|---|---|
| under-lid zigzag, mean \|Δu\| / max / sign changes (x = 0.5, 0.75 < y < 0.95) | 0.020 / 0.035 / **6 of 8** | 0.038 / 0.058 / 7 of 8 |
| rms vs Ghia at t = 10 (u, v) | 0.1252, 0.1384 | 0.1276, 0.1440 |
| field difference for y < 0.9: rms / max, u | 6.6e−3 / 2.8e−2 | |
| … v | 6.3e−3 / 2.3e−2 | |
| … p | 7.1e−2 / 8.3e−2 | |
| lid-corner nodes | u = 0 | u = 1 (after the corner averaging) |

The Fortran code shows the same node-to-node oscillation, at the same
location, with the same sign pattern, at about half the amplitude. The bulk
fields agree to 1 % of the lid speed; the only O(1) differences are at the
two lid-corner nodes, where the Fortran driver keeps u = 0 and the Python
port keeps u = 1, which is the corner regularisation each code chose — and
the corner singularity is what drives the oscillation, so the amplitude
difference is consistent with that choice. Two independently written
implementations of the same functional, plus the assembled direct-solve
prototype, now agree: the oscillation is a property of the least-squares
formulation at small momentum weight, not of any implementation.
