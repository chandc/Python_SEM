# Backward-Facing Step: Validation Ladder and Reference Data

Plan of record, 2026-08-29.  Status of the projection-path (RK3-CN) BFS
branch and every credible comparison target, tiered by what our
spanwise-PERIODIC formulation can fairly be compared against.
Companion docs: `DIVERGENCE_AND_CONSISTENCY.md`, `ARMALY_VALIDATION.md` (2D).
Citation details in this file are from working memory: VERIFY against the
papers before promoting any number to a pass/fail gate.

## 0. What is built (this week)

- Outflow OBC v1 for the projection path (`lssem3d/project.py`): velocity
  free at edge code 4 (natural zero-traction), pressure Dirichlet phi = 0;
  the do-nothing pair, which COINCIDES with Dong's energy-stable OBC when
  no backflow crosses the boundary (the Theta stabiliser term is zero).
  Gate 1 passed: Poiseuille pass-through at 1e-5 for 5 units.
  Gate 2 passed: vortex exits with monotone decay, no reflection
  (figure `scratch/obc_vortex_streamlines.png`).
- Dong Theta backflow stabiliser = v2, to be added when a case puts
  recirculation AT the boundary (see OUTFLOW_DONG_OBC_PLAN.md; note its
  warning that Dong's mechanism is specifically the backflow term).
- Parametric L-shaped step mesh (`lssem2d.mesh.build_bfs`): inlet channel +
  two-block expansion, coordinate-matched connectivity, bc codes
  1 = wall, 3 = inlet (Dirichlet by lifting), 4 = outflow.  Verified:
  gather-scatter self-adjoint to machine precision on the L topology.
- Outflow-aware pressure p-multigrid (`hpmg` outflow_p): 48 CG/step total
  against 1300 with one-level FDM.
- Gate 3 IN PROGRESS: Armaly Re = 389 (see below).

## 1. Tier 1 -- laminar, Re < ~400 (flow genuinely 2D): usable NOW

| reference | data | our comparison |
|---|---|---|
| Armaly, Durst, Pereira, Schonung (1983) JFM 127 | x_r(Re) for 70<Re<8000 (Re = U D/nu, D = 2h), midplane LDA profiles | Gate 3 running: quasi-2D (Nz=4), ER 1.94, Re 389; pass x_r/S = 8.1 +/- 0.4 (in-house 2D: 8.145, expt 8.05) |
| Lee & Mateescu (1998) | ER 1.17/2.0, Re < 3000; lower AND upper-wall separation/reattachment | second observable at Re where the upper bubble exists |
| Tihon et al. (electrodiffusion) | wall shear rate at the lower wall vs Re | apples-to-apples: our x_r diagnostic IS wall shear |

Caveat for the whole tier: above Re ~ 400 Armaly's flow is 3D and
SIDEWALL-driven (aspect ratio 36 notwithstanding; Williams & Baker 1997);
a spanwise-periodic simulation should NOT be tuned to match transitional
Armaly x_r -- the classic 2D-vs-experiment discrepancy is physics, not error.

## 2. Tier 2 -- 3D onset, spanwise-periodic (our formulation's home turf)

| reference | result | our gate |
|---|---|---|
| Barkley, Gomes, Henderson (2002) JFM 473 | primary 3D instability of the ER=2 steady flow: STEADY bifurcation, Re_c ~ 748, lambda_z ~ 6.9 S, mode localised in the primary bubble | (a) sub-critical null test: Re ~ 600, Lz = 6.9 S, seeded 3D perturbation must DECAY; (b) growth-rate bracket of Re_c; (c) wavelength selectivity |
| Kaiktsis, Karniadakis, Orszag (1991) JFM 231 -> (1996) JFM 321 | 1991 unsteadiness was numerically induced; flow is CONVECTIVELY unstable (noise amplifier) below global onset | the sternest OBC test available: any outflow reflection re-seeds the amplifier; sustained oscillation below onset at increasing resolution = numerics, full stop |
| Beaudoin et al. (2004) | experimental steady streamwise vortices on a step | corroboration for the Barkley mode's structure/wavelength |
| Biswas, Breuer, Durst (2004) | laminar 2D/3D across expansion ratios incl. sidewalls | cross-checks; bridge toward literal-Armaly geometry |

## 3. Tier 3 -- turbulent (the long game)

| reference | data | prerequisite |
|---|---|---|
| Jovic & Driver (1994), NASA | Re_h ~ 5000 step: Cf, profiles; THE experimental companion of Le, Moin & Kim (1997) DNS (which used periodic span -- matches our formulation, unlike transitional Armaly) | turbulent inflow generation (recycling plane or synthetic); resolution per k_max*eta ladder |
| Chan & Mittal (1996) CTR | in-house LES heritage at Re = 5100, Smagorinsky + Van Driest | the historical anchor: reproducing it with the modern solver closes a 30-year loop |

## 4. Sidewall (literal Armaly transitional) -- out of scope for Fourier-z

Matching Armaly's 400 < Re < 6600 x_r requires no-slip side walls
(Williams & Baker 1997; Nie & Armaly 2004): a wall-bounded z direction,
i.e. replacing the Fourier span with a third SEM direction.  Different
code branch; document only.

## 5. Order of execution

1. Gate 3 plateau + grid-refinement pass (graded x, N 9-10) -> archive.
2. Tier-2(a) sub-critical null test (also the OBC amplifier test).
3. Tier-2(b/c) Re_c and lambda_z capture vs Barkley.
4. Dong Theta stabiliser (v2) when backflow at the exit first appears.
5. Tier 3 upon turbulent-inflow capability.

---

## 6. Gate 3 result (2026-08-29): PASSED

Armaly Re = 389, quasi-2D (Nz = 4), impulsive start, dt = 4e-3, run to t = 50.

| source | x_r/S |
|---|---|
| this gate (N = 7, uniform x) | **8.11** |
| in-house 2D validated (P = 10, graded) | 8.145 |
| Armaly 1983 experiment | 8.05 +/- 0.7 |

Transient: 5.98 (t=10) -> overshoot 8.71 (t=20) -> plateau 8.11.  CG = 48
per step with the outflow-aware p-multigrid (one-level FDM needed ~1300).
Archive: `results/bfs_armaly_re389/` (Dropbox tree): state, x_r trace,
streamline figures, README.

### Steady-state field checks (measured on the archived state)

- **Spanwise content: identically zero.**  Every k_z != 0 mode of u, v, p
  is 0.0 exactly, and w = 0: a pure 2D solution embedded in the 3D solver.
  Correct physics at Re = 389 (below the ~748 onset) AND a no-spurious-3D
  check over 12,500 steps.  Differences between fields appear only above
  onset (the Barkley mode at lambda_z ~ 6.9 S).
- **Divergence.**  Weak (solver-controlled): 4.5e-3 relative.  Strong:
  0.11 relative globally -- but the maximum sits at exactly (0, 0.94), the
  re-entrant step corner, and **98.4% of |div|^2 lies within 1.5 units of
  it**: the corner stress singularity, where gradients are analytically
  unbounded.  Away from the corner the field is pointwise solenoidal to
  solver tolerance.  Integral quantities (x_r) are unaffected at this
  grid; corner-sensitive quantities need the graded refinement pass --
  exactly why the validated 2D mesh graded toward the step.

---

## 7. Refinement pass and the corner-mask defect (2026-08-29, in progress)

Executing sec 5 steps 1-2 (graded mesh + N=9; Re=600 base for the Tier-2a
null test) surfaced two defects.  Both are fixed and both are general:

**7.1 CFL must come from the mesh, not the hand.**  The 1.6-power grading
shrinks the first outlet column 5.3x; two runs launched with hand-picked dt
died immediately (one at 3e-3 from step one, one at 1e-3 as the inflow
ramp crossed ~35%).  The driver now computes dt = 0.35 x unit-CFL from the
inflow profile evaluated ON the actual mesh, and a C1 cosine inflow ramp
(--ramp) softens impulsive starts.  Measured: N=9 graded dt = 7.9e-4,
N=7 graded dt = 1.27e-3.

**7.2 Mask consistency at multiblock corners (the real find).**  Forensic
tracing localised an e-folding-0.05 explosion of v at exactly the step
corner (0, 0.94).  Cause: the corner node's three element copies carried
INCONSISTENT Dirichlet masks -- the inlet-channel copy (south edge = step
top) and outlet-bottom copy (west edge = step face) masked, but the
outlet-top copy (west edge = interior fluid) FREE.  The no-slip condition
leaks through the free copy under gather-scatter and the masked operator
loses symmetry on the assembled space -- the pin_dof docstring's warning,
materialised.  The uniform-mesh Gate 3 run was marginally stable with the
same latent leak; grading resolved the corner dynamics and detonated it.

Fix (build_masks): after edge masking,
    leaked = gs(1 - mask) > 0.5;  mask[leaked] = 0
-- any node masked in one copy is masked in all copies (Dirichlet wins at
corners), for every boundary type at once.  Verified: zero inconsistent
copies on the graded mesh; a no-op on corner-free meshes (channel, TGV),
so no validated result changes.  Dynamic retest: the reproducible t=1.1
blow-up now runs clean to t=3 (2.7x past the death point).

**Standing consequence for Gate 3**: its passed run carried the latent
leak (stable, but present).  The N=9 graded rerun in flight doubles as
re-validation: x_r near 8.11-8.145 confirms the number; a shift would make
the refined value the number of record.

**7.3 Re-validation result (N=9 graded, run complete).**  tend=60 with
the mask fix in place; the startup eddy exited the OBC at t~30 and the
primary bubble's approach fitted x_r(t) = A - B exp(-t/tau) over
t=35-60 (rms 4e-4):

    A = 8.108 (+/- 0.01),  tau = 38

vs coarse gate 8.11, 2D reference 8.145, Armaly 8.05 +/- 0.7.  **Gate 3
re-validated after the corner-mask fix; grid-converged at the 0.1%
level.**  The coarse and refined runs bracket the reference from below
by ~0.4%, consistent with the remaining outflow-length truncation.
Archive: results/bfs_armaly_re389/ (final_N9refined_t60.npz).

## 8. Tier 1b passed: Erturk Re=600 comparison (2026-08-30)

Two runs, consistent naming `_bfs_er{194,200}_re600`:

- **ER=1.94, L_out=18, tend=80** -- the OBC stress test.  The upper-wall
  bubble sheds eddies at this Re; two full shedding/exit cycles passed
  through the open boundary with no reflection (wall-shear crossings
  collapse 3 -> 1 as each eddy exits).  Domain ends inside the upper
  bubble, so no steady x_r from this run (by design; kept as the OBC
  record).
- **ER=2.0, L_out=32, tend=100** -- the literature-standard geometry.
  Startup eddy exited at t~48; monotone single-crossing approach
  thereafter.  Exponential fit over t>50/60/70 windows:

      x_r/h = 10.2 +/- 0.1   (tau ~ 70)

  vs **Erturk (2008) 2D steady: 10.05** -> +1.5%, inside the combined
  extrapolation + inlet-condition uncertainty.  Together with Gate 3
  (Re=389: 8.11 vs 8.145) the reattachment curve's slope with Re is
  captured across the laminar range.

Archives: results/bfs_er194_re600/, results/bfs_er200_re600/ (final
fields, crossing histories, x_r fit figure).  Next tier: 2a Barkley
null test (L_z=6.9S seeded perturbation must decay at Re~600).
