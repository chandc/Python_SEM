# Cylinder at Re = 100 on a curvilinear mesh, and the dt limit of balanced weighting

*Recorded 2026-09-17. The first case in this project needing both halves of the
curvilinear port at once: a body the Cartesian code cannot represent, and a wake
that must leave through a real outflow condition.*

---

## 1. The production run

Box topology (`curvi.build_cylinder_box`): a body-fitted O-ring on the cylinder
whose outer edge is a square, inside eight rectangular blocks. 240 elements,
N = 8, 62 592 dof. Only the body is curved; every boundary carrying a condition
is axis aligned, so no-slip, free-stream inlet, **Dong outflow** and symmetry all
use existing codes and plan step 8 stays off the critical path.

dt = 0.1, balanced weighting w_mom = w_mass = √dt, seeded with a 5 % v
perturbation even in y, run to t = 150 in 8.72 h.

**Forces need no derivative and no numerical normal.** On a no-slip wall the
velocity vanishes, so the traction collapses to

    sigma . n  =  -p n  +  nu * omega * t,    t = z x n

because omega *is* du_t/dn there — and in VVP omega is a SOLVED VARIABLE, not a
derivative of the velocity. The body is a circle, so n is analytic.

| quantity | measured | published, unbounded | |
|---|---|---|---|
| Strouhal St | **0.1688 ± 0.0002** | 0.164–0.167 | +1.3 % |
| mean C_D | **1.3823** | 1.32–1.35 | inside 1.36–1.40 with this domain's 5 % blockage |
| C_L amplitude | **0.3378 ± 0.0041** | 0.32–0.34 | in range |

17 saturated cycles; the last several identical to four decimals (period 5.922,
St 0.1689, mean C_D 1.3834). The phase portrait closes on itself — a limit
cycle, not a drifting state. C_D oscillates at twice the C_L frequency, which the
code reproduces without being told. Pointwise |div u| is 2.1e−02 max, 1.6e−03
rms. Peak |omega| beyond x = 15 is 0.81 against 22.2 at the wall, so the Dong
outflow absorbs the street without reflection.

**St is 1.3 % high and that is NOT settled.** Blockage raises Strouhal, so the
sign is right, but the simple gap-velocity estimate overshoots in both
directions: published 0.164 would appear here as 0.164/(1−β) = 0.1726, and our
0.1688 corrects back to 0.1604, below the published range. Blockage is plausible
and probably dominant; it is not demonstrated. **Do not quote 0.1688 as agreement
on the strength of the blockage argument alone.**

---

## 2. Balanced weighting cannot be refined toward dt → 0

A dt sweep was set up to test whether temporal resolution moves St. **dt = 0.05
diverged**, while dt = 0.1 and dt = 0.2 ran cleanly — the opposite of the usual
expectation for an implicit scheme, and worth understanding rather than working
around.

With w_mom = w_mass = √dt the coefficients are

    a_mass = w_mass*fac1/dt = fac1/√dt        (grows as dt falls)
    a_flux = w_mom = √dt                      (shrinks as dt falls)

while the continuity row keeps weight 1. So refining dt drives momentum up
against a constraint that never moves, and simultaneously shrinks the pressure
block, which scales as a_flux².

`GARTLING_VALIDATION.md` §6 had already measured the consequence over 34 runs
with no crossover: **a_mass ≤ 6.05 bounded, a_mass ≥ 12.1 divergent.** With BDF2
(fac1 = 1.5):

| dt | a_mass | a_flux² | predicted | observed |
|---|---|---|---|---|
| 0.20 | 3.354 | 0.200 | bounded | completed |
| 0.10 | 4.743 | 0.100 | bounded | completed, 8.72 h |
| **0.05** | **6.708** | **0.050** | **past 6.05** | **diverged at t = 157** |

**The failure signature identifies the mechanism.** At t = 157 the run reported
C_L = −3.47 with **|u|max still 1.322** — an entirely normal velocity field. C_L
is built from p and omega *on the wall*, so the instability appeared in the
pressure and reached the velocity two time units later. That is an
under-weighted pressure block losing uniqueness, not a CFL or nonlinear failure.

### The remedy, which the repo already had

Artificial compressibility (`ls_pseudo_p`) gives the continuity row its own
coefficient so the two can be balanced:

    (1/(beta² dtau_p)) (p - p_prev)  +  div u ,     kappa_p = 1/dtau_p

with **dtau_p = 1/a_mass** making the rows scale together at every dt. Its
reference is the previous SUB-ITERATE, not the previous time level, and
`solver._drop_pseudo` removes kappa_p*p from the residual — so at sub-iteration
convergence p = p_prev, the term vanishes identically, and time accuracy is
preserved. **That requires the sub-iterations to converge:** with max_newton = 1
there is nothing to converge and it degenerates into the physical-time
compressible form. The AC runs here use max_newton = 3.

### Consequence for the method, beyond this experiment

This is a property of the formulation, not of the cylinder. **Balanced weighting
has a lower limit on usable dt that is unrelated to stability in the usual
sense** — √dt → 0 drives the pressure weight to zero. Any temporal-refinement
study under balanced weighting is confounded twice over: dt changes the time step
*and* the momentum-versus-constraint weighting. Measured on the Ghia cavity, the
same confound moved the RMS by 11 % between dt = 0.25 and 0.1, both converged.

Two clean designs exist and they answer different questions:

* **fixed w_mom = w_mass** (a constant, not √dt): a_flux is constant, the steady
  functional is unchanged, dt_eff = dt is preserved, and only the temporal
  discretisation varies. This is the honest temporal-refinement study.
* **balanced √dt plus AC at dtau_p = 1/a_mass**: keeps the production weighting
  and remediates the imbalance it creates. This tests the method as actually run.

---

## 3. Still open

* whether the 1.3 % in St is blockage — needs H = 10 → 20, roughly halving β
* p-refinement N = 8 → 10, since the wake elements grow at ratio 1.45 downstream
  and the roll-up at x ≈ 1–3 sets the frequency
* upstream extent Lu = 10 D is on the short side; fold into the same mesh change
* the four 45° corner elements where the O-ring meets the square are the
  strongly non-affine case that costs convergence *rate* (see the interpolation
  study in `SILENT_FAILURES.md` §3 and `scratch/curvi_error_source.py`); they
  show no visible scar in the vorticity field, which is reassuring but not
  quantitative
