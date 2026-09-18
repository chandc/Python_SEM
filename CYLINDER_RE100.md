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

## 4. The domain is NOT too small — a literature check that retracts my own hypothesis

*Added 2026-09-17, after the dt and N sweeps closed off discretisation error.*

With both sweeps converged (below), the 1–2 % Strouhal offset had to be something
other than discretisation, and blockage was the obvious candidate. **It is not,
and the definitive study says so.**

### Behr, Hastreiter, Mittal & Tezduyar (1995), *CMAME* **123**, 309–316

Re = 100, five *nested* meshes differing only in the lateral distance A, solved
with **two independent** stabilised finite-element formulations
([PDF](https://home.iitk.ac.in/~smittal/publi_&_present/sm_journals/inco_flow_past.pdf),
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0045782594007367)).
The cylinder is 2 units in their Fig. 1, so A = 16 units is **8 diameters**.

| mesh | A | A/D | St (space–time VP) | St (VP–stress) | mean C_D | C_L amp |
|---|---|---|---|---|---|---|
| M090 | 9.0 | 4.5 | 0.1711 | 0.1739 | 1.455 / 1.473 | 0.395 / 0.388 |
| M125 | 12.5 | 6.25 | 0.1658 | 0.1690 | 1.402 / 1.421 | 0.379 / 0.374 |
| M160 | 16.0 | **8** | 0.1641 | 0.1672 | 1.384 / 1.403 | 0.374 / 0.369 |
| M240 | 24.0 | 12 | 0.1624 | 0.1661 | 1.372 / 1.392 | 0.371 / 0.367 |
| M320 | 32.0 | 16 | 0.1624 | 0.1661 | 1.370 / 1.389 | 0.371 / 0.366 |

> *"the lateral boundaries should be removed from the cylinder by a distance of
> **8 cylinder diameters**. If this is not the case, the computed Strouhal number
> will have an artificially high value."*

Quantities "stabilize within 1 % of the final values only as the distance A
exceeds 16.0", and M240 and M320 are identical to four decimals.

### What that does to our result

**Our domain exceeds their recommendation on every axis** — lateral **10 D**
against their 8 D, upstream 10 D against 8 D, downstream 25 D against 22.5 D. So
the lateral extent is adequate and **blockage does not explain the offset.**
I had proposed a wider-domain run as the next experiment; that is now withdrawn,
and the arithmetic behind it (H ≈ 52 for a 0.5 % shift) was answering a question
that did not need asking.

Against their converged values, the comparison also improves:

| | Behr et al. (A = 32) | ours | deviation |
|---|---|---|---|
| mean C_D | 1.370 / 1.389 | **1.3823** | **+0.3 %** |
| C_L amplitude | 0.371 / 0.366 | 0.38 | +3 % |
| St | 0.1624 / 0.1661 | 0.1698 | +2.5 % |

Our C_D sits **inside** their two-formulation range. The 1.32–1.35 band quoted in
§1 comes from other sources with different domains and is the less apt
comparison for a 2-D computation in a finite box.

### The number that reframes the whole offset

**Their two formulations differ by 0.0037 in St — 2.3 % — on identical meshes**,
as large as the entire domain effect from A = 9 to 32. They warn explicitly that
a dissipative time integrator lowering St can combine with confinement raising it
to produce agreement with experiment that "is not indicative of overall accuracy
of the solution method".

So our 2.5 % offset sits inside the formulation-to-formulation scatter the
literature itself reports for this benchmark. The open question is no longer the
domain; it is whether the offset is a property of the least-squares formulation,
and that needs a different instrument than a bigger box.

---

## 5. The dt and N sweeps: St is converged on both axes

All with AC at $\kappa_p = a_{\rm mass}/2$, seeded from saturated limit cycles,
St by **two estimators sharing no code** — FFT with Hann window and quadratic
peak interpolation, and C_L zero crossings over whole cycles. The C_D spectrum
must peak at exactly twice the C_L frequency, which is physics rather than a
property of either estimator, and is reported as a validity check.

| dt | $a_{m mass}$ | St (FFT) | St (crossings) | C_D/C_L harmonic |
|---|---|---|---|---|
| 0.10 (no AC) | 4.74 | 0.1689 | 0.1688 ± 0.0001 | 2.000 |
| 0.10 +AC | 4.74 | 0.1688 | 0.1688 ± 0.0006 | 1.999 |
| 0.05 | 6.71 | 0.1697 | 0.1698 ± 0.0002 | 1.994 |
| 0.025 | 9.49 | 0.1701 | 0.1701 ± 0.0004 | 2.002 |
| **0.0125** | **13.42** | 0.1699 | 0.1698 ± 0.0003 | 2.000 |

| N | dof | St (FFT) | St (crossings) |
|---|---|---|---|
| 6 | 35 424 | 0.1697 | 0.1698 ± 0.0003 |
| 8 | 62 592 | 0.1697 | 0.1698 ± 0.0002 |
| 10 | 97 440 | 0.1701 | 0.1698 |

**St = 0.1698 ± 0.0005.** A 4× reduction in dt and a 2.75× increase in degrees of
freedom move it by under 0.3 %.

**The dt = 0.0125 row is the weighting result.** $a_{m mass}$ = 13.42 is inside
the band where `GARTLING_VALIDATION.md`'s 34-run sweep had **16 of 16 divergent**
without AC, and it ran clean at 5.26 s/step with zero PCG warnings, returning the
same St as dt = 0.1. Balanced weighting plus AC reaches a time step the
formulation cannot otherwise touch, with the answer unchanged.

*Cost note.* AC is not an overhead here, it is the enabler: at dt = 0.05 measured
27.2 s/step and 66–85 CG iterations WITHOUT it (before diverging), against
12.6 s/step and 14–15 iterations with it. `compute_jacobi` has no `a33` entry at
all without AC, so the pressure block of $L^{\mathsf T}L$ is unscaled; AC supplies
exactly that block.

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
