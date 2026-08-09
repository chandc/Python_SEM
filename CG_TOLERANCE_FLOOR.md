# The linear-solve tolerance floor

Study date: 2026-08-08. Found while investigating why the pure steady form
(`w_mass = 0`) appeared to have a ~20x accuracy penalty. It did not — the
penalty was the linear-solve tolerance, and the same floor limits other results
across this project.

Reproduce: `scratch/pois_wmass0.py`; the diagnostic runs are inline in the
session and reproduced in §2–§4 below.

---

## 1. The mechanism

`pcg_solve` ([lssem2d/solver.py](./lssem2d/solver.py)) stops on

```
cgstol = max(cgsfac * res0, tol)          tol defaults to 1e-6
```

and `newton_step` calls it **without passing `tol`**, so every solve in this
project has used an absolute floor of `1e-6`.

Early in a run `res0` is large and the relative term `cgsfac*res0` binds. Once
the residual falls, **the absolute floor takes over and `cgsfac` stops mattering
entirely**. That is why an earlier sweep of `cgsfac` from 1e-3 to 1e-8 changed
nothing — it was tightening a term that had already stopped binding. Both floors
must be lowered together.

---

## 2. How it was found

The pure steady form (`w_mass = 0`, so the momentum row is exactly `w_mom*N(U)`)
converged in 7–11 Newton iterations to a profile error of ~8e-03, about 20x worse
than legacy dt=1. Three hypotheses were tested and **all three refuted**:

| hypothesis | test | result |
|---|---|---|
| CG-tolerance limited | swept `cgsfac` 1e-3 → 1e-8 | no change — refuted |
| mass term regularises low velocity modes | `u`-block diagonal, with and without | 8.3282 vs 8.2944, a 0.4% change — refuted |
| constant-velocity mode is under-constrained | Rayleigh quotient on that mode | 0.84844 vs 0.82998, 2% — refuted |

The decisive measurement was to evaluate the least-squares functional at the
converged state and at the analytic solution:

```
J(converged) = 7.82e-09
J(exact)     = 5.94e-27      <- machine zero
```

**18 orders of magnitude.** The minimum *is* the exact solution (Poiseuille has
identically zero residual: `p_x = -0.12` and `nu*om_y = +0.12` cancel exactly),
so this was a solver failure, not a formulation one. The error was also entirely
localised on the free outflow plane — every field's maximum deviation at x=10.00
— the soft-mode signature measured elsewhere at ~8300x softer than generic.

That points at the linear solve, and the floor is `tol`.

---

## 3. Impact, by configuration

Poiseuille Re=100, order 8, 10x2, profile error at the outlet:

| config | `tol` = 1e-6 | `tol` = 1e-10 | improvement |
|---|---|---|---|
| legacy dt=0.5 | 9.67e-02 | 7.60e-02 | 1.3x |
| legacy dt=1.0 | 5.26e-04 | 1.72e-04 | 3x |
| **steady, `w_mass=0`** | **8.46e-03** | **2.90e-04** | **29x** |

With **both** floors tightened (`cgsfac=1e-8`, `tol=1e-10`), the steady form
reaches **3.14e-07 in 10 Newton iterations** — 27,000x better than its
default-tolerance result:

| preconditioner | iters | J | profile error |
|---|---|---|---|
| Jacobi | 10 | 2.18e-19 | **3.14e-07** |
| p-MG | 10 | 3.48e-17 | 1.04e-06 |

(p-MG converges to a slightly *worse* functional here; at these tolerances the
extra V-cycle arithmetic is not buying anything on a well-conditioned problem.)

---

## 4. What this changes

**The central Poiseuille conclusion survives.** At matched tight tolerance,
dt=0.5 still gives 7.6e-02 against dt=1's 1.7e-04 — a **440x gap**. That is
genuine least-squares weighting error. The dt=0.05 catastrophe (98% velocity
error) is likewise barely tolerance-sensitive, because there the pressure block
of `L^T L` is near-singular — an unconverged solve is not the problem.

**The claim that the steady form is ~20x less accurate was WRONG.** It was
entirely tolerance-limited. At matched tolerance the steady form gives 2.90e-04
in **18 iterations** against legacy dt=1's 1.72e-04 in **600 steps** — comparable
accuracy, ~33x cheaper. Solved tightly it is the most accurate configuration
measured anywhere in this project.

**A systematic caveat on everything measured with default settings.** Results
reporting errors in the 1e-4 to 1e-3 range may be tolerance-limited rather than
discretisation-limited. That plausibly includes some BFS mass-conservation and
exit-pressure figures, which sit in exactly that band. The large effects are far
too big to be explained this way and stand unaffected:

| effect | size | tolerance-explicable? |
|---|---|---|
| Poiseuille dt spread | 1875x | no — **re-run tight: 212,061x**, see below |
| BFS bubble response to weighting | 30% | no |
| cavity vs Poiseuille sensitivity | 5.9x vs 1875x | no |
| BFS exit pressure spread, Jacobi vs p-MG | 5x | probably not, but unchecked |
| BFS mass conservation differences | ~0.5–1% | **unchecked** |

> **Resolved for Poiseuille (2026-08-09).** The full dt sweep was re-run with
> both floors lowered (`scratch/dt_tight.py`). The dt effect is *larger* than
> published, not smaller: dt=1 improves 113x to 4.65e-06 and dt=0.05 is
> unchanged at 98.5%. Table in
> [STEADY_FORM_STUDY.md](./STEADY_FORM_STUDY.md) §6. The BFS rows above remain
> unchecked.

---

## 5. Recommendation, not yet implemented

`newton_step` should pass `tol` through to `pcg_solve`, and the default is
arguably too loose at `1e-6`. As it stands there is no way to tighten the
absolute floor without monkeypatching `pcg_solve`, which is what every diagnostic
in this document had to do.

A reasonable change: give `newton_step` and `step_bdf` a `cg_tol` argument
defaulting to the present `1e-6` (so nothing changes silently), and document that
accuracy studies should lower it together with `cgsfac`.
