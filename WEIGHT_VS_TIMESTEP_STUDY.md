# Separating the least-squares weight from the time step

Study date: 2026-08-08. Uses the `w_mom` / `w_mass` parameters added in
[POISEUILLE_DT_STUDY.md](./POISEUILLE_DT_STUDY.md) §4 to answer a question the
earlier dt sweeps could not: **when dt changes the answer, is that the momentum
weighting or the time step?** Legacy couples them (`weight = dt`), so every
previous sweep moved both at once.

Reproduce: `scratch/bfs_wsweep.py` (BFS 2-D sweep), `scratch/cavity_dt.py`
(cavity sensitivity), raw results in `scratch/bfs_wsweep.json`.

---

## Executive summary

1. **On the BFS the two effects OPPOSE each other, and the weighting wins.**
   Raising the weight shrinks the separation bubble 30%; raising the effective
   time step grows it 19%. The legacy sweep moved both together and the bubble
   shrank 42% — so the time step was working *against* the weighting the whole
   time.

2. **Reattachment is not a useful discriminator.** `x_r/h` spans 8.184–8.207
   across the entire two-dimensional sweep — 0.28%. Use the upper-wall bubble.

3. **Mass conservation tracks `a_mass`, not `a_flux`.**

4. **The stability boundary is flow-dependent, not a property of the
   discretisation.** Identical coefficients `(a_mass, a_flux) = (3.0, 1.0)`
   diverge on Poiseuille and converge comfortably on the BFS.

5. **The cavity is ~300x less dt-sensitive than Poiseuille** (5.9x vs 1875x),
   because it is lid-driven rather than pressure-driven.

6. **Large effective time step reproduces the truncated-outflow pathology on the
   LONG domain** — 31.8% reversed flow at `dt_eff = 2.0`.

---

## 1. Method

At nominal `dt`, to impose momentum weight `W` and effective time step `T`:

```
w_mom = W          w_mass = dt*W/T          (dt_eff = dt*w_mom/w_mass)
```

Long domain only (`cnos_long_grid.dat`, 72 elem, order 10, L/h = 17) — the
truncated domain is multi-valued and would bury the signal. Re=389, continuous
developed IC, free outflow with the SE-corner pressure pin, p-MG at `pc = p/2`,
nominal dt = 0.5. `(W, T) = (0.5, 0.5)` **is** legacy dt=0.5 and anchors both rows.

Exact references needing no knowledge of the true solution: `Qout/Qin = 1`,
`rms div = 0`, `max|u| <= 1.5` (the inlet peak), `exit reversed = 0%`.

Guards, learned from earlier runs that ground for hours: convergence on
`|dU|/dt_eff` (**not** `/dt` — they differ once `w_mass` is set), a hard
`max|u| > 10` divergence trip, and a step cap sized from `dt_eff`.

---

## 2. Row A — `dt_eff` = 0.5 fixed, weight varied

| W | a_mass | a_flux | Qout/Qin | rms div | x_r/h | p spread | outcome |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.75 | 0.25 | **0.9972** | **2.04e-02** | 8.196 | 0.259 | converged |
| 0.5 | 1.50 | 0.50 | 0.9925 | 4.83e-02 | 8.190 | 0.247 | converged (= legacy) |
| 1.0 | 3.00 | 1.00 | 0.9849 | 8.40e-02 | 8.198 | 0.236 | converged |
| 2.0 | 6.00 | 2.00 | — | — | — | — | **diverged, max\|u\|=10.6 at step 48** |

## 3. Row B — weight = 0.5 fixed, `dt_eff` varied

| dt_eff | a_mass | a_flux | steps | Qout/Qin | rms div | x_r/h | p spread | exit rev |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 3.000 | 0.5 | 683 | 0.9900 | 5.22e-02 | 8.184 | 0.213 | 0.0% |
| 0.5 | 1.500 | 0.5 | 323 | 0.9925 | 4.83e-02 | 8.190 | 0.247 | 0.0% |
| 1.0 | 0.750 | 0.5 | 160 | 0.9939 | 4.56e-02 | 8.203 | 0.251 | 0.0% |
| 2.0 | 0.375 | 0.5 | 115 | **0.9947** | **4.39e-02** | 8.207 | **0.692** | **31.8%** |

---

## 4. The result: the two effects oppose each other

The upper-wall separation bubble — the metric that moved 42% in the original dt
sweep, against `x_r`'s 1.4%:

| | bubble length/h | peak du/dy |
|---|---|---|
| **weight** 0.25 → 1.0 (`dt_eff` fixed) | 2.971 → **2.082** (shrinks 30%) | 0.247 → 0.123 (halves) |
| **`dt_eff`** 0.25 → 2.0 (weight fixed) | 2.419 → **2.867** (grows 19%) | 0.165 → 0.210 |
| legacy dt 0.1 → 5 (**both** together) | 2.893 → 1.683 (shrinks 42%) | 0.235 → 0.082 |

**In the legacy sweep the time step was pushing against the weighting, and the
weighting still dominated.** This confirms the attribution in
`PRECONDITIONER_AND_DT_STUDY.md` §5 — that the bubble responds to momentum
weighting — and shows it was *understated*: at fixed `dt_eff` the weight alone
moves the bubble more than the combined sweep did.

For reference, Chan's published bubble is 1.82 and the Fortran gives 2.404 at
dt=0.5. W=1.0 lands at 2.082, closer to Chan than either — **but at 1.51% mass
loss against legacy's 0.75%.** The trade-off is now explicit and tunable rather
than hidden inside dt.

### Reattachment is the wrong metric

`x_r/h` = 8.184 … 8.207 across the whole sweep, a **0.28%** spread. It barely
responds to either axis. An earlier reading of Row A alone as "the weighting is
not responsible for `x_r`" was over-stated — nothing much moves `x_r` on the long
domain, which is why it agreed to 0.5% with Fortran throughout the original dt
study while the bubble was off by 20%.

### Mass conservation follows `a_mass`

Row A: `a_mass` 0.75 → 3.0, `Qout/Qin` 0.9972 → 0.9849 (worse).
Row B: `a_mass` 3.0 → 0.375, `Qout/Qin` 0.9900 → 0.9947 (better).
Consistent in both directions — the mass term competes with the continuity row.

### The stability boundary is flow-dependent

BFS diverges at `a_flux = 2.0`, i.e. `(a_mass, a_flux) = (6.0, 2.0)`.
**Poiseuille failed at `(3.0, 1.0)` — coefficients the BFS handles comfortably.**
So the boundary is not a property of the discretisation; it depends on the flow.
A prediction that the Poiseuille boundary would transfer was made and refuted.

### Large `dt_eff` reproduces the truncated-outflow pathology on the LONG domain

At `dt_eff = 2.0`: 31.8% of the outflow plane has inflow and the exit pressure
spread jumps 0.25 → 0.69. Previously this was seen only on the L/h = 5 domain,
where it is geometric (the plane cuts the bubble). Here the domain is long enough
— the pathology is induced purely by the effective time step. The peak `du/dy` of
0.4524 in that row is this contamination reaching the upper wall, not a bubble
effect.

---

## 5. Why the cavity barely cares

`scratch/cavity_dt.py`, Re=1000, 4x4 order 8, accuracy = centreline `u` vs
Ghia 1982:

| dt | RMS vs Ghia | % of u_max | rms div | p spread |
|---|---|---|---|---|
| 0.05 | 6.58e-02 | 6.58% | 6.82e-02 | 0.8165 |
| 0.1 | 8.29e-02 | 8.29% | 7.21e-02 | 0.7898 |
| 0.5 | 6.80e-02 | 6.80% | 1.39e-01 | 0.7466 |
| 1.0 | 4.57e-02 | 4.57% | 2.30e-01 | 0.7286 |
| 2.0 | **3.94e-02** | **3.94%** | 4.23e-01 | 0.6807 |
| 5.0 | 2.34e-01 | 23.38% | 8.30e-01 | 0.3889 |

| | accuracy spread over dt |
|---|---|
| **cavity** | **5.9x** |
| Poiseuille | 1875x |

**~300x less sensitive.** The explanation is what drives the flow:

- **Poiseuille and the BFS are pressure-driven.** The solution is a response to
  `dp/dx`. Pressure appears only in the momentum rows, so under-weighting them
  corrupts the *driving force* — hence 98% velocity error at dt=0.05.
- **The cavity is lid-driven.** The forcing is a velocity boundary condition;
  pressure is passive, enforcing incompressibility rather than driving anything.
  Degrading it degrades `p` without wrecking `u`.

The cavity's divergence error still climbs monotonically (6.8e-02 → 8.3e-01), so
the pressure/constraint balance degrades exactly as the theory predicts — the
velocity simply does not depend on it. Two secondary reasons point the same way:
the cavity is closed, so there is no outflow plane and none of the soft modes
measured at ~8300x softer than generic; and its walls constrain the velocity far
more tightly than a free outflow.

> This also corrects a comment previously carried in `lssem2d/lssem.py`, which
> grouped "cavity, Poiseuille" together as cases where the weighting is
> harmless. That holds for the cavity. It is badly wrong for Poiseuille.

---

## 6. Practical guidance

- **Nothing in the tested space beats legacy overall on the BFS.** Higher weight
  moves the bubble toward Chan but costs mass conservation and diverges by
  `a_flux = 2.0`.
- **Judge by the quantity you care about.** Bubble/separation features respond to
  the momentum weight; mass conservation and divergence respond to `a_mass`.
  They pull in opposite directions.
- **Do not use `x_r` to tune anything** on the long domain — it is insensitive.
- **Keep `dt_eff <= 1`** on an open domain: at 2.0 the outflow degenerates even
  when the domain is long enough.
- **Expect flow-dependent stability limits.** The safe weight on one problem is
  not the safe weight on another.
