# Fortran vs NumPy: 2D Poisson SEM Benchmark

Comparison of `sem_2d.f90` against `sem_2d_oo.py` (`NumpySolver`) on the matrix-free
2D Poisson problem, with the codes verified equivalent before any timing.

**Date:** 2026-08-02
**Machine:** Apple M3 Max (12P + 4E), macOS 26.5.2
**Toolchain:** GNU Fortran 15.1.0 (Homebrew GCC), NumPy 1.26.4 (OpenBLAS), Python 3 (anaconda3)

---

## 1. Headline result

As originally built, **NumPy beat Fortran by 1.3–2.8×**, with the gap widening as `p` grew.

This was **not** a language result. It was a build defect: `-framework Accelerate` was
linked but **inert**, because gfortran does not route `MATMUL` to BLAS without
`-fexternal-blas`. The as-built binary contained no `dgemm` reference at all.

With the flags corrected, the ranking reverses at the ends of the `p` range — Fortran is
~1.3× faster at `p=15` and ~1.5× at `p=7` — but the two are **at parity at `p=10–12`**.

The honest summary: a correctly built Fortran is *competitive with, or modestly ahead of,*
NumPy here — not dramatically so, because NumPy's inner loop is already BLAS.

See **§7** for the precise scope of the "comparable performance" conclusion, and for a
counterexample in this same project where it does **not** hold.

---

## 2. Equivalence verification (done before timing)

Both codes solve

$$-\nabla^2 u = 32\pi^2 \sin(4\pi x)\sin(4\pi y) \quad\text{on}\quad \Omega = [-1,1]^2,
\qquad u = 0 \ \text{on}\ \partial\Omega$$

with identical numerics:

| | `sem_2d.f90` | `sem_2d_oo.py` |
| :--- | :--- | :--- |
| Domain | `L_x=L_y=2`, origin at −1 | `linspace(-L/2, L/2)`, `L=2` — same |
| Operator | `matmul(matmul(K_1dx,P),M_1dy) + …` | `K_1dx @ u @ M_1dy.T + …` — same |
| Assembly | `apply_dss` (x-exchange, y-exchange, BC zeroing) | `dss_np` — same |
| Preconditioner | diagonal `M_y·K_x + K_y·M_x`, DSS'd, inverted | same |
| Inner product | multiplicity weight `W` | same (0/1 ownership vs 0.5/0.5 — equivalent post-DSS) |
| Exit test | `sqrt(rsnew) < 1e-11` | `np.sqrt(rsnew) < tol`, `tol=1e-11` — same |
| Timer scope | `system_clock` around CG loop only | `perf_counter` around `solve()` only |

**Iteration counts matched exactly at every configuration tested**, and `L_inf` errors
agreed digit-for-digit:

| Config | p | Fortran iters | NumPy iters |
| :--- | ---: | ---: | ---: |
| E=10×10 | 7 / 10 / 12 / 15 | 29 / 56 / 74 / 101 | 29 / 56 / 74 / 101 |
| E=20×20 | 7 / 10 / 12 / 15 | 29 / 57 / 74 / 100 | 29 / 57 / 74 / 100 |
| E=5×15 | 15 | 177 | 177 |

This is what makes the comparison meaningful: it measures implementation speed, not
algorithm. Iteration counts were obtained from an instrumented copy of the Fortran
(it does not print `iter`); all timings used the unmodified binary.

---

## 3. Results

All times in **milliseconds, best of 5**, solve loop only, `tol = 1e-11`.

### E = 10×10

| p | DOF | iters | `-O3` (as built) | `+ -mcpu=native` | `+ external BLAS` | **NumPy** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 5,041 | 29 | 3.02 | 2.39 | **1.64** | 2.4 |
| 10 | 10,201 | 56 | 11.67 | 9.38 | 7.78 | **7.9** |
| 12 | 14,641 | 74 | 23.72 | 18.75 | 14.38 | **13.6** |
| 15 | 22,801 | 101 | 56.79 | 37.52 | **17.15** | 22.4 |

### E = 20×20

| p | DOF | iters | `-O3` (as built) | `+ external BLAS` | **NumPy** |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 19,881 | 29 | 11.56 | **6.38** | 7.4 |
| 10 | 40,401 | 57 | 46.53 | 29.00 | **26.4** |
| 12 | 58,081 | 74 | 92.14 | 51.96 | **46.5** |
| 15 | 90,601 | 100 | 218.99 | **63.66** | 79.5 |

### E = 30×30 (the configuration `run_fortran_sweep.py` uses)

| p | `-O3` (as built) | new flags | speedup |
| ---: | ---: | ---: | ---: |
| 4 | 2.99 | 3.23 | 0.93× |
| 5 | 7.73 | 6.49 | 1.19× |
| 7 | 27.41 | 14.88 | 1.84× |
| 10 | 107.72 | 66.18 | 1.63× |
| 15 | 514.14 | 144.67 | **3.55×** |

The largest single gain in the whole study is the last row: **514 → 145 ms**.

> **Note on the "external BLAS" columns for E=10 and E=20:** those runs used
> `-fblas-matmul-limit=1`. The shipped build uses `=6` (see §5); at E=30×30 the two are
> within run-to-run noise for all `p ≥ 5`, so the columns are representative.

---

## 4. What was ruled out, and how

Two plausible explanations were tested and **eliminated by measurement**, not assumed:

**Precision — ruled out.** A quad-precision bug was found in a *different* code
(`F90_SEM/Poisson_2D_mf_MultiElement`, where `-fdefault-real-8` without
`-fdefault-double-8` promoted `KIND(1.0D0)` to 16 bytes and silently invoked
`libquadmath`, costing ~70×). **That bug does not apply here.** `sem_2d.f90` declares
everything as explicit `real(8)`, a kind neither flag touches, and the build never used
those flags. Verified directly:

```
nm sem_2d_f90 | grep -ciE "tf3|quadmath"   →  0
grep -nE "^\s*real\s*(::|,)" sem_2d.f90    →  none (all explicit real(8))
```

**Threading — ruled out.** NumPy pinned to a single thread was within noise
(`p=15`, E=10×10: 21.88 ms default vs 23.32 ms single-threaded). At 16×16 blocks
OpenBLAS stays serial regardless.

**The actual cause — confirmed.** `nm sem_2d_f90 | grep -ci dgemm` returned **0** on the
as-built binary. Fortran looped element-by-element into gfortran's *internal* `matmul`
on small `(p+1)×(p+1)` blocks, while `apply_K` in NumPy batches all `E²` elements into a
single call that lands in OpenBLAS. This is why the gap tracked `p`: it is the $O(p^3)$
tensor contraction. Enabling `-fexternal-blas` (count becomes 1) reverses the result.

---

## 5. The fix, and why the matmul limit is 6

`run_fortran_sweep.py` now builds with:

```bash
gfortran -O3 -mcpu=native -funroll-loops \
         -fexternal-blas -fblas-matmul-limit=6 \
         sem_2d.f90 -o sem_2d_f90 -framework Accelerate
```

`-fblas-matmul-limit` sets the crossover at which `MATMUL` is handed to DGEMM. Forcing it
on unconditionally (`=1`) **regresses the small-`p` end**, where DGEMM call overhead
dominates a 4×4 or 5×5 block. Since the sweep runs `p=3..15`, this matters. Measured at
E=30×30 (ms, best of 5):

| p | old `-O3` | `lim=1` | `lim=5` | `lim=6` | `lim=7` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.601 | 1.531 | 1.365 | 1.475 | 1.447 |
| 4 | 2.994 | **3.992** | 3.200 | 3.226 | 3.113 |
| 5 | 7.728 | 6.136 | 6.196 | 6.492 | 6.497 |
| 6 | 14.537 | 13.384 | 13.408 | 13.360 | 12.104 |
| 7 | 27.406 | 14.844 | 14.748 | 14.878 | 14.833 |
| 10 | 107.721 | 65.951 | 66.317 | 66.182 | 66.166 |
| 15 | 514.142 | 143.934 | 145.035 | 144.667 | 146.350 |

BLAS first pays off at `p=5`, and `lim=1` costs 33% at `p=4`. Setting `=6` puts the
crossover at 6×6 blocks (`p ≥ 5`). Above `p ≥ 5` the limit value is within noise, so the
choice only protects the low end.

Accuracy is unaffected — `L_inf` is identical to all 17 digits at `p=4` and `p=15`, and
differs only in the last two digits at `p=7` from DGEMM's summation order.

---

## 6. Corrections to `SEM_BENCHMARK_REPORT.md`

This document supersedes parts of the earlier report. Three specific claims there are
wrong or no longer hold:

1. **§4 states NumPy "delegates the tensor contraction … directly to Apple's Accelerate
   BLAS framework."** This is incorrect for this environment. `numpy.show_config()`
   reports `name: openblas`. The performance argument still holds — the work does go to
   an optimized BLAS — but it is OpenBLAS, not Accelerate.

2. **§4 and §6 claim NumPy "matches the raw speed of compiled Fortran" (a "Tie").** That
   conclusion was an artifact of the BLAS-inert Fortran build. Correctly built, Fortran
   is ~1.3× faster at `p=15`; the two genuinely tie only at `p=10–12`.

3. **§5's absolute timings do not reproduce.** Re-measuring the *same* as-built binary at
   the table's own configuration (E=5×15, 17,176 DOF, `p=15`) gives 75 ms against the
   table's 171 ms; NumPy gives 30 ms against the table's 54 ms. Both columns are ~2×
   faster now, so the table's *ratio* (3.2× vs my 2.5×) is broadly consistent and the
   ranking it reports is not in question — but the absolute numbers should not be quoted.
   Provenance of that run is unclear; treat the tables in this document as current.

For reference, at that configuration (E=5×15, `p=15`, 17,176 DOF, 177 iterations):

| | time |
| :--- | ---: |
| Fortran, as built (`-O3`) | 75.1 ms |
| NumPy | 29.9 ms |
| Fortran, corrected flags | **22.1 ms** |

---

## 7. Scope: when "comparable performance" holds — and when it does not

It **is** fair to say Fortran and NumPy are comparable in performance *on this problem*.
Two qualifications are load-bearing, and the claim is wrong without them.

### Qualification 1 — the Fortran must be correctly built

Against the as-built binary the claim is false in the opposite direction: NumPy led by up
to 2.8×. "Comparable" describes the corrected build only. Stating it unqualified is the
error §6 documents in the earlier report.

### The measured spread

Ratio of corrected Fortran to NumPy (>1 = Fortran faster):

| config | p=7 | p=10 | p=12 | p=15 |
| :--- | ---: | ---: | ---: | ---: |
| E=10×10 | 1.46× | 1.02× | 0.95× | 1.31× |
| E=20×20 | 1.16× | 0.91× | 0.90× | 1.25× |

Full spread across every matched point: **0.90× to 1.46×**, with half the points inside
±10%. Neither language wins consistently — they trade the lead depending on `p`. For a
compiled-vs-interpreted comparison this is comfortably "comparable."

### Qualification 2 — this is a property of the *workload*, not of the languages

The result is not a testament to Python. Both codes spend nearly all their time in the
same kind of optimized dense DGEMM — NumPy through OpenBLAS, Fortran through Accelerate —
with a thin driver layer around it. What is really being measured is *OpenBLAS vs
Accelerate DGEMM, plus dispatch overhead*. Python's interpreter cost is amortized to
irrelevance because a single `@` in `apply_K` does the work of all `E²` elements at once.

So the conclusion holds **because this problem is dominated by dense matmul on blocks
large enough to amortize dispatch.** Remove that condition and it fails.

### Counterexample from this same project

The VVP cavity solve (`lssem2d`, $Re=100$, matched graded mesh, analytic `dge`) measured:

| | per time step |
| :--- | ---: |
| Fortran | 21.6 ms |
| Python | 57.8 ms |
| | **2.7× gap** |

Same two languages, same style of matrix-free operator, same machine — but a much lower
flop-to-dispatch ratio, and the parity disappears. The `p=3–4` end of the Poisson sweep
(§5) shows the same effect in miniature.

### Recommended phrasing

> For the 2D Poisson benchmark, a correctly built Fortran and the NumPy port perform
> within roughly ±25% of each other, because both delegate the dominant tensor
> contraction to an optimized BLAS.

The unqualified version — *"Python is as fast as Fortran"* — does not survive contact
with the VVP case and should not be quoted from this document.

---

## 8. Caveats

- **`-mcpu=native` is machine-specific.** Drop it if the binary must be portable; it
  accounts for roughly a third of the gain (at `p=15`, E=10×10: 56.79 → 37.52 ms from
  `-mcpu=native` alone, then → 17.15 ms with BLAS).
- **Both codes waste ~half of `apply_K`.** `M_1dx` and `M_1dy` are diagonal but stored
  and multiplied as full matrices, making those two contractions $O(p^3)$ where $O(p^2)$ would
  do. This penalizes both languages equally, so the comparison stands — but roughly half
  the flops are avoidable on either side, and this is the single largest remaining
  optimization for both.
- **Parity in the middle.** The reversal is clearest at the ends of the `p` range; at
  `p=10–12` the two are within a few percent on both meshes.
- **Single machine, single BLAS.** Results on Accelerate-backed NumPy, or on x86, may
  differ.

---

## 9. Reproducing

```bash
cd /Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo

# Fortran, as-built (the defective build, for comparison)
gfortran -O3 sem_2d.f90 -o sem_old -framework Accelerate

# Fortran, corrected
gfortran -O3 -mcpu=native -funroll-loops -fexternal-blas -fblas-matmul-limit=6 \
         sem_2d.f90 -o sem_2d_f90 -framework Accelerate

# confirm BLAS is actually wired in (0 = inert, 1 = live)
nm sem_2d_f90 | grep -ci dgemm

# run: <binary> <matrix file> <E_x> <E_y> <max_iters>
./sem_2d_f90 matrices_p15.txt 10 10 20000
```

NumPy side, matched to the above:

```python
from sem_2d_oo import ReferenceElement, Mesh2D, NumpySolver, ProblemDefinition
import numpy as np, time

prob = ProblemDefinition(lambda x, y: np.sin(4*np.pi*x)*np.sin(4*np.pi*y),
                         lambda x, y: 32*np.pi**2*np.sin(4*np.pi*x)*np.sin(4*np.pi*y))
ref  = ReferenceElement(15)
mesh = Mesh2D(10, 10, L_x=2.0, L_y=2.0, ref_el=ref)
ue, F = prob.evaluate(mesh)
s = NumpySolver(mesh); b = s.dss_np(F)
s.solve(b, max_iters=5)                      # warm up, then time solve() only
t0 = time.perf_counter(); u, it = s.solve(b, max_iters=20000, tol=1e-11)
print(1000*(time.perf_counter()-t0), "ms", it, "iters")
```
