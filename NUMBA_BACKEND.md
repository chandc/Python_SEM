# The numba backend

Fused `@njit` kernels for the VVP operator, available as an **option**. NumPy
remains the default — nothing changes unless numba is asked for explicitly.

Implemented 2026-08-07. Supersedes the design in
[NUMBA_INTEGRATION_PROPOSAL.md](./NUMBA_INTEGRATION_PROPOSAL.md), whose kernel
source is stale (see §5 below).

---

## 1. Using it

Two ways, both opt-in:

```bash
LSSEM_BACKEND=numba python your_script.py        # process-wide, read at import
```

```python
import lssem2d
lssem2d.set_backend('numba')     # at runtime, any time, reversible
lssem2d.get_backend()            # -> 'numba'
lssem2d.available('numba')       # probe without committing
```

The public names `apply_L` / `apply_LT` are unchanged, so every existing call
site and test picks up whichever backend is active with no edits.

Optional: pay JIT compilation up front rather than inside the first timed step.

```python
from lssem2d import kernels_numba
kernels_numba.warmup(state)      # requires update_linearisation() first
```

### Failure behaviour

Requesting `numba` when it is not importable **raises `ImportError`**; it does
not fall back silently. A silent fallback turns a missing dependency into an
unexplained 3x slowdown, which is precisely what corrupts a benchmark. Use
`available()` to branch deliberately.

---

## 2. Measured speedup

Apple M3 Max, Python 3.12.7, NumPy 2.4.6 (Accelerate), numba 0.66.0.
Per operator application, microseconds:

| mesh | DOF | | `apply_L` | `apply_LT` | `apply_A` |
|---|---|---|---|---|---|
| 6x6 order 7 | 9,216 | numpy | 79.0 | 125.3 | 227.9 |
| | | numba | 21.6 | 29.6 | 70.5 |
| | | **speedup** | **3.65x** | **4.24x** | **3.23x** |
| 6x6 order 8 | 11,664 | **speedup** | 4.40x | 3.91x | **3.38x** |
| 8x8 order 10 | 30,976 | **speedup** | 2.75x | 2.71x | **2.41x** |
| 8x8 order 12 | 43,264 | **speedup** | 2.31x | 1.96x | **1.92x** |

**The speedup declines with resolution** — 3.23x at 9k DOF down to 1.92x at 43k
— because NumPy's BLAS becomes progressively more efficient on larger blocks
while the fused kernel's advantage (avoiding call overhead on tiny blocks) is
largest when the blocks are small. The proposal measured only the 9k mesh and so
reported the best case.

End-to-end, cavity Re=1000, 4x4 order 8, dt=1.0, run to `max|dU| < 1e-8`:

| backend | steps | wall | speedup |
|---|---|---|---|
| numpy | 139 | 13.65 s | 1.00x |
| numba | 140 | 5.25 s | **2.60x** |

Steady states agree to **1.0e-04 max, 0.0077% of the u-range** (per component:
u 5.6e-06, v 4.8e-06, p 2.4e-06, om 1.0e-04). The one-step difference in
trajectory is the `fastmath` reordering, see §3.

### Interaction with the preconditioner

These are NumPy-matvec-relative numbers. Because numba makes the matvec ~2-3x
cheaper, it shifts the Jacobi/p-MG crossover in
[PRECONDITIONER_AND_DT_STUDY.md](./PRECONDITIONER_AND_DT_STUDY.md) toward
**larger** meshes: preconditioners that trade extra matvecs for fewer iterations
get relatively more expensive. Re-measure the crossover before relying on it
under numba.

---

## 3. `fastmath`

On by default; `LSSEM_FASTMATH=0` disables it. It permits floating-point
reassociation, which is worth a meaningful part of the speedup.

Per call the difference is ~1e-16 (round-off). Over a full run the trajectory
shifts slightly — 140 steps instead of 139 above — while steady states agree to
0.008% of the range. Turn it off if you need run-for-run bit-comparability with
the NumPy path rather than per-call agreement.

### A trap that was found and fixed

**numba's on-disk cache does not key on njit flags.** With `cache=True`, a kernel
compiled under `fastmath=True` is silently reused when `fastmath=False` is
requested, so `LSSEM_FASTMATH=0` appears to do nothing. Verified directly —
whichever flavour compiled first determined the result for both:

```
cache cleared, fastmath=0 first:  both runs -> -5.64503634916568409e-01
cache cleared, fastmath=1 first:  both runs -> -5.64503634916567965e-01
```

The differing checksums confirm the flag does change the arithmetic; the cache
was masking it. Fixed by giving each flavour its own cache directory
(`lssem2d/__nbcache__/{fastmath,strict}`, gitignored), set before `numba` is
imported since numba reads its configuration at import time.

---

## 4. Files

| file | role |
|---|---|
| `lssem2d/backend.py` | `get_backend` / `set_backend` / `available`, env resolution |
| `lssem2d/kernels_numba.py` | the two fused kernels, buffers, `warmup` |
| `lssem2d/lssem.py` | NumPy bodies renamed `_apply_L_numpy` / `_apply_LT_numpy`; public `apply_L` / `apply_LT` dispatch |
| `lssem2d/__init__.py` | re-exports the backend controls (new; package was previously namespace-only) |
| `lssem2d/tests/test_backend_parity.py` | the parity gate |

Dispatch is resolved once per backend switch via a listener registered with
`backend.register`, not per call, so there is no per-application lookup cost.
The numba work buffers (`_nb_D`, `_nb_su`, `_nb_c`) are allocated lazily on
first use, so selecting NumPy costs no extra memory.

---

## 5. The weighting bug this nearly reintroduced

**The kernel source in the proposal document is stale and must not be copied
verbatim.** It was written 2026-08-02, before the least-squares row-weighting fix.

It computes the momentum rows as `idt*u + N(u)` with `idt = fac1/dt`. The correct
form, matching `lssem_baseline.f90` `rhs()`, is `fac1*u + dt*N(u)`. These are the
same equation but differ by a factor `dt` **as least-squares rows**, so the old
form over-weights momentum by `1/dt` against continuity and the vorticity
definition. It is harmless where the residual is ~0 (cavity, Poiseuille) and
**diverges on under-resolved cases such as the BFS**.

The delivered kernels carry `f1` and `dtl` separately in `_kernel_L`, and
`_kernel_LT` scales the two momentum components of `su` by `dtl` on read — the
exact transpose, since the new operator is `R_new = S R_old` with
`S = diag(dt, dt, 1, 1)`.

Because the cavity cannot detect this class of error, the parity test runs at
**dt = 0.1, 1.0 and 0.0** (the steady form) across four meshes, rather than at one
convenient value.

---

## 6. Test coverage

`lssem2d/tests/test_backend_parity.py`, 16 tests, all skipped gracefully when
numba is absent:

- numba vs NumPy for `apply_L` and `apply_LT`, 4 meshes x 3 dt, tolerance 1e-13
  (measured ~2e-16, three orders of headroom)
- strided linearisation velocities — `newton_step` passes `fu = U[..., 0]`, which
  is a strided view; numba accepts it but compiles a slower path, so `_C()`
  guards it and this test pins the behaviour
- `set_backend` actually switches dispatch, and switching back reproduces the
  NumPy result **exactly** (`array_equal`, not a tolerance)
- default is `numpy`; unknown names raise
- self-adjointness under numba, `<x,Ay>` vs `<y,Ax>`, tolerance 1e-12

> The adjointness test projects its vectors first (`gather_scatter(v)/mult`, then
> mask). Random *local* arrays are not continuous and so are not in the
> operator's domain; testing with them shows a spurious ~4% asymmetry in **both**
> backends. That cost a false alarm during the original investigation.

Full suite: **47 passed** under `numpy`, under `numba`, and under
`numba + LSSEM_FASTMATH=0`.

(`lssem2d/tests/test_solver.py` remains excluded — it fails to collect on an
unrelated pre-existing issue, importing `cg_solve` which no longer exists.)
