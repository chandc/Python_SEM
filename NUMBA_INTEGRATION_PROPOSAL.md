# Proposal: fused numba kernels for the VVP operator

> **STATUS UPDATE (2026-08-07): IMPLEMENTED, with changes.** The numba backend
> now exists as an **option**, not a replacement — NumPy remains the default.
> See `NUMBA_BACKEND.md` for the delivered design and measurements.
>
> Two things in this document are superseded and must not be copied verbatim:
>
> 1. **The kernel source in §4a uses the pre-fix least-squares row weighting**
>    (`idt = fac1/dt` applied to `u`, instead of `fac1*u + dt*(...)`). That is the
>    weighting bug that made the BFS diverge before t=23. The delivered kernels in
>    `lssem2d/kernels_numba.py` carry `f1` and `dtl` separately and scale the two
>    momentum components of `su` on read in `_kernel_LT`.
> 2. **§5a's "no fallback path" was rejected.** A dispatch layer (`lssem2d/backend.py`)
>    was built instead, so §5b (the stranded anaconda interpreter) no longer applies.

**Status: PROPOSAL — not implemented.** Nothing in `lssem2d/` has been changed.
The kernels below were written and verified in a scratch area; the full source is
reproduced here so the work survives independently of that scratch directory.

**Date:** 2026-08-02
**Machine:** Apple M3 Max (12P + 4E), macOS 26.5.2
**Measured under:** a clean `uv` env — Python 3.12.7, NumPy 2.4.6 (Accelerate),
numba 0.66.0, scipy 1.18.0

---

## 1. The case for it, in numbers

Per operator application, on 36 elements at order 7 (9,216 DOF — the same mesh as the
Fortran's `cavity_36_7_elem_grid.dat`):

| | current | fused numba | speedup |
| :--- | ---: | ---: | ---: |
| `apply_L` | 75.2 µs | 19.5 | **3.86×** |
| `apply_LT` | 111.8 µs | 25.9 | **4.32×** |
| **`apply_A`** | **209.5 µs** | **60.3** | **3.47×** |

End-to-end, lid-driven cavity $Re=100$, dt=0.1, 6×6 at N=8, run to steady state
(`max|dU| < 1e-7`):

| config | steps | wall | speedup | max\|u−u_ref\| |
| :--- | ---: | ---: | ---: | ---: |
| current kernels, cgsfac=0 | 216 | 33.9 s | 1.00× | reference |
| current kernels, cgsfac=0.3 | 219 | 13.1 s | 2.59× | 2.6e-06 |
| fused numba, cgsfac=0 | 221 | 15.3 s | 2.21× | 1.2e-05 |
| **fused numba, cgsfac=0.3** | 215 | **7.2 s** | **4.73×** | 5.4e-07 |

### Why it closes the Fortran gap

A matched Fortran VVP matvec (`pmg_clean`, same mesh, same operator, diagonal
preconditioner) measures **53.6 µs** when built optimized. See
[FORTRAN_VS_NUMPY_BENCHMARK.md](./FORTRAN_VS_NUMPY_BENCHMARK.md) for that methodology.

    Python now      209.5 µs   ->  3.91x slower than Fortran
    Python fused     60.3 µs   ->  1.13x slower than Fortran

That is effectively parity — and smaller than the 1.74× the Fortran itself gains
just from dropping `-fcheck=bounds`, so which is "faster" depends on build flags.

Earlier in this work I claimed parity was not reachable. **That was wrong.** It holds
for NumPy — `apply_L` really is near NumPy's ceiling, which is why the batching and
numexpr ideas bought so little — but a fused `@njit` kernel *is* the Fortran loop,
compiled, so the flop-to-dispatch argument does not constrain it.

---

## 2. Where the time goes, and what does NOT work

Composition of `apply_A` before any change:

| | µs | % |
| :--- | ---: | ---: |
| `apply_LT` | 111.0 | 52.2% |
| `apply_L` | 76.2 | 35.8% |
| `gather_scatter` | 14.5 | 6.8% |
| mask ×2 | 4.3 | 2.0% |

Within `apply_L`: 8 derivative matmuls 52%, 4 `su` expressions 31%, copies 10%.
Derivative matmuls across both operators are **54% of the whole matvec**. At 8×8
blocks the BLAS call overhead dominates the arithmetic, which is why fusing only the
elementwise algebra caps out — the contractions have to be fused too.

Options measured and **rejected**:

| option | result |
| :--- | :--- |
| **numexpr** | **0.49× / 0.27× / 0.15×** at 1/4/8 threads — *slower*, monotonically worse with threads |
| **numba `prange`** | **0.82×** — *slower* than serial |
| batched derivatives (8 matmul calls → 2) | 1.26× — real but subsumed by full fusion |
| numba on elementwise only | 2.64× — subsumed by full fusion |

numexpr fails because the arrays are ~18 KB; there is nothing to amortize its chunking
and threading against. `prange` losing independently confirms the Apple-Silicon thread
contention noted in `SEM_BENCHMARK_REPORT.md` §4. **Use serial `@njit` only.**

---

## 3. Verification already performed

**Correctness** — relative error of fused vs current, random states, four meshes:

| mesh | `apply_L` | `apply_LT` | `apply_A` |
| :--- | ---: | ---: | ---: |
| 36 elem, order 7 | 2.6e-16 | 1.7e-16 | 1.8e-16 |
| 36 elem, order 8 | 1.8e-16 | 1.3e-16 | 4.0e-16 |
| 16 elem, order 5 | 2.2e-16 | 1.8e-16 | 3.5e-16 |
| 15 elem, order 9 | 1.5e-16 | 1.4e-16 | 1.4e-16 |

**Self-adjointness** — `⟨x,Ay⟩` vs `⟨y,Ax⟩` in the multiplicity-weighted inner product,
on continuous masked test vectors: **1.98e-16** fused vs 3.97e-16 current. CG stays valid.

> Note on how that test must be set up: random *local* arrays are not continuous and are
> therefore not in the operator's domain. Testing with them shows a spurious ~4%
> asymmetry in **both** implementations. Project the test vectors first
> (`gather_scatter(v)/mult`, then apply the mask). This cost me a false alarm.

---

## 4. Proposed design

Three files change. No runtime dispatch, no fallback path — see §5 for the assumption
that buys.

### Constraint that shapes it

Five files import `apply_L`/`apply_LT` directly from `lssem2d.lssem`
(`tests/test_adjoint.py`, `tests/test_lssem.py`, `tests/verification.py`,
`scratch/test_dge.py`, `scratch/test_jacobi.py`). `tests/test_adjoint.py` is the
mandatory gate named in `BUILD_PROMPTS_PYTHON.md`. **Keep the public names in place**
so every existing call site and test exercises the new kernels unchanged.

### 4a. New: `lssem2d/kernels_numba.py`

```python
"""Fused numba kernels for the VVP operator.

Each kernel performs the tensor-product contractions AND the elementwise algebra in a
single pass, writing directly into the native (nelem, n, n, 4) layout. This removes the
per-field strided reads, the BLAS call overhead on tiny (n x n) blocks, and every
intermediate temporary.

Serial @njit only -- prange measured 0.82x (slower) at this problem size.

Index conventions, taken from lssem2d/operators.py:
    dUdx(U)[e,i,j] = facx[e] * sum_k D[i,k] U[e,k,j]
    dUdy(U)[e,i,j] = facy[e] * sum_k D[j,k] U[e,i,k]
    DxT (S)[e,i,j] = facx[e] * sum_k D[k,i] S[e,k,j]
    DyT (S)[e,i,j] = facy[e] * sum_k D[k,j] S[e,i,k]
"""
import os
import numpy as np
from numba import njit

_FASTMATH = os.environ.get("LSSEM_FASTMATH", "1") == "1"


@njit(fastmath=_FASTMATH, boundscheck=False, cache=True)
def _kernel_L(U, D, facx, facy, wq, fu, fv, dfux, dfuy, dfvx, dfvy, nu, idt, out):
    NE, n = U.shape[0], U.shape[1]
    for e in range(NE):
        fx = facx[e]; fy = facy[e]
        for i in range(n):
            for j in range(n):
                ux = 0.0; vx = 0.0; px = 0.0; ox = 0.0
                uy = 0.0; vy = 0.0; py = 0.0; oy = 0.0
                for k in range(n):
                    dik = D[i, k]; djk = D[j, k]
                    ux += dik*U[e, k, j, 0]; vx += dik*U[e, k, j, 1]
                    px += dik*U[e, k, j, 2]; ox += dik*U[e, k, j, 3]
                    uy += djk*U[e, i, k, 0]; vy += djk*U[e, i, k, 1]
                    py += djk*U[e, i, k, 2]; oy += djk*U[e, i, k, 3]
                ux *= fx; vx *= fx; px *= fx; ox *= fx
                uy *= fy; vy *= fy; py *= fy; oy *= fy
                u = U[e, i, j, 0]; v = U[e, i, j, 1]; om = U[e, i, j, 3]
                w = wq[e, i, j]; a = fu[e, i, j]; b = fv[e, i, j]
                out[e, i, j, 0] = (idt*u + a*ux + b*uy
                                   + u*dfux[e, i, j] + v*dfuy[e, i, j] + px + nu*oy)*w
                out[e, i, j, 1] = (idt*v + a*vx + b*vy
                                   + u*dfvx[e, i, j] + v*dfvy[e, i, j] + py - nu*ox)*w
                out[e, i, j, 2] = (ux + vy)*w
                out[e, i, j, 3] = (om + uy - vx)*w


@njit(fastmath=_FASTMATH, boundscheck=False, cache=True)
def _kernel_LT(su, D, facx, facy, fu, fv, dfux, dfuy, dfvx, dfvy, nu, idt, out):
    NE, n = su.shape[0], su.shape[1]
    for e in range(NE):
        fx = facx[e]; fy = facy[e]
        for i in range(n):
            for j in range(n):
                tx1 = 0.0; tx2 = 0.0; tx3 = 0.0; tx4 = 0.0; txg1 = 0.0; txg3 = 0.0
                ty1 = 0.0; ty2 = 0.0; ty3 = 0.0; ty4 = 0.0; tyg2 = 0.0; tyg4 = 0.0
                for k in range(n):
                    dki = D[k, i]; dkj = D[k, j]
                    s1x = su[e, k, j, 0]; s2x = su[e, k, j, 1]
                    tx1 += dki*s1x
                    tx2 += dki*s2x
                    tx3 += dki*su[e, k, j, 2]
                    tx4 += dki*su[e, k, j, 3]
                    txg1 += dki*fu[e, k, j]*s1x        # Dx^T(fu*su1)
                    txg3 += dki*fu[e, k, j]*s2x        # Dx^T(fu*su2)
                    s1y = su[e, i, k, 0]; s2y = su[e, i, k, 1]
                    ty1 += dkj*s1y
                    ty2 += dkj*s2y
                    ty3 += dkj*su[e, i, k, 2]
                    ty4 += dkj*su[e, i, k, 3]
                    tyg2 += dkj*fv[e, i, k]*s1y        # Dy^T(fv*su1)
                    tyg4 += dkj*fv[e, i, k]*s2y        # Dy^T(fv*su2)
                tx1 *= fx; tx2 *= fx; tx3 *= fx; tx4 *= fx; txg1 *= fx; txg3 *= fx
                ty1 *= fy; ty2 *= fy; ty3 *= fy; ty4 *= fy; tyg2 *= fy; tyg4 *= fy
                s1 = su[e, i, j, 0]; s2 = su[e, i, j, 1]; s4 = su[e, i, j, 3]
                out[e, i, j, 0] = (idt*s1 + dfux[e, i, j]*s1 + dfvx[e, i, j]*s2
                                   + tx3 + txg1 + ty4 + tyg2)
                out[e, i, j, 1] = (idt*s2 + dfuy[e, i, j]*s1 + dfvy[e, i, j]*s2
                                   - tx4 + txg3 + ty3 + tyg4)
                out[e, i, j, 2] = tx1 + ty2
                out[e, i, j, 3] = s4 - nu*tx2 + nu*ty1


def _C(a):
    return a if a.flags.c_contiguous else np.ascontiguousarray(a)


def apply_L(state, U, fu, fv):
    m = state.mesh
    idt = state.fac1 / state.dt if state.dt != 0 else 0.0
    _kernel_L(_C(U), state._nb_D, m.facx, m.facy, m.wq, _C(fu), _C(fv),
              state.dfu_dx, state.dfu_dy, state.dfv_dx, state.dfv_dy,
              state.nu, idt, state._nb_su)
    return state._nb_su


def apply_LT(state, su, fu, fv):
    m = state.mesh
    idt = state.fac1 / state.dt if state.dt != 0 else 0.0
    _kernel_LT(_C(su), state._nb_D, m.facx, m.facy, _C(fu), _C(fv),
               state.dfu_dx, state.dfu_dy, state.dfv_dx, state.dfv_dy,
               state.nu, idt, state._nb_c)
    return state._nb_c


def warmup(state):
    """Pay JIT compilation once at setup, not inside a timed loop or the first step."""
    m = state.mesh; n = m.N + 1
    Z = np.zeros((m.nelem, n, n, 4)); z = np.zeros((m.nelem, n, n))
    apply_LT(state, apply_L(state, Z, z, z), z, z)
```

The `_C()` guard is not cosmetic: `newton_step` passes `fu = U[..., 0]`, a **strided
view**. Numba accepts it but compiles a slower path. The 2.21× end-to-end above was
measured *with* that handicap, so this should recover a little more.

### 4b. `lssem2d/lssem.py`

Demote the NumPy bodies to reference implementations and re-export the kernels:

```python
def _apply_L_reference(state, U, fu, fv):
    """NumPy reference. Not used in production; tests/test_backend_parity.py
       validates the numba kernels against this."""
    ...   # existing body, verbatim

def _apply_LT_reference(state, su, fu, fv):
    ...   # existing body, verbatim

from .kernels_numba import apply_L, apply_LT      # noqa: F401  (production path)
```

In `SolverState.__init__`:

```python
self._nb_D  = np.ascontiguousarray(D)
self._nb_su = np.empty((nelem, n, n, 4))
self._nb_c  = np.empty((nelem, n, n, 4))
```

**Also move the existing `u_c` / `su0_c` / `c0_c` buffers from `get_global_mask` into
`__init__`.** They are currently allocated inside a method that early-returns when the
mask is cached, so any path calling `apply_L` before the mask is built raises
`AttributeError`. That is a live bug in the NumPy path today, independent of this
proposal, and the reference implementation needs those buffers to keep working.

### 4c. New: `tests/test_backend_parity.py`

```python
import numpy as np, pytest
from lssem2d import lssem

@pytest.mark.parametrize("N,EX,EY", [(7,6,6), (8,6,6), (5,4,4), (9,3,5)])
def test_numba_matches_reference(N, EX, EY):
    state, U, fu, fv = _make_case(N, EX, EY)
    su_ref = lssem._apply_L_reference(state, U, fu, fv).copy()
    c_ref  = lssem._apply_LT_reference(state, su_ref, fu, fv).copy()
    su_new = lssem.apply_L(state, U, fu, fv).copy()
    c_new  = lssem.apply_LT(state, su_ref, fu, fv).copy()
    assert np.max(np.abs(su_ref-su_new)) / np.max(np.abs(su_ref)) < 1e-13
    assert np.max(np.abs(c_ref -c_new )) / np.max(np.abs(c_ref )) < 1e-13
```

Those four meshes are the ones already verified at ~1e-16, so 1e-13 leaves three
orders of headroom.

---

## 5. Decisions needing sign-off

### 5a. No fallback path — requires a dependency manifest

Dropping the `try/except` guard, `HAVE_NUMBA`, a `backend.py`, and the config key is
what makes this a 3-file change instead of 5. That is only safe if numba is guaranteed
present, and **right now it is not**: `sem_demo` has no `pyproject.toml`, no
`requirements.txt`, and the existing `.venv` contains mlx, torch, and matplotlib but
**not numba**. Today the assumption is luck, not a project property.

```toml
# pyproject.toml
[project]
name = "lssem2d"
requires-python = ">=3.11"
dependencies = ["numpy>=2.0", "scipy", "numba>=0.60", "tomli", "matplotlib"]
```

```bash
uv venv && uv pip install -e .
```

Without the manifest, an `ImportError` at import time replaces the graceful
degradation being removed.

### 5b. This strands the anaconda interpreter

**numba is currently unusable in `/Users/danielchan/opt/anaconda3`** — the simplest
possible `@njit` fails there with an unbox error, independent of any code here. With no
fallback, anything run through that interpreter stops working entirely rather than
falling back to NumPy. If notebooks or scripts depend on it, either fix that install or
keep the dispatch layer.

### 5c. `fastmath=True` — recommended on, but it is a real trade

It reorders floating-point arithmetic. Per call the difference is ~1e-16, but over
~85,000 CG iterations the trajectory shifts: 221 steps instead of 216. Steady states
agree to 0.001% of the u-range, and the fused+cgsfac run actually landed *closest* to
the reference (5.4e-07). If bit-reproducibility against current results matters more
than the speed, default it off. Exposed as `LSSEM_FASTMATH=0` either way.

### 5d. `cache=True`

Persists compiled code to `__pycache__` so compilation is ~1–2 s once rather than per
process start. It silently degrades to no caching when the directory is not writable.

---

## 6. Explicitly out of scope

- `gather_scatter` (6.8%) and the mask (2.0%) stay on the NumPy/scipy path. Folding
  them into the kernels is a further ~9% and can follow later.
- In-place CG vector ops (~2%) and the redundant `compute_jacobi` call in `step_bdf`
  (~1%, its result is overwritten by `newton_step` on its second line) are separate
  small cleanups.
- **The preconditioner.** At cgsfac=0.3 the solver still takes ~122 CG iterations per
  step. That is the LSSEM normal-equation conditioning (`cond(𝓛ᵀ𝓛) = cond(𝓛)²`) and no
  amount of kernel work touches it. It remains the largest single lever on total
  runtime and is independent of everything proposed here.
