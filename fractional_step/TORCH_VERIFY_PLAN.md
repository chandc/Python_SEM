# Verifying the PyTorch/CUDA backend on the Spark — plan

Phase 1 of `GPU_PORT_PLAN.md`, expanded. Written **before** porting, because the
point of this phase is to produce a **go/no-go for Phase 2** — the device-resident
solver is 2–3 days and carries the schedule risk, and it should not start until
the operator has been measured on real hardware rather than extrapolated from
einsum proxies.

---

## 0. What the container already gives us

Probed in `chandc/unsloth-dgx-spark:latest` (the image the §7O numbers were
measured in):

| | |
|---|---|
| torch | 2.10.0a0+nv25.11, CUDA, GB10 sm_121, FP64 verified |
| numpy | 2.1.0 |
| **numba** | **0.62.1** |
| scipy | 1.16.3 |
| pytest | 8.1.1 |

**numba being present is the useful surprise.** It means all three backends can
be compared *on one machine, against one numpy version* — so a discrepancy is
attributable to the backend rather than to the platform. The Mac has numpy 2.4.6
and numba 0.66.0; comparing across both at once would confound three variables.

---

## 0b. Blocker to clear first

`lssem3d/kernels_torch.py` currently does `from . import operator as OP` at module
scope while `operator._bind_backend` imports `kernels_torch` — a **circular
import**, which fails as soon as the backend is selected:

```
ImportError: cannot import name 'apply_L' from partially initialized module
             lssem3d.kernels_torch (most likely due to a circular import)
```

`kernels_numba.py` has the same structure and gets away with it only because of
import ordering. Fix both properly: take the handful of constants
(`NVAR`, `NROW`, field indices) as module-level literals or a lazy import, so a
kernel module never imports the module that binds it.

---

## V0 — environment gate: does the container reproduce the Mac?

**Nothing about torch is trusted until this passes.**

1. `rsync` the repo to `~/lssem3d_src` on the Spark.
2. Run the **existing 200-test suite** in the container under `numpy` and `numba`.

**Gate:** 200/200 pass, or every failure explained.

**Why this comes first.** If a test fails here it is a numpy-version, numba-version
or aarch64 issue — nothing to do with the port. Discovering that *after* torch
parity fails would send the debugging in the wrong direction. Two known
differences to watch: **numba 0.62.1 vs the Mac's 0.66.0**, and the `__nbcache__`
directory must not be carried over by rsync — it is keyed on neither the numba
version nor `fastmath` (§7M), so a stale cache from the Mac would be silently
reused.

---

## V1 — operator parity: torch(GPU) vs numpy, in the same container

Extend `test_backend_parity.py` to a third backend. The harness already sweeps
what matters and each case closes a specific silent failure (§7M):

| exercised | why |
|---|---|
| `kap ≠ 0` | a backend that drops AC still converges — to the wrong continuity equation |
| `rw ≠ 1`, incl. `w7 = 1e-4` | dropping row weights looks 5× *slower*, not wrong |
| `wq = None` | the unweighted operator the symmetry tests use |
| `k_z = 0` **and** `≠ 0` | half the `i·k` terms vanish at `k_z = 0`, hiding a sign error |
| `facx ≠ facy` | a swapped metric cannot pass on a non-square mesh |

**Plus two assertions this backend specifically needs**, and they are the whole
lesson of L15 — *verify what you are measuring*:

* the tensors inside the kernel are on **`cuda`**, not silently on CPU;
* their dtype is **`float64`**, not downcast.

MLX inverted a conclusion by silently casting float64→float32, and Legate
silently resolved a CPU-only build that still reported a GPU. Torch will not do
either, but the cost of asserting is nil and the cost of not asserting is a
retraction.

**Tolerance must be measured, not assumed.** GPU reductions reorder summation, so
exact agreement is impossible — the same reason numba shifts CG by ±1 iteration.
Expect ~1e-13–1e-14 relative rather than the 1e-16 numba achieves on CPU.

**Gate:** ≤1e-12 relative on every case, with the *observed* worst case recorded
in the section so a later regression is visible.

---

## V2 — cross-machine check

Same seeded input, **Mac numpy** vs **Spark torch**, looser tolerance.

V1 deliberately holds the machine fixed to isolate the backend. V2 deliberately
does not, to catch anything platform-level that an in-container comparison is
blind to. Failing V2 while passing V1 would point at numpy 2.1.0 vs 2.4.6 or at
aarch64, and would be worth knowing before any production run.

---

## V3 — real-operator performance, replacing every proxy so far

**Every GPU number quoted to date is an einsum proxy** — §7N and §7O timed
contractions with `ux+uy` standing in for the eight-row assembly, the `wq`/`rw`
multiplies and the transpose recombination. This measures the actual
`apply_L` + `apply_Lᵀ`.

Three shapes, all FP64:

| shape | dof | why |
|---|---|---|
| 48³ (36 elem, N=8, nk=25) | 1.02 M | the Re=400 TGV reference |
| 88³ (121 elem, N=8, nk=45) | 6.17 M | the §7O comparison point |
| **minimal channel** (108 elem, N=8, nk=17) | **2.08 M** | **the case we actually want to run** |

Four configurations: **Spark torch GPU**, **Spark numba CPU**, **Spark numpy**,
**Mac numba** (the incumbent). The Spark-numba column is the one that makes this
interpretable — it separates "the GPU is fast" from "the Grace CPU is slow."

Known reference points to beat: Mac numba is **9.4 ms** at 48³ and **59.6 ms** at
88³ for the complete fused operator.

---

## V4 — the go/no-go, stated in advance

Deciding the threshold *before* seeing the number is the point.

| V3 result (real operator, minimal-channel shape) | decision |
|---|---|
| torch GPU **≥ 3×** Mac numba | **proceed to Phase 2** — device-resident solver |
| **1.5–3×** | proceed only with the fused-CUDA-kernel path costed in; a naive port will disappoint |
| **< 1.5×** | **stop.** Re-open the alternatives: attack the 4786 CG/stage iteration count, or run reduced-scope on the Mac |

Rationale for 3×: the minimal channel is ~35 days on the Mac. A 3× operator gain
does not translate to 3× end-to-end — `gs`, the CG vector operations and the FFTs
are untouched by Phase 1 — so below 3× at the operator level the end-to-end
figure will not justify 2–3 days of Phase 2 plus its risk.

**What V3 still does not establish**, and must not be read as establishing:

* **gather-scatter is not measured** — irregular indexing, currently
  `scipy.sparse`, and the awkward part of any GPU port (`GPU_PORT_PLAN.md` §2).
* **the CG loop is not measured** — the operator is ~99% of a *CPU* step, but on
  GPU the untouched pieces become a larger fraction, exactly as they did for
  numba (§7M: 34% of a CG iteration ended up outside the fused kernel).
* the numbers say nothing about whether the run stays turbulent.

---

## Mechanics

```bash
# Mac -> Spark.  Exclude the numba cache: it keys on neither numba version nor
# fastmath, so a stale one would be silently reused (sec 7M).
rsync -av --exclude '__pycache__' --exclude '__nbcache__' --exclude '.venv' \
      --exclude 'scratch/*.npz' \
      /Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/ Spark:~/lssem3d_src/

# In-container.  --user so nothing lands root-owned; src read-only; results to a
# bind mount that survives --rm (GPU_PORT_PLAN.md sec 4).
ssh Spark 'mkdir -p ~/lssem_runs/verify && docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v ~/lssem3d_src:/src:ro -v ~/lssem_runs/verify:/out -w /src \
  chandc/unsloth-dgx-spark:latest \
  python3 -m pytest lssem3d/tests -q'
```

Note `/src` is mounted **read-only**, so pytest cannot write `.pyc` there —
`PYTHONDONTWRITEBYTECODE=1`, or copy into the container first.

---

## RESULTS (2026-08-22) — V0–V4 executed

### 0b. Circular import — FIXED

Both kernel modules now restate `NV`, `NR` and the field indices instead of
importing `operator.py`. Restating means they can *drift*, so
`test_kernel_constants_match_operator` pins them.

### V0 — environment gate: **PASS, 232/232**

One failure on the first attempt, and it was exactly what V0 is for:
`scratch/spectrum.py` hardcoded `/Users/danielchan/...` and is imported by
`test_spectrum_tool.py`, so the *suite* was unportable. Fixed to derive from
`__file__`. **~290 such references remain across ~135 scratch scripts** — only
this one blocks the tests, but any driver that must run on the Spark needs the
same treatment.

### V1 — torch(GPU) vs numpy parity: **PASS**

| | worst relative error |
|---|---|
| `apply_L`, full sweep (`kap`, `rw`, `wq`, `k_z=0` and `≠0`, `facx≠facy`) | **3.478e-16** |
| `apply_LT` | **2.799e-16** |
| full `normal_op` — numba / torch | **1.837e-16 / 1.837e-16** |

Device asserted `cuda` (NVIDIA GB10), dtype asserted `float64`.

**The predicted tolerance was wrong, pessimistically.** This plan expected
~1e-13–1e-14 from GPU reduction reordering; the measurement is 1e-16 —
indistinguishable from the numba backend, which suggests the residual is the
NumPy reference's own rounding rather than either backend.

### V3 — the real operator, replacing every proxy

`apply_L` + `apply_Lᵀ`, FP64, no einsum stand-ins:

| shape | dof | Spark numpy | Spark numba | **torch device-resident** | Mac numba |
|---|---|---|---|---|---|
| 48³ TGV ref | 1.02 M | 69.8 ms | 7.9 ms | **2.8 ms** | 10.1 ms |
| **minimal channel** | 2.08 M | 140.0 ms | 16.8 ms | **5.9 ms** | 19.4 ms |
| 88³ | 6.17 M | 380.7 ms | 57.2 ms | **17.6 ms** | 68.5 ms |

**torch GPU vs Mac numba: 3.6× / 3.3× / 3.9×** — consistent across an order of
magnitude in problem size. Against the Spark's *own* CPU it is 2.8–3.2×, so the
gain is the GPU and not a slow host.

**H2D/D2H is confirmed as the thing Phase 2 must eliminate.** Timing the NumPy
facade — which copies host→device→host per call, i.e. what Phase 1 alone would
deliver — gives **386.0 ms at 88³ against 17.6 ms device-resident, a 21.9×
penalty**, essentially erasing the GPU. (The facade column is erratic at the
smaller shapes and should not be read closely; the 88³ figure is the one that
matters and it is unambiguous.)

### V4 — the go/no-go: **PROCEED, but only just**

The threshold was set in advance at **≥3× Mac numba** on the minimal-channel
shape. Measured: **3.3×**. That clears it — barely.

**Read that number honestly.** 3.3× on the operator will *not* be 3.3×
end-to-end. §7M measured 34% of a CG iteration falling outside the fused kernel
on CPU; on GPU the operator is faster, so the untouched remainder — `gs`, the CG
vector operations, the FFTs — becomes a *larger* fraction. A realistic
end-to-end expectation is **2–2.5×**, i.e. the minimal channel drops from ~35
days to roughly **14–17 days**. Substantial, not transformative.

The upside case is unchanged and is now the main reason to continue: torch's
kernels are **unfused**, making ~30 passes over the state. §7M got **7.6×** from
fusing exactly that on the CPU, and `_kernel_L` is already written as explicit
scalar arithmetic — the form a CUDA kernel wants. A fused CUDA kernel is where
another 2–3× would come from.

---

## PHASE 2 — device-resident solver: DONE, and it beat the V4 forecast

`lssem3d/device.py` is the seam: an array-namespace dispatch plus a torch
gather-scatter. `solver3d.py`'s `gs`, `_dot`, `multiplicity_weight`,
`make_continuous` and `pcg` now run unchanged on NumPy arrays *or* torch tensors,
so there is one `pcg` rather than two that could drift.

**Whole preconditioned CG solve, identical iteration counts:**

| case | dof | Spark numpy | Spark numba | **torch GPU** | vs Spark numba | vs **Mac numba** |
|---|---|---|---|---|---|---|
| 48³-ish | 1.02 M | 52.2 s | 12.5 s | **3.0 s** | 4.13× | — |
| **minimal channel** | 2.08 M | 222.9 s | 52.2 s | **12.3 s** | **4.23×** | **3.7×** (46.0 s) |

CG counts 1370/1370/1370 — like-for-like.

**V4 forecast 2–2.5× end-to-end and was too pessimistic.** That estimate assumed
`gs`, the CG vector operations and the reductions would stay on the host and
become a larger fraction once the operator sped up. Phase 2 moved them onto the
device too, so instead of eroding the operator's 3.3× the whole solve *improved*
on it — **4.23×** against the Spark's own CPU, **3.7×** against the Mac.

### Three bugs the tests caught, all of the silent kind

**406 host transfers in 200 CG iterations.** `np.asarray(kz)` inside the kernel
facade calls `.numpy()` on a tensor — a synchronising copy, twice per iteration.
Correct answers, and the entire GPU advantage spent on PCIe. This is precisely
the failure a numerical unit test passes; it took a test that *counts transfers*.

**The facade forced every input to `device()`.** A caller holding CPU tensors got
a CUDA result, which then met a CPU mask: `Expected all tensors to be on the same
device`. On the Spark it raised; in a device-resident loop it would instead have
inserted a silent transfer per call. The backend does not choose where data
lives — the caller does.

**An `id(mesh)`-keyed cache.** CPython reuses ids after garbage collection, so a
new mesh could receive a dead one's index map — silently wrong gather-scatter
whenever the shapes happened to be compatible. Found via a one-off flaky failure
that three subsequent clean runs would have let us dismiss. Now a
`WeakKeyDictionary`.

### Gates

* 237/237 in-container **on the GPU**, with the transfer test genuinely
  exercising CUDA (it builds tensors on `KT.device()`, so a CPU-only run would
  prove nothing).
* Solutions agree to what the solve itself defines: two CG runs stopping at the
  same *residual* differ in the *solution* by up to κ·tol, and κ ≈ 1e4. A fixed
  `err < 1e-6` gate failed a correct port on first run (measured 1.19e-04 at
  tol = 1e-8). The test now asserts the disagreement **shrinks with tolerance** —
  a real defect would not.

### Still not established

`convect`/FFT are untouched (Phase 3), so a full *time step* is not yet measured
— only the solve, which is ~99% of it on CPU but a smaller share on GPU. And
nothing here says the minimal channel stays turbulent.

**Implication for the run:** ~35 days on the Mac becomes roughly **9–10 days**
at 3.7×, before Phase 3.

---

## Order of work

| | | risk |
|---|---|---|
| 0b | fix the circular import in both kernel modules | none |
| V0 | rsync; existing suite green in-container | low — but must be first |
| V1 | torch parity + device/dtype assertions | low |
| V2 | cross-machine spot check | low |
| V3 | real-operator timing, four configurations | none — measurement |
| V4 | apply the go/no-go above | — |

Roughly a day, and it either unlocks Phase 2 or saves the 2–3 days Phase 2 would
have cost.
