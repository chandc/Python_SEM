# Running the channel case on the GB10 with PyTorch — plan, cost, and I/O

Companion to `3D_STATUS.md` §7N (why the Spark, not MLX) and §7O (why PyTorch,
not cuPyNumeric). This is the *how*, the *how long*, and the part that bites
people: **getting results out of the container onto disk.**

---

## 0. The strategic question, answered first

The port buys roughly **3–7× over what the Mac already does today with numba** —
not the 85× that a naive `torch 5.3 ms vs numpy 450 ms` comparison suggests. That
ratio is against the *unoptimised* NumPy path, and we no longer run that path.

| target | Mac + numba (today) | GB10 + torch (ported) | worth 6–8 days of work? |
|---|---|---|---|
| Stage 5 channel (validation) | seconds | seconds — **GPU loses** at 0.06 M dof | no |
| **minimal channel** (4×12 elem, Nz=24) | ~14 h | ~2–5 h | **probably not** |
| **full M7** (20×12 elem, Nz=128) | ~15–30 days | ~3–10 days | **yes** |

**If the goal is the minimal channel, do not port.** The Mac runs it overnight
with the numba backend that already exists and is already validated. The port
pays for itself only on full M7, or if M7-class runs will be *repeated*.

Error bars on the runtime column are wide — see §5.

---

## 1. What actually has to move to the GPU

The trap is thinking this is "port the matvec." It is not. **The entire CG loop
must be device-resident**; a single host↔device copy per iteration would cost
more than the matvec saves. At ~680 CG iterations per step, one 49 MB round trip
per iteration is ~65 GB of PCIe traffic per step.

| component | today | port | risk |
|---|---|---|---|
| `operator.apply_L` / `apply_LT` | numba fused | `kernels_torch.py` | **low** — the numba kernel is an exact spec |
| `solver3d.gs` (gather-scatter) | `scipy.sparse` Q, Qᵀ | `index_add_` + gather on `mesh.gidx` | **medium** — see §2 |
| `solver3d._dot`, CG vector ops | numpy | torch, on device | low |
| `solver3d.pcg` | numpy | dtype/device-agnostic rewrite | low |
| `jacobi_diagonal_analytic` | numpy | compute once on CPU, `.to(dev)` | low |
| `convect.convective` + `fourier` | numpy FFT | `torch.fft` | low |
| `bc.build_mask`, mesh, LGL | numpy | **stays on CPU**, transferred once | none |
| drivers (`channel3d*.py`) | numpy | device-aware | low |

`lssem2d` is **not** touched — same constraint as every other backend.

---

## 2. Gather-scatter: the one genuinely new piece

`lssem2d.assembly.gather_scatter` does `QT @ (Q @ U_flat)` with scipy sparse.
Q is a pure gather matrix — each local dof maps to exactly one global dof — so
the whole operation is a segmented sum and a broadcast:

```python
idx = torch.as_tensor(mesh.gidx.reshape(-1), device=dev)     # (nlocal,)

def gs(U):                                    # (nel, n, n, nv, nk)
    flat = U.reshape(-1, U.shape[-2]*U.shape[-1])
    g = torch.zeros(nglobal, flat.shape[1], dtype=U.dtype, device=dev)
    g.index_add_(0, idx, flat)                # gather (Q)
    return g[idx].reshape(U.shape)            # scatter (Q^T)
```

**Two hazards, both real:**

**Non-determinism.** `index_add_` on CUDA uses atomics, so the summation order
varies run to run and results are **not bit-reproducible**. This project's whole
validation method is bit-level parity against a reference, so that matters.
`torch.use_deterministic_algorithms(True)` forces a deterministic kernel at some
cost — turn it **on for the parity tests**, and decide separately whether to
keep it in production.

**FP64 atomics.** Supported on GB10, but slower than FP32 atomics. `gs` runs
once per matvec, so if it turns out to dominate, the fallback is a precomputed
CSR `torch.sparse.mm`, which is deterministic and atomic-free.

---

## 3. Step by step

### Phase 1 — torch backend for the operator (≈1 day, low risk)

1. `lssem3d/kernels_torch.py`: `apply_L` / `apply_LT`, same signatures as
   `kernels_numba.py`. Write the **per-field 4-D einsum form** — §7O measured it
   at **5.3 ms vs 10.1 ms** for the "obviously vectorised" 5-D batched form.
   Benchmark both anyway; the preference is shape-sensitive and inverted between
   libraries.
2. `backend.py`: `VALID = ('numpy', 'numba', 'torch')`, extend `_bind_backend`.
3. `test_backend_parity.py`: add torch to the existing sweep — `kap≠0`, row
   weights, `wq=None`, `k_z=0` **and** `≠0`, `facx≠facy`. The harness exists;
   this is a parametrize change.

**Gate:** torch matches NumPy to ≤1e-12 relative on every case.

### Phase 2 — device-resident solver (≈2–3 days, medium risk)

4. `gs` per §2, with a CPU-vs-GPU equivalence test on a **periodic** mesh —
   `test_bc.py` already has `_periodic_mesh` and the shared-node fixtures that
   caught the one-copy pin bug.
5. `normal_op`, `_dot`, `multiplicity_weight`, `make_continuous`, `pcg` made
   device-agnostic (`xp`-style dispatch, or a thin array-namespace shim).
6. Assert **no host↔device transfer inside the CG loop** — a `torch.cuda`
   sync-count or profiler check in the test, not a code review.

**Gate:** `pcg` on GPU reaches the same residual in ±2 iterations of the NumPy
path (exact match is impossible — different summation order, same reason as
numba, §7M).

### Phase 3 — convection and FFT (≈1 day, low risk)

7. `torch.fft.rfft`/`irfft` with the 3/2-rule dealiasing, preserving the layout
   assertions in `convect.py`.

**Gate:** the existing dealiasing negative control still fails when disabled.

### Phase 4 — driver and I/O (≈1 day) — see §4

8. `scratch/channel3d_torch.py`: device selection, checkpointing, restart.

### Phase 5 — validation (≈1 day, non-negotiable)

9. Re-run the cases with **known answers**, on GPU:
   * **Stokes decay** — analytic σ = 9.3137399, both `kz0` and `span` families.
     `span` fires every `i·k_z` term the channel leaves dormant.
   * **M2 cavity** at `k_z = 0`.
   * **Stage 5 channel**, against `validate_row7_stage5.json` — same `E/E0`,
     same mean-profile error.

**Gate:** σ to the same 8 significant figures the numba backend reproduced.
Anything less and the port is not done.

### Phase 6 — production

10. One step timed on the real grid → **re-price §5 before committing** to a
    multi-day run.

**Total: ~6–8 engineering days**, and Phase 2 carries the schedule risk.

---

## 4. Docker output — the part that loses runs

Container filesystems are destroyed by `--rm`. **Everything written outside a
bind mount is gone**, including a week of checkpoints.

### The run command

```bash
# ON THE SPARK. Host directory first -- bind mounts do not create it with your
# ownership, and a root-created directory is a permissions fight later.
mkdir -p ~/lssem_runs/m7_001

docker run -d --name m7_001 \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v ~/lssem_runs/m7_001:/runs \
  -v ~/lssem3d_src:/src:ro \
  -w /src \
  -e LSSEM3D_BACKEND=torch \
  -e PYTHONUNBUFFERED=1 \
  chandc/unsloth-dgx-spark:latest \
  python3 scratch/channel3d_torch.py --out /runs --checkpoint-every 100
```

**Every flag that matters, and why:**

| flag | why |
|---|---|
| `-d` | detached — an ssh drop must not kill a multi-day run |
| `--user $(id -u):$(id -g)` | **without it the container runs as root and every output file is root-owned on the host** — you cannot `rsync` them back without sudo |
| `-v ~/lssem_runs/...:/runs` | the only writes that survive |
| `-v ~/lssem3d_src:/src:ro` | source read-only — the container cannot corrupt it |
| `--ipc=host`, `--ulimit` | NVIDIA's recommendation; the base image warns without them |
| `-e PYTHONUNBUFFERED=1` | otherwise `docker logs` shows nothing for hours |
| `--gpus all` | no GPU otherwise |

Note `--restart` is deliberately **absent**: a diverged run should not be
auto-restarted into the same divergence. Restart from a checkpoint, deliberately.

### What the driver must write, all under `/runs`

* `checkpoint_<step>.npz` every N steps — **the run's insurance.** Standing rule
  in this project: never re-solve to recover a field.
* `run.log` — per-step `t`, `E`, `CG`, wall, in the format `tgv3d_re400.log`
  uses, so existing plotting works unchanged.
* `config.json` — grid, `dt`, `nu`, backend, image digest, git SHA. A result
  whose configuration is not recorded alongside it is not reproducible.
* `diag.npz` — the time series, written incrementally.

Write to `tmp` then `os.replace()` — an atomic rename means a crash mid-write
cannot corrupt the last good checkpoint.

### Monitoring and retrieval

```bash
ssh Spark 'docker logs -f m7_001'                        # live
ssh Spark 'tail -5 ~/lssem_runs/m7_001/run.log'          # cheaper
rsync -av Spark:~/lssem_runs/m7_001/ scratch/m7_001/     # pull to the Mac
```

Disk: the Spark has 1.9 TB free. Full-M7 checkpoints are ~140 MB each
(17.7 M dof × 8 B); at every 100 steps over 20 k steps that is **~28 GB**. Fine,
but not free — prune or thin as the run proceeds.

---

## 5. Turnaround estimate, and why the error bars are wide

**Measured inputs** (`3D_STATUS.md` §7M, §7N, §7O):

| | |
|---|---|
| Mac numba, complete fused operator @ 6.17 M dof | 59.6 ms |
| torch GB10, derivative contractions @ 6.17 M dof | 5.3 ms |
| Re=400 TGV reference, 48³ | 27.1 s/step, measured over 1312 steps |

**Estimates:**

| target | Mac numba | GB10 torch |
|---|---|---|
| minimal channel (4×12 elem, Nz=24, ~0.7 M dof) | ~14 h | **~2–5 h** |
| full M7 (20×12 elem, Nz=128, ~17.7 M dof) | ~15–30 days | **~3–10 days** |

**Four things widen these**, in descending order:

1. **Row assembly, `wq`/`rw` and `gs` are not in the 5.3 ms.** They are
   elementwise and bandwidth-bound — at 141.8 GB/s and 49 MB of state, a few ms.
   The Mac numba figure **includes** them, so the true ratio is smaller than
   59.6/5.3 = 11× suggests. I use **3–7×** as the defensible range.
2. **CG iteration count under *h*-refinement is unmeasured.** §8.6 established
   iterations are flat in the implicit coefficient; that says nothing about
   refining the mesh. Bracketed [flat … 2×].
3. **`dt` must shrink with the CFL limit** as the grid refines.
4. **The reference runs used mode-parallelism**, so part of the naive ratio is
   already spent.

**Therefore Phase 6 step 10 exists.** One timed step on the real grid replaces
every assumption above with a measurement, and costs minutes. Do not commit a
multi-day run to this table.

---

## 6. What could sink it

| risk | signal | mitigation |
|---|---|---|
| `gs` atomics dominate | profile shows `index_add_` ≫ matvec | precomputed CSR `torch.sparse.mm` |
| non-deterministic results | parity test flaps | `use_deterministic_algorithms(True)` |
| host↔device copy in the CG loop | GPU util low, wall ≫ predicted | sync-count assertion in the test |
| torch einsum slow at *our* shapes | Phase 1 benchmark | try both forms (§7O); fall back to a CUDA extension |
| FP64 throttling bites harder than measured | Phase 1 | already characterised: 4.4×, and it still wins |

The escape hatch if einsum disappoints: **`_kernel_L` is already written as
explicit scalar arithmetic over `(elem, node, mode)`** — which is exactly the
shape of a CUDA kernel. §7M got 7.6× from fusing on the CPU; the same fusion in
CUDA is the upside case, and the code is already in the form a port needs.
