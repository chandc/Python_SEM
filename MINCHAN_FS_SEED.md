# FOSLS-3D minimal channel, seeded from the converged fractional-step field

Run record. Started 2026-09-02. Code: `scratch/minchan_seed_fs.py` (loader and
gates), `scratch/minchan.py` (driver). Source field:
`~/lssem_fs/scratch/_minchan_stat_E/chk_latest.npz` on the DGX Spark.

---

## 1. Why seed at all

Both prior FOSLS minimal-channel attempts were retired **without reaching
turbulence** (`3D_STATUS.md` §7P):

| run | outcome |
|---|---|
| `minchan_001` | 14 h, healthy on every logged diagnostic, carrying **relative divergence 1.1e−01** from a non-solenoidal random trip |
| `minchan_002` | 16 h to t = 1.170, every budget closed — and **relaminarised**, because making the trip solenoidal also made it 1.8× weaker (`rms_w` 0.243 → 0.135) and dropped it below the bypass-transition threshold |

A cold start to sustained turbulence was priced at **~12 days**. The
fractional-step code already has the state, so seeding skips the transient
entirely.

## 2. The source field

`_minchan_stat_E`, RK3 fractional step, 79 h of accumulated runtime:

| | |
|---|---|
| **Re_τ** | **180.3** (ν = 1/180 exactly) |
| u_τ | **1.0014 ± 0.0205** (last 2000 of 7401 samples) |
| statistics | t = 3.00 → 15.95, `nsamp` = 7401 |
| U⁺ at y⁺ = 1.0 | 1.048 (law of the wall: 1.004) |
| U⁺ at y⁺ = 30 / 50 / 100 | 14.12 / 15.35 / 17.18 |
| `u_rms` peak | **2.860** (KMM ≈ 2.7) |

Mean profile follows `U⁺ = y⁺` in the sublayer and tracks `2.5 ln y⁺ + 5.2` in
the log layer ~3–4% high — normal for a *minimal* channel, which over-predicts
the intercept.

## 3. Why the transfer is a copy, not a project

Both codes share `RE_TAU=180, DELTA=1, LX=π, LZ=0.34π, FX=1` and
`N=8, ex=6, ey=18, nz=32` → **108 elements, 9 nodes, 17 modes**. The
fractional-step `minchan.py` is a direct descendant of the FOSLS one.

Fractional step stores 4 complex fields; FOSLS carries 7 as 14 split-real:

| FOSLS slot | source |
|---|---|
| `U_, V_, W_` | direct copy |
| `P_` | direct copy |
| **`OX_, OY_, OZ_`** | **derived — discrete curl** (`channel3d._set_vorticity`) |

The curl must be **discrete, not analytic**: FOSLS carries vorticity as
independent unknowns whose definition rows `R₁–R₃` are least-squares residuals.
Verified exact — `|R_ωx| = |R_ωy| = 0.000e+00` before masking.

## 4. Gates on the converted field

| gate | value | reference |
|---|---|---|
| finite | yes | |
| `u_τ` | **1.0017** | source 1.0014 ± 0.0205 ✓ |
| `U_bulk` | 15.890 | KMM ≈ 15.6 (1.9% high) |
| **`rms_w`** | **0.8887** | `minchan_002` relaminarised below ~0.135 — **6.6× above** |
| strong-form `‖div u‖/‖u‖` | 9.89e−02 | see §5 |
| **J** | 2.809 | see §5 |

### 4.1 The J breakdown, against a control

| | J | continuity | ω_x | ω_y |
|---|---|---|---|---|
| **imported** | 2.809 | 48.4% | 9.5% | 41.8% |
| `minchan.py`'s **own** IC | 0.242 | **50.5%** | 2.1% | **44.4%** |

**The row distribution is essentially identical**, so the imported field is
structurally no worse than what the driver itself produces. J is 11.6× larger
only because the field is genuinely turbulent (`rms_w` 0.889 vs ~0.13).

The ω-row share is **not** an import artefact: `_set_vorticity` is exact, and the
residual appears only after masking, because velocities are zeroed at no-slip
walls while ω is left free by design (`bc.py`). Present in both fields.

## 5. The divergence, and why it is not `minchan_001` again

The source logs `div ≈ 2e−01` and the converted field measures
`‖div u‖/‖u‖ = 9.9e−02` — the same order that retired `minchan_001`. **It is not
the same defect.** Fractional step enforces `∇·u = 0` **weakly**, as a projection
against the pressure test space, so a **strong-form pointwise** measure reads
large by construction. `minchan_001`'s divergence was a genuine property of a
non-solenoidal random initial condition, not a form mismatch.

**This remains the one thing to watch.** FOSLS *penalises* `div u` and never
removes it (lesson L5), so the continuity row's 48% share of J is real work the
solver must do at t = 0. Whether it settles or fights is the first question the
run answers — and `J` is logged from step one precisely so the answer is visible,
which is what §7P's L16 says was missing the first two times.

## 6. Timestep

The imported field is fully turbulent, so CFL is tighter than for a laminar start:

| dt | CFL | |
|---|---|---|
| 2.0e−03 | 2.926 | **unstable** (RKW3 limit √3 = 1.732) — this is what `minchan_001` used |
| 1.0e−03 | 1.463 | OK |
| **8.0e−04** | **1.170** | **chosen** |
| 5.0e−04 | 0.731 | OK |
| 3.5e−04 | 0.512 | the source run's dt |

## 7. Status

**Set up; not yet launched.** IC written to
`scratch/fs_seed/minchan_ic_from_fs.npz`.
