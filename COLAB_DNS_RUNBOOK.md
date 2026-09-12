# Runbook: the $Re_\tau = 180$ minimal-channel DNS on Colab, one night at a time

The production run behind Section 10 of the paper (`PAPER_DRAFT.md`): VVP-FOSLS,
6×18 spectral elements at $N=8$ with 32 Fourier modes in $z$, continuing run01
from $t=4.96$ toward $t=30$.

**Notebook:** [`colab/lssem_channel_dns.ipynb`](colab/lssem_channel_dns.ipynb) →
<https://colab.research.google.com/github/chandc/Python_SEM/blob/main/colab/lssem_channel_dns.ipynb>
**Supervisor:** [`colab/run_channel_dns.py`](colab/run_channel_dns.py)
**Statistics:** [`colab/stats_window.py`](colab/stats_window.py)
**Time-step justification:** [`scratch/dns_timescales.py`](scratch/dns_timescales.py),
REFERENCE_DATA_RE180.md §7

---

## 1. The shape of the problem, and why the tooling looks like this

A Colab session is shorter than this run and can end without warning. Everything
below follows from taking that as the design constraint rather than fighting it:

| constraint | consequence |
|---|---|
| a session is ~10–12 h; the run is 2–4 nights | the supervisor sets its own wall-clock deadline and stops **before** the VM is reclaimed, so the session ends with a clean checkpoint rather than a dead kernel |
| the VM's disk dies with it | checkpoints, log and statistics are copied to Drive every 10 minutes; the local disk is only a fast scratch |
| a browser disconnect ends the session | a disconnect costs at most the steps since the last checkpoint (100) and the last sync (10 min) — never the run |
| restarts must not restart the *flow* | `minchan.run` reloads the convective history (`Nprev`) and the running statistics with the state; `ZETA[0] = 0` means stage 0 never uses carried convective history, so a restart is exact even if $\Delta t$ changes |
| a forgotten upload could waste a night | starting from the tripped initial condition requires an explicit `--from-scratch`; otherwise the supervisor refuses to start rather than silently beginning a fresh transition that is not comparable with run01 |

## 2. Nightly procedure

**First night:** cells 1 (GPU), 2 (code), 3 (Drive), 5 (smoke test, ~2 min),
then 6 with `HOURS` set a little below the expected session length.

**Every night after:** cells 1, 2, 3, 6. Cell 3 prints how far the run has got
and the last log lines; cell 6 finds the newest checkpoint on Drive, copies it
locally and continues. The smoke test is only worth repeating after a code
change.

**Any morning:** cell 7 (which limit set the step) and cell 8 (statistics over a
window that excludes the transient).

Checks before leaving it: the GPU is an A100 (fp64 on an L4 or T4 is 1/32–1/64
of its fp32 rate — about 20× slower than the numbers below); the seed checkpoint
line in cell 3 says `True` on the first night; and the first log line has
appeared, which means the preconditioner built.

## 3. What it runs, and why these are not defaults

| setting | value | why |
|---|---|---|
| weighting | `legacy` | Convection is explicit RKW3, so each implicit stage is a Stokes **projection** with a non-solenoidal right-hand side and the constraint rows must dominate. The balanced weighting that Sections 3–6 of the paper prove correct for implicit convection is **wrong here**: it drives the divergence from 8.6e−4 to 3.3e−1 in one step (BALANCED_CONDENSED_PLAN.md §1.5). |
| preconditioner | `vsbatch`, `share_precond=1` | Condensed vertex-patch Schwarz with a $p=2$ coarse space: 72 CG iterations per stage against Jacobi's 4675, step identical to Jacobi in every logged digit. One build at the middle stage's $c$ serves all three (the three differ by 1.39×), for a third of the memory and build time. |
| $\Delta t$ | 8e−4 | run01's value, physics-limited (§5), and unchanged so tonight's samples merge with run01's. |
| checkpoint | every 100 steps | ~3–18 min of work at risk, 24 MB each. Written atomically (temp name, then rename), so a sync never copies a half-written file. |
| CG tolerance | 1e−6 relative | production setting; unchanged from run01. |

## 4. Where the data lives

```
MyDrive/lssem_data/checkpoint_0006200.npz    run01 at t = 4.96 — the seed, needed once
MyDrive/lssem_dns/                           the run
    checkpoint_XXXXXXX.npz                   newest 3 + every 2500 steps (24 MB each)
    stats_XXXXXXX.npz                        every sync, kept forever (~15 kB)
    run.log  diag.npz  config.json           appended/overwritten each sync
    window.png                               written by cell 8
```

`run.log` is **append mode** and will hold several nights' worth: split it on the
`# minimal channel` header lines before analysing, or on the `[Ns]` elapsed field
resetting.

## 5. Statistics, and the transient

`PlaneStats` accumulates running **sums** from $t=0$, so the average carried in
any one checkpoint includes run01's start-up transient and is not the production
number. It does not need to be reset: sums are additive, so

$$\langle q\rangle_{[a,b]}=\frac{\text{sums}_b-\text{sums}_a}{n_b-n_a}$$

is exact for any window. That is why a small stats file is archived under its
step number at every sync — the averaging window is chosen *after* the run, from
the $u_\tau$ history, rather than guessed before it. `colab/stats_window.py`
differences two snapshots (or two checkpoints, which carry the same sums under
`stats_*` keys) and prints the peaks against the literature marks plus the
total-stress balance

$$-\langle u'v'\rangle^+ + \mathrm{d}U^+/\mathrm{d}y^+ = 1-y/\delta,$$

which needs no reference data and indicts $u_\tau$, the forcing balance or the
averaging rather than the physics when it fails. Verified on run01: the window
$t=2.4$–$5.0$ gives $u'_{\rm rms}$ 2.84, $w'$ 1.03, $v'$ 0.89, $Re_\tau$ 182,
balance closing to 0.075 — noisier than the full-record 0.023, as a 2.6-turnover
window should be.

## 6. Reading the log

```
t=   4.968 u_tau=0.9937 U_b=15.840 rms_w=0.9201 E=897.30 eps=103.54 div=8.64e-04 conv=0.000 CFL=1.11 CG=72 [612s]
```

| field | healthy | what it means if not |
|---|---|---|
| `u_tau` | ≈ 1 | prescribed by the forcing, so a drift toward 0 means the near-wall cycle has died. The minimal box is intermittent **by design** (Jiménez & Moin), so this is a finding to report, not necessarily a bug — but it must be seen while it happens. The driver warns below 0.5. |
| `CFL` | < 1.732 | above the RKW3 limit the explicit convection is unstable; the driver flags it. |
| `div` | ~1e−3 | the projection is doing its work. A jump of two orders means the weighting or the preconditioner is wrong for this stage. |
| `CG` | ≈ 72/stage | a climb means the shared preconditioner has gone stale for the current field; rebuild by restarting the session. |
| `rms_w` | ≈ 0.9–1.0 | spanwise fluctuation; collapse precedes a `u_tau` collapse. |
| `[Ns]` | — | cumulative seconds **within this session**, not since the run began. |

## 7. Cost, and how many nights

Per-step cost depends on which apply path the build uses (BALANCED_CONDENSED_PLAN.md
§8b): measured 11.5 s/step on an A100 before the matrix-product apply, 25.4 s on
a GB10 with it, and ~2 s projected on an A100 with matrix products plus CUDA-graph
replay. A 10-hour session is therefore roughly 3,000–18,000 steps, i.e. 2.4–14
turnovers, and $t=4.96\to30$ takes two to four nights. The supervisor prints the
measured rate 30 steps in, along with what fits in the remaining budget.

Stopping early is legitimate: statistics converge like $1/\sqrt{T}$ and cell 8
gives the profiles from whatever has accumulated, with the total-stress balance
saying whether the average is good enough to quote.

## 8. Failure modes and recovery

| symptom | cause | what to do |
|---|---|---|
| supervisor exits with "no checkpoint found" | the seed is not on Drive | upload `checkpoint_0006200.npz` to `MyDrive/lssem_data/`; do **not** pass `--from-scratch` unless a fresh transition is actually wanted |
| "not an A100" in cell 1 | Colab gave a different device | change the runtime type, or accept ~20× slower fp64 |
| session dies mid-run | Colab reclaimed the VM | nothing to repair: rerun cells 1–3 and 6, which resume from the newest checkpoint on Drive |
| `CG` climbing above ~100 | preconditioner stale for the evolved field | restart the session so it is rebuilt at the current state |
| `nan` / `BLEWUP` in the log | CFL exceeded or a solver failure | the driver aborts on a non-finite state; resume from the last good checkpoint with a smaller `--dt` and report it — at this $\Delta t$ it should not happen |
| Drive full | 24 MB checkpoints | the supervisor already keeps only the newest 3 plus every 2500 steps; delete older archives by hand if needed |

## 9. Why the step is 8e−4 (summary; full measurement in REFERENCE_DATA_RE180.md §7)

Measured on run01's own fields, dissipation from velocity gradients rather than
the stored $\omega$ (an independent VVP unknown, left free at the walls where
dissipation peaks): minimum Kolmogorov time $\tau_\eta=5.2\times10^{-3}$ at the
wall, convective CFL 1.11 of the RKW3 limit $\sqrt3$. Physics allows
$5.2\times10^{-4}$ ($\tau_\eta/10$); stability allows $1.24\times10^{-3}$.
**Physics binds, by 2.4×.** The production $8\times10^{-4}$ resolves $\tau_\eta$
by 6.5× at $\Delta t^+=0.14$, running at 64 % of the stability limit.

Two things follow for this project. Explicit convection is defensible at this
Reynolds number *because* the two limits are close; at higher $Re_\tau$ the CFL
limit tightens with the grid while $\tau_\eta^+$ only falls like
$Re_\tau^{-1/2}$, and implicit convection — with the weighting result that comes
with it — starts to matter. And a DNS is **forced** to small $\Delta t$, hence to
large $c=\mathrm{fac}_1/\Delta t$, which is exactly the regime where pointwise
relaxation cannot see the divergence-free modes: the step cannot be enlarged to
suit the solver, so the solver has to be built for the step.

## 10. When $t=30$ is reached

1. `colab/stats_window.py A B --out window.png` over the stationary window, with
   A chosen past the transient from the $u_\tau$ history.
2. Compare against the five reference databases (REFERENCE_DATA_RE180.md §1,
   §4) — mean profile, $u'v'w'$ rms, $-\langle u'v'\rangle$, and the vorticity
   rms, which is the FOSLS-specific test since $\omega$ is a primary unknown.
3. Keep the final checkpoint and the full stats snapshot series; they are the
   figure-7 source for the paper.
4. Section 10 of `PAPER_DRAFT.md` is written around this run; only its statistics
   paragraph is outstanding.
