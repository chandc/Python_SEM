# Review: Kim–Moin RK4-CN implementation vs the RK3-CN path

Date: 2026-08-24.  Companion: `SCHEME_COMPARISON.md` (equations side by side).
Code under review: `lssem3d/project.py::step_kim_moin` against D. C. Chan
thesis ch. 2 eqs (2.7)–(2.12); contrast `substage` (RKW3/SMR).

## 1. The implementation is faithful

Verified line by line: Jameson $\alpha = (1/4, 1/3, 1/2, 1)$ on convection only
(2.7); skew form $H = \tfrac12(\nabla\cdot(uu) + u\cdot\nabla u)$ (2.8); one CN
viscous solve (2.9); one projection (2.11)–(2.12).  Components verified in
isolation: CN reproduces analytic diffusive decay to 3.6e-14; RK4 skew
convection conserves energy to 6.5e-9 / 20 steps; projection scaling exact.
No transcription mistake.  As written, the scheme is stable.

## 2. The blow-ups were additions, not the scheme

The thesis method is pressure-free.  A grafted incremental `pc` +
rotational update `pc += phi - nu*div(u*)` fed the divergence floor
(weak-Laplacian projection is inexact) back into the pressure every step:
anti-dissipative at $\Delta t/2$, blow-up at step 184.  Dropping the
rotational term restored stability; dropping the whole graft was correct,
since measurement shows it buys nothing (velocities agree to 1.3e-5,
balances to 4 decimals).

## 3. The 23.6% excess dissipation is structural, not a bug

The four-stage convective sweep runs with no pressure gradient.  Starting
div-free, nothing opposes the compressive part of $u\cdot\nabla u$; the field
drifts $\Delta t\nabla p$ off the manifold.  Skew conserves *total* energy, so
the gradient mode's energy $\tfrac12\Delta t^2\|\nabla p\|^2$ is skimmed from
the physical field; the projection can only discard it.  Quantitative check
with the exact TGV pressure ($\|\nabla p\|^2 = 19.38$): predicted excess
fraction 0.2365 vs measured 0.2362; inviscid 20-step loss predicted 6.24e-3 vs
measured 6.23e-3; halves exactly with $\Delta t$ (0.214/0.107/0.054).

## 4. Why RK3-CN converged

Its pressure gradient sits inside the momentum RHS of every substage,
simultaneous with convection: the gradient mode is cancelled as produced,
never exceeds the lag error $O(\Delta t^2\|\partial_t\nabla p\|)$, and the
energy leak goes as its square — hence the measured ~700x gap (balance 1.0003
vs 1.236), not the ~8x a per-substage step-size argument predicts.

## 5. Regime, not method

Kim–Moin's home regime is CFL-limited fine-grid turbulence, where
$\tfrac12\Delta t\|\nabla p\|^2 / 2\nu\Omega$ is small.  Early TGV is the bad
corner: smooth flow permits large $\Delta t$ while $2\nu\Omega$ is minimal.
The run's own trajectory confirms: balance 1.236 at $t=0$ falling to 1.012 as
enstrophy grew 13x.

## 6. Remedy (implemented 2026-08-24)

Add the lagged $-\nabla p^{n-1}$ to $H$ inside the Jameson stages — the
interior analogue of the wall correction (2.10), zero extra solves; carry
$p^n = p^{n-1} + \phi^n$.  Gate: balance 1.2365 (step 1, $p=0$) then 1.0000
from step 2, 1.0043 at step 200; $|\phi|$ collapses 2450 to 0.4.  Defect gone.
