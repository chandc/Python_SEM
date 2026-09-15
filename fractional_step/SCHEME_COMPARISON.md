# RK4-CN (Kim–Moin) vs RK3-CN (RKW3 substage): side-by-side

Both advance $u^n \to u^{n+1}$ over $\Delta t$ on the same spatial machinery —
SEM weak operators, Fourier in $z$, skew-symmetric convection
$N(u) = -\tfrac12[\nabla\cdot(uu) + u\cdot\nabla u]$, weak pressure Poisson,
PCG — and differ only in the time splitting.  Implementations:
`lssem3d/project.py` — `step_kim_moin` (A) and `substage` (B).
Reference for A: D. C. Chan, thesis ch. 2, eqs (2.7)–(2.12).
Reference for B: Spalart, Moser & Rogers (1991).
Companion analysis: `KIM_MOIN_REVIEW.md`.

## Scheme A — Kim–Moin RK4-CN (one fractional step per $\Delta t$)

**A1. Convection — Jameson four-stage RK, full step, no solves.**

$$u^{(0)} = u^n, \qquad
u^{(\ell)} = u^n + \Delta t\,\alpha_\ell\left[N(u^{(\ell-1)}) - \nabla p^{n-1}\right],
\qquad \alpha = (\tfrac14, \tfrac13, \tfrac12, 1)$$

$$u^p = u^{(4)}$$

The $-\nabla p^{n-1}$ term is the 2026-08-24 stage-pressure remedy; the thesis
original (2.7) has $N$ alone — see "energy defect" below.

**A2. Diffusion — ONE Crank–Nicolson solve over the whole step (2.9).**

$$\frac{u^* - u^p}{\Delta t} = \frac{\nu}{2}\nabla^2(u^* + u^n)
\;\;\Longrightarrow\;\;
\Big[\tfrac{2}{\Delta t}M + \nu(K + k_z^2 M)\Big]u^*
= \tfrac{2}{\Delta t}M u^p - \nu(K + k_z^2 M)u^n$$

Wall value (2.10), inert for periodic TGV:
$\hat u\vert_w = u^{n+1}\vert_w + \Delta t\,\nabla\phi^{n-1}\vert_w$.

**A3. Projection — ONE Poisson solve, full-step scale (2.11)–(2.12).**

$$\nabla^2\phi^n = \frac{\nabla\cdot u^*}{\Delta t}, \qquad
u^{n+1} = u^* - \Delta t\,\nabla\phi^n, \qquad
p^n = p^{n-1} + \phi^n$$

Per step: 1 Helmholtz + 1 Poisson.  **Measured 0.97 s/step** (Spark GB10, 88^3).

## Scheme B — RKW3-CN substage (three fractional steps per $\Delta t$)

Substages $k = 1,2,3$ with

$$\gamma = (\tfrac{8}{15}, \tfrac{5}{12}, \tfrac34),\quad
\zeta = (0, -\tfrac{17}{60}, -\tfrac{5}{12}),\quad
\alpha = (\tfrac{29}{96}, -\tfrac{3}{40}, \tfrac16),\quad
\beta = (\tfrac{37}{160}, \tfrac{5}{24}, \tfrac16)$$

each advancing $h_k = (\gamma_k + \zeta_k)\Delta t$, with $\sum h_k = \Delta t$,
$\alpha_k + \beta_k = \gamma_k + \zeta_k$ (asserted at import), and
$c_k = 1/(\beta_k\Delta t)$.

**B1. Explicit assembly — convection AND pressure together, viscous explicit half.**

$$r^k = u^{k-1}
+ \Delta t\left[\gamma_k N(u^{k-1}) + \zeta_k N(u^{k-2})\right]
+ \alpha_k\Delta t\,\nu\nabla^2 u^{k-1}
- \beta_k\Delta t\,\nabla p^{k-1}$$

**B2. Diffusion — CN across the substage (implicit half $\beta_k$).**

$$(c_k - \nu\nabla^2)\,\hat u = c_k\,r^k
\;\;\Longleftrightarrow\;\;
\big[c_k M + \nu(K + k_z^2 M)\big]\hat u = c_k M r^k$$

**B3. Projection — substage scale $\beta_k\Delta t$, rotational pressure update.**

$$\nabla^2\phi = \frac{\nabla\cdot\hat u}{\beta_k\Delta t}, \qquad
u^k = \hat u - \beta_k\Delta t\,\nabla\phi, \qquad
p^k = p^{k-1} + \phi - \nu\,\nabla\cdot\hat u$$

Per step: 3 Helmholtz + 3 Poisson (3 velocity preconditioners, one per $c_k$).
**Measured 2.75 s/step.**

## Contrast

| | A: RK4-CN Kim–Moin | B: RKW3-CN substage |
|---|---|---|
| convective stages / step | 4 (Jameson), full $\Delta t$ | 3 (SMR), $h_k$ each |
| convective stability limit | CFL $\le 2\sqrt2$ | CFL $\le \sqrt3$ |
| viscous solves / step | 1, at $\lambda = 2/\Delta t$ | 3, at $\lambda = c_k$ |
| projections / step | 1, scale $\Delta t$ | 3, scale $\beta_k\Delta t$ |
| pressure in momentum | thesis: none; remedy: lagged $\nabla p^{n-1}$ in the RK stages | $\beta_k\Delta t\,\nabla p^{k-1}$, simultaneous with convection |
| pressure update | $p^n = p^{n-1} + \phi$ (plain) | rotational, $-\nu\nabla\cdot\hat u$ term |
| drift off div-free manifold before projection | thesis: $\Delta t\|\nabla p\|$; remedy: $\Delta t\|\nabla(p - p^{n-1})\| = O(\Delta t^2)$ | $O(\Delta t^2)$, corrected 3x/step |
| energy-balance excess (TGV Re=800, $\Delta t = 0.00567$) | thesis form: **23.6%**, clean $O(\Delta t)$ (0.214/0.107/0.054 under halving); with stage pressure: **0.25% at step 100** | 0.03% |
| wall accuracy (measured) | order ~2.2 via (2.10) | order ~1.6 |
| cost / step | **0.97 s** | 2.75 s |

## The one governing difference

Where the pressure force acts.  In B it sits inside every momentum RHS,
cancelling the compressive part of $N$ as it is produced; what reaches each
projection is $O(\Delta t^2)$ and the discarded energy is $O(\Delta t^4)$ per
step.  In thesis-A no pressure acts until after the four-stage sweep: the field
drifts $\Delta t\,\nabla p$ off the divergence-free manifold, and — because the
skew form conserves *total* energy — the energy in that gradient mode,
$\tfrac12\Delta t^2\|\nabla p\|^2$ per step, is skimmed from the physical field.
The projection can only discard it; no post-hoc pressure treatment recovers it
(measured: incremental-in-CN changes the velocity by 1.3e-5 and the balance
not at all).  Predicted excess $\tfrac12\Delta t\|\nabla p\|^2 / 2\nu\Omega$ =
0.2365 vs measured 0.2362 at $t = 0$, with $\|\nabla p\|^2 = 19.38$ from the
exact TGV pressure.

The stage-pressure remedy moves A's pressure to where B keeps it, at zero
extra solves, and eliminates the energy defect while the flow is smooth
(balance 1.0000 from step 2, vs 1.236).  **But on this spatial discretisation
it is NOT stable into the sharpening flow.**  Both variants -- accumulated
$p^n = p^{n-1} + \phi^n$ and memoryless $\nabla^2 p^n = \nabla\cdot N(u^n)$
re-solved each step -- destabilise at $t \approx 5.1$, exactly where the TGV
vortex sheets roll up (accumulated: enstrophy spike, balance 0.68, solver
stall at step ~900; memoryless: $|\mathrm{div}\,u|/|u|$ grows 3e-3
$\to$ 1.4 over steps 300..1000 and saturates -- no NaN, but garbage).
The pressure-free original passes the same window untroubled.

**Root cause: discrete operator incompatibility.**  The pressure is solved
with the weak Laplacian $K$, but the stages take divergence with the strong
operator $D$, and $D\!\cdot\!G \ne K$ on collocated C0 SEM -- the same
mismatch behind the divergence floor and the broken `dg_pressure`.  So
$\nabla\cdot(N - \nabla p) \ne 0$ at high wavenumber, the four-stage sweep
integrates that residue at full-$\Delta t$ amplitude, and the (weak)
projection cannot see it.  Kim & Moin's staggered finite-difference mesh
satisfies $D\!\cdot\!G = L$ exactly -- operator compatibility is designed
into their grid, and is why the scheme's long turbulent history does not
transfer as-is to this discretisation.  `substage` tolerates the mismatch
by re-projecting 3x per step at 1/5 the amplitude.

**Status: for periodic production use scheme B.**  The principled fix for A
is the consistent SEM pressure operator $E = D M^{-1} D^T$ (Deville-Fischer-
Mund $P_N$-$P_N$ pseudo-Laplacian), which would remove the floor everywhere
and preserve A's 3x cost advantage.

## Failed remedies, kept for the record

- **Incremental pressure in the CN solve** (with the required factor 2 on the
  $M\nabla p$ term): velocity change 1.3e-5, balance unchanged — the energy is
  skimmed during the sweep, before the CN solve runs.
- **Rotational update $p \mathrel{+}= \phi - \nu\nabla\cdot u^*$ in A**: feeds
  the divergence floor back into $p$ each step; anti-dissipative by
  $\Delta t/2$, blow-up at step 184.  (B tolerates the same term because it
  re-projects 3x per step at $\tfrac15$ the amplitude.)
- **$\Delta t$ refinement alone**: works ($O(\Delta t)$, ratios 2.00) but needs
  ~40x smaller step for 1% balance.
- **D.G consistent projection** (`dg_pressure`): NaN at step 2 in A; operator
  asymmetric (2.8e-2) — broken, do not use.
