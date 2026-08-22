# CORIA-CFD TGV benchmark, Re = 1600 (step 2)

Source: <https://benchmark.coria-cfd.fr/index.php/Step_2> ("3D single-component
Taylor-Green vortex"), raw data from <.../index.php/Results>.

Case: triply periodic $(2\pi L_0)^3$ box, $u_0 = L_0 = 1$, $\nu = 6.25\times10^{-4}$
so $Re = u_0L_0/\nu = 1600$; standard TGV initial condition; integrated to
$t = 20\,\tau_\mathrm{ref}$. The database's own reference solution is the
pseudo-spectral RLPK code at $512^3$, published by van Rees et al. (2011);
three codes are compared against it (YALES2 finite-volume, Nek5000 spectral
element, DINO).

Files here: YALES2 at $512^3$ for four Courant numbers -- a *temporal*
resolution study at fixed spatial resolution.

    YALES2_512_CFL0.10.txt   0.20  0.30  0.60
    columns:  1:t   2:KE   3:eps          (2001 rows, dt = 0.01, t = 0..20)

VERIFIED ON DOWNLOAD, and both checks passed exactly:

  * `KE(0)` = 0.12499706 against the analytic 1/8 -- so KE is the **mean
    kinetic energy density**, the same normalisation this repo uses, with no
    ambiguity of the kind that had to be resolved for Gourianov et al.
    (`TGV_VALIDATION.md` sec 8.1).
  * `eps(t = 0.02)` = 4.6875e-04 against the analytic
    $2\nu\Omega_0/V = 2\nu\cdot 3/8$ = 4.6875e-04, to eight digits. (The first
    two rows, 1.81e-04 and 3.25e-04, are one-sided start-up artefacts of their
    time differencing -- do not use them.)

Peak dissipation in this data: **0.01276 at t = 8.95** ($t/T_0$ = 1.42).

WHY IT IS NOT YET USED FOR A DIRECT COMPARISON: the database is Re = 1600 only,
and our TGV runs are Re = 100 / 400 / 800. Resolving Re = 1600 properly needs
$k_\mathrm{max}\eta \gtrsim 1$, i.e. $\gtrsim 192^3$ ($\eta$ = 0.0118 here) --
weeks of wall clock at the measured 88^3 rate. See `TGV_VALIDATION.md` sec 9.
