"""The 3D counterpart of figs/chan_fig1_pref.png: Stokes decay, p-refinement.

    uv run --quiet python scratch/stokes3d_pref.py run <N>     # dt sweep at one N
    uv run --quiet python scratch/stokes3d_pref.py span        # spanwise-mode check
    uv run --quiet python scratch/stokes3d_pref.py plot        # assemble the figure

Configuration is the production verdict of 3D_STATUS.md sec 8.2: ROW WEIGHTS ON,
NO OPERATOR-AC.  Each run integrates the exact Stokes eigenmode on the periodic
channel (2x4 elements, Nz = 8) and fits sigma from ln(E/E0); the analytic rate
is sigma = 9.3137399 for BOTH mode families measured here --

    kz0   alpha = 1, k_z = 0   the 2D mode embedded in the 3D code
    span  alpha = 0, k_z = 1   no x-dependence; only v, w, omega_x live, and
                               every i*k_z term is exercised

which must agree by symmetry (k^2 = alpha^2 + k_z^2 = 1 in both).  Left panel:
both families decaying on the single analytic line -- the 3D content the 2D
figure could not show.  Right panel: relative error in sigma vs dt per N, with
the slope-2 reference (the design order: RK3 is the convective half only, CN
caps the mixed scheme at 2), the shared temporal regime, and the spatial floor
dropping with N.

Every solve is guarded against hitting max_iter -- a capped solve promotes the
preconditioner into the scheme (3D_STATUS.md sec 8.4) and its 'result' is
wherever CG stopped.  Runs save to scratch/stokes3d_pref_N<N>.npz.
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SC)
os.chdir(ROOT)
import numpy as np
import stokes3d as SD
from lssem3d import parallel as PAR

SIGMA = SD.SIGMA_2D
DTS = (0.01, 0.005, 0.0025, 0.00125, 0.000625)
TEND = 0.05
MAXIT = 60000

# Guard: refuse capped solves rather than fitting sigma to garbage.
_pcg = PAR.pcg
def _guarded(*a, **kw):
    x, it, r = _pcg(*a, **kw)
    assert it < kw.get('max_iter', MAXIT), f'CG CAPPED at {it}'
    return x, it, r
PAR.pcg = _guarded


def sweep(N, mode='kz0', dts=DTS):
    s = SD.setup(N=N)
    U0, meta = SD.initial_state(s, mode=mode)
    rows = []
    for dt in dts:
        t0 = time.perf_counter()
        r = SD.measure_sigma(s, U0, dt, 0.0, tend=TEND, rowweight=True,
                             tol=1e-12, max_iter=MAXIT)
        assert r['status'] == 'ok', r
        r['err'] = abs(r['sigma'] - SIGMA)/SIGMA
        r['wall'] = time.perf_counter() - t0
        rows.append(r)
        print(f"N={N} {mode}: dt={dt:g}  sigma={r['sigma']:.6f}  "
              f"rel err={r['err']:.3e}  CG={r['cg']}  {r['wall']:.0f}s", flush=True)
    np.savez(f'{SC}/stokes3d_pref_{mode}_N{N}.npz',
             dts=np.array([r['dt'] for r in rows]),
             errs=np.array([r['err'] for r in rows]),
             sigmas=np.array([r['sigma'] for r in rows]),
             cg=np.array([r['cg'] for r in rows]),
             ts=np.array(rows[2]['ts']), Es=np.array(rows[2]['Es']),
             ts_f=np.array(rows[-1]['ts']), Es_f=np.array(rows[-1]['Es']),
             N=N, mode=mode, tend=TEND)


def plot():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    NS = [6, 8, 10]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.6))

    # ---- left: decay trajectories on the analytic line ----
    tt = np.linspace(0, TEND, 200)
    ax1.plot(tt, -2.0*SIGMA*tt, 'k-', lw=2.2,
             label=f'analytical   $\\sigma$ = {SIGMA:.6f}')
    d8 = np.load(f'{SC}/stokes3d_pref_kz0_N8.npz')
    ax1.plot(d8['ts'][::1], np.log(d8['Es']/d8['Es'][0]), 'o', ms=7, mfc='none',
             color='C0', label='kz0 mode ($\\alpha$=1, $k_z$=0), dt = 0.0025')
    dsp = np.load(f'{SC}/stokes3d_pref_span_N8.npz')
    ax1.plot(dsp['ts'], np.log(dsp['Es']/dsp['Es'][0]), 's', ms=7, mfc='none',
             color='C1', label='SPAN mode ($\\alpha$=0, $k_z$=1), dt = 0.0025')
    ax1.set_xlabel('Time'); ax1.set_ylabel('Natural Log of Total Kinetic Energy  $E/E_0$')
    ax1.set_title('3D Stokes decay, periodic channel\n2$\\times$4 elements, order 8, '
                  '$N_z$ = 8 — row weights on, AC off')
    ax1.legend(loc='lower left'); ax1.grid(alpha=0.3)

    # ---- right: error vs dt per N ----
    marks = {6: 'o', 8: 's', 10: '^'}
    slopes = {}
    for N in NS:
        d = np.load(f'{SC}/stokes3d_pref_kz0_N{N}.npz')
        dts, errs = d['dts'], d['errs']
        # fit over the temporal regime (drop the finest point, where the
        # spatial floor may bite)
        sl = np.polyfit(np.log(dts[:-1]), np.log(errs[:-1]), 1)[0]
        slopes[N] = sl
        ax2.loglog(dts, errs, marks[N]+'-', ms=8, mfc='none',
                   label=f'N = {N}   (slope {sl:.2f})')
    dsp = np.load(f'{SC}/stokes3d_pref_span_N8.npz')
    ax2.loglog(dsp['dts'], dsp['errs'], 'd--', ms=7, color='C3', alpha=0.8,
               label=f"SPAN mode, N = 8 (slope "
                     f"{np.polyfit(np.log(dsp['dts'][:-1]), np.log(dsp['errs'][:-1]), 1)[0]:.2f})")
    ref = np.array([2e-3, 2e-2])
    ax2.loglog(ref, 1.68e-4*(ref/0.01)**2, 'k--', lw=1.5, label='slope 2 reference')
    ax2.axvspan(1.1e-3, 1.1e-2, color='0.9', zorder=0)
    ax2.text(4.5e-3, 1.6e-6, 'temporal regime\n(all orders coincide)',
             ha='center', fontsize=10)
    ax2.annotate('spatial floor emerging at N = 6',
                 xy=(6.25e-4, 1.03e-6), xytext=(1.05e-3, 4.2e-7),
                 fontsize=9, arrowprops=dict(arrowstyle='->', lw=1.0))
    ax2.set_xlabel('Time Step Size'); ax2.set_ylabel('Relative error in $\\sigma$')
    ax2.set_title('Temporal accuracy vs polynomial order — RKW3/CN design order is 2')
    ax2.legend(loc='upper left', fontsize=9); ax2.grid(alpha=0.3, which='both')

    e10 = np.load(f'{SC}/stokes3d_pref_kz0_N10.npz')['errs']
    fig.suptitle(
        f"3D counterpart of Chan (1996) Fig. 1 (cf. figs/chan_fig1_pref.png).  "
        f"Row weights on, operator-AC OFF.  At dt = 6.25e-4, N = 10: rel err in "
        f"$\\sigma$ = {e10[-1]:.2e}.\n"
        f"The kz0 and SPAN families must share $\\sigma$ = {SIGMA:.6f} by symmetry "
        f"($k^2 = \\alpha^2 + k_z^2 = 1$): the SPAN curve is the check every "
        f"$k_z$ = 0 test is blind to.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = f'{ROOT}/figs/stokes3d_pref.png'
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'plot'
    if mode == 'run':
        sweep(int(sys.argv[2]))
    elif mode == 'span':
        sweep(8, mode='span')
    elif mode == 'plot':
        plot()
    else:
        raise SystemExit(f'unknown mode {mode!r}')
