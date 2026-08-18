"""Does the laminar channel's a_mass exemption survive a NON-ZERO residual?

    uv run --quiet python scratch/chan_amass_sweep.py <parabolic|uniform> <dt> <off|half|match|VALUE>

WHY THIS EXISTS.  GARTLING_VALIDATION.md sec 8 says the periodic/laminar channel
is exempt from the a_mass instability *because its residual is essentially zero*:
Poiseuille is exactly representable (J = 5.94e-27), so all four rows of the
functional vanish together and the weighting has nothing to trade off.  The BFS
sits at L2(div u) = 2.3e-02 and there the weights decide which row is sacrificed,
giving the measured threshold (stable <= 6.05, divergent >= 12.1).

3D_DEVELOPMENT_PLAN.md sec 0.2 turns on whether that exemption transfers to a
TURBULENT channel, whose residual is emphatically not zero.  This sweep is the
cheapest 2D proxy for that question:

  inlet = parabolic  -> exact solution representable, residual ~ 0   [CONTROL]
  inlet = uniform    -> flow must develop, rms div ~ 8e-02           [TEST]

Same grid, same everything else.  If the exemption is really about the residual,
the parabolic runs stay clean to arbitrarily large a_mass while the uniform runs
start failing near the BFS threshold.

a_mass = w_mass*fac1/dt = 1.5/dt at w_mom = w_mass = 1, so the dt values probe:

    dt = 0.025 -> a_mass = 60      (AC survived this on the BFS)
    dt = 0.0125 -> a_mass = 120    (AC failed at this on the BFS)
    dt = 0.005 -> a_mass = 300     (never tested anywhere)

The largest a_mass ever measured on ANY channel here is 30, and the 3D CFL
estimate is 150-1500, so this also closes a range that the plan currently has to
call "unmeasured" rather than "clean".

Runs to tmax = 15: on the BFS every divergent case blew up by t = 10-45, so this
window catches the failure mode without paying for a full steady solve.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from pois_ac import run                      # has a __main__ guard; safe to import

GRID = 'grids/channel_L12_12x2_N10_grid.dat'
TMAX, CAP, NSUB = 15.0, 6000, 5

if __name__ == '__main__':
    inlet = sys.argv[1]
    dt = float(sys.argv[2])
    kspec = sys.argv[3]
    a_mass = 1.5/dt
    if kspec == 'off':
        dtau = None
    elif kspec == 'half':
        dtau = 2.0/a_mass
    elif kspec == 'match':
        dtau = 1.0/a_mass
    else:
        dtau = 1.0/float(kspec)
    t0 = time.perf_counter()
    r = run(dt, dtau, cap=CAP, nsub=NSUB, tmax=TMAX, grid=GRID, inlet=inlet)
    kp = r.get('kappa_p', 0.0)
    print(f"{inlet:>10}{dt:>8g}{a_mass:>8.4g}{kp:>9.4g}{r['status']:>14}"
          f"{r['steps']:>7}{r.get('maxu', float('nan')):>9.4f}"
          f"{r.get('maxv', float('nan')):>10.2e}"
          f"{r.get('rms_div', float('nan')):>10.2e}"
          f"{r.get('l2_u', float('nan')):>10.2e}"
          f"{time.perf_counter()-t0:>8.0f}s", flush=True)
