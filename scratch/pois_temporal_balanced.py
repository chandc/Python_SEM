"""Temporal order with the BALANCED weighting w_mom = w_mass = sqrt(dt).
Same harness as pois_temporal.py (start-up plane Poiseuille, exact unsteady
solution, BDF2 seeded exactly); only the weights differ.  Reports the fitted
slope of the rms u error vs dt, next to the legacy and w=1 values of the
2026-08-12 study (2.039 / 2.042 at N=14)."""
import os, sys, json
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import pois_temporal as PT
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S

def run_w(N, dt, w):
    m = PT.build(N)
    st = SolverState(m, diff_matrix(N), nu=PT.NU, dt=dt, fac1=1.0, w_mom=w, w_mass=w)
    _, a_flux, _ = ls_coeffs(st)
    f = np.zeros((m.nelem, N+1, N+1, 4)); f[..., 0] = a_flux*2.0*PT.NU
    nsteps = int(round((PT.TEND-PT.T0)/dt))
    U = PT.state_at(m, N, PT.T0); hist = [U, PT.state_at(m, N, PT.T0-dt)]
    pin = (0, (N+1)//2, (N+1)//2)
    for s in range(nsteps):
        U = S.step_bdf(st, hist, time=PT.T0+s*dt, max_newton=PT.NEWTON, newton_tol=PT.NTOL, newton_factor=0.0, f_known=f,
                       pin_p=pin, cgsfac=PT.CGSFAC, cg_tol=PT.CGTOL, cg_max_iter=PT.CGMAX, line_search=False)
        if not np.all(np.isfinite(U)): return None
    du = U[..., 0] - PT.state_at(m, N, PT.TEND)[..., 0]
    return float(np.sqrt((du**2).mean()))

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    print(f'balanced weighting w_mom = w_mass = sqrt(dt), N={N}, t {PT.T0} -> {PT.TEND}')
    print(f'{"dt":>10} {"w=sqrt(dt)":>11} {"a_mass":>8} {"a_flux":>8} {"rms u err":>12}')
    errs = []
    for dt in PT.DTS:
        w = float(np.sqrt(dt)); e = run_w(N, dt, w); errs.append(e)
        print(f'{dt:>10g} {w:11.4f} {1.5/np.sqrt(dt):8.2f} {w:8.4f} {e:12.4e}', flush=True)
    print(f'slope (all 5): {PT.slope(PT.DTS, errs):.3f}   slope (coarse 3): {PT.slope(PT.DTS[:3], errs[:3]):.3f}')
    print('reference (TEMPORAL_ACCURACY_STUDY.md, N=14): legacy 2.039, w=1 2.042')
