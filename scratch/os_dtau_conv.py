"""Is the dtau bias in the OS growth rate an artifact of UNDER-CONVERGED Newton?

The fixed point of the dtau-augmented normal equations satisfies

    L^T R + kappa E^T R = 0,     E = diag(1,1,0,0)

so the bias is O(kappa*R) and vanishes as R -> 0.  The Fig. 2 driver runs
max_newton = 2 with newton_tol = 0.0 -- exactly two sub-iterations, never
converged -- so R is finite at the end of every step and kappa*R is injected
1000 times.  If that is the mechanism, deepening the sub-iteration should
recover the undamped growth rate.

Control (dtau=None, nsub=6) separates "more Newton moves the answer" from
"the dtau bias went away".
"""
import os, sys, time
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.environ.setdefault('WFRAC', '0.15')   # Chan's mesh: 0.3 / 1.4 / 0.3
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S
from os_base2 import build, base_state, RE, NU, DT
from os_dtau import perturbation, SIGMA_REF, TEND

N = 14

def run(dtau, nsub):
    m = build(N); n = N+1
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0, dtau=dtau)
    _, a_flux, _ = ls_coeffs(st)
    U0 = base_state(m, N)
    P, _ = perturbation(m, N)
    U = U0 + P
    f = np.zeros_like(U0); f[..., 0] = a_flux*2.0*NU

    def epert(Uf):
        du = Uf[..., 0]-U0[..., 0]; dv = Uf[..., 1]
        return 0.5*float(np.sum((du*du+dv*dv)*m.wq))

    hist = [U]; pin = (0, n//2, n//2)
    nst = int(round(TEND/DT))
    ts, Es = [0.0], [epert(U)]
    t0 = time.perf_counter()
    for s in range(nst):
        U = S.step_bdf(st, hist, time=s*DT, max_newton=nsub,
                       newton_tol=0.0, newton_factor=0.0, f_known=f,
                       pin_p=pin, cgsfac=0.01, cg_tol=1e-14,
                       cg_max_iter=2000, line_search=False)
        if not np.all(np.isfinite(U)):
            return None
        ts.append((s+1)*DT); Es.append(epert(U))
    ts, Es = np.array(ts), np.array(Es)
    k = len(ts)//4
    sig = 0.5*np.polyfit(ts[k:], np.log(Es[k:]/Es[0]), 1)[0]
    return sig, time.perf_counter()-t0

print(f"OS growth rate, N={N}, dt={DT}, exact {SIGMA_REF}")
print(f"{'dtau':>8}{'kappa':>9}{'nsub':>6}{'sigma':>15}{'err':>11}{'wall':>8}")
for dtau, nsub in ((None, 2), (10.0, 2), (None, 6), (10.0, 6)):
    r = run(dtau, nsub)
    if r is None:
        print(f"{str(dtau):>8}{'':>9}{nsub:>6}{'  DIVERGED':>15}"); continue
    sig, w = r
    kap = 0.0 if dtau is None else DT/dtau
    print(f"{str(dtau):>8}{kap:>9.4f}{nsub:>6}{sig:>15.8f}"
          f"{(sig-SIGMA_REF)/SIGMA_REF:>10.3%}{w:>7.0f}s", flush=True)
