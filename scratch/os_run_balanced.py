"""Orr-Sommerfeld growth rate (Chan 1996 Fig. 2) with the BALANCED momentum
weighting w_mom = w_mass = sqrt(dt) versus legacy (a_mass=fac1, a_flux=dt).

Same harness as os_run_w.py (mesh 1x3, y: 0.6/0.8/0.6, Re=7500, alpha=1,
amp 1e-4, TEND=100).  Usage:

    python -u scratch/os_run_balanced.py --N 10 --dt 0.1 --w balanced

Saves scratch/os_bal_<w>_N<N>_dt<dt>.npz with t, E/E0, sigma.
"""
import os, sys, time, argparse
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S
from os_base2 import build, base_state, RE, NU
from os_run_w import perturbation, SIGMA_REF, TEND


def run(N, dt, w, tend=TEND, verbose=True, cgsfac=0.01):
    m = build(N)
    n = N+1
    if w == 'balanced':
        wv = np.sqrt(dt)
        st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, w_mom=wv, w_mass=wv)
    else:
        st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0)
    a_mass, a_flux, _ = ls_coeffs(st)
    if verbose:
        print(f"  {w}: N={N} dt={dt} a_mass={a_mass:.4g} a_flux={a_flux:.4g}", flush=True)
    U0 = base_state(m, N)
    P, c = perturbation(m, N)
    U = U0 + P
    f = np.zeros_like(U0)
    f[..., 0] = a_flux*2.0*NU           # forcing carries the momentum-row weight

    def epert(Uf):
        du = Uf[..., 0] - U0[..., 0]
        dv = Uf[..., 1] - U0[..., 1]
        return 0.5*float(np.sum((du*du + dv*dv)*m.wq))

    hist = [U]
    pin = (0, n//2, n//2)
    nsteps = int(round(tend/dt))
    ts, Es, its = [0.0], [epert(U)], []
    t0 = time.perf_counter()
    rep = max(1, nsteps//10)
    for s in range(nsteps):
        U = S.step_bdf(st, hist, time=s*dt, max_newton=2,
                       newton_tol=0.0, newton_factor=0.0, f_known=f,
                       pin_p=pin, cgsfac=cgsfac, cg_tol=1e-14,
                       cg_max_iter=4000, line_search=False)
        if not np.all(np.isfinite(U)):
            print(f"  NaN at step {s+1}", flush=True)
            return dict(status='NaN')
        ts.append((s+1)*dt); Es.append(epert(U))
        if verbose and (s+1) % rep == 0:
            print(f"    N={N} dt={dt} {w:8s} step {s+1:6d} t={(s+1)*dt:7.1f}  "
                  f"ln(E/E0) = {np.log(Es[-1]/Es[0]):+.5f}  "
                  f"{time.perf_counter()-t0:6.0f}s", flush=True)
    ts, Es = np.array(ts), np.array(Es)
    k = len(ts)//4
    sig = 0.5*np.polyfit(ts[k:], np.log(Es[k:]/Es[0]), 1)[0]
    return dict(status='ok', sigma=sig, ts=ts, Es=Es,
                err=abs(sig-SIGMA_REF)/SIGMA_REF, wall=time.perf_counter()-t0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=10)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--w', choices=('legacy', 'balanced'), default='balanced')
    ap.add_argument('--tend', type=float, default=TEND)
    ap.add_argument('--cgsfac', type=float, default=0.01)
    a = ap.parse_args()
    r = run(a.N, a.dt, a.w, a.tend, cgsfac=a.cgsfac)
    tag = (f"{a.w}_N{a.N}_dt{a.dt:g}" + (f"_cg{a.cgsfac:g}" if a.cgsfac != 0.01 else "")).replace('.', 'p')
    if r['status'] == 'ok':
        print(f"RESULT {a.w:8s} N={a.N:3d} dt={a.dt:<6g} sigma={r['sigma']:.8f} "
              f"err={r['err']:.3%} lnE_T={np.log(r['Es'][-1]/r['Es'][0]):+.5f} "
              f"wall={r['wall']:.0f}s", flush=True)
        np.savez(f"{SC}/os_bal_{tag}.npz", t=r['ts'], e=r['Es']/r['Es'][0],
                 sigma=np.array([r['sigma']]))
    else:
        print(f"RESULT {a.w} N={a.N} dt={a.dt} {r['status']}")
