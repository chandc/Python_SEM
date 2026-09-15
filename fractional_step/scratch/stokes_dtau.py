"""Chan (1996) Fig. 1: Stokes decay in a periodic channel.

Setup per the paper: two elements streamwise over [0, 2pi], four wall-to-wall,
6th-order Legendre, periodic in x, no-slip walls, pressure fixed at one point,
IC = the analytic Stokes eigenmode.  Chan's three time steps are 0.0025,
0.00125 and 0.000625, and he reports sigma = 9.313316 (0.0045% error).

Our eigenproblem gives sigma = 9.3137399 for alpha = 1, nu = 1, half-height 1.

Solver configuration is the F77 one: legacy weights, nsub = 2 with no Newton
convergence test, no line search, cgsfac = 0.01, cg_tol = 1e-14, nitcgs = 1000,
Jacobi preconditioner.

The amplitude check matters: our solver always carries u.grad u, so the Stokes
limit only holds if the perturbation is small.  Each dt is run at two amplitudes
and the rates must agree.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from stokes_ic import stokes_ic, energy, slowest_mode, ALPHA, NU

LX, LY = 2.0*np.pi, 2.0          # half-height 1 -> full height 2
N, EX, EY = int(os.environ.get('NORD', '6')), 2, 4
TEND = 0.1                        # Fig. 1 x-axis
SIGMA_REF = 9.313316              # Chan
CHAN_DTS = (0.0025, 0.00125, 0.000625)
# Wider span for the temporal-accuracy panel, to match Chan's own right-hand
# figure.  The window has to adapt: sigma = 9.31 means E ~ exp(-18.6 t), so a
# large dt needs a SHORT integration or the energy underflows, while still
# giving enough samples to fit a slope.  nsteps = max(6, 0.1/dt) capped at
# T = 0.6 keeps E/E0 above ~1e-5 everywhere.
ACC_DTS = (0.0025,)


def build():
    m = build_channel(LX, LY, EX, EY, N, bcs=(0, 0, 1, 1))   # W/E interior, S/N no-slip
    m.periodic_x = LX
    m.compute_global_indices()
    return m


def run(dt, amp, verbose=False):
    m = build()
    n = N + 1
    # eigenmode is defined on y in [-1, 1]; the mesh spans [0, 2]
    ymid = 0.5*(m.ynod.min() + m.ynod.max())
    shifted = type(m).__new__(type(m))
    shifted.__dict__ = dict(m.__dict__)
    shifted.ynod = m.ynod - ymid
    U0, info = stokes_ic(shifted, amp=amp)

    # pressure pinned at one interior node, as the paper states
    pin = (0, n//2, n//2)

    _dtv = os.environ.get('DTAU','none')
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, dtau=(None if _dtv=='none' else float(_dtv)))
    U = U0.copy(); hist = [U]
    E0 = energy(m, U)
    ts, Es = [0.0], [E0]
    nsteps = max(6, int(round(TEND/dt)))
    if nsteps*dt > 0.6:                    # keep E/E0 above ~1e-5
        nsteps = max(6, int(0.6/dt))
    t0 = time.perf_counter()
    for s in range(nsteps):
        U = S.step_bdf(st, hist, time=s*dt, max_newton=2,
                       newton_tol=0.0, newton_factor=0.0,
                       pin_p=pin, cgsfac=0.01, cg_tol=1e-14,
                       cg_max_iter=1000, line_search=False)
        if not np.all(np.isfinite(U)):
            return dict(dt=dt, amp=amp, status='NaN')
        ts.append((s+1)*dt); Es.append(energy(m, U))
    wall = time.perf_counter()-t0

    ts, Es = np.array(ts), np.array(Es)
    # fit ln(E/E0) = -2 sigma t over the second half, past any startup transient
    k = len(ts)//2
    A = np.polyfit(ts[k:], np.log(Es[k:]/E0), 1)
    sigma = -0.5*A[0]

    D = diff_matrix(N)
    div = np.sqrt(np.mean((dUdx(np.ascontiguousarray(U[..., 0]), D, m.facx) +
                           dUdy(np.ascontiguousarray(U[..., 1]), D, m.facy))**2))
    return dict(dt=dt, amp=amp, status='ok', sigma=sigma, ts=ts, Es=Es, E0=E0,
                err=abs(sigma-SIGMA_REF)/SIGMA_REF, wall=wall,
                nsteps=nsteps, EE=Es[-1]/E0, div=float(div), info=info)


if __name__ == '__main__':
    b1, c, _ = slowest_mode()
    print("Chan (1996) Fig. 1 -- Stokes decay, periodic channel")
    print(f"  mesh {EX}x{EY} elements, order {N}, x in [0, {LX:.4f}] periodic, "
          f"y in [0, {LY}] (half-height 1)")
    print(f"  analytic:  beta_1 = {b1:.7f},  sigma = {NU*(ALPHA**2+b1**2):.7f}")
    print(f"  Chan:      sigma = {SIGMA_REF}\n")
    print(f"{'dt':>10}{'amp':>9}{'steps':>7}{'sigma':>13}{'err vs Chan':>13}"
          f"{'E(T)/E0':>11}{'rms div':>11}{'wall':>8}")
    traces = {}
    for dt in ACC_DTS:
        for amp in ((1e-3, 5e-4) if dt in CHAN_DTS else (1e-3,)):
            r = run(dt, amp)
            if r['status'] != 'ok':
                print(f"{dt:>10.6f}{amp:>9.0e}{'':>7}{r['status']:>13}")
                continue
            print(f"{dt:>10.6f}{amp:>9.0e}{r['nsteps']:>7}{r['sigma']:>13.6f}"
                  f"{r['err']:>12.3%}{r['EE']:>11.4f}{r['div']:>11.2e}{r['wall']:>7.1f}s")
            sys.stdout.flush()
            if amp == 1e-3:
                traces[dt] = (r['ts'], r['Es']/r['E0'], r['sigma'])
    np.savez(f'{SC}/stk_dtau.npz',
             **{f'dt{k:g}_{a}': v for k, (t, e, sg) in traces.items()
                for a, v in (('t', t), ('e', e), ('s', np.array([sg])))})
    print(f'\nsaved stokes_traces_N{N}.npz')
    np.savez(f'{SC}/stokes_acc_N{N}.npz',
             dts=np.array(ACC),
             sig=np.array([traces[k][2] for k in ACC]))
