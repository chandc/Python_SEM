"""Temporal accuracy of the 3D stepper, done properly: two regimes, one warm start.

    uv run --quiet python scratch/ac_temporal2.py [warmup|order]

WHAT WENT WRONG THE FIRST TIME (`ac_temporal_order.py`).  The initial condition
sets p = 0, which is inconsistent with the velocity field, so pressure relaxes
over a fixed number of STEPS.  Refining dt then gives more relaxation, and
successive solutions moved APART -- negative order.  That was the initial
condition, not the scheme.  Confirmed directly: at fixed physical time,
max|p| = 2.94e-03 -> 3.93e-03 -> 4.69e-03 as dt was refined, still climbing.

FIX 1 -- a pressure-consistent start.  Warm up once from the analytic state with
converged sub-iterations, and use that single field as the initial condition for
EVERY refinement run.  Pressure is then equilibrated before the clock starts, and
all runs share it exactly.

FIX 2 -- and this one is conceptual, not a bug.  A convergence study must refine
ONE FIXED SCHEME.  In production kappa_p = a_mass = 6/dt, so refining dt CHANGES
THE EQUATION; that is not a convergence study.  Hence two regimes:

  KAPPA FIXED       kappa_p held constant while dt refines.  A genuine
                    convergence study of a fixed scheme.  Expect ~2, since
                    Stage 4 showed Crank-Nicolson is the limiter.  This
                    validates the harness -- if it does not give ~2, the
                    measurement is broken and nothing else here can be trusted.

  KAPPA = a_mass    the production setting, kappa_p ~ 1/dt.  div u was measured
                    FLAT under refinement here (O(1) in dt), so the prediction
                    is that successive solutions stop approaching each other:
                    the differences PLATEAU and the apparent order falls to ~0.

A plateau is itself the signature of non-convergence, so no expensive AC-off
reference solution is needed -- which matters, since AC-off costs ~10x per step.

Reported for velocity and pressure separately: AC acts through the continuity
row, so pressure is where damage appears first and a combined norm would hide it.
"""
import os, sys, time, json
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import channel3d as C
from lssem3d import operator as OP, timestep as T

GRID = dict(N=6, ex=3, ey=3, nz=16, re=180.0)
AMP = 0.05
DT_WARM, NSUB_WARM = 0.002, 12
KAP_REF = T.a_mass_worst(0.004)      # the fixed kappa_p, and the warm-up's
DTS = [0.008, 0.004, 0.002, 0.001]
TEND = 0.032                          # divides every dt in DTS exactly
WARM = f'{SC}/ac_temporal2_warm.npz'


def _blank_prev(s, U):
    return np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)


def warmup(nstep=40, verbose=True):
    """Equilibrate the pressure, then freeze the field as the shared start.

    Converged sub-iterations throughout: the point is to arrive at a state whose
    pressure genuinely satisfies its own continuity row, so that no run in the
    refinement study is still paying off an initial transient.
    """
    s = C.setup(**GRID)
    U = C.initial_state(s, amp=AMP)
    Minv = C.make_precond(s, DT_WARM, KAP_REF)
    Nprev = _blank_prev(s, U)
    traj = []
    for i in range(nstep):
        U, Nprev, _ = C.step(s, U, Nprev, DT_WARM, KAP_REF, Minv=Minv,
                             tol=1e-12, max_iter=30000, nsub=NSUB_WARM,
                             sub_tol=1e-13)
        pmax = float(np.abs(OP.to_complex(U)[..., OP.P_, :]).max())
        traj.append((i+1, pmax, C.divergence(s, U)))
        if verbose and (i+1) % 5 == 0:
            print(f'  warm step {i+1:3d}  max|p|={pmax:.6e}  '
                  f'rms|div u|={traj[-1][2]:.4e}', flush=True)
    np.savez(WARM, U=U, traj=np.array(traj))
    if verbose:
        p = [t[1] for t in traj]
        drift = abs(p[-1]-p[-5])/max(abs(p[-1]), 1e-30)
        print(f'\n  max|p| over the last 5 steps drifts {drift:.2%} '
              f'-- equilibrated if this is small')
        print(f'  wrote {WARM}')
    return U, s


def integrate(s, U0, dt, kap_mode, nsub, tend=TEND):
    kap = KAP_REF if kap_mode == 'fixed' else T.a_mass_worst(dt)
    Minv = C.make_precond(s, dt, kap)
    U = U0.copy()
    Nprev = _blank_prev(s, U)
    for _ in range(int(round(tend/dt))):
        U, Nprev, _ = C.step(s, U, Nprev, dt, kap, Minv=Minv, tol=1e-12,
                             max_iter=30000, nsub=nsub, sub_tol=1e-13)
    return U


def diffs(sols):
    out = []
    for a, b in zip(sols[:-1], sols[1:]):
        ca, cb = OP.to_complex(a), OP.to_complex(b)
        v = sum(float(np.sum(np.abs(ca[..., f, :]-cb[..., f, :])**2))
                for f in (OP.U_, OP.V_, OP.W_))
        p = float(np.sum(np.abs(ca[..., OP.P_, :]-cb[..., OP.P_, :])**2))
        out.append((np.sqrt(v), np.sqrt(p)))
    return out


def order(seq):
    return [np.log2(a/b) if b > 0 else float('nan')
            for a, b in zip(seq[:-1], seq[1:])]


def main():
    if not os.path.exists(WARM):
        print('=== warm-up (pressure equilibration) ===')
        warmup()
    d = np.load(WARM)
    U0 = d['U']
    s = C.setup(**GRID)
    print(f'\n=== refinement from the warmed state, t={TEND}, dt={DTS} ===')
    res = {}
    for kap_mode in ('fixed', 'amass'):
        for nsub in (1, 3):
            tag = f'kappa={kap_mode}, nsub={nsub}'
            sols = []
            for dt in DTS:
                t0 = time.perf_counter()
                sols.append(integrate(s, U0, dt, kap_mode, nsub))
                print(f'  [{tag}] dt={dt:<7g} ({time.perf_counter()-t0:.0f}s)',
                      flush=True)
            dv = [x[0] for x in diffs(sols)]
            dp = [x[1] for x in diffs(sols)]
            res[tag] = dict(dv=dv, dp=dp, ov=order(dv), op=order(dp))
            print(f'    velocity |dU| {[f"{x:.3e}" for x in dv]}  '
                  f'order {[f"{x:.2f}" for x in order(dv)]}')
            print(f'    pressure |dU| {[f"{x:.3e}" for x in dp]}  '
                  f'order {[f"{x:.2f}" for x in order(dp)]}\n', flush=True)
    with open(f'{SC}/ac_temporal2.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('  kappa=fixed  -> order ~2 validates the harness (CN-limited).')
    print('  kappa=amass  -> a PLATEAU (order -> 0) is the production setting')
    print('                  failing to converge in time, which is the concern.')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'warmup':
        warmup()
    else:
        main()
