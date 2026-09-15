"""Does AC degrade the TEMPORAL ORDER of the real 3D stepper -- and does
sub-iteration restore it?

    uv run --quiet python scratch/ac_temporal_order.py

THE GAP THIS CLOSES.  Stage 4's temporal gate was (correctly, for its purpose)
restated onto a SCALAR model problem, in order to separate the RKW3 coefficient
table from Crank-Nicolson as the order limiter.  That test is blind to AC: it has
no pressure and no continuity row.  So no temporal-order measurement of the
actual 3D PDE stepper with AC switched on exists anywhere in this project.

That matters because AC without sub-iterations leaves
`div u = -kappa_p*(p - p_prev)`, which is O(1) in dt (kappa_p ~ 1/dt cancels
Delta p ~ dt) -- measured flat at ~1.46e-02 across an 8x refinement.  Whether
that O(1) divergence error also contaminates the VELOCITY at O(1) -- destroying
temporal convergence -- or stays confined to a mode the least-squares balance
absorbs, is exactly what this measures.

METHOD -- Richardson self-convergence, so no exact solution is needed:

    p = log2( ||U(dt) - U(dt/2)|| / ||U(dt/2) - U(dt/4)|| )

All runs integrate to the SAME physical time, so the differences are pure
temporal error.  Velocity and pressure are reported separately: AC acts through
the continuity row, so pressure is where any damage should appear first, and
averaging it into one norm with the velocity would hide that.

Expected: ~2 (Crank-Nicolson limited, per Stage 4).  A value well below 2 for
nsub=1 that recovers toward 2 for nsub=3 would show AC damaging the time
accuracy and sub-iteration repairing it.
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
import channel3d as C
from lssem3d import operator as OP, timestep as T

GRID = dict(N=6, ex=3, ey=3, nz=16, re=180.0)
AMP, TEND = 0.05, 0.08
DTS = [0.008, 0.004, 0.002, 0.001]
KAP_FIXED = None      # set below: kappa_p held FIXED across dt (see note)


def run(dt, ac, nsub, kap_fixed):
    """Integrate to TEND.  kappa_p is held FIXED across dt on purpose.

    a_mass = 6/dt changes with dt, but if kappa_p were tied to it the operator
    itself would change between refinements and the comparison would no longer
    isolate temporal error.  Fixing kappa_p keeps one scheme being refined.
    """
    s = C.setup(**GRID)
    U = C.initial_state(s, amp=AMP)
    kap = kap_fixed if ac else 0.0
    Minv = C.make_precond(s, dt, kap)
    Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
    for _ in range(int(round(TEND/dt))):
        U, Nprev, _ = C.step(s, U, Nprev, dt, kap, Minv=Minv, tol=1e-12,
                             max_iter=30000, nsub=nsub, sub_tol=1e-13)
    return U, s


def norms(Ua, Ub):
    """(velocity, pressure) L2 differences, kept separate."""
    a, b = OP.to_complex(Ua), OP.to_complex(Ub)
    vel = sum(float(np.sum(np.abs(a[..., f, :] - b[..., f, :])**2))
              for f in (OP.U_, OP.V_, OP.W_))
    pre = float(np.sum(np.abs(a[..., OP.P_, :] - b[..., OP.P_, :])**2))
    return np.sqrt(vel), np.sqrt(pre)


def order(seq):
    return [np.log2(seq[i]/seq[i+1]) if seq[i+1] > 0 else float('nan')
            for i in range(len(seq)-1)]


if __name__ == '__main__':
    kap = T.a_mass_worst(DTS[0])          # fixed kappa_p for every refinement
    print(f'Re={GRID["re"]:g}  t={TEND}  dt={DTS}  kappa_p fixed at {kap:.0f}\n')
    for lab, ac, nsub in (('AC on, nsub=1 (today)', True, 1),
                          ('AC on, nsub=3', True, 3)):
        sols = []
        for dt in DTS:
            t0 = time.perf_counter()
            U, s = run(dt, ac, nsub, kap)
            sols.append(U)
            print(f'  [{lab}] dt={dt:<7g} done ({time.perf_counter()-t0:.0f}s)',
                  flush=True)
        dv = [norms(sols[i], sols[i+1])[0] for i in range(len(sols)-1)]
        dp = [norms(sols[i], sols[i+1])[1] for i in range(len(sols)-1)]
        print(f'\n  === {lab} ===')
        print(f'    |dU| velocity : {[f"{x:.4e}" for x in dv]}')
        print(f'    order         : {[f"{x:.3f}" for x in order(dv)]}')
        print(f'    |dU| pressure : {[f"{x:.4e}" for x in dp]}')
        print(f'    order         : {[f"{x:.3f}" for x in order(dp)]}\n',
              flush=True)
    print('  Stage 4 predicts ~2 (Crank-Nicolson limited).  Well below 2 at')
    print('  nsub=1, recovering at nsub=3, would convict AC.')
