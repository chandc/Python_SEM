"""Does sub-iterating repair AC's unsteady error -- and what does it cost?

    uv run --quiet python scratch/ac_subiter.py

THE POINT.  AC is a STEADY-STATE device unless it is sub-iterated.  The
continuity row solves

    kappa_p*p + div u = kappa_p*p_prev   =>   div u = -kappa_p*(p - p_prev)

With nsub = 1 and p_prev from the previous time level, (p - p_prev) ~ dp/dt*dt
while kappa_p ~ 1/dt, so the product is **O(1) in dt** -- refining the time step
does not reduce it.  Sub-iterating refreshes p_prev from the previous
SUB-ITERATE; on convergence p = p_prev, the AC term vanishes identically, and
div u = 0 is recovered at the current time level.  That is the whole
justification for using AC on an unsteady problem, and it has never been tested
in this project -- 2D or 3D.

Three configurations, same physical time, same everything else:

    AC off             -- no AC term at all.  Gives the FLOOR: the divergence
                          a least-squares formulation carries anyway, since
                          continuity is a weighted row, not a hard constraint.
    AC on, nsub = 1    -- what the driver does today.
    AC on, nsub > 1    -- dual time stepping, the correct formulation.

Reading it:
  * nsub=1 divergence >> AC-off, falling toward AC-off as nsub grows
        -> AC without sub-iteration IS injecting error, and sub-iteration fixes it
  * all three equal
        -> the divergence is spatial discretisation error and AC is exonerated

COST is reported alongside, because that is the real decision.  AC buys a large
reduction in CG iterations per solve; sub-iteration spends some of it back.  The
question for M7 is whether AC + sub-iterations is still cheaper than AC-off at
equal accuracy -- so total CG iterations, not just the error, is tabulated.
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
AMP, TEND, DT = 0.05, 0.04, 0.004


def run(ac, nsub, dt=DT, tend=TEND):
    s = C.setup(**GRID)
    U = C.initial_state(s, amp=AMP)
    a = T.a_mass_worst(dt)
    kap = a if ac else 0.0
    Minv = C.make_precond(s, dt, kap)
    Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
    nstep = int(round(tend/dt))
    tot, t0 = 0, time.perf_counter()
    for _ in range(nstep):
        U, Nprev, its = C.step(s, U, Nprev, dt, kap, Minv=Minv, tol=1e-11,
                               max_iter=30000, nsub=nsub, sub_tol=1e-12)
        tot += its
    return dict(ac=ac, nsub=nsub, div=C.divergence(s, U), cg=tot,
                wall=time.perf_counter()-t0,
                epert=C.perturbation_energy(s, U),
                meanerr=C.mean_profile_error(s, U))


if __name__ == '__main__':
    print(f'Re={GRID["re"]:g}  dt={DT}  a_mass={T.a_mass_worst(DT):.0f}  '
          f't={TEND}  ({int(TEND/DT)} steps)\n')
    print(f"{'config':>26}{'rms|div u|':>13}{'vs AC-off':>11}"
          f"{'CG its':>9}{'wall s':>9}{'E_pert':>11}")
    rows = []
    ref = None
    for lab, ac, nsub in (('AC off', False, 1),
                          ('AC on, nsub=1 (today)', True, 1),
                          ('AC on, nsub=3', True, 3)):
        r = run(ac, nsub)
        r['label'] = lab
        if ref is None:
            ref = r['div']
        rows.append(r)
        print(f"{lab:>26}{r['div']:>13.4e}{r['div']/ref:>11.2f}"
              f"{r['cg']:>9}{r['wall']:>9.1f}{r['epert']:>11.4e}", flush=True)

    off = rows[0]
    print('\n  AC-off is the least-squares divergence FLOOR (continuity is a')
    print('  weighted row, never a hard constraint), so "vs AC-off" ~ 1.0 is')
    print('  the target, not zero.')
    best = min(rows[1:], key=lambda r: r['div'])
    print(f"\n  best AC config: {best['label']}  "
          f"div ratio {best['div']/off['div']:.2f}, "
          f"CG {best['cg']} vs {off['cg']} for AC-off "
          f"({off['cg']/max(best['cg'],1):.2f}x)")
