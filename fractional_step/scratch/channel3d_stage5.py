"""STAGE 5 decision gate, in 3D: do the a_mass and CFL windows overlap?

    uv run --quiet python scratch/channel3d_stage5.py <dt> <laminar|perturbed> <ac|noac> [tag]

THE COLLISION, stated so the sweep direction makes sense.  With RKW3/CN,

    a_mass = 1/(beta_2 * dt) = 6/dt          (worst stage)
    CFL    ∝ dt

so the two constraints pull in OPPOSITE directions and the feasible window is

    6/a_max  <  dt  <  dt_CFL

where a_max is the largest a_mass that is stable and dt_CFL is set by the
explicit convection (limit sqrt(3) for RKW3).  A window exists iff
a_max > 6/dt_CFL.  **Instability from a_mass therefore appears at SMALL dt**,
which is why the sweep runs dt DOWNWARD -- the opposite of the usual intuition
that smaller steps are safer.

WHY THE LAMINAR CASE IS A CONTROL, NOT EVIDENCE.  Poiseuille is exactly
representable here (verified: the base flow holds to 2.2e-15), so the
least-squares residual is ~0, and plan sec 0.2 says that is exactly the
condition under which the a_mass mechanism stays hidden.  A clean laminar sweep
means nothing on its own; the perturbed case is the measurement.  Both are run
so the difference between them is visible rather than assumed.

DETECTION.  Per-step, not per-unit-time: the 2D failures blew up at step 29-36
regardless of dt, so a fixed STEP budget catches the mechanism, while a fixed
physical time would give the small-dt cases many more chances to fail and
confound the comparison.
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import channel3d as C
from lssem3d import operator as OP, timestep as T, convect as CV, fourier as FR

NSTEP = 200          # 2D failures appeared by step ~33; this is ~6x margin
AMP = 0.05
GRID = dict(N=6, ex=3, ey=3, nz=16, re=180.0)
GROWTH_FAIL = 1.0e3          # energy ratio that counts as divergence


def dt_for_cfl(s, U, target=None):
    """Largest dt meeting the RKW3 imaginary-axis limit, from the actual field."""
    target = T.cfl_limit() if target is None else target
    Uphys = FR.to_physical(OP.to_complex(U), s['nz'])
    return float(CV.max_dt_for_cfl(Uphys, s['D'], s['m'].facx, s['m'].facy,
                                   s['lz'], s['nz'], target))


def run_case(dt, kind, ac, nstep=NSTEP, grid=None, verbose=True,
             rowweight=False):
    s = C.setup(**(grid or GRID))
    U = C.initial_state(s, amp=AMP if kind == 'perturbed' else 0.0)
    a_mass = T.a_mass_worst(dt)
    kap = a_mass if ac else 0.0
    Minv = C.make_precond(s, dt, kap, rowweight=rowweight)
    Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)

    e0 = C.perturbation_energy(s, U)
    rec = dict(dt=dt, a_mass=a_mass, kind=kind, ac=bool(ac), kap=kap,
               rowweight=bool(rowweight),
               cfl0=C.cfl(s, U, dt), dt_cfl=dt_for_cfl(s, U), e0=e0,
               grid=grid or GRID, nstep=nstep)
    status, hist, t0 = 'OK', [], time.perf_counter()
    cg_tot = 0
    for i in range(nstep):
        # tol = 1e-6: the measured policy (3D_STATUS.md sec 7F) -- identical
        # accuracy to 1e-12 at ~40% of the iterations
        U, Nprev, it = C.step(s, U, Nprev, dt, kap, Minv=Minv, tol=1e-6,
                              max_iter=20000, rowweight=rowweight)
        cg_tot += it
        if not np.all(np.isfinite(U)):
            status = 'BLEWUP'; break
        e = C.perturbation_energy(s, U)
        me = C.mean_profile_error(s, U)
        if e0 > 0 and e/e0 > GROWTH_FAIL:
            status = 'DIVERGED'; break
        if me > 10.0:
            status = 'MEANLOST'; break
        if (i+1) % 50 == 0:
            hist.append(dict(step=i+1, its=int(it), e_ratio=(e/e0 if e0 else 0.0),
                             meanerr=me))
            if verbose:
                print(f'  step {i+1:4d}  its={it:5d}  E/E0={e/e0 if e0 else 0:.4e}'
                      f'  meanerr={me:.3e}', flush=True)
    rec.update(status=status, steps=i+1, wall=time.perf_counter()-t0, hist=hist,
               cg_per_step=cg_tot/max(i+1, 1),
               e_end=C.perturbation_energy(s, U) if np.all(np.isfinite(U)) else None,
               meanerr_end=C.mean_profile_error(s, U) if np.all(np.isfinite(U)) else None)
    return rec


if __name__ == '__main__':
    dt = float(sys.argv[1])
    kind = sys.argv[2] if len(sys.argv) > 2 else 'perturbed'
    ac = (sys.argv[3] if len(sys.argv) > 3 else 'ac') == 'ac'
    rowweight = 'rw' in sys.argv[4:]
    rest = [a for a in sys.argv[4:] if a != 'rw']
    tag = rest[0] if rest else (f'{kind}_dt{dt:g}_{"ac" if ac else "noac"}'
                                + ('_rw' if rowweight else ''))
    print(f'=== dt={dt:g}  a_mass={T.a_mass_worst(dt):.1f}  {kind}  '
          f'AC={"on" if ac else "off"}  rowweights={"on" if rowweight else "off"}'
          f' ===', flush=True)
    r = run_case(dt, kind, ac, rowweight=rowweight)
    print(f'--> {r["status"]} after {r["steps"]} steps  '
          f'(CFL0={r["cfl0"]:.3f}, dt_CFL={r["dt_cfl"]:.4f}, '
          f'CG/step={r["cg_per_step"]:.0f}, {r["wall"]:.0f}s, '
          f'{r["wall"]/max(r["steps"],1):.2f} s/step)', flush=True)
    with open(f'scratch/stage5_{tag}.json', 'w') as f:
        json.dump(r, f, indent=1)
    print(f'wrote scratch/stage5_{tag}.json')
