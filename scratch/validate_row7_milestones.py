"""Re-validate M2 and Stage 5 on the row-7-down-weighted operator.

    uv run --quiet python scratch/validate_row7_milestones.py <m2|stage5> [args]

Both milestones were measured with w7 = 1.  The operator has changed, so their
verdicts have to be re-established rather than assumed.  The two cases are
expected to behave DIFFERENTLY, and that difference is itself the check:

  M2, cavity at k_z = 0
      omega_x = omega_y = 0 identically, so row 7 is inert.  Expect NO change --
      same RMS, same iteration count.  If M2 moves, the weighting is leaking
      into a place it has no business touching.

  Stage 5, channel with a roll perturbation
      max|omega_x| ~ 8, so the near-null cluster IS excited.  Expect the same
      physics (stability, decay) at ~10x fewer iterations.  If the physics
      moves, the down-weighting is not the free lunch the single-solve test
      suggested.
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
from lssem3d import operator as OP


def stage5(dt=0.01, kind='perturbed', nstep=200):
    """Channel: same physics, fewer iterations?"""
    import channel3d_stage5 as S5
    out = {}
    _orig = OP.momentum_row_weights
    for w7 in (1.0, OP.ROW7_WEIGHT):
        # Patch the FUNCTION, not the module constant: `def f(c, w7=ROW7_WEIGHT)`
        # binds the default at DEFINITION time, so reassigning OP.ROW7_WEIGHT
        # afterwards changes nothing -- an A/B built that way silently compares
        # a configuration against itself.
        OP.momentum_row_weights = (lambda c, _w=w7, _f=_orig: _f(c, w7=_w))
        t0 = time.perf_counter()
        r = S5.run_case(dt, kind, False, nstep=nstep, verbose=False,
                        rowweight=True)
        r['wall'] = time.perf_counter()-t0
        out[w7] = r
        OP.momentum_row_weights = _orig
        print(f"  w7={w7:<8g} {r['status']:>9}  {r['steps']:>4} steps  "
              f"CG/step={r['cg_per_step']:>7.0f}  {r['wall']:>6.0f}s  "
              f"E/E0={(r['e_end']/r['e0'] if r['e0'] else float('nan')):.4e}  "
              f"meanerr={r['meanerr_end']:.2e}", flush=True)
    a, b = out[1.0], out[OP.ROW7_WEIGHT]
    print(f"\n  iteration gain : {a['cg_per_step']/max(b['cg_per_step'],1):.1f}x")
    print(f"  wall gain      : {a['wall']/max(b['wall'],1e-9):.1f}x")
    if a['e0'] and b['e0']:
        ra, rb = a['e_end']/a['e0'], b['e_end']/b['e0']
        print(f"  physics        : E/E0 {ra:.6f} vs {rb:.6f}  "
              f"(rel diff {abs(ra-rb)/ra:.2e})")
    print(f"  status         : {a['status']} vs {b['status']}")
    json.dump({str(k): {kk: vv for kk, vv in v.items() if kk != 'hist'}
               for k, v in out.items()},
              open('scratch/validate_row7_stage5.json', 'w'), indent=1, default=str)


def m2(cflt=0.8, tmax=2.0):
    """Cavity at k_z = 0: the NEGATIVE control for the row-7 fix.

    This case has omega_x = omega_y = 0 identically, so row 7 reduces to
    d_x*0 + d_y*0 + i*k*omega_z with k = 0 -- identically zero.  The near-null
    cluster is never excited, so the weight should make NO difference at all.

    That asymmetry is the real test.  A fix that speeds up everything is
    suspicious; one that speeds up exactly the cases whose mechanism it targets,
    and leaves the others bit-for-bit alone, is doing what the theory says.
    """
    import cavity3d_kz0 as CAV
    out = {}
    _orig = OP.momentum_row_weights
    for w7 in (1.0, OP.ROW7_WEIGHT):
        OP.momentum_row_weights = (lambda c, _w=w7, _f=_orig: _f(c, w7=_w))
        t0 = time.perf_counter()
        (ru, rv), status, steps, wall = CAV.run(cflt, tmax, 1.0)
        OP.momentum_row_weights = _orig
        out[w7] = dict(rms_u=float(ru), rms_v=float(rv), status=str(status),
                       steps=int(steps), wall=time.perf_counter()-t0)
        print(f'  w7={w7:<8g} RMS u={ru:.6e}  RMS v={rv:.6e}  '
              f'{status}, {steps} steps, {out[w7]["wall"]:.0f}s', flush=True)
    a, b = out[1.0], out[OP.ROW7_WEIGHT]
    du = abs(a['rms_u']-b['rms_u'])/max(a['rms_u'], 1e-30)
    dv = abs(a['rms_v']-b['rms_v'])/max(a['rms_v'], 1e-30)
    print(f"\n  RMS u rel diff : {du:.3e}")
    print(f"  RMS v rel diff : {dv:.3e}")
    print(f"  wall ratio     : {a['wall']/max(b['wall'],1e-9):.2f}x")
    print('\n  EXPECTED: no change.  omega_x = omega_y = 0 here, so row 7 is')
    print('  inert and the cluster it creates is never excited.')
    json.dump({str(k): v for k, v in out.items()},
              open('scratch/validate_row7_m2.json', 'w'), indent=1)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'stage5'
    if what == 'm2':
        tmax = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
        print(f'M2 re-validation, cavity k_z = 0, AC on, to t={tmax}')
        print('Expect: NO change (omega_x = omega_y = 0, so row 7 is inert)\n')
        m2(0.8, tmax)
    elif what == 'stage5':
        dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        print(f'STAGE 5 re-validation, channel dt={dt}, {n} steps, AC off, '
              f'row weights on')
        print('Expect: SAME physics, ~10x fewer iterations '
              '(max|omega_x| ~ 8, so the cluster IS excited)\n')
        stage5(dt, 'perturbed', n)
