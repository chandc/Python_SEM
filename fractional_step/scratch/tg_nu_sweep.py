"""Which (row weighting, AC) recipe survives at M7's viscosity?

    TG_NU=<nu> uv run --quiet python scratch/tg_nu_sweep.py

THE PROBLEM THIS SETTLES.  Three benchmarks disagree about the right
least-squares row weighting, and they disagree along the viscosity axis:

    Stokes decay   nu = 1       legacy weights, AC off   -> order 2.00
    Taylor-Green   nu = 0.1     legacy weights, AC off   -> order 2.00
    cavity         nu = 1e-3    w_mom = 1, AC ON         -> 25 CG/step
                                (legacy is 27x worse; AC off is 12320)

**M7 runs at nu = 1/180 = 5.6e-3** -- far closer to the cavity than to either
case where order 2.00 has actually been demonstrated.  So the configuration M7
needs is the one carrying the AC accuracy problem, and the two configurations
with verified accuracy sit at viscosities M7 will never see.

Taylor-Green is the right instrument because it is the only case with an exact
unsteady solution, ACTIVE convection, and a free nu -- so accuracy and cost can
both be measured at the viscosity that matters.
"""
import os, sys, time, json
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import taylorgreen as TG

DT = 0.05
N = 12

if __name__ == '__main__':
    print(f'Taylor-Green, nu={TG.NU:g}, N={N}, dt={DT}, t=0.4  '
          f'(M7 runs at nu = 1/180 = 5.6e-3)')
    print(f"{'row weights':>13}{'AC':>6}{'L2 err':>12}{'CG':>10}{'capped':>8}"
          f"{'wall s':>9}")
    out = []
    for rw in (True, False):
        for ac in (False, True):
            t0 = time.perf_counter()
            try:
                r = TG.run(DT, rw, ac, grid=dict(N=N))
            except Exception as e:
                print(f"{str(rw):>13}{str(ac):>6}   ERROR {e}"); continue
            w = time.perf_counter()-t0
            if r['status'] != 'ok':
                print(f"{str(rw):>13}{str(ac):>6}   {r['status']}"); continue
            out.append(dict(nu=TG.NU, rw=rw, ac=ac, **{k: r[k] for k in
                                                       ('err', 'cg', 'capped')}))
            print(f"{str(rw):>13}{str(ac):>6}{r['err']:>12.4e}{r['cg']:>10}"
                  f"{('YES' if r['capped'] else 'no'):>8}{w:>9.0f}", flush=True)
    with open(f'scratch/tg_nu_{TG.NU:g}.json', 'w') as f:
        json.dump(out, f, indent=1)
