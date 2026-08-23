"""Does the row-weight candidate move the answer on an UNDER-RESOLVED case?

    python scratch/rowweight_ab.py [ex ey nz N] [nstep]

The validation ladder passed the candidate (mom x10, vort x0.1) with
bit-identical numbers -- but that is expected and proves less than it looks.
For a NEARLY CONSISTENT least-squares system the minimiser is almost
independent of the row weights, and the gate cases are smooth and well
resolved, so their residual is tiny.

The production case is not.  Re = 1600 on 128^3 gives k_max*eta ~ 0.76:
deliberately under-resolved at peak dissipation, which is exactly where the
least-squares residual is LARGEST and where a change of weighting can move the
solution rather than just the conditioning.  This runs both weightings on that
configuration and compares what actually matters:

  divergence ||div u||   -- the direct test.  Buying iterations by caring less
                            about the continuity row would show up here first.
  balance -dE/dt / 2nuOm -- parameter-free; 1.0 means the resolved cascade
                            accounts for all the dissipation.
  field difference       -- if the two solutions differ by much more than the
                            CG tolerance, the weighting is changing the answer.

VERDICT RULE, fixed in advance so it cannot be rationalised afterwards: adopt
only if the candidate's divergence is no worse than the baseline's and its
balance is no further from 1.0.  A faster run that drifts on either is the
constraints being sold for speed.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import tgv_gpu_run as TG
from lssem3d import operator as OP, solver3d as S3, timestep as T

argv = [a for a in sys.argv[1:] if not a.startswith('-')]
ex, ey, nz, N = (int(v) for v in (argv[:4] or [16, 16, 128, 8]))
nstep = int(argv[4]) if len(argv) > 4 else 10
import lssem3d; lssem3d.set_backend('cupy')
ops = TG.Ops('cupy')
cfg = dict(nu=6.25e-4, N=N, ex=ex, ey=ey, nz=nz, tend=1.0, snap=1.0,
           cfl=1.0, tol=1e-6)
s = TG.setup(cfg, ops.t, ops)
U_ic = ops.to_dev(TG.ic_tgv(s))
dt = 0.0039
print(f'{ex}x{ey} N={N} Nz={nz}, dt={dt}, {nstep} steps\n')


def divergence(U):
    """L2 norm of row 0 of L0, which IS the divergence (kap = 0)."""
    Rr = OP.apply_L(U, s['Dg'], s['fxg'], s['fyg'], s['kzg'], cfg['nu'], 0.0)
    d = Rr[..., 0, :] + 1j*Rr[..., OP.NROW, :]
    return float(ops.t.sqrt(ops.t.sum(ops.t.abs(d)**2 * s['wqg'][..., None])))


def run(w_mom, w_vort, label):
    OP.MOM_WEIGHT, OP.VORT_WEIGHT = w_mom, w_vort
    U = U_ic.copy()
    Minv, rws = TG.precond(s, dt)
    Np_ = ops.zeros_c(tuple(OP.to_complex(U).shape[:-2]) + (3, s['nk']),
                      OP.to_complex(U))
    E0, Om0, _ = TG.diagnostics(s, U)
    prevE, prevOm = E0, Om0
    out = dict(bal=[], div=[divergence(U)], its=[], E=[E0], Om=[Om0])
    t0 = time.perf_counter()
    for i in range(nstep):
        tot = 0
        for k in range(T.NSTAGE):
            U, Np_, it = TG.stage(s, U, Np_, k, dt, Minv, rws[k], cfg['tol'],
                                  check_every=10)
            tot += it
        E, Om, _ = TG.diagnostics(s, U)
        out['bal'].append((-(E - prevE)/dt)/(2*cfg['nu']*0.5*(Om + prevOm)))
        prevE, prevOm = E, Om
        out['its'].append(tot); out['E'].append(E); out['Om'].append(Om)
        out['div'].append(divergence(U))
        print(f'  {label}  step {i+1:2d}  E={E:.8f}  Om={Om:.4f}  '
              f'bal={out["bal"][-1]:.6f}  |div|={out["div"][-1]:.3e}  '
              f'CG={tot}', flush=True)
    ops.sync()
    out['wall'] = (time.perf_counter()-t0)/nstep
    out['U'] = U
    return out

base = run(1.0, 1.0, 'base')
print()
cand = run(10.0, 0.1, 'cand')

print(f'\n{"":22} {"baseline":>14} {"candidate":>14}')
for key, fmt in (('E', '.8f'), ('Om', '.4f')):
    print(f'  final {key:<16} {base[key][-1]:>14{fmt}} {cand[key][-1]:>14{fmt}}')
bb, cb = np.mean(base['bal']), np.mean(cand['bal'])
bd, cd = base['div'][-1], cand['div'][-1]
print(f'  mean balance         {bb:>14.6f} {cb:>14.6f}')
print(f'  final |div u|        {bd:>14.3e} {cd:>14.3e}')
print(f'  CG per step          {np.mean(base["its"]):>14.0f} {np.mean(cand["its"]):>14.0f}')
print(f'  s per step           {base["wall"]:>14.1f} {cand["wall"]:>14.1f}')
d = float(ops.t.abs(base['U'] - cand['U']).max()/ops.t.abs(base['U']).max())
print(f'\n  field difference (relative): {d:.3e}   (CG tolerance {cfg["tol"]:.0e})')
print(f'  speedup: {np.mean(base["its"])/max(np.mean(cand["its"]),1):.2f}x fewer '
      f'iterations, {base["wall"]/cand["wall"]:.2f}x wall clock')

div_ok = cd <= bd*1.05
bal_ok = abs(cb - 1.0) <= abs(bb - 1.0)*1.05
print(f'\n  divergence no worse : {"YES" if div_ok else "NO"}')
print(f'  balance  no further : {"YES" if bal_ok else "NO"}')
print(f'  VERDICT: {"ADOPT" if (div_ok and bal_ok) else "REJECT -- the iterations are being bought with physics"}')
