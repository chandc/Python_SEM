"""How much is the current preconditioner buying, and can row weights do better?

    python scratch/precond_headroom.py [ex ey nz N] [--backend cupy]

PART A -- HEADROOM.  The production case runs ~1580 CG iterations per stage
with a scalar-diagonal Jacobi.  Whether a better preconditioner is worth
building depends entirely on what the current one already buys:

    unpreconditioned ~20000  ->  Jacobi is doing the heavy lifting; a better
                                 one fights for scraps
    unpreconditioned ~3000   ->  Jacobi barely helps; there is a lot on the
                                 table

PART B -- ROW WEIGHTS.  These cost NOTHING per iteration, unlike a V-cycle
(the 2D study measured p-MG cutting iterations 9.9x for ~14 matvecs of cost,
a net 1.25x).  ROW7_WEIGHT = 1e-4 was found empirically and helped materially;
the momentum rows use an analytic 1/c^2 and nobody has swept the rest.

  *** A ROW WEIGHT IS NOT A PRECONDITIONER. ***  It changes the least-squares
  functional, so it changes the DISCRETE SOLUTION -- a weighting that
  converges faster may simply be solving a different, easier, wronger problem.
  Iteration counts here are a screening tool only.  Anything promising must
  then pass the validation ladder (Stokes sigma, temporal order, and the
  parameter-free energy balance) before it goes anywhere near a production run.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import tgv_gpu_run as TG          # main() is guarded, so importing is safe
from lssem3d import operator as OP, solver3d as S3, timestep as T

argv = [a for a in sys.argv[1:] if not a.startswith('-')]
backend = 'cupy'
if '--backend' in sys.argv:
    backend = sys.argv[sys.argv.index('--backend') + 1]
ex, ey, nz, N = (int(v) for v in (argv[:4] or [8, 8, 64, 8]))

import lssem3d; lssem3d.set_backend(backend)
ops = TG.Ops(backend)
cfg = dict(nu=6.25e-4, N=N, ex=ex, ey=ey, nz=nz, tend=1.0, snap=1.0,
           cfl=1.0, tol=1e-6)
s = TG.setup(cfg, ops.t, ops)
U0 = ops.to_dev(TG.ic_tgv(s))
Np0 = ops.zeros_c(tuple(OP.to_complex(U0).shape[:-2]) + (3, s['nk']),
                  OP.to_complex(U0))
dt = 0.0039
c = T.implicit_coeff(dt, 0)
shape = (s['m'].nelem, N+1, N+1, OP.NVAR_R, s['nk'])
print(f'{ex}x{ey} N={N} Nz={nz}: {U0.size if backend=="cupy" else U0.numel():,} dof, '
      f'dt={dt}, c={c:.1f}\n')

def build(rw):
    d = S3.jacobi_diagonal_analytic(shape, s['D'], s['m'].facx, s['m'].facy,
                                    s['kz'], cfg['nu'], c, s['m'], s['mask'],
                                    s['m'].wq, 0.0, rw=rw)
    return ops.to_dev(S3.jacobi_inverse(d, s['mask']))

def run(rw, M_inv, cap):
    _, _, it = TG.stage(s, U0, Np0, 0, dt, {0: M_inv}, ops.to_dev(rw),
                        cfg['tol'], max_iter=cap, check_every=10)
    return it

rw0 = OP.momentum_row_weights(c)
print('PART A -- what is the Jacobi worth?')
CAP = 40000
j = run(rw0, build(rw0), CAP)
print(f'  Jacobi (current)     {j:6d} iterations')
n = run(rw0, None, CAP)
tag = '  *** HIT THE CAP -- true count is higher ***' if n >= CAP else ''
print(f'  unpreconditioned     {n:6d} iterations{tag}')
print(f'  -> the current preconditioner is worth {n/max(j,1):.1f}x')
print(f'     A 14x14 nodal block-Jacobi costs ~0.25 matvec to apply, so it\n'
      f'     needs only ~1.25x fewer iterations than {j} to pay for itself.\n')

print('PART B -- row-weight sweep (screening only; see the header)')
def weights(w_cont=1.0, w_vort=1.0, w_mom=1.0, w7=None):
    rw = np.ones(OP.NROW)
    rw[0] = w_cont
    rw[1:4] = w_vort
    rw[4:7] = w_mom/(c*c)
    rw[7] = OP.ROW7_WEIGHT if w7 is None else w7
    return rw

base = run(weights(), build(weights()), CAP)
print(f'  baseline (current weighting)          {base:6d}')
for label, kw in (
        ('row 7 (div omega)      w7=1.0',      dict(w7=1.0)),
        ('row 7                  w7=1e-2',     dict(w7=1e-2)),
        ('row 7                  w7=1e-6',     dict(w7=1e-6)),
        ('row 7                  w7=0.0',      dict(w7=0.0)),
        ('momentum rows          x 0.1',       dict(w_mom=0.1)),
        ('momentum rows          x 10',        dict(w_mom=10.0)),
        ('continuity row         x 0.1',       dict(w_cont=0.1)),
        ('continuity row         x 10',        dict(w_cont=10.0)),
        ('vorticity-def rows     x 0.1',       dict(w_vort=0.1)),
        ('vorticity-def rows     x 10',        dict(w_vort=10.0)),
        # The two winners point the same way -- MORE weight on momentum
        # relative to the constraint rows -- so try them together and further
        # along that axis.  If the trend continues without limit, that is a
        # warning sign, not a result: it would mean the constraints are simply
        # being discarded, and the ladder will say so.
        ('mom x10 + vort x0.1',                dict(w_mom=10.0, w_vort=0.1)),
        ('mom x100 + vort x0.01',              dict(w_mom=100.0, w_vort=0.01)),
        ('mom x100',                           dict(w_mom=100.0)),
        ('vort x0.01',                         dict(w_vort=0.01))):
    rw = weights(**kw)
    it = run(rw, build(rw), CAP)
    d = (base - it)/base*100
    mark = '   <<< better' if it < base*0.9 else ('  worse' if it > base*1.1 else '')
    print(f'  {label:<38}{it:6d}  ({d:+5.1f}%){mark}')
print('\n  Reminder: a lower count here may mean an easier problem, not a\n'
      '  better one.  Validate any candidate on the ladder before adopting it.')
print('  In particular, weighting momentum UP relative to the continuity and\n'
      '  vorticity-definition rows buys iterations by caring less about the\n'
      '  constraints -- which is exactly what the divergence error and the\n'
      '  energy balance would pay for.  If the count keeps falling as the\n'
      '  ratio grows without bound, that is the tell.')
