"""Does the FDM preconditioner beat point-Jacobi, and does the gap grow with N?

    python scratch/fdm_bench.py [Nlist] [nz] [--backend cupy] [--fixed-dof]

The separability proof (scratch/fdm_structure.py) says FDM applies EXACTLY to
every field-diagonal block.  It says nothing about whether dropping the field
coupling -- 37% of the operator by Frobenius norm -- leaves a useful
preconditioner.  That is what this measures, on the real assembled operator
with real boundary conditions, in CG ITERATIONS rather than condition numbers
(an earlier attempt at condition numbers measured its own tolerance -- see
scratch/precond_ceiling.py, kept as a record of how that fails).

Two things decide it.  The RATIO at a given N says whether FDM pays for its
cost: an application is four small contractions, roughly one matvec, so it
needs better than ~2x to be worth using at all.  The TREND matters more --
point-Jacobi's iterations grow as N^1.01 at fixed dof, so a preconditioner
whose advantage GROWS with N is the entire point, and one that wins only a
constant factor is not worth the complexity.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

backend = 'numpy'
if '--backend' in sys.argv:
    backend = sys.argv[sys.argv.index('--backend') + 1]
fixed = '--fixed-dof' in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith('-')
        and a not in ('cupy', 'numpy', 'torch')]
Ns = [int(v) for v in (argv[0].split(',') if argv else ['4', '6', '8'])]
nz = int(argv[1]) if len(argv) > 1 else 8

import lssem3d; lssem3d.set_backend(backend)
import tgv_gpu_run as TG
from lssem3d import operator as OP, solver3d as S3, timestep as T, fdm
from lssem3d import device as DEV

ops = TG.Ops(backend)
# elements per side, chosen to hold total dof roughly constant as N rises
FIXED = {4: 8, 6: 6, 8: 5, 10: 4, 12: 3, 14: 3, 16: 3}
dt, CAP = 0.0039, 40000
print(f'backend={backend}, Nz={nz}, dt={dt}'
      f'{", fixed dof" if fixed else ""}\n')
print(f'{"N":>3} {"elem":>5} {"dof":>11} {"Jacobi":>8} {"FDM":>8} {"ratio":>7}'
      f'  {"sym resid":>10}')
rows = []
for N in Ns:
    ne = FIXED.get(N, 4) if fixed else 4
    cfg = dict(nu=6.25e-4, N=N, ex=ne, ey=ne, nz=nz, tend=1.0, snap=1.0,
               cfl=1.0, tol=1e-6)
    s = TG.setup(cfg, ops.t, ops)
    U0 = ops.to_dev(TG.ic_tgv(s))
    Np0 = ops.zeros_c(tuple(OP.to_complex(U0).shape[:-2]) + (3, s['nk']),
                      OP.to_complex(U0))
    c = T.implicit_coeff(dt, 0)
    rw = OP.momentum_row_weights(c)
    shape = (s['m'].nelem, N+1, N+1, OP.NVAR_R, s['nk'])
    dj = S3.jacobi_diagonal_analytic(shape, s['D'], s['m'].facx, s['m'].facy,
                                     s['kz'], cfg['nu'], c, s['m'], s['mask'],
                                     s['m'].wq, 0.0, rw=rw)
    Mj = ops.to_dev(S3.jacobi_inverse(dj, s['mask']))
    Mf = fdm.build(s['m'], N, s['kz'], cfg['nu'], c, rw, s['maskg'], like=U0)
    # A preconditioner for CG MUST be symmetric -- but symmetric IN THE SPACE
    # AND INNER PRODUCT CG ACTUALLY USES, which took two tries to get right.
    #
    # CG's vectors live in discontinuous element storage but are constrained to
    # the CONTINUOUS subspace (normal_op ends with gather-scatter), and _dot
    # weights by 1/multiplicity so that redundantly-stored shared nodes are
    # counted once.  Probing with random vectors tests neither: they are not
    # continuous, and a plain sum is not the inner product.  That combination
    # reports ~1e-2 for a preconditioner that is symmetric to 1e-17 on the
    # vectors CG will actually hand it -- a false alarm that flags every row.
    rng = np.random.default_rng(0)
    mw = DEV.to_device(S3.multiplicity_weight(s['m'], shape), U0)
    xc = S3.gs(s['m'], ops.to_dev(rng.standard_normal(shape)))*s['maskg']
    yc = S3.gs(s['m'], ops.to_dev(rng.standard_normal(shape)))*s['maskg']
    dot = lambda a, b: float(ops.t.sum(a*b*mw))
    sym = abs(dot(xc, Mf(yc)) - dot(yc, Mf(xc)))/max(abs(dot(xc, Mf(xc))), 1e-300)
    rwd = ops.to_dev(rw)
    _, _, itj = TG.stage(s, U0, Np0, 0, dt, {0: Mj}, rwd, cfg['tol'],
                         max_iter=CAP, check_every=10)
    _, _, itf = TG.stage(s, U0, Np0, 0, dt, {0: Mf}, rwd, cfg['tol'],
                         max_iter=CAP, check_every=10)
    dof = U0.numel() if backend == 'torch' else U0.size
    flag = '  <-- SYMMETRY FAIL, ignore the counts' if sym > 1e-10 else ''
    print(f'{N:>3} {ne:>5} {dof:>11,} {itj:>8} {itf:>8} '
          f'{itj/max(itf,1):>7.2f}  {sym:>10.2e}{flag}', flush=True)
    rows.append((N, itj, itf))
if len(rows) >= 4:
    lg = lambda i: np.polyfit(np.log([r[0] for r in rows]),
                              np.log([r[i] for r in rows]), 1)[0]
    pj, pf = lg(1), lg(2)
    print(f'\n  Jacobi scales as N^{pj:.2f},  FDM as N^{pf:.2f}')
    print('  ' + ('FDM appears to flatten the slope -- CHECK IT HOLDS when more '
                  'orders are added' if pf < pj - 0.15 else
                  'no slope advantage: FDM wins (if at all) only a constant factor'))
    print('  (A fit over five erratic points once showed FDM at N^0.88 against\n'
          '   Jacobi N^1.31; adding N = 14 turned it into N^1.51 against N^1.39.\n'
          '   Do not act on a slope from a short sweep.)')
elif len(rows) > 1:
    print(f'\n  {len(rows)} points is too few to fit a slope, and a slope from a\n'
          f'  short sweep of this quantity has already been wrong once -- run at\n'
          f'  least 4-5 orders before reading a trend.')
    print(f'\n  An FDM application is four small contractions, roughly one\n'
          f'  matvec, so a ratio above ~2 is a real win and below ~2 is not.')
