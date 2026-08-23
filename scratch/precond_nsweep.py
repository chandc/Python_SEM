"""How do CG iterations scale with polynomial order N?

    python scratch/precond_nsweep.py [ex ey nz] [Nlist]

The question a single-N headroom measurement CANNOT answer -- and the one that
decides whether a better preconditioner is worth building.

A point-Jacobi preconditioner removes metric and element-size variation.  It
does NOT address the N-dependence of the spectral-element condition number,
which grows like O(N^3) for the stiffness operator and is SQUARED here because
a least-squares formulation solves the normal equations.  Fast diagonalisation,
element block-Jacobi and overlapping Schwarz exist precisely because they give
near-N-INDEPENDENT convergence.

So the payoff from a better preconditioner is not a constant factor: it is the
difference between two SLOPES, and it grows with every order added.  Since
spectral accuracy is the entire reason to use SEM, that slope is the number
that matters.

dt is held FIXED across the sweep, so c is fixed and the comparison is about
the operator's conditioning rather than the time step.  Iterations are reported
per stage-0 solve, with the unpreconditioned count alongside so we can see
whether Jacobi's RELATIVE benefit grows or is merely constant while the
absolute cost climbs.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import tgv_gpu_run as TG
from lssem3d import operator as OP, solver3d as S3, timestep as T

argv = [a for a in sys.argv[1:] if not a.startswith('-')]
ex, ey, nz = (int(v) for v in (argv[:3] or [4, 4, 32]))
Ns = [int(v) for v in argv[3].split(',')] if len(argv) > 3 else [4, 6, 8, 10, 12, 14]
import lssem3d; lssem3d.set_backend('cupy')
ops = TG.Ops('cupy')
dt, CAP = 0.0039, 60000
print(f'{ex}x{ey} elements, Nz={nz}, dt={dt} fixed across the sweep\n')
print(f'{"N":>3} {"dof":>12} {"Jacobi":>8} {"none":>9} {"ratio":>7}  '
      f'{"its/N^a":>9}')
rows = []
for N in Ns:
    cfg = dict(nu=6.25e-4, N=N, ex=ex, ey=ey, nz=nz, tend=1.0, snap=1.0,
               cfl=1.0, tol=1e-6)
    s = TG.setup(cfg, ops.t, ops)
    U0 = ops.to_dev(TG.ic_tgv(s))
    Np0 = ops.zeros_c(tuple(OP.to_complex(U0).shape[:-2]) + (3, s['nk']),
                      OP.to_complex(U0))
    c = T.implicit_coeff(dt, 0)
    rw = OP.momentum_row_weights(c)
    shape = (s['m'].nelem, N+1, N+1, OP.NVAR_R, s['nk'])
    d = S3.jacobi_diagonal_analytic(shape, s['D'], s['m'].facx, s['m'].facy,
                                    s['kz'], cfg['nu'], c, s['m'], s['mask'],
                                    s['m'].wq, 0.0, rw=rw)
    Minv = ops.to_dev(S3.jacobi_inverse(d, s['mask']))
    rwd = ops.to_dev(rw)
    _, _, itj = TG.stage(s, U0, Np0, 0, dt, {0: Minv}, rwd, cfg['tol'],
                         max_iter=CAP, check_every=10)
    _, _, itn = TG.stage(s, U0, Np0, 0, dt, {0: None}, rwd, cfg['tol'],
                         max_iter=CAP, check_every=10)
    dof = U0.size
    cap = '+' if itn >= CAP else ' '
    print(f'{N:>3} {dof:>12,} {itj:>8} {itn:>8}{cap} {itn/max(itj,1):>7.1f}',
          flush=True)
    rows.append((N, itj, itn))

# slope of log(iterations) vs log(N) -- the number the whole question turns on
lm = np.polyfit(np.log([r[0] for r in rows]), np.log([r[1] for r in rows]), 1)
print(f'\n  Jacobi iterations scale as N^{lm[0]:.2f}')
n0, n1 = rows[0][0], rows[-1][0]
print(f'  over N = {n0} -> {n1} that is {rows[-1][1]/rows[0][1]:.1f}x more '
      f'iterations, at fixed element count')
print(f'\n  An N-independent preconditioner (fast diagonalisation, element\n'
      f'  block-Jacobi, overlapping Schwarz) would flatten that slope.  The\n'
      f'  payoff is therefore not a constant factor but {lm[0]:.2f} powers of N,\n'
      f'  and it compounds with every order added -- which is the case for\n'
      f'  building one, and is invisible to a measurement at a single N.')
print(f'\n  Caveat: this holds N-per-element fixed and the mesh fixed, so dof\n'
      f'  grows with N.  Part of the growth is size, not conditioning.  A\n'
      f'  fixed-dof sweep (coarsening the mesh as N rises) separates them and\n'
      f'  is the honest follow-up if the slope here looks strong.')
