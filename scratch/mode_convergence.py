"""How much work does the batched GPU solve waste on already-converged modes?

    python scratch/mode_convergence.py [N ne nz] [--backend cupy]

lssem3d/parallel.py already exploits the one place this algorithm is
embarrassingly parallel -- Fourier modes are independent inside the implicit
solve -- and measures 6.7x on 12P+4E CPU cores.  It also records the second
reason chunking helps:

    pcg exits on all(rn < target), so a mode that converged long ago keeps
    iterating until the WORST mode catches up.

The GPU path cannot do that.  Every mode lives in one batched array and they
all iterate until the slowest is done, so a mode that converged at iteration
300 still pays for iterations 301..1600.  High-k modes are strongly damped by
their kz^2 term and converge early, which is exactly the case where this bites.

This solves each mode SEPARATELY and reports the spread.  The waste is

    batched cost  = max(iters) * nk        (what the GPU pays now)
    ideal cost    = sum(iters)             (what per-mode exit would cost)

and the ratio bounds what chunking could recover.  A chunked GPU solve trades
some batch size for less wasted work, so the real gain sits between 1 and this
bound -- but the bound is what says whether it is worth building at all.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

backend = 'numpy'
if '--backend' in sys.argv:
    backend = sys.argv[sys.argv.index('--backend') + 1]
argv = [a for a in sys.argv[1:] if not a.startswith('-')
        and a not in ('cupy', 'numpy', 'torch')]
N, ne, nz = (int(v) for v in (argv[:3] or [8, 4, 32]))

import lssem3d; lssem3d.set_backend(backend)
import tgv_gpu_run as TG
from lssem3d import operator as OP, solver3d as S3, timestep as T

ops = TG.Ops(backend)
dt = 0.0039
cfg = dict(nu=6.25e-4, N=N, ex=ne, ey=ne, nz=nz, tend=1.0, snap=1.0,
           cfl=1.0, tol=1e-6)
s = TG.setup(cfg, ops.t, ops)
U0 = ops.to_dev(TG.ic_tgv(s))
Np0 = ops.zeros_c(tuple(OP.to_complex(U0).shape[:-2]) + (3, s['nk']),
                  OP.to_complex(U0))
c = T.implicit_coeff(dt, 0)
rw = OP.momentum_row_weights(c)
rwd = ops.to_dev(rw)
nk = s['nk']
shape = (s['m'].nelem, N+1, N+1, OP.NVAR_R, nk)
dj = S3.jacobi_diagonal_analytic(shape, s['D'], s['m'].facx, s['m'].facy,
                                 s['kz'], cfg['nu'], c, s['m'], s['mask'],
                                 s['m'].wq, 0.0, rw=rw)
Mj = ops.to_dev(S3.jacobi_inverse(dj, s['mask']))
_, _, batched = TG.stage(s, U0, Np0, 0, dt, {0: Mj}, rwd, cfg['tol'],
                         max_iter=40000, check_every=1)
print(f'{ne}x{ne} N={N} Nz={nz}, {nk} modes\n')
print(f'batched solve (all modes together): {batched} iterations\n')

# per-mode: restrict every array to one mode and solve that alone
b_full = S3.gs(s['m'], S3.normal_op(U0, s['Dg'], s['fxg'], s['fyg'], s['kzg'],
                                    cfg['nu'], c, s['m'], s['maskg'],
                                    s['wqg'], 0.0, rwd))*s['maskg']
its = []
print(f'{"k":>4} {"iters":>7}')
for k in range(nk):
    kzk = s['kzg'][k:k+1]
    mk = s['maskg'][..., k:k+1]
    bk = b_full[..., k:k+1]
    Mk = Mj[..., k:k+1]
    _, it, _ = S3.pcg(bk, s['Dg'], s['fxg'], s['fyg'], kzk, cfg['nu'], c,
                      mesh=s['m'], mask=mk, M_inv=Mk, tol=cfg['tol'],
                      max_iter=40000, wq=s['wqg'], rw=rwd, check_every=1)
    its.append(it)
    if nk <= 20 or k % max(1, nk//12) == 0 or k == nk-1:
        print(f'{k:>4} {it:>7}')
its = np.array(its)
now = its.max()*nk
ideal = its.sum()
print(f'\n  slowest mode {its.max()} iterations (k={int(its.argmax())}), '
      f'fastest {its.min()} (k={int(its.argmin())}), spread {its.max()/max(its.min(),1):.1f}x')
print(f'  batched cost  = {its.max()} x {nk} = {now:,} mode-iterations')
print(f'  per-mode cost = sum            = {ideal:,} mode-iterations')
print(f'  UPPER BOUND on what chunking could recover: '
      f'{100*(1-ideal/now):.0f}% of the solve')
print('\n  A chunked GPU solve trades batch size for less waste, so the real\n'
      '  gain is between 1x and that bound.  Below ~20% it is not worth the\n'
      '  complexity; above ~40% it is the largest remaining lever on this path.')
