"""Did PMG's mask bug change the ANSWER, not just the iteration count?

A preconditioner that returns zero on a subspace gives CG search directions
with no component there, so the iterate never moves in those directions and the
solve converges to the solution of a PROJECTED problem.  That is a correctness
failure, not a performance one, and it would be invisible in an iteration
count -- which is exactly what was observed: old and fixed both converge in 27
iterations while their outputs differ by 21%.

Compares the converged solutions of the same stage solve under Jacobi (the
reference), PMG-old (mask-inconsistent) and PMG-fixed.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import tgv_gpu_run as TG
from lssem3d import operator as OP, solver3d as S3, timestep as T, precond as PC

N, ne, nz = 8, 3, 8
ops = TG.Ops('numpy')
cfg = dict(nu=6.25e-4, N=N, ex=ne, ey=ne, nz=nz, tend=1.0, snap=1.0,
           cfl=1.0, tol=1e-9)
s = TG.setup(cfg, np, ops)
U0 = TG.ic_tgv(s)
Np0 = np.zeros(OP.to_complex(U0).shape[:-2] + (3, s['nk']), dtype=complex)
dt = 0.0039
c = T.implicit_coeff(dt, 0)
rw = OP.momentum_row_weights(c)
shape = (s['m'].nelem, N+1, N+1, OP.NVAR_R, s['nk'])
dj = S3.jacobi_diagonal_analytic(shape, s['D'], s['m'].facx, s['m'].facy,
                                 s['kz'], cfg['nu'], c, s['m'], s['mask'],
                                 s['m'].wq, 0.0, rw=rw)
Mj = S3.jacobi_inverse(dj, s['mask'])
common = dict(rw=rw, orders=(8, 4, 2), deg=6)
old = PC.PMG(s['m'], s['nk'], nz, cfg['nu'], c, s['kz'], **common)
new = PC.PMG(s['m'], s['nk'], nz, cfg['nu'], c, s['kz'], mask=s['mask'],
             **common)
sols = {}
for nm, M in (('jacobi', Mj), ('pmg-old', old), ('pmg-fixed', new)):
    U, _, it = TG.stage(s, U0, Np0, 0, dt, {0: M}, rw, cfg['tol'],
                        max_iter=20000, check_every=1)
    sols[nm] = U
    print(f'  {nm:<10} {it:>5} iterations')
ref = sols['jacobi']
sc = np.abs(ref).max()
print()
for nm in ('pmg-old', 'pmg-fixed'):
    d = np.abs(sols[nm] - ref).max()/sc
    print(f'  {nm:<10} vs jacobi: max relative difference {d:.3e}'
          f'   {"SAME ANSWER" if d < 1e-8 else "*** DIFFERENT ANSWER ***"}')
# where does any difference live?
d = np.abs(sols['pmg-old'] - ref)
if d.max() > 0:
    f = int(np.unravel_index(d.argmax(), d.shape)[3])
    k = int(np.unravel_index(d.argmax(), d.shape)[4])
    nm = {OP.U_: 'u', OP.V_: 'v', OP.W_: 'w', OP.OX_: 'ox', OP.OY_: 'oy',
          OP.OZ_: 'oz', OP.P_: 'p'}.get(f % OP.NVAR, '?')
    print(f'\n  largest pmg-old deviation is in field {nm} '
          f'({"imag" if f >= OP.NVAR else "real"}), mode k={k}')
