"""Wall clock: Jacobi vs the CORRECTED PMG, setup and per-solve separated.

7K measured PMG at 0.28-0.48x Jacobi on wall time -- but with the
mask-inconsistent preconditioner, and without separating setup from per-solve.
Both matter here:

  SETUP amortises to nothing.  The coarse operator depends on (c, nu, kz, rw),
  and c = implicit_coeff(dt, k) takes exactly THREE values that repeat every
  step, so three factorisations serve a run of thousands of steps.  The driver
  already builds preconditioners once per stage outside the time loop.

  PER-SOLVE does not amortise.  A V-cycle costs ~2*deg fine-level smoother
  matvecs (13 at deg = 6) against Jacobi's one, and that is paid every
  iteration of every solve.  This is what 7K identified as the killer, and it
  is the number that decides the question.

Reported at two tolerances, because the mask bug was invisible at 1e-6 and
fatal at 1e-9 -- the corrected preconditioner should now behave at both.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
import tgv_gpu_run as TG
from lssem3d import operator as OP, solver3d as S3, timestep as T, precond as PC

N, ne, nz = 8, 3, 8
ops = TG.Ops('numpy')
dt = 0.0039
c = T.implicit_coeff(dt, 0)
print(f'numpy, {ne}x{ne} elements, N={N}, Nz={nz}\n')
for tol in (1e-6, 1e-9):
    cfg = dict(nu=6.25e-4, N=N, ex=ne, ey=ne, nz=nz, tend=1.0, snap=1.0,
               cfl=1.0, tol=tol)
    s = TG.setup(cfg, np, ops)
    U0 = TG.ic_tgv(s)
    Np0 = np.zeros(OP.to_complex(U0).shape[:-2] + (3, s['nk']), dtype=complex)
    rw = OP.momentum_row_weights(c)
    shape = (s['m'].nelem, N+1, N+1, OP.NVAR_R, s['nk'])

    t0 = time.perf_counter()
    dj = S3.jacobi_diagonal_analytic(shape, s['D'], s['m'].facx, s['m'].facy,
                                     s['kz'], cfg['nu'], c, s['m'], s['mask'],
                                     s['m'].wq, 0.0, rw=rw)
    Mj = S3.jacobi_inverse(dj, s['mask'])
    setup_j = time.perf_counter() - t0

    t0 = time.perf_counter()
    Mp = PC.PMG(s['m'], s['nk'], nz, cfg['nu'], c, s['kz'], rw=rw,
                orders=(8, 4, 2), deg=6, mask=s['mask'])
    setup_p = time.perf_counter() - t0

    print(f'tol = {tol:.0e}')
    print(f'  {"":<10} {"setup":>9} {"solve":>9} {"iters":>7} {"ms/iter":>9}')
    res = {}
    for nm, M in (('jacobi', Mj), ('pmg-fixed', Mp)):
        TG.stage(s, U0, Np0, 0, dt, {0: M}, rw, tol, max_iter=40000,
                 check_every=1)                       # warm
        t0 = time.perf_counter()
        _, _, it = TG.stage(s, U0, Np0, 0, dt, {0: M}, rw, tol,
                            max_iter=40000, check_every=1)
        el = time.perf_counter() - t0
        setup = setup_j if nm == 'jacobi' else setup_p
        res[nm] = (el, it)
        print(f'  {nm:<10} {setup:>8.2f}s {el:>8.3f}s {it:>7} '
              f'{1e3*el/max(it,1):>9.2f}')
    ej, ij = res['jacobi']; ep, ip = res['pmg-fixed']
    print(f'  -> PMG needs {ij/max(ip,1):.1f}x fewer iterations and '
          f'{ep/ej:.2f}x the wall time '
          f'({"WINS" if ep < ej else "LOSES"})')
    print(f'  -> per iteration PMG costs {(ep/max(ip,1))/(ej/max(ij,1)):.1f}x '
          f'Jacobi, against ~13 smoother matvecs expected\n')
print('Setup is one-off per RK stage and amortises over a whole run; the\n'
      'per-solve column is what recurs.  PMG also has NO GPU PATH, so on the\n'
      'production solver this comparison is moot until it is ported.')
