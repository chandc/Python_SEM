"""Kovasznay: accuracy + timing + Mflops, against Chan (1996) Tables 1 and 2.

FLOP MODEL (counted from the source, not guessed).  Per CG iteration the work is
one apply_A = apply_L followed by apply_LT, plus the CG vector operations.

  Derivative applications (lssem.py):
    apply_L  : u_x u_y v_x v_y p_x p_y om_x om_y                        =  8
    apply_LT : c0 4 (DxT su3, DxT fu*su1, DyT su4, DyT fv*su1)
               c1 4, c2 2, c3 2                                         = 12
                                                                   total = 20
  Each application, per element: D(n x n) @ f(n x n) = n^2(2n-1) flops,
  then scaled by the metric = n^2 more, so exactly 2n^3.

  Pointwise, per node:
    apply_L  : su0 14, su1 14, su2 2, su3 3                             = 33
    apply_LT : prescale 2, c0 11, c1 11, c2 1, c3 4                     = 29
                                                                   total = 62
  CG vector ops per iteration over ndof = 4*nelem*n^2:
    p.Ap 3, x+=ap 2, r-=aAp 2, ||r|| 3, z=Mr 1, r.z 3, p=z+bp 2         = 16/dof
                                                            = 64 per node

  flops/CG-iteration = nelem * (20*2n^3 + 62n^2 + 64n^2)
                     = nelem * (40 n^3 + 126 n^2)

This counts the CG inner loop only -- Newton residual assembly, BC application
and preconditioner setup are excluded, so the rate is a LOWER BOUND on the true
delivered rate.  Chan's Mflops almost certainly came from a hardware counter and
would include everything, so his figure is the more inclusive of the two.
"""
import sys, os
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC); sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import kov
from lssem2d import backend


def flops_per_cg(nelem, N):
    n = N+1
    return nelem*(40.0*n**3 + 126.0*n**2)


CHAN_P = {4: (6.44e-2, 1.31e-2, 0.211, 19, 8.7, 25.7),
          9: (1.56e-6, 3.58e-7, 3.76e-6, 19, 30.3, 59.7),
          14: (9.22e-13, 4.72e-13, 1.47e-11, 26, 353, 96)}
CHAN_H = {(15, 10): (5.49e-2, 8.34e-3, 0.25, 18, 31.6, 52.4),
          (30, 20): (1.07e-2, 1.77e-3, 7.29e-2, 19, 258, 59.7),
          (60, 40): (1.56e-3, 2.69e-4, 1.66e-2, 19, 1916, 60.5)}

print(f"backend: {getattr(backend, 'active', lambda: '?')() if callable(getattr(backend,'active',None)) else getattr(backend,'_active','numpy')}")
print(f"flops/CG-iter = nelem*(40 n^3 + 126 n^2);  CG-loop only (lower bound)\n")
print(f"{'case':>9}{'elem':>6}{'N':>4}{'pts':>7}{'steps':>6}{'CG':>9}{'wall':>8}"
      f"{'Gflop':>9}{'Mflops':>9}{'eps_u':>11}{'eps_v':>11}{'eps_p':>11}"
      f"{'|Chan steps':>12}{'time':>8}{'Mflops':>8}{'eps_u':>11}")
rows = []
for (nex, ney, N) in [(4, 2, 4), (4, 2, 9), (4, 2, 14)]:
    r = kov.run(nex, ney, N, 1e-12, cap=60, cg_tol=1e-13)
    fl = flops_per_cg(r['nelem'], N)*r['cg']
    ref = CHAN_P[N]
    print(f"{'N=%d'%N:>9}{r['nelem']:>6}{N:>4}{r['npts']:>7}{r['steps']:>6}{r['cg']:>9}"
          f"{r['wall']:>7.2f}s{fl/1e9:>9.3f}{fl/r['wall']/1e6:>9.1f}"
          f"{r['eu']:>11.3e}{r['ev']:>11.3e}{r['ep']:>11.3e}"
          f"{ref[3]:>12d}{ref[4]:>7.1f}s{ref[5]:>8.1f}{ref[0]:>11.2e}", flush=True)
for (nex, ney) in [(15, 10), (30, 20), (60, 40)]:
    r = kov.run(nex, ney, 2, 1e-12, cap=60, cg_tol=1e-13)
    fl = flops_per_cg(r['nelem'], 2)*r['cg']
    ref = CHAN_H[(nex, ney)]
    print(f"{'%dx%d'%(nex,ney):>9}{r['nelem']:>6}{2:>4}{r['npts']:>7}{r['steps']:>6}{r['cg']:>9}"
          f"{r['wall']:>7.2f}s{fl/1e9:>9.3f}{fl/r['wall']/1e6:>9.1f}"
          f"{r['eu']:>11.3e}{r['ev']:>11.3e}{r['ep']:>11.3e}"
          f"{ref[3]:>12d}{ref[4]:>7.1f}s{ref[5]:>8.1f}{ref[0]:>11.2e}", flush=True)
