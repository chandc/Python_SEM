"""Is the N=14 Kovasznay error a SOLVER floor or the discrete solution itself?

If eps_u moves with cg_tol, we are solver-limited and Chan's 9.22e-13 is
reachable.  If it is flat, our discrete problem genuinely differs from his.
kov.py has a __main__ guard, so importing it runs nothing.
"""
import sys, os
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
import kov

print(f"{'N':>4}{'cg_tol':>10}{'status':>7}{'steps':>7}{'CG':>8}{'wall':>8}"
      f"{'eps_u':>12}{'eps_v':>12}{'eps_p':>12}{'res':>10}")
for N in (9, 14):
    for ct in (1e-10, 1e-12, 1e-14, 1e-16):
        r = kov.run(4, 2, N, 1e-13, cap=60, cg_tol=ct)
        print(f"{N:>4}{ct:>10.0e}{r['status']:>7}{r['steps']:>7}{r['cg']:>8}"
              f"{r['wall']:>7.1f}s{r['eu']:>12.3e}{r['ev']:>12.3e}{r['ep']:>12.3e}"
              f"{r['res']:>10.1e}", flush=True)
