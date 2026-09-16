"""Is the convergence actually exponential?  Extend N until the two models split.

    uv run python scratch/curvi_rate_check.py

An R^2 comparison between an exponential and an algebraic fit over a span of N
smaller than a factor of three cannot separate them, and should not have been
quoted as if it could.  The discriminating test needs no fitting at all:

    exponential  e ~ exp(-bN)   ->  the DROP PER ORDER is constant,
                                    and the implied algebraic order q rises
                                    without bound
    algebraic    e ~ N^-q       ->  q is constant, and the drop per order falls

So the quantity to watch is q over successive intervals.  On the annulus (G3) it
runs 13.0 -> 24.8 and is still climbing: exponential.  On the DEFORMED box (G2)
it runs 5.8, 8.6, 15.8, 15.7, 14.9 -- climbing and then FLAT over the last three
intervals, which is the algebraic signature and is why this script exists.  Two
readings are possible and only more N separates them:

  * a crossover between two exponentials -- the solution error, which is fast,
    giving way to the geometry error of the collocation metrics, which is
    exponential too but slower.  Then q resumes climbing.
  * a genuine algebraic tail, which would mean the metrics are not consistent
    and G2's PASS was premature.  Then q stays flat.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np
import lssem2d
import curvi_g2 as g2
import curvi_g3 as g3


def table(tag, Ns, es):
    print(f'\n{tag}')
    print(f'  {"N":>3s} {"error":>12s} {"drop":>9s} {"local b":>8s} {"implied q":>10s}')
    for i, (N, e) in enumerate(zip(Ns, es)):
        if i == 0:
            print(f'  {N:3d} {e:12.3e}')
            continue
        dN = Ns[i] - Ns[i-1]
        rat = es[i-1]/e
        print(f'  {N:3d} {e:12.3e} {rat:9.1f} {np.log(rat)/dN:8.2f} '
              f'{np.log(rat)/np.log(Ns[i]/Ns[i-1]):10.1f}')


def main():
    lssem2d.set_backend('numpy')
    g2._F = g2._fields()

    # How distorted is the deformed mesh, really?  A near-zero Jacobian would
    # explain a slow rate without any inconsistency in the metrics.
    print('deformed mesh Jacobian (G2, amp = 10 %):')
    for N in (8, 12, 16):
        m = g2.make_mesh(N, 2, 'deformed')
        print(f'   N = {N:2d}   min J = {m.jacq.min():.4e}   max J = {m.jacq.max():.4e}'
              f'   max/min = {m.jacq.max()/m.jacq.min():.3f}')

    Ns2 = [8, 10, 12, 14, 16, 18]
    es2 = []
    for N in Ns2:
        e = g2.solve(N, 2, 'deformed')
        es2.append(e['uv'])
        print(f'   G2 deformed N = {N:2d}: {e["uv"]:.4e}  ({e["wall"]:.0f} s)', flush=True)
    table('G2 deformed box, extended', Ns2, es2)

    Ns3 = [8, 9, 10, 11]
    es3 = []
    for N in Ns3:
        e = g3.solve(N)
        es3.append(e['uv'])
        print(f'   G3 annulus  N = {N:2d}: {e["uv"]:.4e}  ({e["wall"]:.0f} s)', flush=True)
    table('G3 annulus, extended (watching for the round-off floor)', Ns3, es3)


if __name__ == '__main__':
    main()
