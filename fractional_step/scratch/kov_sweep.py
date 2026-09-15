"""Sweeps for the Kovasznay figures: h-refinement, spectral convergence, cost.

  p-sweep : N = 2..14 on the paper's 4x2 mesh, with BOTH CG guards, so the
            accuracy ceiling of the absolute guard is visible against the true
            spectral curve.
  h-sweep : 4 meshes at N = 2 and at N = 4, giving two algebraic curves whose
            slopes should be ~N+1.

Cost is recorded as wall time AND as modelled flops
(nelem*(40 n^3 + 126 n^2) per CG iteration), the latter being
hardware-independent and so the only fair axis against Chan's 1995 numbers.
"""
import sys, os, json
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC); sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import kov
import lssem2d.solver as S
from cg_rel import pcg_rel

_orig = S.pcg_solve
flops = lambda ne, N: ne*(40.0*(N+1)**3 + 126.0*(N+1)**2)
out = {'p_abs': [], 'p_rel': [], 'h2': [], 'h4': []}


def one(nex, ney, N, rel, cap=60):
    S.pcg_solve = pcg_rel if rel else _orig
    try:
        r = kov.run(nex, ney, N, 1e-12, cap=cap, cg_tol=1e-13)
    finally:
        S.pcg_solve = _orig
    r['gflop'] = flops(r['nelem'], N)*r['cg']/1e9
    r['mflops'] = r['gflop']*1e3/r['wall']
    r['nex'], r['ney'] = nex, ney
    return r


def show(tag, r):
    print(f"  {tag:>16}  N={r['N']:>2} elem={r['nelem']:>5} pts={r['npts']:>6} "
          f"steps={r['steps']:>3} CG={r['cg']:>7} {r['wall']:>7.2f}s "
          f"{r['gflop']:>8.3f}Gf {r['mflops']:>7.0f}Mf  eps_u={r['eu']:.3e}",
          flush=True)


print("p-sweep, 4x2 mesh, ABSOLUTE guard (the shipped pcg_solve)")
for N in (2, 4, 6, 8, 10, 12, 14):
    r = one(4, 2, N, rel=False); out['p_abs'].append(r); show('abs', r)

print("\np-sweep, 4x2 mesh, RELATIVE guard")
for N in (2, 4, 6, 8, 10, 12, 14):
    r = one(4, 2, N, rel=True); out['p_rel'].append(r); show('rel', r)

print("\nh-sweep at N = 2")
for (a, b) in ((8, 5), (15, 10), (30, 20), (60, 40)):
    r = one(a, b, 2, rel=False); out['h2'].append(r); show(f'{a}x{b}', r)

print("\nh-sweep at N = 4")
for (a, b) in ((4, 2), (8, 5), (15, 10), (30, 20)):
    r = one(a, b, 4, rel=False); out['h4'].append(r); show(f'{a}x{b}', r)

keep = ('N', 'nelem', 'npts', 'steps', 'cg', 'wall', 'gflop', 'mflops',
        'eu', 'ev', 'ep', 'res', 'status', 'nex', 'ney')
with open(f'{SC}/kov_sweep.json', 'w') as f:
    json.dump({k: [{a: v[a] for a in keep} for v in vs] for k, vs in out.items()},
              f, indent=1)
print('\nsaved kov_sweep.json')
