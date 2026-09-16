"""Does the captured V-cycle give the same answer, and how much faster?

    python colab/fs_graph_check.py                    # correctness + speed
    python colab/fs_graph_check.py --steps 20         # and an end-to-end step time

CORRECTNESS FIRST, AND THE BAR IS BIT-EXACTNESS.  Graph replay re-executes the
same kernels in the same order against the same pointers, so the captured
V-cycle should not merely agree to solver tolerance -- it should agree to the
last bit.  Anything else means a pointer moved, which is the failure mode that
produces wrong numbers rather than errors, so this refuses the speed-up unless
the difference is exactly zero or at round-off.

Three vectors are tried, including the real right-hand side the solver sees, and
each is applied twice to catch a graph whose result depends on what the previous
call left in its buffers.
"""
import argparse
import os
import runpy
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = os.path.join(_R, 'fractional_step')
sys.path.insert(0, FS)
sys.path.insert(0, os.path.join(FS, 'scratch'))
os.chdir(FS)

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restart', default=os.path.join(
        _R, 'results/minchan_re180_E/state_t15.95.npz'))
    ap.add_argument('--repeat', type=int, default=20)
    ap.add_argument('--steps', type=int, default=0,
                    help='also time this many full steps with and without the graph')
    a = ap.parse_args()

    t0 = float(np.load(a.restart)['t'])
    sys.argv = ['fs_minchan_stats.py', '--restart', a.restart, '--backend', 'cupy',
                '--tend', repr(t0), '--outdir', '/tmp/fs_graphcheck', '--consistent']
    print('building the production setup (zero steps)...', flush=True)
    G = runpy.run_path(os.path.join('scratch', 'fs_minchan_stats.py'),
                       run_name='__main__')

    import cupy as cp
    from cupyx.profiler import benchmark
    from lssem3d.graph_vcycle import maybe_graph

    s, Uc, PJ = G['s'], G['Uc'], G['PJ']
    Mp = s['Mp']
    mask = s['mask_p']

    # the right-hand side the solver actually presents, plus two synthetic ones
    div = PJ.divergence(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'])
    real_rhs = (cp.concatenate([div.real, div.imag], axis=3)
                if div.dtype.kind == 'c' else div)
    real_rhs = cp.ascontiguousarray(real_rhs*mask).astype(cp.float64)
    rng = cp.random.default_rng(0)
    cases = [('solver right-hand side', real_rhs),
             ('random', (rng.standard_normal(real_rhs.shape)*mask).astype(cp.float64)),
             ('ones', (cp.ones_like(real_rhs)*mask))]

    ref = [Mp(r).copy() for _, r in cases]          # uncaptured reference first

    Mg = maybe_graph(Mp, real_rhs, enable=True, verbose=True)
    if Mg is Mp:
        raise SystemExit('graph capture did not engage; nothing to check')

    print(f'\n{"case":26s} {"max |graph - ref|":>18s} {"rel":>10s}  verdict')
    worst = 0.0
    for (name, r), z0 in zip(cases, ref):
        for rep in range(2):                        # twice: catch buffer reuse bugs
            z = Mg(r)
            d = float(cp.abs(z - z0).max())
            rel = d/max(float(cp.abs(z0).max()), 1e-300)
            worst = max(worst, rel)
            if rep:
                tag = 'BIT-EXACT' if d == 0.0 else ('round-off' if rel < 1e-13
                                                    else '** MISMATCH **')
                print(f'{name:26s} {d:18.3e} {rel:10.2e}  {tag}')
    if worst > 1e-13:
        raise SystemExit('\nREFUSED: the captured V-cycle does not reproduce the '
                         'reference.  Do not use it.')

    print('\ncorrectness passed; timing the apply')
    for lab, fn in (('uncaptured', lambda: Mp(real_rhs)),
                    ('graph      ', lambda: Mg(real_rhs))):
        fn()
        b = benchmark(fn, n_repeat=a.repeat, n_warmup=3)
        print(f'  {lab}  {b.cpu_times.mean()*1e3:8.3f} ms')

    if a.steps:
        print(f'\nend-to-end: {a.steps} steps with and without the graph')
        import subprocess
        for flag in ('', '--graph'):
            cmd = [sys.executable, os.path.join(_R, 'colab', 'fs_profile.py'),
                   '--steps', str(a.steps), '--backend', 'cupy', '--consistent']
            if flag:
                cmd.append(flag)
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 cwd=_R).stdout
            row = [l for l in out.splitlines() if 'phases total' in l or 'pressure' in l]
            print(f'  {"with graph" if flag else "baseline  "}: ' + ' | '.join(r.strip() for r in row))


if __name__ == '__main__':
    main()
