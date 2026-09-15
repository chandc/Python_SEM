"""Is the fractional-step pressure solve GPU-bound or host-bound?  Measure both.

    python colab/fs_dispatch_probe.py [--backend cupy] [--consistent]

THE OBSERVATION THIS EXISTS TO EXPLAIN.  The consistent path costs 19.96 s/step
on an A100 and 7.65 s/step on a GB10, at identical iteration counts (88 pressure
iterations per substage on both).  The A100 has roughly 46x the GB10's fp64
throughput and 14x its bandwidth, so if the solve were limited by either, it
would win by a large factor.  Losing by 2.6x leaves one candidate: the loop is
limited by the HOST issuing kernels, and the GB10's tightly coupled CPU issues
them faster.

The vendored tree already documents this for its OWN least-squares operator
(`lssem3d/cupy_graph.py`): one `normal_op` cost 11.45 ms "regardless of problem
size -- 0.53 M dof and 6.17 M dof time identically", from which 84 % of the wall
clock was inferred to be dispatch.  A fix was written -- `pcg_graph`, CUDA-graph
capture of the CG inner loop -- and wired only into a check script, never into
the production solver, and in any case it wraps `solver3d.normal_op` rather than
the Helmholtz/E solves the projection path actually uses.

So the inherited number is not evidence about THIS solve.  This measures it.

HOW.  `cupyx.profiler.benchmark` reports CPU and GPU time separately for the
same call.  Applied to the fine-level E operator and to one V-cycle:

    gpu_time / cpu_time near 1   -> the device is the limit; a faster GPU helps
    gpu_time / cpu_time far below 1 -> the host is the limit; graph capture is
                                       worth the work and a faster GPU is not

The setup is not reimplemented: the production driver runs through `runpy` with
`--tend` equal to the restart time, so it builds everything and takes zero steps,
and the objects are taken from the module globals it returns.
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
    ap.add_argument('--backend', default='cupy')
    ap.add_argument('--consistent', action='store_true')
    ap.add_argument('--restart', default=os.path.join(
        _R, 'results/minchan_re180_E/state_t15.95.npz'))
    ap.add_argument('--repeat', type=int, default=20)
    a = ap.parse_args()

    t0 = float(np.load(a.restart)['t'])
    argv = ['fs_minchan_stats.py', '--restart', a.restart, '--backend', a.backend,
            '--tend', repr(t0), '--outdir', '/tmp/fs_probe']
    if a.consistent:
        argv.append('--consistent')
    sys.argv = argv
    print('building the production setup (zero steps)...', flush=True)
    G = runpy.run_path(os.path.join('scratch', 'fs_minchan_stats.py'),
                       run_name='__main__')

    s, Mp, Uc = G['s'], G['s']['Mp'], G['Uc']
    PJ = G['PJ']
    rng = np.random.default_rng(0)

    if a.backend != 'cupy':
        raise SystemExit('this probe needs --backend cupy (CPU has no dispatch gap)')
    import cupy as cp
    from cupyx.profiler import benchmark

    # a representative right-hand side in the pressure space
    div = PJ.divergence(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'])
    r = cp.ascontiguousarray(
        cp.concatenate([div.real, div.imag], axis=3)
        if div.dtype.kind == 'c' else div)
    r = (r*s['mask_p']).astype(cp.float64)

    print(f'\ndevice: {cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}')
    print(f'pressure dofs per mode: {r[..., 0].size:,}, modes: {r.shape[-1]}\n')

    tests = [('V-cycle preconditioner  M(r)', lambda: Mp(r))]
    lv = getattr(Mp, 'lv', None)
    if lv:
        tests.insert(0, ('fine-level operator     E(v)', lambda: lv[0].A(r)))

    print(f'{"call":28s} {"cpu ms":>9s} {"gpu ms":>9s} {"gpu/cpu":>9s}  verdict')
    for name, fn in tests:
        fn()                                      # warm up the memory pool
        b = benchmark(fn, n_repeat=a.repeat, n_warmup=3)
        cpu, gpu = b.cpu_times.mean()*1e3, b.gpu_times.mean()*1e3
        ratio = gpu/max(cpu, 1e-12)
        verdict = ('DEVICE-bound' if ratio > 0.8 else
                   'HOST-bound (dispatch)' if ratio < 0.4 else 'mixed')
        print(f'{name:28s} {cpu:9.3f} {gpu:9.3f} {ratio:9.2f}  {verdict}')

    print('\nReading: gpu/cpu well below 1 means the GPU finishes and waits while')
    print('Python issues the next kernel.  In that regime a faster GPU buys nothing')
    print('and CUDA-graph capture of the CG inner loop buys most of the wall clock.')


if __name__ == '__main__':
    main()
