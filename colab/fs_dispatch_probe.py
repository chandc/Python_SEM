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

HOW -- AND HOW THE FIRST VERSION OF THIS WAS WRONG.  It compared
`cupyx.profiler.benchmark`'s cpu_times against its gpu_times and called a ratio
near 1 "device-bound".  That test cannot discriminate: gpu_times is the elapsed
time between two CUDA events on the stream, and that interval INCLUDES the idle
gaps while the host issues the next kernel.  A launch-bound loop and a
device-bound loop both report gpu ~ cpu.  Measured on an A100 it duly returned
1.00 for both calls, which says nothing.

What does discriminate is kernel-level accounting: sum the time the GPU spends
inside kernels and compare it with the wall clock.  `nsys` reports both, plus the
kernel count, so `--nsys` runs the same applies under it.  Without nsys the probe
still reports the wall time per apply against this problem's bandwidth floor,
which bounds how much room there is without saying where it went.

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
        print(f'{name:28s} {cpu:9.3f} {gpu:9.3f} {ratio:9.2f}  '
              f'{"(ratio is uninformative -- see the module docstring)":s}')

    # ---- THE TEST THAT DISCRIMINATES, and it needs no nsys ----
    # The V-cycle already holds the SAME operator at three polynomial orders.
    # Work per apply scales with the dof count, roughly (p+1)^2 per element;
    # launch cost does not scale at all.  So:
    #     time roughly proportional to dofs -> the device is doing the work
    #     time roughly FLAT across a 9x range of dofs -> the host is the limit
    # This is the same test `cupy_graph.py` used when it found one matvec costing
    # "11.45 ms regardless of problem size".
    if lv and len(lv) > 1:
        print(f'\n{"level":>7s} {"order":>6s} {"dofs/mode":>11s} {"ms":>9s} '
              f'{"ms/Mdof":>10s}')
        base = None
        for i, l in enumerate(lv):
            shp = l.shape
            v = cp.zeros(shp, dtype=cp.float64)
            v[...] = 1.0
            v = v*l.mask
            try:
                l.A(v)
                b = benchmark(lambda: l.A(v), n_repeat=a.repeat, n_warmup=3)
            except Exception as e:
                print(f'{i:7d} {"?":>6s}  level apply failed: {type(e).__name__}')
                continue
            ms = b.cpu_times.mean()*1e3
            nd = int(np.prod(shp[:-1]))*shp[-1]
            if base is None:
                base = (ms, nd)
            print(f'{i:7d} {getattr(l, "p", "?"):>6} {nd/shp[-1]:11,.0f} {ms:9.3f} '
                  f'{ms/(nd/1e6):10.2f}')
        if base:
            print('\n  flat ms down the levels -> launch-bound: fix by fusing or')
            print('  capturing the sequence.  ms falling with dofs -> the kernels')
            print('  are doing real work and the fix is fewer/better kernels.')

    # How much room is there?  Bandwidth floor for one fine-level E apply.
    np_ = r[..., 0].size*r.shape[-1]
    mb = np_*8/1e6
    traffic = 2*mb + 4*3*mb                    # G, M^-1, G^T with temporaries
    floor = traffic*1e6/1.555e12*1e3           # ms at the A100's 1555 GB/s
    print(f'\npressure field {mb:.1f} MB; generous traffic for one E apply '
          f'{traffic:.0f} MB -> {floor:.3f} ms at 1555 GB/s')
    print('So the measured apply is far above the floor.  WHERE it goes needs')
    print('kernel-level accounting -- rerun with --nsys, or under:')
    print('  nsys profile --stats=true -t cuda python colab/fs_dispatch_probe.py ...')
    print('and compare total kernel time with wall time; a large gap is launch')
    print('overhead (graph capture helps), a small gap with thousands of tiny')
    print('kernels is fusion (graph capture helps less, kernel work helps more).')


if __name__ == '__main__':
    main()
