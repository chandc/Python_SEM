"""Batched vertex-patch preconditioner (precond=vsbatch) on an NVIDIA GPU: the
test sequence for a Colab A100 (BALANCED_CONDENSED_PLAN.md step 4.5, gate
4.5; FP64_ON_APPLE_GPU.md sec 5 item 2).  fp64 throughout.

    python colab/run_vsbatch_a100.py [--quick] [--ckpt PATH] [--out DIR] [--backends torch,cuda]

Runs, in order, and writes a markdown summary to --out:
  1. device + fp64 rates for the two batched operations the apply is made of
  2. scratch/vsbatch_check.py on the production channel mesh (6x18 N=8, 17
     modes): factor sharing, build time, apply time per CG iteration, CG count
  3. scratch/minchan.py price: seconds per RKW3 step, Jacobi vs vsbatch, on the
     torch backend (and the fused 'cuda' backend if it compiles here)
  4. if --ckpt points at run01's checkpoint_0006200.npz: a two-step restart
     with vsbatch, to be compared with the Mac/GB10 Jacobi line
        t=4.961 u_tau=0.9937 U_b=15.840 rms_w=0.9201 E=897.30 eps=103.54 div=8.64e-04
     (both must agree to the printed digits; CG ~72 vs 4675).

--quick uses a 2x4 N=6 nz=8 mesh for steps 2-3 (a smoke test of the path, no
performance meaning).  Mac reference numbers (M3 Max CPU, fp64): apply 0.21 s
per iteration, full step 70 s vsbatch / ~80 s Jacobi (numba operator).
GB10 (Docker, fp64): apply 271 ms, vsbatch step 60 s at 71 CG/stage; run01's Jacobi step 60 s.
"""
import argparse, os, subprocess, sys, time, json, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def sh(cmd, env=None, log=None):
    e = dict(os.environ); e.update(env or {})
    print(f'\n$ {" ".join(cmd)}   [{" ".join(f"{k}={v}" for k, v in (env or {}).items())}]', flush=True)
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    out = p.stdout + p.stderr
    print('\n'.join(l for l in out.splitlines() if 'Warning' not in l), flush=True)
    if log is not None:
        log.append((cmd, out))
    return out


def fp64_rates(dev):
    import torch
    print(f'\n== 1. device {dev}: {torch.cuda.get_device_name(0) if dev.startswith("cuda") else "cpu"}')
    def tm(f, n=5):
        f(); (torch.cuda.synchronize() if dev.startswith('cuda') else None)
        t0 = time.perf_counter()
        for _ in range(n): f()
        (torch.cuda.synchronize() if dev.startswith('cuda') else None)
        return (time.perf_counter() - t0)/n
    n, k, b = 1302, 96, 17
    A = torch.randn(b, n, n, dtype=torch.float64, device=dev); A = A @ A.transpose(1, 2) + n*torch.eye(n, dtype=torch.float64, device=dev)
    L = torch.linalg.cholesky(A); B = torch.randn(b, n, k, dtype=torch.float64, device=dev)
    t = tm(lambda: torch.cholesky_solve(B, L)); print(f'   batched cholesky_solve 17 x ({n}^2, {k} rhs) fp64: {t*1e3:7.2f} ms = {b*4*n*n*k/t/1e9:6.0f} GFLOP/s   (Mac CPU: 24 ms / 230, GB10: 18 ms / 600)')
    t = tm(lambda: torch.bmm(A, B)); print(f'   batched GEMM           17 x ({n}^2 @ {n}x{k}) fp64: {t*1e3:7.2f} ms = {b*2*n*n*k/t/1e9:6.0f} GFLOP/s   (Mac CPU: 17 ms / 332, GB10: 18 ms / 307)')
    N = 4096; a = torch.randn(N, N, dtype=torch.float64, device=dev)
    t = tm(lambda: a @ a, 3); print(f'   DGEMM {N}: {2*N**3/t/1e12:5.2f} TFLOP/s   (A100 ~9.7-19, GB10 0.21, M3 Max CPU 0.41)')
    return dict(device=torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu')


def parse_price(out):
    m = re.search(r'step 1:\s*([\d.]+) s \((\d+) CG\)\s+step 2:\s*([\d.]+) s \((\d+) CG\)', out)
    return dict(step1=float(m.group(1)), cg=int(m.group(2)), step2=float(m.group(3))) if m else None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out', default='colab_results'); ap.add_argument('--backends', default='torch,cuda')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    import torch, datetime
    _commit = subprocess.run(['git', 'log', '--oneline', '-1'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print(f'code: {_commit}   started {datetime.datetime.now():%Y-%m-%d %H:%M:%S}', flush=True)
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['LSSEM3D_DEVICE'] = dev
    os.makedirs(a.out, exist_ok=True)
    R = dict(quick=a.quick, device=dev)
    R.update(fp64_rates(dev))
    mesh = dict(N='6', EX='2', EY='4', NZ='8') if a.quick else dict(N='8', EX='6', EY='18', NZ='32')
    price_args = ['N=6', 'ex=2', 'ey=4', 'nz=8'] if a.quick else []

    print('\n== 2. preconditioner build/apply on the channel mesh (device probes + apply; operator on the device)')
    out = sh([sys.executable, '-u', 'scratch/vsbatch_check.py'], env=dict(mesh, REF='0', LSSEM3D_BACKEND=('torch' if dev.startswith('cuda') else 'numpy'), LSSEM3D_DEVICE=dev))
    R['check'] = [l.strip() for l in out.splitlines() if 'VertexSchwarzBatched3D' in l or 'build batched' in l or 'pcg with' in l]

    print('\n== 3. seconds per RKW3 step: Jacobi vs vsbatch')
    R['price'] = {}
    for be in a.backends.split(','):
        for pc in ('jacobi', 'vsbatch'):
            out = sh([sys.executable, '-u', 'scratch/minchan.py', 'price', f'precond={pc}'] + price_args,
                     env=dict(LSSEM3D_BACKEND=be, LSSEM3D_DEVICE=dev))
            r = parse_price(out)
            R['price'][f'{be}/{pc}'] = r if r else 'failed: ' + out.strip().splitlines()[-1][:200]

    if a.ckpt and os.path.exists(a.ckpt):
        print('\n== 4. two-step restart from run01 with vsbatch (compare with the Jacobi line in the docstring)')
        outdir = os.path.join(a.out, 'restart_vsbatch'); os.makedirs(outdir, exist_ok=True)
        out = sh([sys.executable, '-u', 'scratch/minchan.py', 'run', f'out={outdir}', 'nstep=6202', 'dt=0.0008', 'every=2',
                  f'resume={a.ckpt}', 'weighting=legacy', 'precond=vsbatch'], env=dict(LSSEM3D_BACKEND=a.backends.split(',')[0], LSSEM3D_DEVICE=dev))
        R['restart'] = [l for l in out.splitlines() if 'u_tau=' in l]
    else:
        R['restart'] = 'skipped (no --ckpt)'

    md = [f"# vsbatch on {R['device']} ({'quick' if a.quick else 'channel 6x18 N=8 nz=32'})", '', '## fp64 rates and build/apply', ''] + [f'    {l}' for l in R['check']]
    md += ['', '## seconds per step (step 2 = steady-state timing)', '', '| backend / precond | step 1 | step 2 | CG/stage |', '|---|---|---|---|']
    for k, v in R['price'].items():
        md.append(f'| {k} | {v["step1"]:.1f} s | {v["step2"]:.1f} s | {v["cg"]} |' if isinstance(v, dict) else f'| {k} | {v} | | |')
    md += ['', '## restart from run01 (Mac Jacobi reference: t=4.961 u_tau=0.9937 U_b=15.840 rms_w=0.9201 E=897.30 eps=103.54 div=8.64e-04 CG=4675)', '']
    md += [f'    {l}' for l in R['restart']] if isinstance(R['restart'], list) else [f'    {R["restart"]}']
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    json.dump(R, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1, default=str)
    print('\n'.join(md))
