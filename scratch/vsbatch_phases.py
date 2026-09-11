"""Per-phase timing of the batched apply on the device (channel mesh by default)."""
import os, sys, time
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'torch')
import numpy as np, torch
import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
from lssem2d.mesh import build_channel
from lssem3d import bc as BC, operator as OP, fourier as FR, solver3d as S3
from lssem3d.precond import VertexSchwarzBatched3D
if __name__ == '__main__':
    N, ex, ey, nz = int(os.environ.get('N', 8)), int(os.environ.get('EX', 6)), int(os.environ.get('EY', 18)), int(os.environ.get('NZ', 32))
    nu, c = 1/180., 5405.4
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1)); m.periodic_x = np.pi; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0); BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    kz = FR.wavenumbers(nz, 0.34*np.pi); rw = OP.momentum_row_weights(c)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    bat = VertexSchwarzBatched3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask, verbose=True)
    r = torch.as_tensor(S3.gs(m, np.random.default_rng(0).standard_normal(shape))*mask, device=bat.dev)
    bat(r); bat(r)
    bat.timing = {}
    n = 10
    if bat.dev.type == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): bat(r)
    if bat.dev.type == 'cuda': torch.cuda.synchronize()
    tot = (time.perf_counter() - t0)/n*1e3
    print(f'apply {tot:.1f} ms on {bat.dev} ({bat.n_groups} mode groups); phases per apply:')
    for k, v in bat.timing.items():
        if k != 'start': print(f'   {k:16s} {v/n:7.2f} ms')
    if os.environ.get('TORCHPROF', '0') == '1':
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if bat.dev.type == 'cuda' else [])
        bat.timing = None
        with profile(activities=acts) as prof:
            for _ in range(3): bat(r)
            if bat.dev.type == 'cuda': torch.cuda.synchronize()
        key = 'cuda_time_total' if bat.dev.type == 'cuda' else 'cpu_time_total'
        print(prof.key_averages().table(sort_by=key, row_limit=18, max_name_column_width=60))
