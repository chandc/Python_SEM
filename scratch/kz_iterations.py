"""Does the preconditioner use k_z?  CG iterations per Fourier mode.

    uv run --quiet python scratch/kz_iterations.py [nz] [N] [ex]

3D_DEVELOPMENT_PLAN.md sec 4 states the expectation and the diagnosis together:
iterations-per-solve should FALL with k_z, because the k_z^2 term makes the
operator more diagonally dominant as the mode gets higher; a FLAT profile is
evidence the preconditioner is not using k_z at all.

This is the cheapest high-information measurement available before Stage 5 and
M7, and it is deliberately done BEFORE optimising the Jacobi diagonal: if the
profile is flat, the preconditioner is the wrong object to be optimising, and
the analytic-diagonal work would be effort spent on the wrong thing.

METHOD.  Each mode is solved as its OWN single-mode problem (nmode = 1) rather
than reading a batched count, because the batched `pcg` reports only the worst
mode -- it iterates until ALL modes converge, so per-mode counts are invisible
there by construction.  The same spatial RHS is used at every k_z, so the only
thing varying is the wavenumber.

Recorded both preconditioned and unpreconditioned: the unpreconditioned column
is the control.  If BOTH fall with k_z, that is the operator becoming better
conditioned and says nothing about the preconditioner.  The preconditioner's
contribution is the RATIO of the two.
"""
import os, sys, time, json
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d import timestep as T

RE = 180.0
NU = 1.0/RE
DT = 5e-3
LZ = 2.0*np.pi
TOL = 1e-8
MAXIT = 6000


def run(nz=64, N=8, ex=4):
    mesh = build_channel(1.0, 1.0, ex, ex, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    kz_all = FR.wavenumbers(nz, LZ)
    c = T.implicit_coeff(DT, 2)          # worst stage -- the one that sets a_mass
    kap = c                              # AC: kappa_p = a_mass (M5)
    n = N+1

    # one fixed spatial RHS pattern, reused at every k_z so only k varies
    rng = np.random.default_rng(0)
    pattern = rng.standard_normal((mesh.nelem, n, n, OP.NVAR_R, 1))

    # Build the mask for the FULL mode set once and slice one column per probe.
    # Do not call build_mask(mesh, 1, ...) here: nmode == 1 is defined to mean
    # "k_z = 0 alone" and freezes the whole imaginary half, which is right for
    # the M2 cavity but wrong for a single probe at k_z != 0.  Slicing the real
    # mask is also exactly what a production run does.
    mask_full = BC.build_mask(mesh, len(kz_all), pin_p=True, nz=nz)

    rows = []
    print(f'Re={RE}  dt={DT:g}  c=a_mass={c:.1f}  kappa_p={kap:.1f}  '
          f'tol={TOL:g}\n{ex}x{ex} elements, N={N}, Nz={nz}\n')
    print(f"{'idx':>4}{'k_z':>9}{'its(jac)':>10}{'its(none)':>11}"
          f"{'ratio':>8}{'t_jac s':>9}")
    for idx, k in enumerate(kz_all):
        kz = np.array([k])
        mask = np.ascontiguousarray(mask_full[..., idx:idx+1])
        b = S3.gs(mesh, pattern*mask)*mask

        diag = S3.jacobi_diagonal(b.shape, D, mesh.facx, mesh.facy, kz, NU, c,
                                  mesh, mask, mesh.wq, kap)
        Minv = 1.0/np.maximum(diag, 1e-30)

        t0 = time.perf_counter()
        _, it_j, _ = S3.pcg(b, D, mesh.facx, mesh.facy, kz, NU, c, mesh, mask,
                            Minv, TOL, MAXIT, None, mesh.wq, kap)
        t_j = time.perf_counter()-t0
        _, it_n, _ = S3.pcg(b, D, mesh.facx, mesh.facy, kz, NU, c, mesh, mask,
                            None, TOL, MAXIT, None, mesh.wq, kap)
        rows.append(dict(idx=idx, kz=float(k), it_jac=int(it_j),
                         it_none=int(it_n), t=t_j))
        print(f'{idx:>4}{k:>9.2f}{it_j:>10}{it_n:>11}'
              f'{it_n/max(it_j,1):>8.2f}{t_j:>9.2f}')

    js = [r['it_jac'] for r in rows]
    ns = [r['it_none'] for r in rows]
    print(f'\n  jacobi  : k_z=0 -> {js[0]},  max {max(js)},  '
          f'k_z max -> {js[-1]}')
    print(f'  none    : k_z=0 -> {ns[0]},  max {max(ns)},  '
          f'k_z max -> {ns[-1]}')
    trend = js[-1]/max(js[0], 1)
    print(f'\n  iterations at highest k_z / at k_z=0  =  {trend:.3f}')
    print('  << 1 : falls with k_z, as the k_z^2 diagonal dominance predicts')
    print('  ~ 1  : FLAT -- the preconditioner is not using k_z')
    with open('scratch/kz_iterations.json', 'w') as f:
        json.dump(rows, f, indent=1)
    print('\n  wrote scratch/kz_iterations.json')
    return rows


if __name__ == '__main__':
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 64,
        int(sys.argv[2]) if len(sys.argv) > 2 else 8,
        int(sys.argv[3]) if len(sys.argv) > 3 else 4)
