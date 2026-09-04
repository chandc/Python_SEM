"""Is the w7=1 soft cluster DISCRETE (deflatable) or a whole SUBSPACE?

sec 7J read "ranks 0-1, worth 10.5x" and inferred a couple of stray modes.  But
R_7 = div(omega) annihilates EVERY discretely divergence-free vorticity field,
and that is a subspace of dimension O(N^3), not a pair of vectors.  If the soft
set is a subspace, deflation is dead on arrival: you cannot deflate a constant
fraction of the dof.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np, importlib.util
_s = importlib.util.spec_from_file_location('r7', os.path.join(_R,'scratch','fosls3d_row7.py'))
r7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r7)
from lssem3d import backend

def spec(A):
    d = np.diag(A).copy(); d[d <= 0] = 1.0
    Dm = np.diag(1.0/np.sqrt(d)); e = np.linalg.eigvalsh(Dm@A@Dm)
    return e[e > 0]

def main():
    backend.set_backend('numpy')
    kzv = 2.0*np.pi/(0.34*np.pi)
    print(f'k_z={kzv:.2f}, 2x2 elements, c=525.  "soft" = lambda below lambda_min(w7=1e-4),')
    print('i.e. modes that row 7 at full weight pushes below the down-weighted floor.\n')
    print(f'{"N":>2} {"dof":>5} | {"n_soft":>6} {"% of dof":>8} | '
          f'{"defl 2":>7} {"defl n_soft":>11} | {"cond@1":>9} {"cond@1e-4":>9}')
    print('-'*76)
    for N in (4, 6, 8):
        m, D, kz, mask = r7.setup(N, 2, 2, kz_val=kzv)
        A1, *_ = r7.dense_A(m, D, kz, mask, 1.0)
        A4, *_ = r7.dense_A(m, D, kz, mask, 1.0e-4)
        e1, e4 = spec(A1), spec(A4)
        floor = e4[0]
        ns = int((e1 < floor).sum())
        c1 = e1[-1]/e1[0]
        d2 = c1/(e1[-1]/e1[2]) if len(e1) > 2 else 1.0
        dn = c1/(e1[-1]/e1[ns]) if ns < len(e1) else float('inf')
        print(f'{N:2d} {A1.shape[0]:5d} | {ns:6d} {100*ns/A1.shape[0]:7.1f}% | '
              f'{d2:6.1f}x {dn:10.1f}x | {c1:9.2e} {e4[-1]/e4[0]:9.2e}')
    print('\ndefl 2      = cond gain from deflating the 2 softest modes (sec 7J\'s reading)')
    print('defl n_soft = cond gain from deflating the WHOLE soft set')

if __name__ == '__main__':
    main()
