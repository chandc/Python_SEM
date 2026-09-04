"""Can a POINT-BLOCK smoother keep the FOSLS-correct w7=1 and still be fast?

sec 7J's mechanism is that R_7 wrecks the JACOBI DIAGONAL, not A.  Jacobi rescales
each dof independently and is blind to the omega_x-omega_y-omega_z coupling that
div(omega)=0 introduces -- so it sees a huge diagonal and a near-null mode.  The
14x14 point block at each node sees that coupling and can invert it.  If block
Jacobi at w7=1 recovers most of what point Jacobi loses, the fix is to keep the
formulation FOSLS prescribes and change the SMOOTHER, not the weight.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np, scipy.linalg as sla, importlib.util
_s = importlib.util.spec_from_file_location('r7', os.path.join(_R,'scratch','fosls3d_row7.py'))
r7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r7)
from lssem3d import backend

def cond_pt(A):
    d = np.diag(A).copy(); d[d <= 0] = 1.0
    Dm = np.diag(1.0/np.sqrt(d)); e = np.linalg.eigvalsh(Dm@A@Dm); e = e[e > 0]
    return e[-1]/e[0]

def cond_blk(A, nodes):
    """P = block-diag(A) over the 14 fields at each node; cond of P^-1/2 A P^-1/2."""
    grp = {}
    for c, nd in enumerate(nodes): grp.setdefault(nd, []).append(c)
    Ph = np.zeros_like(A)
    for idx in grp.values():
        ix = np.ix_(idx, idx)
        w, V = np.linalg.eigh(0.5*(A[ix] + A[ix].T))
        w = np.maximum(w, 1e-300*max(w.max(), 1.0))
        Ph[ix] = V @ np.diag(w**-0.5) @ V.T
    e = np.linalg.eigvalsh(Ph @ A @ Ph); e = e[e > 0]
    return e[-1]/e[0]

def main(N=6):
    backend.set_backend('numpy')
    dkz = 2.0*np.pi/(0.34*np.pi)
    print(f'N={N}, 2x2 elements, c=525.  cond of the PRECONDITIONED operator.\n')
    print(f'{"k_z":>7} | {"pt-Jac w7=1":>11} {"pt-Jac 1e-4":>11} | '
          f'{"blk w7=1":>10} {"blk 1e-4":>10} | {"blk@1 vs pt@1e-4":>17}')
    print('-'*80)
    out = []
    for n in (1, 2, 4):
        kzv = n*dkz
        m, D, kz, mask = r7.setup(N, 2, 2, kz_val=kzv)
        res = {}
        for w7 in (1.0, 1.0e-4):
            A, B, mwf, shape = r7.dense_A(m, D, kz, mask, w7)
            nodes = r7.dense_A.nodes
            res[w7] = (cond_pt(A), cond_blk(A, nodes))
        p1, b1 = res[1.0]; p4, b4 = res[1e-4]
        print(f'{kzv:7.2f} | {p1:11.3e} {p4:11.3e} | {b1:10.3e} {b4:10.3e} | '
              f'{b1/p4:16.2f}x')
        out.append((kzv, p1, p4, b1, b4))
    np.savez_compressed(os.path.join(_R,'scratch','fosls3d_row7_block.npz'), rows=np.array(out))
    print('\nlast column < ~1 means: block-Jacobi at the FOSLS-correct w7=1 is as well')
    print('conditioned as point-Jacobi at the down-weighted w7=1e-4, i.e. we can put')
    print('row 7 back at full weight for the price of a 14x14 node solve.')

if __name__ == '__main__':
    main()
