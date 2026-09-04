"""k_z sweep -- at k_z=0 the term i*k_z*om_z drops out of R_7, so the k_z=0
result cannot settle whether R_7 matters.  Channel LZ=0.34*pi -> dkz=5.88."""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np, scipy.linalg as sla, importlib.util
_s = importlib.util.spec_from_file_location('r7', os.path.join(_R,'scratch','fosls3d_row7.py'))
r7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r7)
from lssem3d import backend, operator as OP

def main(N=6):
    backend.set_backend('numpy')
    dkz = 2.0*np.pi/(0.34*np.pi)
    print(f'N={N}, 2x2 elements, c=525, nu=1/180;  dkz={dkz:.3f} (LZ=0.34*pi)\n')
    print(f'{"k_z":>7} | {"c2/c1 w7=1":>11} {"c2/c1 1e-4":>11} {"ratio":>6} | '
          f'{"condJ w7=1":>10} {"condJ 1e-4":>10} {"ratio":>6} | softest-J: p/om/u')
    print('-'*104)
    out = []
    for n in (0, 1, 2, 4):
        kzv = n*dkz
        m, D, kz, mask = r7.setup(N, 2, 2, kz_val=kzv)
        r = {}
        for w7 in (1.0, 1.0e-4):
            A, B, mwf, shape = r7.dense_A(m, D, kz, mask, w7)
            H = r7.dense_H(m, D, kz, B, mwf, shape)
            ev = sla.eigvalsh(A, H)
            d = np.diag(A).copy(); d[d <= 0] = 1.0
            Dm = np.diag(1.0/np.sqrt(d))
            ej, Vj = np.linalg.eigh(Dm @ A @ Dm)
            k = ej > 0; ej, Vj = ej[k], Vj[:, k]
            r[w7] = (ev[-1]/ev[0], ej[-1]/ej[0], B @ (Dm @ Vj[:, 0]), shape)
        q = r[1.0][2].reshape(r[1.0][3]); tot = float((q**2).sum())
        f = lambda i: 100*float((q[..., i, :]**2).sum())/tot
        pc = f(OP.P_); oc = f(OP.OX_)+f(OP.OY_)+f(OP.OZ_); uc = f(OP.U_)+f(OP.V_)+f(OP.W_)
        print(f'{kzv:7.2f} | {r[1.0][0]:11.4e} {r[1e-4][0]:11.4e} '
              f'{r[1e-4][0]/r[1.0][0]:6.3f} | {r[1.0][1]:10.3e} {r[1e-4][1]:10.3e} '
              f'{r[1.0][1]/r[1e-4][1]:6.2f} | {pc:5.1f}% {oc:5.1f}% {uc:5.1f}%')
        out.append((kzv, r[1.0][0], r[1e-4][0], r[1.0][1], r[1e-4][1], pc, oc, uc))
    np.savez_compressed(os.path.join(_R,'scratch','fosls3d_row7_kz.npz'), rows=np.array(out))
    print('\n"ratio" under c2/c1 = how much WORSE the H^1 ellipticity gets at w7=1e-4 (1.0 = no cost).')
    print('"ratio" under condJ = how much BETTER Jacobi conditioning gets at w7=1e-4.')

if __name__ == '__main__':
    main()
