"""p-trend for the two row-7 measurements.  See scratch/fosls3d_row7.py."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np, scipy.linalg as sla, importlib.util
_s = importlib.util.spec_from_file_location('r7', os.path.join(_R,'scratch','fosls3d_row7.py'))
r7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r7)
from lssem3d import backend, operator as OP

def comp(q):
    tot = float((q**2).sum())
    f = lambda i: 100*float((q[..., i, :]**2).sum())/tot
    return f(OP.P_), f(OP.OX_)+f(OP.OY_), f(OP.U_)+f(OP.V_)

def main():
    backend.set_backend('numpy')
    print(f'{"N":>2} {"dof":>5} | {"c2/c1 w7=1":>11} {"c2/c1 1e-4":>11} | '
          f'{"cond J w7=1":>11} {"cond J 1e-4":>11} | {"gap":>5} {"k":>2} '
          f'{"defl x":>7} | softest-J mode: p / om / u')
    print('-'*118)
    out = []
    for N in (4, 6, 8):
        m, D, kz, mask = r7.setup(N, 2, 2)
        r = {}
        for w7 in (1.0, 1.0e-4):
            A, B, mwf, shape = r7.dense_A(m, D, kz, mask, w7)
            H = r7.dense_H(m, D, kz, B, mwf, shape)
            ev = sla.eigvalsh(A, H)
            d = np.diag(A).copy(); d[d <= 0] = 1.0
            Dm = np.diag(1.0/np.sqrt(d))
            ej, Vj = np.linalg.eigh(Dm @ A @ Dm)
            k = ej > 0; ej, Vj = ej[k], Vj[:, k]
            r[w7] = (ev[-1]/ev[0], ej[-1]/ej[0], ej, B @ (Dm @ Vj[:, 0]), shape)
        ej = r[1.0][2]
        g = ej[1:21]/ej[:20]; kk = int(np.argmax(g)) + 1
        defl = (ej[-1]/ej[0])/(ej[-1]/ej[kk])
        pc, oc, uc = comp(r[1.0][3].reshape(r[1.0][4]))
        print(f'{N:2d} {A.shape[0]:5d} | {r[1.0][0]:11.4e} {r[1e-4][0]:11.4e} | '
              f'{r[1.0][1]:11.3e} {r[1e-4][1]:11.3e} | {g.max():5.2f} {kk:2d} '
              f'{defl:6.1f}x | {pc:5.1f}% {oc:5.1f}% {uc:5.1f}%')
        out.append((N, A.shape[0], r[1.0][0], r[1e-4][0], r[1.0][1], r[1e-4][1],
                    g.max(), kk, defl, pc, oc, uc))
    np.savez_compressed(os.path.join(_R,'scratch','fosls3d_row7_trend.npz'),
                        rows=np.array(out))
    print('\ncols: c2/c1 = H^1 ellipticity ratio;  cond J = cond(D^-1 A) Jacobi;')
    print('      gap/k = largest lambda_(k+1)/lambda_k in the lowest 20 at w7=1;')
    print('      defl x = cond reduction from deflating those k modes.')

if __name__ == '__main__':
    main()
