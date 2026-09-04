"""Where is the KNEE in cond(w7)?  We want the LARGEST w7 that still sits on the
conditioning floor: row 7's accuracy contribution scales with w7, its
conditioning damage saturates, so anything above the knee is paid for nothing and
anything below the knee throws away accuracy for free.  1e-4 may be far below it."""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np, importlib.util
_s = importlib.util.spec_from_file_location('r7', os.path.join(_R,'scratch','fosls3d_row7.py'))
r7 = importlib.util.module_from_spec(_s); _s.loader.exec_module(r7)
from lssem3d import backend

def cond(A):
    d = np.diag(A).copy(); d[d <= 0] = 1.0
    Dm = np.diag(1.0/np.sqrt(d)); e = np.linalg.eigvalsh(Dm@A@Dm); e = e[e > 0]
    return e[-1]/e[0]

def main(N=6):
    backend.set_backend('numpy')
    dkz = 2.0*np.pi/(0.34*np.pi)
    W = (1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5)
    print(f'N={N}, 2x2 elements, c=525.  cond(D^-1 A), and CG-iteration proxy sqrt(cond)\n')
    hdr = '   k_z |' + ''.join(f'{w:>10.0e}' for w in W)
    print(hdr); print('-'*len(hdr))
    tab = []
    for n in (1, 2, 4):
        kzv = n*dkz
        m, D, kz, mask = r7.setup(N, 2, 2, kz_val=kzv)
        row = [cond(r7.dense_A(m, D, kz, mask, w)[0]) for w in W]
        tab.append(row)
        print(f'{kzv:6.2f} |' + ''.join(f'{c:10.2e}' for c in row))
    tab = np.array(tab)
    print('\nrelative to the w7=1e-4 floor (1.00 = on the floor):')
    print(hdr); print('-'*len(hdr))
    for i, n in enumerate((1, 2, 4)):
        f = tab[i]/tab[i, W.index(1e-4)]
        print(f'{n*dkz:6.2f} |' + ''.join(f'{v:10.2f}' for v in f))
    np.savez_compressed(os.path.join(_R,'scratch','fosls3d_row7_knee.npz'),
                        w7=np.array(W), cond=tab)
    print('\nCG cost scales ~sqrt(cond), so a factor 4 in cond is only 2x in iterations.')

if __name__ == '__main__':
    main()
