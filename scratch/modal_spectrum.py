"""Is (x,y) aliasing polluting the run?

z IS dealiased (3/2 rule, convect.py).  (x,y) is NOT, and this tree's
convective() has no skew-symmetric option -- the fractional-step code that made
the seed uses skew=True.  u.grad u sampled nodally is a degree-(2N-1) product
collapsed onto degree N, which aliases.

THE DIAGNOSTIC.  Expand each element's field on Legendre modes in x and y.  A
resolved spectral element shows modal energy decaying EXPONENTIALLY toward the
highest mode.  Aliasing (or under-resolution) shows up as the tail flattening or
turning UP -- energy piling into the modes that cannot be represented properly.

Compares the FOSLS run against its own fractional-step SEED, which was made with
the skew form: same mesh, same physics, different convective treatment.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
import numpy as np

def legendre_vandermonde(x, N):
    """V[i, n] = P_n(x_i) on the GLL nodes."""
    V = np.zeros((len(x), N+1))
    V[:, 0] = 1.0
    if N >= 1: V[:, 1] = x
    for n in range(1, N):
        V[:, n+1] = ((2*n+1)*x*V[:, n] - n*V[:, n-1])/(n+1)
    return V

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR
    from lssem2d.lgl import lgl_nodes
    import minchan as MC
    s = MC.setup(); m = s['m']; N = s['N']; nz = s['nz']
    xg = lgl_nodes(N)[0] if isinstance(lgl_nodes(N), tuple) else lgl_nodes(N)
    V = legendre_vandermonde(np.asarray(xg, float).ravel()[:N+1], N)
    Vi = np.linalg.inv(V)

    def modal(U):
        """RMS Legendre-mode amplitude vs polynomial degree, averaged over
        elements, z planes and the two directions."""
        C = OP.to_complex(U)
        u = FR.to_physical(np.ascontiguousarray(C[..., OP.U_:OP.U_+1, :]), nz)[..., 0, :]
        # u is (nelem, N+1, N+1, nz).  Transform both element directions.
        a = np.einsum('pi,eijz->epjz', Vi, u)
        a = np.einsum('qj,epjz->epqz', Vi, a)
        e = (a**2).mean(axis=(0, 3))            # (N+1, N+1) mean over elem, z
        # collapse to a 1-D spectrum vs max(p,q)
        deg = np.maximum.outer(np.arange(N+1), np.arange(N+1))
        return np.array([np.sqrt(e[deg == d].mean()) for d in range(N+1)])

    print(f'Legendre modal spectrum of u, N={N} (degrees 0..{N}), '
          f'mean over {m.nelem} elements and {nz} z-planes\n')
    runs = [('fractional-step SEED (skew)', 'scratch/fs_seed/seed_ckpt.npz')]
    for f in sorted(glob.glob('scratch/run01_ck/checkpoint_*.npz'))[-1:]:
        d = np.load(f)
        runs.append((f'FOSLS run01 t={float(d["t"]):.2f} (no skew)', f))
    print(f'{"degree":>7} ' + ' '.join(f'{n:>13}' for n, _ in runs) + '   ratio')
    sp = {}
    for name, f in runs:
        sp[name] = modal(np.load(f)['U'])
    n0, n1 = runs[0][0], runs[1][0]
    for d in range(N+1):
        print(f'{d:7d} ' + ' '.join(f'{sp[n][d]:13.4e}' for n, _ in runs) +
              f'   {sp[n1][d]/max(sp[n0][d],1e-300):6.2f}')
    for name, _ in runs:
        a = sp[name]
        print(f'\n  {name}:')
        print(f'    decay over the last 4 degrees: {a[-4]/a[-1]:8.1f}x  '
              f'(exponential decay => large; ~1 or <1 => tail pile-up)')
        print(f'    highest/peak mode ratio:       {a[-1]/a.max():.3e}')

if __name__ == '__main__':
    main()
