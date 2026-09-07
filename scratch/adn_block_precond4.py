"""Part 4: is the vorticity elimination (Schur complement S_u) worth anything
on its own?  kappa and CG iterations of Jacobi on S_u (u only, 6 real fields)
vs Jacobi on the full A, and the diagonal-vs-Rayleigh check that identifies
the bad modes as the discretely div-free, smooth ones (H(div) kernel)."""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sla
from adn_block_precond import pcg_ritz, VEL, VOR, PRE
from adn_block_precond2 import build

print(f'{"kz":>6} {"p":>3} | {"jacobi A":>14} | {"jacobi S_u":>14} | {"jacobi Hdiv":>14} | softest S_u eigvec: |div u|/|u|, |curl u|/|u| (scaled by p^2/h)')
for kcol in (0, 1):
    for N in (4, 6, 8, 12):
        A, free, field, kz = build(2, 2, N, 5405.4, kcol)
        rwc = np.zeros(8); rwc[1:4] = 1.0
        Ac, _, _, _ = build(2, 2, N, 5405.4, kcol, rw_override=rwc)
        A = A[free][:, free].tocsr(); Ac = Ac[free][:, free].tocsr(); field = field[free]
        iu, iw = (np.flatnonzero(np.isin(field, g)) for g in (VEL, VOR))
        Aww = spla.splu(A[iw][:, iw].tocsc()); Auw = A[iu][:, iw]; Awu = A[iw][:, iu]
        Su = A[iu][:, iu].toarray() - Auw@np.column_stack([Aww.solve(Awu[:, j].toarray().ravel()) for j in range(len(iu))])
        Su = 0.5*(Su + Su.T); Hd = (A[iu][:, iu] - Ac[iu][:, iu]).toarray()
        rng = np.random.default_rng(0)
        bA = A@rng.standard_normal(A.shape[0]); bA /= np.linalg.norm(bA)
        bS = Su@rng.standard_normal(len(iu)); bS /= np.linalg.norm(bS)
        rA = pcg_ritz(A, lambda r: r/A.diagonal(), bA)
        rS = pcg_ritz(sp.csr_matrix(Su), lambda r: r/np.diag(Su), bS)
        rH = pcg_ritz(sp.csr_matrix(Hd), lambda r: r/np.diag(Hd), bS)
        # softest Jacobi-preconditioned mode of S_u: what does it look like?
        d = np.diag(Su); w, V = sla.eigh(Su/np.sqrt(np.outer(d, d)))
        v = V[:, 0]/np.sqrt(d)
        Kdiv = Hd - np.diag(np.diag(Hd))*0  # placeholder to keep names honest
        divq = float(v@(Hd@v) - v@v)                 # v^T K_div v  (Hd = M + K_div, M~lumped ~ v^T M v)
        curlq = float(v@(Ac[iu][:, iu].toarray()@v))  # v^T K_curl v
        Mq = float(v@v)
        print(f'{kz:6.2f} {N:3d} | {rA[0]:5d} {rA[1]:8.1e} | {rS[0]:5d} {rS[1]:8.1e} | {rH[0]:5d} {rH[1]:8.1e} |  div {np.sqrt(max(divq,0)/Mq):8.2e}  curl {np.sqrt(max(curlq,0)/Mq):8.2e}', flush=True)
    print()
