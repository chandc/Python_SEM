"""Where does kappa_p pay for itself?  The Uzawa trade, measured on both sides.

Our AC continuity row is  kappa_p*(p - p_prev) + div u = 0, i.e.

    div u + kappa_p * p = kappa_p * p_prev

which is an ITERATED PENALTY (Uzawa) method with penalty parameter
epsilon = kappa_p -- see Kean, Xie & Xu, arXiv:2201.03978, and the references
there to Shen, Bercovier & Engelman, and Hughes, Liu & Brooks.  That framing
predicts a two-sided trade, and both sides are measured here:

  * SMALLER kappa_p  -> FASTER outer (sub-iteration) convergence, because the
    Jacobian's spurious +kappa shrinks and the damped fixed point contracts
    harder.  Measured: rate 0.715 / 0.870 / 0.940 at kappa_p = 1.19/2.37/4.74.
  * SMALLER kappa_p  -> WORSE conditioned linear system, so more PCG iterations
    ("choosing epsilon too small will cause numerical conditioning problems").

Production sits at kappa_p = a_mass/2 with 16 PCG iterations against 85 with no
AC at all, so the Schwarz+coarse preconditioner is absorbing the conditioning
cost easily -- which suggests there is headroom to spend on the outer rate.  The
question this answers is whether the TOTAL work, (sub-iterations to converge) x
(PCG per sub-iteration), has an optimum below the current setting.

WHAT THIS DOES NOT TEST: stability.  ARTIFICIAL_COMPRESSIBILITY.md's window was
established over long runs, and a handful of steps cannot see a slow divergence.
A kappa_p that wins here still has to survive a production run before it is
adopted.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import io
import contextlib
import re
import numpy as np

import lssem2d
import lssem2d.solver as S
from lssem2d import curvi
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
from pmg_ghia_cavity import snapshot

SEED = 'scratch/_cyl_Xu20_N6dt0.1ac/final.npz'
DT, NIT, NSTEP = 0.1, 12, 2
KFRACS = (None, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 1.0)   # None = AC off


def one(kfrac):
    m = curvi.build_cylinder_box(N=6, xs_upstream=curvi.nested_upstream_edges(20.0, 2))
    m.compute_global_indices()
    D = diff_matrix(m.N)
    w = np.sqrt(DT)
    st = SolverState(m, D, nu=0.01, dt=DT, fac1=1.0, w_mom=w, w_mass=w)
    st.fac1 = 1.5
    a_mass, _, _ = ls_coeffs(st)
    st.dtau_p = (1.0/(kfrac*a_mass)) if kfrac else None
    c = {'n': 0, 'pre': None}

    def factory(s_, fu, fv, Mi, pp):
        if c['pre'] is None or c['n'] % 40 == 0:
            sn = snapshot(s_, fu, fv)
            fu_, fv_ = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
            c['pre'] = VertexSchwarzCondensed2D(sn, fu_, fv_, pin_p=pp,
                        coarse=make_coarse(sn, fu_, fv_, Mi, pp))
        c['n'] += 1
        return c['pre']
    st.precond_factory = factory

    z = np.load(SEED, allow_pickle=True)
    h = [z['U0'].copy(), z['U0'].copy()]
    t = float(z['t'])
    inlet = lambda x, y, tt: 1.0
    last = ''
    for _ in range(NSTEP):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            S.step_bdf(st, h, time=t + DT, max_newton=NIT, newton_tol=1e-14,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=False,
                       cgsfac=1e-6, cg_tol=1e-12, cg_max_iter=8000,
                       line_search=False, verbose=True)
        t += DT
        last = buf.getvalue()
    d = [float(x) for x in re.findall(r'change = ([0-9.eE+-]+)', last)]
    it = [int(x) for x in re.findall(r'PCG iters = (\d+)', last)]
    return a_mass, d, it


def main():
    lssem2d.set_backend('numpy')
    print(f'seed {SEED}, dt = {DT}, {NIT} sub-iterations, {NSTEP} steps\n')
    print(f'{"kappa_p":>9s} {"/a_mass":>8s} {"rate":>7s} {"it2":>9s} '
          f'{"PCG/it":>7s} {"n to 1e-6":>10s} {"CG work":>9s} {"vs prod":>8s}')
    prod = None
    for kf in KFRACS:
        a_mass, d, it = one(kf)
        kp = kf*a_mass if kf else 0.0
        # asymptotic rate from the last four ratios; AC off converges outright
        rs = [d[i+1]/d[i] for i in range(len(d)-1) if d[i] > 0]
        rate = float(np.median(rs[-4:])) if len(rs) >= 4 else float('nan')
        pcg = float(np.mean(it[-4:])) if it else float('nan')
        # ORDER MATTERS.  A sequence that actually converges ends at round-off,
        # where consecutive ratios are noise and can exceed 1 -- so the
        # convergence test must come FIRST, or the one case that converges is
        # the one case scored as a failure.
        if min(d) < 1e-6:                         # converged outright
            n6 = float(next(i+1 for i, v in enumerate(d) if v < 1e-6))
            work = n6*float(np.mean(it[:int(n6)]))
        elif not np.isfinite(rate) or rate >= 1.0 or rate <= 0.0:
            n6, work = float('nan'), float('nan')
        else:
            n6 = 2 + np.log(1e-6/d[1])/np.log(rate)
            work = n6*pcg
        if kf == 0.5:
            prod = work
        tag = f'{kp:9.3f} {(kf if kf else 0):8.4f}'
        print(f'{tag} {rate:7.3f} {d[1]:9.2e} {pcg:7.1f} {n6:10.1f} '
              f'{work:9.0f} {work/prod if prod else float("nan"):7.2f}x',
              flush=True)
    print('\nCG work = (sub-iterations to reach 1e-6) x (PCG iterations each).')
    print('Stability is NOT tested here -- see the module docstring.')


if __name__ == '__main__':
    main()
