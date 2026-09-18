"""What tightening the NONLINEAR iteration buys, measured field by field.

The production cylinder runs take 3 Newton sub-iterations with AC and 2 without,
and neither is a convergence criterion: newton_factor = 0.0 makes the ratio test
unreachable and newton_tol = 1e-11 is never met, so the cap is simply what runs.
With AC the sub-iteration contracts at a measured 0.863 per iteration and cannot
be converged at any affordable count; without AC the same solver reaches 3e-10
in five.  This script asks what that costs, in the quantity the paper is about.

DIVERGENCE IS THE POINT.  Velocity and the forces are integrated quantities and
forgive a lot; the pointwise divergence is what the least-squares formulation
claims as its advantage over a projection method, and it is the first thing an
un-converged pressure should damage.  It is reported here as both an rms over
the field (volume weighted, so the huge outer cells cannot dominate) and a max.

THE noAC REFERENCE IS NOT TRUSTWORTHY -- read only the within-AC trend.  It was
included because the AC sub-iteration cannot be converged at any affordable
count while the AC-off iteration reaches 3e-10 in five, which made it look like
the only honest reference available.  It is not: AC off at this resolution
(N = 6, Xu = 20) DIVERGES at t = 450, and already at 4 steps its pressure field
has max|p| = 1.20 against 0.58 for AC, where stagnation pressure at U = 1 is
0.5.  The no-AC column here is a scheme on its way to blowing up, so the 7.6 %
C_D difference between the columns measures that, not a defect in AC.

WHAT THIS SCRIPT ACTUALLY ESTABLISHED: within AC, going from 3 sub-iterations to
20 moves C_D by 0.04 % and improves divergence rms by 17 % (plateauing at n = 8),
with divergence max unchanged.  Against a 4.3 % gap to the literature C_D, that
is 100x too small to matter.  NOTE this is a 4-step, single-phase comparison of
FORCES; it says nothing directly about St, which is a frequency over many
cycles and has to be measured by running to statistics.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np

import lssem2d
import lssem2d.solver as S
from lssem2d import curvi
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
from pmg_ghia_cavity import snapshot
import curvi_cyl_re100 as drv

SEED = 'scratch/_cyl_Xu20_N6dt0.1ac/final.npz'
DT, NSTEP = 0.1, 4
CASES = ((1, True), (2, True), (3, True), (5, True), (8, True), (12, True),
         (20, True), (2, False), (3, False), (5, False), (8, False))
REF = (8, False)


def build():
    m = curvi.build_cylinder_box(N=6, xs_upstream=curvi.nested_upstream_edges(20.0, 2))
    m.compute_global_indices()
    return m, diff_matrix(m.N)


def divergence(m, D, U):
    """|div u| on a curvilinear mesh: d/dx = rx d/dr + sx d/ds, likewise d/dy."""
    ur = np.einsum('ai,eij->eaj', D, U[..., 0])
    us = np.einsum('bj,eij->eib', D, U[..., 0])
    vr = np.einsum('ai,eij->eaj', D, U[..., 1])
    vs = np.einsum('bj,eij->eib', D, U[..., 1])
    d = m.rx*ur + m.sx*us + m.ry*vr + m.sy*vs
    rms = float(np.sqrt((d**2*m.wq).sum()/m.wq.sum()))   # volume weighted
    return rms, float(np.abs(d).max())


def run(nit, ac):
    m, D = build()
    w = np.sqrt(DT)
    st = SolverState(m, D, nu=0.01, dt=DT, fac1=1.0, w_mom=w, w_mass=w)
    st.fac1 = 1.5
    a_mass, _, _ = ls_coeffs(st)
    st.dtau_p = (1.0/(0.5*a_mass)) if ac else None
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
    for _ in range(NSTEP):
        S.step_bdf(st, h, time=t + DT, max_newton=nit, newton_tol=1e-14,
                   newton_factor=0.0, custom_inlet=inlet, pin_p=False,
                   cgsfac=1e-6, cg_tol=1e-12, cg_max_iter=8000,
                   line_search=False)
        t += DT
    return m, D, h[0]


def main():
    lssem2d.set_backend('numpy')
    m, D = build()
    wn = drv.wall_nodes(m, D)
    out = {}
    print(f'{NSTEP} steps from {SEED}, dt = {DT}\n')
    print(f'{"case":>12s} {"div rms":>10s} {"div max":>10s} {"C_D":>9s} {"C_L":>9s} '
          f'{"|du| vs ref":>11s} {"|dp| vs ref":>11s}')
    for nit, ac in CASES:
        _, _, U = run(nit, ac)
        out[(nit, ac)] = U
    Ur = out[REF]
    rr, rm = divergence(m, D, Ur)
    for nit, ac in CASES:
        U = out[(nit, ac)]
        rms, mx = divergence(m, D, U)
        cd, cl = drv.forces(U, wn, 0.01)
        du = np.abs(U[..., 0] - Ur[..., 0]).max()/np.abs(Ur[..., 0]).max()
        dp = np.abs(U[..., 2] - Ur[..., 2]).max()/np.abs(Ur[..., 2]).max()
        tag = f'{"AC" if ac else "noAC"} n={nit}'
        star = '  <- ref' if (nit, ac) == REF else ''
        print(f'{tag:>12s} {rms:10.3e} {mx:10.3e} {cd:9.5f} {cl:+9.5f} '
              f'{du:11.2e} {dp:11.2e}{star}')
    print(f'\nreference {REF[0]} sub-iterations without AC: div rms {rr:.3e}, '
          f'max {rm:.3e}')
    print('Production is "AC n=3".  |du|,|dp| are max-norm relative to the '
          'reference field.')


if __name__ == '__main__':
    main()
