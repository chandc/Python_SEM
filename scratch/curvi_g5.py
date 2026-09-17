"""Gate G5 — the vertex-patch preconditioner survives a curvilinear mesh.

    uv run python scratch/curvi_g5.py

CURVILINEAR_2D_PLAN.md G5.  G2-G4 established that the DISCRETISATION is right,
using exact linear solves throughout so that no preconditioner could flatter or
spoil the answer.  This gate asks the complementary question, and it is the one
that decides whether the branch is usable rather than merely correct: does the
solver still converge in a number of iterations that does not grow with N?

The requirement is FLATNESS IN N, not a particular count.  A preconditioner that
captures the local operator gives an iteration count independent of the
polynomial order; one that does not gives a count growing like some power of N,
and on the normal equations of a least-squares system that power is brutal.  So
the affine mesh is run alongside as the control at every order, and the quantity
that matters is the ratio.  The plan allows up to ~1.5x degradation on a
strongly deformed mesh; more than that means the patch blocks are no longer
capturing the local operator.

WHAT HAD TO BE FIXED TO GET HERE.  `element_blocks` probes `apply_L`/`apply_LT`
with local unit vectors, so it inherited the curvilinear operator for free.  The
COARSE level did not: `PMG2` builds it with a shallow `copy` of the mesh and then
lowers `N`, which on a curvilinear mesh leaves the fine-order metric fields
attached to a coarse-order state.  The coarse geometry is now the fine geometry
interpolated to the coarse nodes -- not a re-evaluation of the mapping -- so both
levels describe the same domain to the accuracy each can represent.

THE LID-DRIVEN CAVITY IS THE RIGHT CASE because `curvi.deform` leaves the domain
boundary alone: the cavity is the same unit square with the same lid in both
runs, and the ONLY difference between the two columns is whether the elements
are rectangles.  Any change in iteration count is therefore attributable.

THE ROTATED CAVITY IS NOT A VALID CASE, and the first version of this gate ran it
anyway -- reporting 44 to 48 iterations against the affine 32, which reads as a
1.5x curvature penalty and is nothing of the kind.  Boundary code 2 prescribes
the CARTESIAN pair (u, v) = (custom_lid, 0) on the lid.  Rotate the mesh by 30
degrees and the lid rotates with it, but the prescribed velocity does not: (1, 0)
against an outward normal of (-sin 30, cos 30) has a NORMAL component of -0.5.
Every other wall is no-slip, so the net flux through the boundary is -0.5 and the
incompressibility constraint cannot be satisfied anywhere.  A least-squares
method does not diverge on that -- it minimises a residual it can never zero --
so the run completes and returns an iteration count that looks like a
measurement.  It is the solver fighting an inconsistent problem.

Rotation covariance is already established exactly, at 1e-15, by gate G1's T2,
which tests the OPERATOR and so needs no boundary conditions.  What replaces the
rotated column here is the annulus: a genuinely curved domain whose boundary data
is consistent, and which has no affine counterpart at all -- which is the point
of the gate.
"""
import os
import sys
import time as _time

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np

import lssem2d
from lssem2d import curvi
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
import lssem2d.solver as S
import curvi_g3 as g3
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
from pmg_ghia_cavity import snapshot

OUT = os.path.join(_R, 'figs', 'curvi_g5.png')
RE = 1000.0
DT = 1.0
EX = 4
NSTEP, NEWTON = 3, 2
CG_TOL = 1e-10


def mesh(N, kind, amp=0.10):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    if kind == 'deformed':
        curvi.deform(m, amp=amp, kx=2, ky=1)
    elif kind == 'curvi-affine':
        curvi.attach(m, *curvi.affine_coords(m))
    return m


def annulus(N, E_r=3, E_th=4):
    m = curvi.build_annulus(g3.A_IN, g3.B_OUT, E_r, E_th, N, 0.0, np.pi/2,
                            bcs=(1, 1, 1, 1))
    m.compute_global_indices()
    return m


def run(N, kind, amp=0.10):
    if kind == 'annulus':
        m = annulus(N)
        nu, exact = g3.NU, g3.exact_xy
    else:
        m = mesh(N, kind, amp)
        nu, exact = 1.0/RE, None
    st = SolverState(m, diff_matrix(N), nu=nu, dt=DT, fac1=1.0)
    n = N + 1

    def factory(s_, fu, fv, Mi, pp):
        # REBUILT EVERY NEWTON STEP: the linearisation changes, so a frozen
        # preconditioner would be the diagonal of a different operator.
        snap = snapshot(s_, fu, fv)
        fu_, fv_ = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
        return VertexSchwarzCondensed2D(snap, fu_, fv_, pin_p=pp,
                                        coarse=make_coarse(snap, fu_, fv_, Mi, pp))
    st.precond_factory = factory

    its = []
    _orig = S.pcg_solve

    def _spy(*a, **k):
        x, it = _orig(*a, **k)
        its.append(int(it))
        return x, it
    S.pcg_solve = _spy
    U = np.zeros((m.nelem, n, n, 4))
    h = [U.copy(), U.copy()]
    t0 = _time.perf_counter()
    try:
        for _ in range(NSTEP):
            S.step_bdf(st, h, time=0.0, max_newton=NEWTON, newton_tol=1e-12,
                       newton_factor=0.0, pin_p=True, exact_solution=exact,
                       cgsfac=0.0, cg_tol=CG_TOL, cg_max_iter=4000,
                       line_search=False)
    finally:
        S.pcg_solve = _orig
    return dict(mean=float(np.mean(its)), mx=int(np.max(its)),
                wall=_time.perf_counter() - t0,
                umax=float(np.abs(h[0][..., 0]).max()),
                ndof=int(m.gidx.max() + 1)*4)


def main():
    lssem2d.set_backend('numpy')
    Ns = [5, 6, 8, 10, 12]
    kinds = ['affine', 'curvi-affine', 'deformed', 'annulus']
    res = {k: [] for k in kinds}
    print(f'Ghia cavity Re = {RE:.0f}, {EX}x{EX} elements, dt = {DT}, '
          f'condensed vertex patch + p=2 coarse, CG tol {CG_TOL:g}')
    print(f'mean CG iterations per solve (max in brackets)\n')
    print(f'{"N":>3s} {"ndof":>7s} ' + ' '.join(f'{k:>16s}' for k in kinds)
          + f' {"deformed/affine":>16s}')
    for N in Ns:
        row, r0 = [], None
        for k in kinds:
            r = run(N, k)
            res[k].append(r)
            if k == 'affine':
                r0 = r
            row.append(f'{r["mean"]:9.1f} ({r["mx"]:3d})')
        print(f'{N:3d} {res["affine"][-1]["ndof"]:7d} ' + ' '.join(row)
              + f' {res["deformed"][-1]["mean"]/r0["mean"]:16.2f}')

    print('\n(the annulus column is a DIFFERENT problem -- Taylor-Couette on a\n'
          ' curved domain, Re = 5 -- so only its flatness in N is meaningful,\n'
          ' not its ratio to the cavity.)')
    reg = max(abs(a['mean'] - b['mean'])
              for a, b in zip(res['affine'], res['curvi-affine']))
    ratios = [d['mean']/a['mean'] for a, d in zip(res['affine'], res['deformed'])]
    growth = {k: res[k][-1]['mean']/res[k][0]['mean'] for k in kinds}
    print(f'\nflatness, mean iterations at N = {Ns[-1]} over N = {Ns[0]}:')
    for k in kinds:
        print(f'   {k:14s} {res[k][0]["mean"]:6.1f} -> {res[k][-1]["mean"]:6.1f}'
              f'   ({growth[k]:.2f}x)')
    print(f'\nworst deformed/affine iteration ratio: {max(ratios):.2f}  '
          f'(plan allows 1.5)')
    print(f'affine vs curvilinear-affine, same mesh: {reg:.1e} iterations apart')
    ok = max(ratios) < 1.5 and growth['deformed'] < 1.5*growth['affine'] and reg < 1e-9
    print(f'\ngate: the patch preconditioner stays FLAT in N on a curved mesh '
          f'and costs\n      under 1.5x against the Cartesian control -> '
          f'{"PASS" if ok else "FAIL"}')
    figure(Ns, res, kinds)
    return 0 if ok else 1


def figure(Ns, res, kinds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))

    m = mesh(8, 'deformed')
    n = m.nterm
    for el in range(m.nelem):
        for idx in (0, n-1):
            ax[0].plot(m.X[el, idx, :], m.Y[el, idx, :], 'k-', lw=1.0)
            ax[0].plot(m.X[el, :, idx], m.Y[el, :, idx], 'k-', lw=1.0)
    ax[0].plot(m.X.ravel(), m.Y.ravel(), '.', ms=1.2, color='C0', alpha=0.5)
    ax[0].set_aspect('equal'); ax[0].tick_params(labelsize=8)
    ax[0].set_title(f'the cavity mesh, deformed 10 %\n{EX}x{EX} elements, '
                    f'the lid and walls unmoved', fontsize=10)

    style = {'affine': ('C7', 'o', '--'), 'curvi-affine': ('C0', 's', '-'),
             'deformed': ('C3', 'D', '-'), 'annulus': ('C2', '^', '-')}
    for k in kinds:
        c, mk, ls = style[k]
        ax[1].plot(Ns, [r['mean'] for r in res[k]], marker=mk, ls=ls, color=c,
                   ms=5, label=k)
    ax[1].set_xlabel('polynomial order $N$')
    ax[1].set_ylabel('mean CG iterations per solve')
    ax[1].set_ylim(bottom=0)
    ax[1].set_title('G5: flat in $N$ is the requirement,\nnot a particular count',
                    fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    for k in ('deformed',):
        c, mk, _ = style[k]
        ax[2].plot(Ns, [d['mean']/a['mean']
                        for a, d in zip(res['affine'], res[k])],
                   marker=mk, ls='-', color=c, ms=5, label=f'{k} / affine')
    ax[2].axhline(1.0, color='C7', ls=':', lw=1)
    ax[2].axhline(1.5, color='C3', ls='--', lw=1.2, label='plan limit 1.5x')
    ax[2].set_xlabel('polynomial order $N$'); ax[2].set_ylabel('iteration ratio')
    ax[2].set_title('the cost of curvature, per iteration count', fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    ax[2].set_ylim(0, 2.0)

    fig.suptitle('Gate G5 — condensed vertex-patch preconditioner on a '
                 f'curvilinear mesh, Ghia cavity Re = {RE:.0f}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
