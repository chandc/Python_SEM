"""Gate G1 — rotation invariance, the strongest test in the curvilinear plan.

    uv run python scratch/curvi_g1.py

CURVILINEAR_2D_PLAN.md G1.  A rigidly rotated mesh is still AFFINE -- the
Jacobian is constant per element -- but it is not axis-aligned, so every metric
term and every cross term is live while the exact answer is known: it is the
unrotated answer, rotated.  A metric that is merely self-consistent passes G0 and
fails here.

THREE TESTS, in increasing strength.

**T1 — the curvilinear path reproduces the affine path.**  Convert an affine mesh
to curvilinear coordinates without moving a node and apply the operator.  The two
code paths must agree to round-off.  This is the regression that protects every
result already in the paper: if it drifts, the branch has broken the Cartesian
code.

**T2 — the operator is rotation-covariant.**  Let $R_\\theta$ rotate the plane.
Put a field on the unrotated mesh, put its rotated image on the rotated mesh, and
require the residual rows to be the rotated image of the original rows:

    momentum (2 components)  ->  rotates as a vector
    continuity               ->  invariant (a scalar)
    vorticity definition     ->  invariant (curl_z is a pseudo-scalar,
                                 invariant under a PROPER rotation)

This tests the assembled operator, not a solve, so no boundary conditions are
involved and nothing is hidden by a preconditioner or an iteration count.  It is
the sharpest statement available that the metrics and the rows agree.

**T3 — the adjoint pair survives rotation.**  <L U, S> = <U, L^T(wq S)> on every
rotated mesh.  `apply_L` multiplies its rows by `wq`, so `apply_LT` expects the
weight already inside its argument; passing an unweighted S makes this read 89
instead of 1e-16, which is a property of the convention and not of the code.
The least-squares operator is built from adjoint pairs, so this failing would
make every solve wrong while leaving T1 and T2 green.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np

import lssem2d
from lssem2d import curvi
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT
from lssem2d.mesh import build_channel

OUT = os.path.join(_R, 'figs', 'curvi_g1.png')
N, E = 6, 2
NU, DT = 0.01, 1.0


def make(theta=None, deform=None):
    m = build_channel(1.0, 1.0, E, E, N)
    m.compute_global_indices()
    if theta is not None:
        curvi.rotate(m, theta)
    elif deform is not None:
        curvi.deform(m, deform, 2, 1)
    else:
        curvi.attach(m, *curvi.affine_coords(m))
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n))
    fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)
    return m, st, fu, fv


def affine_state():
    m = build_channel(1.0, 1.0, E, E, N)
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n))
    fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)
    return m, st, fu, fv


def field(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def rot_field(U, theta):
    """Rotate a (u, v, p, omega) field: velocity as a vector, p and omega scalars."""
    c, s = np.cos(theta), np.sin(theta)
    out = U.copy()
    out[..., 0] = c*U[..., 0] - s*U[..., 1]
    out[..., 1] = s*U[..., 0] + c*U[..., 1]
    return out


def t1_path_identity():
    """Affine code path vs curvilinear code path on the same, unmoved mesh."""
    ma, sa, fu, fv = affine_state()
    La = apply_L(sa, field((ma.nelem, N+1, N+1, 4)), fu, fv).copy()
    mc, sc, fu2, fv2 = make()
    Lc = apply_L(sc, field((mc.nelem, N+1, N+1, 4)), fu2, fv2)
    return float(np.abs(La - Lc).max()/max(np.abs(La).max(), 1e-300))


def t2_covariance(theta):
    """Rows on the rotated mesh vs the rotated rows on the original mesh."""
    m0, s0, fu, fv = make(theta=0.0)
    mr, sr, fu2, fv2 = make(theta=theta)
    U = field((m0.nelem, N+1, N+1, 4))
    L0 = apply_L(s0, U, fu, fv).copy()
    Lr = apply_L(sr, rot_field(U, theta), fu2, fv2)
    return float(np.abs(Lr - rot_field(L0, theta)).max()
                 / max(np.abs(L0).max(), 1e-300))


def t3_adjoint(theta=None, deform=None):
    m, st, fu, fv = make(theta=theta, deform=deform)
    U = field((m.nelem, N+1, N+1, 4), 0)
    S = field((m.nelem, N+1, N+1, 4), 1)
    lhs = float((apply_L(st, U, fu, fv)*S).sum())
    rhs = float((U*apply_LT(st, S*m.wq[..., None], fu, fv)).sum())
    return abs(lhs - rhs)/max(abs(lhs), 1e-300)


def main():
    lssem2d.set_backend('numpy')
    eps = np.finfo(float).eps

    t1 = t1_path_identity()
    print(f'T1  affine path vs curvilinear path, same mesh : {t1:.3e}'
          f'   ({t1/eps:.1f} eps)')

    angles = np.array([0, 5, 15, 30, 45, 60, 75, 90], float)
    cov = np.array([t2_covariance(np.deg2rad(a)) for a in angles])
    print(f'\nT2  rotation covariance of the operator')
    for a, c in zip(angles, cov):
        print(f'      {a:5.0f}°  {c:.3e}   ({c/eps:6.1f} eps)')

    adj = {'affine': t3_adjoint(theta=0.0),
           'rotated 30°': t3_adjoint(theta=np.pi/6),
           'rotated 45°': t3_adjoint(theta=np.pi/4),
           'deformed 10%': t3_adjoint(deform=0.10)}
    print(f'\nT3  adjoint pair <LU,S> = <U,Lt(wq S)>')
    for k, v in adj.items():
        print(f'      {k:14s} {v:.3e}   ({v/eps:.1f} eps)')

    worst = max(t1, cov.max(), max(adj.values()))
    gate = 1e-12
    print(f'\nworst over all three tests: {worst:.3e}   gate {gate:g} -> '
          f'{"PASS" if worst < gate else "FAIL"}')

    # ---- figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))

    m = make(theta=np.pi/6)[0]
    n = m.nterm
    for e in range(m.nelem):
        for idx in (0, n-1):
            ax[0].plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=0.8)
            ax[0].plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=0.8)
    m0 = make(theta=0.0)[0]
    for e in range(m0.nelem):
        for idx in (0, n-1):
            ax[0].plot(m0.X[e, idx, :], m0.Y[e, idx, :], color='C7', ls='--', lw=0.8)
            ax[0].plot(m0.X[e, :, idx], m0.Y[e, :, idx], color='C7', ls='--', lw=0.8)
    ax[0].set_aspect('equal'); ax[0].set_title('T2 setup: the same mesh, rotated 30°', fontsize=10)
    ax[0].tick_params(labelsize=8)

    ax[1].semilogy(angles, np.maximum(cov, 1e-18), 'o-', color='C0')
    ax[1].axhline(gate, color='C3', ls='--', lw=1.2, label=f'gate {gate:g}')
    ax[1].axhline(eps, color='C7', ls=':', lw=1, label=r'$\epsilon$')
    ax[1].set_xlabel('rotation angle [deg]'); ax[1].set_ylabel('relative error')
    ax[1].set_title('T2 rotation covariance of $L$', fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(1e-18, 1e-10)

    labs = ['T1 path\nidentity'] + [f'T3 {k}' for k in adj]
    vals = [max(t1, 1e-18)] + [max(v, 1e-18) for v in adj.values()]
    ax[2].bar(range(len(vals)), vals, color=['C2'] + ['C0']*len(adj))
    ax[2].set_yscale('log'); ax[2].set_xticks(range(len(labs)))
    ax[2].set_xticklabels(labs, rotation=25, ha='right', fontsize=8)
    ax[2].axhline(gate, color='C3', ls='--', lw=1.2, label=f'gate {gate:g}')
    ax[2].axhline(eps, color='C7', ls=':', lw=1)
    ax[2].set_ylabel('relative error'); ax[2].legend(fontsize=8)
    ax[2].set_title('T1 regression and T3 adjoint pair', fontsize=10)
    ax[2].set_ylim(1e-18, 1e-10); ax[2].grid(alpha=0.3, axis='y')

    fig.suptitle('Gate G1 — a rotated mesh is affine but not axis-aligned: '
                 'every cross term live, the answer known', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'wrote {OUT}')
    return 0 if worst < gate else 1


if __name__ == '__main__':
    sys.exit(main())
