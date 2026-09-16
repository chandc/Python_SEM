"""Gate G0 — the geometry is right before anything is solved on it.

    uv run python scratch/curvi_g0.py

CURVILINEAR_2D_PLAN.md G0.  Every later error is uninterpretable if the metrics
are wrong, and four cheap identities catch nearly every way of getting them
wrong.  All four have answers known exactly in advance -- zero, one, or the
analytic area -- so there is nothing to calibrate and nothing to argue about.

  1. CONSTANT PRESERVED.  grad(1) = 0.  Fails if the two contractions are
     combined with the wrong sign or the metric is stale.
  2. LINEAR EXACT.  grad(x) = (1,0) and grad(y) = (0,1).  This is the discrete
     metric identity in usable form: it holds for COLLOCATION metrics and fails
     for analytically differentiated ones, which is why the plan fixes D2.
  3. DIVERGENCE OF A CONSTANT VECTOR is zero -- the same identity read through
     the operator the solver actually assembles.
  4. AREA.  sum_e sum_ij wq_ij equals the analytic area, which checks the
     Jacobian's magnitude rather than only its consistency.

THE GATE IS SCALE-AWARE, AND THE FIRST VERSION OF IT WAS NOT.  Differentiating a
constant cannot give better than round-off AMPLIFIED BY THE OPERATOR: D has
entries of order N^2 and the metric of order 1/h, so the floor is
eps*||D||*||metric||, not eps.  An absolute threshold therefore fails on a fine
mesh or at high order for no reason -- measured here at 1.4e-13 on a coarse
affine mesh and 1.1e-12 on a twice-refined annulus, with the ratio to that floor
sitting at 4-7 in every case, unchanged by refinement.  So the identities are
reported in ROUND-OFF UNITS, error/(eps*||D||*||metric||), and the gate is that
the ratio is O(10).  A genuine metric bug lands at 1e6 or more and is in no
danger of passing.

And one more that the plan lists under step 2 but belongs here, because it
catches what 1-4 cannot:

  5. ADJOINT.  <ddx(u), v>_wq = <u, ddxT(v)> for random u, v.  A transposed
     index or a dropped weight passes every forward test above and destroys the
     least-squares operator, which is built from adjoint pairs.

Four meshes: affine (the control -- these identities must not degrade), rigidly
rotated (affine but not axis-aligned, so every cross term is live), sinusoidally
deformed, and an annulus with no straight edges anywhere.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np

from lssem2d import curvi
from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel

OUT = os.path.join(_R, 'figs', 'curvi_g0.png')


def meshes(N=8):
    out = []

    m = build_channel(1.0, 1.0, 3, 3, N); m.compute_global_indices()
    X, Y = curvi.affine_coords(m)
    out.append(('affine (control)', curvi.attach(m, X, Y), 1.0))

    m = build_channel(1.0, 1.0, 3, 3, N); m.compute_global_indices()
    out.append(('rotated 30°', curvi.rotate(m, np.pi/6), 1.0))

    m = build_channel(1.0, 1.0, 3, 3, N); m.compute_global_indices()
    out.append(('deformed 10%', curvi.deform(m, amp=0.10, kx=2, ky=1), None))

    m = curvi.build_annulus(0.5, 1.0, 3, 8, N, theta0=0.0, theta1=np.pi/2)
    out.append(('annulus (quarter)', m, np.pi*(1.0**2 - 0.5**2)/4))
    return out


EPS = np.finfo(float).eps


def roundoff_floor(mesh, D):
    """eps * ||D||_inf * ||metric||_inf -- the best a derivative identity can do."""
    g = max(np.abs(mesh.rx).max(), np.abs(mesh.sx).max(),
            np.abs(mesh.ry).max(), np.abs(mesh.sy).max())
    return EPS*np.abs(D).max()*g


def gate(mesh, area_exact):
    D = diff_matrix(mesh.N)
    X, Y = mesh.X, mesh.Y
    one = np.ones_like(X)
    floor = roundoff_floor(mesh, D)
    r = {}
    r['grad(1) = 0'] = max(np.abs(curvi.ddx(one, D, mesh)).max(),
                           np.abs(curvi.ddy(one, D, mesh)).max())/floor
    r['grad(x) = (1,0)'] = max(np.abs(curvi.ddx(X, D, mesh) - 1).max(),
                               np.abs(curvi.ddy(X, D, mesh)).max())/EPS
    r['grad(y) = (0,1)'] = max(np.abs(curvi.ddx(Y, D, mesh)).max(),
                               np.abs(curvi.ddy(Y, D, mesh) - 1).max())/EPS
    # divergence of the constant field (a, b), a and b arbitrary
    a, b = 0.37, -1.21
    r['div(const) = 0'] = np.abs(curvi.ddx(a*one, D, mesh)
                                 + curvi.ddy(b*one, D, mesh)).max()/(floor*max(abs(a), abs(b)))
    area = float(mesh.wq.sum())
    r['area'] = abs(area - area_exact)/area_exact/EPS if area_exact else np.nan
    # adjoint, in the quadrature-weighted inner product
    rng = np.random.default_rng(0)
    u = rng.standard_normal(X.shape)
    v = rng.standard_normal(X.shape)
    for lab, f, fT in (('adjoint x', curvi.ddx, curvi.ddxT),
                       ('adjoint y', curvi.ddy, curvi.ddyT)):
        terms = mesh.wq*f(u, D, mesh)*v
        lhs = float(terms.sum())
        rhs = float((u*fT(mesh.wq*v, D, mesh)).sum())   # wq explicit now
        # Normalise by the CONDITION of the summation, sum|t| / |sum t|.  These
        # are dot products of random fields, so the terms cancel heavily and the
        # achievable relative accuracy is eps times that ratio, not eps.  On the
        # affine mesh the ratio is ~60, which is exactly where an unnormalised
        # test put this line -- a cancellation artefact, not an adjoint error.
        cond = float(np.abs(terms).sum())/max(abs(lhs), 1e-300)
        r[lab] = abs(lhs - rhs)/max(abs(lhs), 1e-300)/(EPS*cond)
    return r, area


def figure(cases, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])

    for k, ((name, m, _), _) in enumerate(zip(cases, results)):
        ax = fig.add_subplot(gs[0, k])
        n = m.nterm
        for e in range(m.nelem):                       # element outlines
            for idx in (0, n-1):
                ax.plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=0.7)
                ax.plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=0.7)
        ax.plot(m.X.ravel(), m.Y.ravel(), '.', ms=0.8, color='C0', alpha=0.5)
        ax.set_aspect('equal'); ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[1, :2])
    tests = [t for t in results[0][0] if t != 'area']
    w = 0.2
    xs = np.arange(len(tests))
    for k, (name, _, _) in enumerate(cases):
        vals = [max(results[k][0][t], 1e-18) for t in tests]
        ax.bar(xs + (k - 1.5)*w, vals, w, label=name)
    ax.set_yscale('log'); ax.set_xticks(xs)
    ax.set_xticklabels(tests, rotation=20, ha='right', fontsize=8)
    ax.axhline(50, color='C3', ls='--', lw=1.2, label='gate: 50 round-off units')
    ax.set_ylabel('error / round-off floor'); ax.legend(fontsize=7, ncol=2)
    ax.set_title('G0 identities, in units of the round-off floor '
                 r'$\epsilon\,\|D\|\,\|metric\|$', fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, 2:])
    m = cases[-1][1]
    sc = ax.tripcolor(m.X.ravel(), m.Y.ravel(), m.jacq.ravel(), shading='gouraud')
    ax.set_aspect('equal'); fig.colorbar(sc, ax=ax, shrink=0.85)
    ax.set_title(r'annulus: Jacobian $J$, exact $= r\,\Delta r\,\Delta\theta/4$',
                 fontsize=10)
    ax.tick_params(labelsize=7)

    fig.suptitle('Gate G0 — curvilinear metrics, before anything is solved', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'\nwrote {OUT}')


def main():
    cases = meshes()
    results = [gate(m, a) for _, m, a in cases]
    tests = list(results[0][0])
    print(f'{"test":20s} ' + ' '.join(f'{n:>18s}' for n, _, _ in cases))
    worst = 0.0
    for t in tests:
        row = []
        for (name, _, area_exact), (r, _) in zip(cases, results):
            v = r[t]
            row.append('        —         ' if np.isnan(v) else f'{v:18.1f}')
            if not np.isnan(v):
                worst = max(worst, v)
        print(f'{t:20s} ' + ' '.join(row))
    print(f'\nall values are MULTIPLES OF THE ROUND-OFF FLOOR, not absolute errors')
    print(f'worst over all tests and meshes: {worst:.1f} round-off units   '
          f'gate is 50 -> {"PASS" if worst < 50 else "FAIL"}')
    for (name, m, ae), (_, area) in zip(cases, results):
        if ae:
            print(f'  area {name:20s} {area:.12f}  exact {ae:.12f}')
    figure(cases, results)
    return 0 if worst < 50 else 1


if __name__ == '__main__':
    sys.exit(main())
