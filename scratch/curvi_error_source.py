"""What actually generates the error on a deformed mesh?

    uv run python scratch/curvi_error_source.py

G2 measured the penalty (935,000x at N = 14, rate 1.43 against 2.72) but not its
cause.  Four mechanisms are available, and they call for different remedies:

  1. REPRESENTATION.  The solver approximates u∘Phi on the reference element, not
     u on the physical one.  If Phi is affine that composition is as analytic as
     u; if Phi is a sine map it is not, and the Legendre coefficients of u∘Phi
     decay more slowly.  This alone would change the exponential RATE.
  2. METRICS.  rx = ys/J and friends are RATIONAL in the nodal coordinates, so
     even a polynomial map has metrics the space cannot represent exactly.
  3. QUADRATURE.  wq = J*w*w, and the integrand u*v*J is not a polynomial, so
     GLL quadrature (exact to degree 2N-1) is inexact -- a variational crime.
  4. RESOLUTION REDISTRIBUTION.  The map compresses in one place and stretches
     in another, and the error follows the worst-resolved region.

THE DISCRIMINATING TEST NEEDS NO SOLVER.  Mechanism 1 is a property of the
discrete SPACE alone: take the exact solution, interpolate it at the nodes, and
measure how far the interpolant sits from it.  No operator, no residual
quadrature, no metrics, no boundary conditions.  If that interpolation error has
the same exponential rate as the solved error, mechanism 1 is the whole story
and 2-4 ride along; if the solved error decays more slowly, the operator adds
something of its own.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np

import lssem2d
from lssem2d import curvi
from lssem2d.lgl import lgl_nodes, diff_matrix
import curvi_g2 as g2

OUT = os.path.join(_R, 'figs', 'curvi_error_source.png')


def _lag(nodes, r):
    n = len(nodes)
    L = np.ones((len(r), n))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[:, i] *= (r - nodes[j])/(nodes[i] - nodes[j])
    return L


def _jac(Xe, Ye, D, Lq):
    xr = Lq @ (D @ Xe) @ Lq.T
    xs = Lq @ (Xe @ D.T) @ Lq.T
    yr = Lq @ (D @ Ye) @ Lq.T
    ys = Lq @ (Ye @ D.T) @ Lq.T
    return xr*ys - xs*yr


def interp_error(N, kind, nq=48):
    """||u - I_N u|| over the mesh, on a fine Gauss sample of each element."""
    m = g2.make_mesh(N, 2, kind)
    xi = lgl_nodes(N)
    D = diff_matrix(N)
    q, wq = np.polynomial.legendre.leggauss(nq)
    Lq = _lag(xi, q)
    X, Y = (m.X, m.Y) if m.curvilinear else g2.coords(m)
    num = den = 0.0
    for e in range(m.nelem):
        ue = g2._F['u'](X[e], Y[e], g2.NU)      # exact solution AT THE NODES
        ui = Lq @ ue @ Lq.T                     # its interpolant, sampled
        xs = Lq @ X[e] @ Lq.T
        ys = Lq @ Y[e] @ Lq.T
        ut = g2._F['u'](xs, ys, g2.NU)          # the true solution there
        w2 = np.outer(wq, wq)*np.abs(_jac(X[e], Y[e], D, Lq))
        num += float((w2*(ui - ut)**2).sum())
        den += float((w2*ut**2).sum())
    return np.sqrt(num/den), m


def main():
    lssem2d.set_backend('numpy')
    g2._F = g2._fields()
    Ns = [4, 6, 8, 10, 12, 14]
    solved = {'affine': [1.544e-2, 1.186e-4, 7.520e-7, 3.850e-9, 1.478e-11, 4.319e-14],
              'deformed': [2.909e-2, 2.801e-3, 2.358e-4, 6.998e-6, 4.028e-7, 4.038e-8]}
    res = {}
    print('INTERPOLATION error of the exact solution in the discrete space')
    print('(no solver, no operator, no residual quadrature)\n')
    print(f'{"N":>3s} {"affine interp":>15s} {"deformed interp":>16s} {"ratio":>8s}'
          f' | {"affine solved":>14s} {"deformed solved":>16s} {"ratio":>10s}')
    for k, N in enumerate(Ns):
        ea, _ = interp_error(N, 'affine')
        ed, _ = interp_error(N, 'deformed')
        res.setdefault('affine', []).append(ea)
        res.setdefault('deformed', []).append(ed)
        print(f'{N:3d} {ea:15.3e} {ed:16.3e} {ed/ea:8.1f} | '
              f'{solved["affine"][k]:14.3e} {solved["deformed"][k]:16.3e} '
              f'{solved["deformed"][k]/solved["affine"][k]:10.1f}')
    x = np.asarray(Ns, float)
    print('\nexponential rate b (err ~ exp(-bN)) fitted over N = 6..14:')
    for lab, d in (('interpolation', res), ('solved', solved)):
        for k in ('affine', 'deformed'):
            b = -np.polyfit(x[1:], np.log(d[k][1:]), 1)[0]
            print(f'   {lab:14s} {k:9s} b = {b:5.2f}')
    md = g2.make_mesh(10, 2, 'deformed')
    print(f'\nthe deformed mesh: J {md.jacq.min():.5f} to {md.jacq.max():.5f} '
          f'(ratio {md.jacq.max()/md.jacq.min():.3f})')
    figure(Ns, res, solved)


def figure(Ns, res, solved):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for k, c in (('affine', 'C7'), ('deformed', 'C3')):
        ax[0].semilogy(Ns, res[k], 'o-', color=c, label=k)
        ax[1].semilogy(Ns, solved[k], 's--', color=c, label=f'{k} solved')
        ax[1].semilogy(Ns, res[k], 'o-', color=c, alpha=0.45,
                       label=f'{k} interpolation')
    ax[0].set_xlabel('$N$'); ax[0].set_ylabel(r'relative $L^2$ error')
    ax[0].set_title('interpolation error alone\nwhat the discrete SPACE can do',
                    fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')
    ax[1].set_xlabel('$N$'); ax[1].set_ylabel(r'relative $L^2$ error')
    ax[1].set_title('interpolation vs solved\nthe gap is what the OPERATOR adds',
                    fontsize=10)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, which='both')
    fig.suptitle('Where the deformed-mesh error comes from', fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
