"""V4 -- lid-driven cavity at Re = 1000 against Ghia, Ghia & Shin (1982).

NEWTON, with continuation in Reynolds number.  A cold Newton start at
Re = 1000 has no reason to converge; each Re is solved from the previous one,
which is standard and also cheap because the Jacobian is reassembled anyway.

THE FUNCTIONAL IS REPORTED at every step.  Under the Cai-Manteuffel-McCormick
equivalence J(U_h) is norm-equivalent to the squared H^1 error, so it is a
sharp a posteriori estimator available for free -- no exact solution needed.
Watching it fall alongside the Newton update is a check that the iteration is
converging to the right thing and not merely stalling.

THE CORNERS ARE SINGULAR.  The lid velocity is discontinuous at the two top
corners, so the exact solution is not in H^1 there and no method converges at
its nominal rate.  That is a property of the problem (paper sec 6.3), and the
comparison with Ghia is made on the centreline profiles as everyone else's is.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
from fosls_fem.mesh import build_rect, NF
from fosls_fem.assemble import assemble_newton, apply_dirichlet, dof_index, functional
from fosls_fem.solve import solve_direct

# Ghia, Ghia & Shin (1982) Table I and II
GH_Y = np.array([0.0000,0.0547,0.0625,0.0703,0.1016,0.1719,0.2813,0.4531,
                 0.5000,0.6172,0.7344,0.8516,0.9531,0.9609,0.9688,0.9766,1.0000])
GH_U = np.array([0.00000,-0.18109,-0.20196,-0.22220,-0.29730,-0.38289,-0.27805,
                 -0.10648,-0.06080,0.05702,0.18719,0.33304,0.46604,0.51117,
                 0.57492,0.65928,1.00000])
GH_X = np.array([1.0000,0.9688,0.9609,0.9531,0.9453,0.9063,0.8594,0.8047,
                 0.5000,0.2344,0.2266,0.1563,0.0938,0.0781,0.0703,0.0625,0.0000])
GH_V = np.array([0.00000,-0.21388,-0.27669,-0.33714,-0.39188,-0.51550,-0.42665,
                 -0.31966,0.02526,0.32235,0.33075,0.37095,0.32627,0.30353,
                 0.29012,0.27485,0.00000])


def cavity(n, re, U0=None, tol=1e-10, maxit=25, verbose=True):
    m = build_rect(0, 1, 0, 1, n, n)
    coef = (0.0, 1.0, 1.0, 1.0/re)                 # steady: a_mass = 0
    lid = np.isclose(m.xy[:, 1], 1.0)
    wall = (np.isclose(m.xy[:, 0], 0) | np.isclose(m.xy[:, 0], 1) |
            np.isclose(m.xy[:, 1], 0)) & ~lid
    bnd = lid | wall
    ub = np.where(lid, 1.0, 0.0)[bnd]
    fixed = np.concatenate([dof_index(np.flatnonzero(bnd), 0),
                            dof_index(np.flatnonzero(bnd), 1),
                            dof_index([0], 2)])
    U = np.zeros(m.ndof) if U0 is None else U0.copy()
    U[dof_index(np.flatnonzero(bnd), 0)] = ub          # satisfy BCs exactly
    U[dof_index(np.flatnonzero(bnd), 1)] = 0.0
    for k in range(maxit):
        A, b = assemble_newton(m, U, coef)
        Ac, bc = apply_dirichlet(A, b, fixed, np.zeros(len(fixed)))  # dU = 0 on bnd
        dU = solve_direct(Ac, bc)
        U += dU
        nrm = np.abs(dU).max()
        J = functional(m, U, coef)
        if verbose:
            print(f'      newton {k+1:2d}: |dU|inf {nrm:.3e}   J {J:.4e}')
        if nrm < tol:
            break
    return m, U, J, k+1


def centreline(m, U, n, field, along):
    """Nodal values along a centreline of the structured mesh."""
    nid = lambda i, j: i*(n+1) + j
    if along == 'y':
        idx = [nid(n//2, j) for j in range(n+1)]
        return m.xy[idx, 1], U[NF*np.array(idx) + field]
    idx = [nid(i, n//2) for i in range(n+1)]
    return m.xy[idx, 0], U[NF*np.array(idx) + field]


print('V4 -- lid-driven cavity, Q1 FOSLS, Newton with Re continuation')
ok = True
U0 = None
for n in (32, 64):
    print(f'  mesh {n}x{n}  ({(n+1)**2*NF} dof)')
    U0 = None
    for re in (100.0, 400.0, 1000.0):
        print(f'    Re = {re:.0f}')
        m, U0, J, its = cavity(n, re, U0)
    y, u = centreline(m, U0, n, 0, 'y')
    x, v = centreline(m, U0, n, 1, 'x')
    eu = np.sqrt(np.mean((np.interp(GH_Y, y, u) - GH_U)**2))
    ev = np.sqrt(np.mean((np.interp(GH_X, x, v) - GH_V)**2))
    print(f'    RMS vs Ghia:  u(y) {eu:.4e}   v(x) {ev:.4e}   '
          f'umin {u.min():.4f} (Ghia -0.3829)   vmin {v.min():.4f} (Ghia -0.5155)')
    last = (eu, ev)
good = last[0] < 5e-2 and last[1] < 5e-2
print(f'\n  {"PASS" if good else "FAIL"}  finest mesh RMS vs Ghia < 5e-2  '
      f'[u {last[0]:.3e}, v {last[1]:.3e}]  (spectral N=10 reaches 1.47e-2)')
sys.exit(0 if good else 1)
