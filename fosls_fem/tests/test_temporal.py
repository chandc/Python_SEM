"""G6 -- temporal order of BDF2, isolated from spatial error.

THE TRICK is to choose a solution that is EXACTLY representable in Q1, so the
spatial error is identically zero and whatever remains is purely temporal.
Linear-in-space fields do that:

    u = (1 + 2x - 3y) g(t)      v = (4 - 5x - 2y) g(t)     div u = 0
    omega = (v_x - u_y) g = -2 g(t)                        omega + u_y - v_x = 0
    p = (0.7 + 1.3x - 0.4y) h(t)

Every field, and every term of the forcing, is linear in space, so the Q1
interpolant is exact and a mesh of ANY size gives the same answer.  A
convergence study that did not do this would measure the sum of two errors and
report whichever dominated -- which at coarse dt looks like second order and at
fine dt stalls at the spatial floor.

BDF2 IS STARTED EXACTLY, from the analytic solution at t0 - dt, so the observed
order is BDF2's own and not a first-order startup contaminating it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
from fosls_fem.mesh import build_rect, NF
from fosls_fem.assemble import dof_index
from fosls_fem.timestep import coeffs, step, BDF2_FAC1

NU, T0, T1 = 0.05, 0.0, 0.4
A = lambda x, y: 1 + 2*x - 3*y
B = lambda x, y: 4 - 5*x - 2*y
C = lambda x, y: 0.7 + 1.3*x - 0.4*y
g  = lambda t: np.sin(t) + 0.5
gp = lambda t: np.cos(t)
h  = lambda t: np.cos(t)

def exact(xy, t):
    x, y = xy[:, 0], xy[:, 1]
    U = np.zeros(len(x)*NF)
    U[0::NF] = A(x, y)*g(t)
    U[1::NF] = B(x, y)*g(t)
    U[2::NF] = C(x, y)*h(t)
    U[3::NF] = -2.0*g(t)
    return U

def forcing(xy, t, a_flux):
    """a_flux * f, with f the momentum forcing that makes the exact fields solve
    the unsteady equations.  omega is constant in space, so its curl vanishes."""
    x, y = xy[:, 0], xy[:, 1]
    a, b = A(x, y), B(x, y)
    fx = a*gp(t) + (2*a - 3*b)*g(t)**2 + 1.3*h(t)
    fy = b*gp(t) + (-5*a - 2*b)*g(t)**2 - 0.4*h(t)
    r = np.zeros((len(x), NF))
    r[:, 2] = a_flux*fx
    r[:, 3] = a_flux*fy
    return r

print('G6 -- BDF2 temporal order, spatial error identically zero')
print(f'{"weighting":>10s} {"dt":>8s} {"steps":>6s} {"L2 err":>11s} {"order":>7s}')
ok = True
for wt in ('balanced', 'legacy'):
    errs, dts = [], []
    for nst in (5, 10, 20, 40):
        dt = (T1 - T0)/nst
        coef, hist = coeffs(dt, NU, wt)      # hist scale comes from ls_coeffs' table
        m = build_rect(0, 1, 0, 1, 2, 2)          # any mesh: spatial error is 0
        bnd = np.unique(np.concatenate([e.ravel() for e in m.edge_tags.values()]))
        fixed = np.concatenate([dof_index(bnd, 0), dof_index(bnd, 1),
                                dof_index([0], 2)])
        Unm1, Un = exact(m.xy, T0 - dt), exact(m.xy, T0)
        t = T0
        for k in range(nst):
            t += dt
            Ue = exact(m.xy, t)
            vals = np.concatenate([Ue[dof_index(bnd, 0)], Ue[dof_index(bnd, 1)],
                                   [Ue[dof_index([0], 2)[0]]]])
            f = forcing(m.xy, t, coef[1])
            Unew, its, J = step(m, Un, Unm1, coef, hist, fixed, vals, f)
            Unm1, Un = Un, Unew
        e = np.sqrt(np.mean((Un - exact(m.xy, T1))**2))
        o = np.log2(errs[-1]/e) if errs else float('nan')
        errs.append(e); dts.append(dt)
        print(f'{wt:>10s} {dt:8.4f} {nst:6d} {e:11.4e} {o:7.2f}')
    r = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    good = abs(r - 2.0) < 0.15
    ok &= good
    print(f'{"":>10s} fitted order {r:.3f}   {"PASS" if good else "FAIL"} '
          f'(BDF2 target 2.00)\n')
print(f'G6 temporal: {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
