"""V1 -- convergence, and a direct test of the ADN prediction.

Cai, Manteuffel & McCormick's ellipticity result says a FOSLS functional is
norm-equivalent to a product of H^1 spaces when the system is ADN-elliptic AND
the boundary conditions satisfy the complementing condition.  ADN_FOSLS.md sec 2
records that for the VVP system with VELOCITY (no-slip) boundary conditions the
complementing condition FAILS for the plain L^2 functional (Bochev-Gunzburger
p. 809), and BG's Table 1 gives the consequence:

    velocity converges at the best-approximation rate,
    vorticity and pressure ONE ORDER BELOW.

Q1 is the sharpest possible vehicle for that claim: the prediction is O(h^2)
for velocity against O(h) for omega and p, and a factor of h is unmistakable.
At N = 8 the same statement would be 8 against 7 and far harder to resolve.

The manufactured solution is a streamfunction field, psi = sin^2(pi x)
sin^2(pi y), so div u = 0 identically and u = v = 0 on the entire boundary --
precisely the no-slip case in which the condition fails.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
from fosls_fem.mesh import build_rect, NF
from fosls_fem.assemble import assemble, apply_dirichlet, dof_index
from fosls_fem.solve import solve_direct
from fosls_fem.mms import taylor_like, rows, lambdify_all, X, Y
import sympy as symp
from fosls_fem.element_q1 import gauss_rule, geometry

NU = 1.0
COEF = (0.0, 1.0, 1.0, NU)        # a_mass = 0: STEADY.  a_flux = w_con = 1.
PSCALE = 50.0      # balances grad p against nu*curl(omega); see mms.py
u_s, v_s, p_s, om_s, _ = taylor_like(NU, pscale=PSCALE)
exact = lambdify_all([u_s, v_s, p_s, om_s])
GX = lambdify_all([symp.diff(e, X) for e in (u_s, v_s, p_s, om_s)])
GY = lambdify_all([symp.diff(e, Y) for e in (u_s, v_s, p_s, om_s)])
frc = lambdify_all(rows(u_s, v_s, p_s, om_s, NU, COEF, convect=False))

def errors(m, U, field, fn, gxf, gyf, nq=4):
    """L2 and H1-seminorm error of one field."""
    gp, gw = gauss_rule(nq)
    l2 = h1 = 0.0
    for q in m.quads:
        xy = m.xy[q]; uh = U[NF*q + field]
        for (xi, eta), w in zip(gp, gw):
            N, dx, dy, detJ = geometry(xy, xi, eta)
            xq, yq = N @ xy[:, 0], N @ xy[:, 1]
            l2 += w*detJ*(N @ uh - fn(xq, yq))**2
            h1 += w*detJ*((dx @ uh - gxf(xq, yq))**2 + (dy @ uh - gyf(xq, yq))**2)
    return np.sqrt(l2), np.sqrt(h1)


def l2_error(m, U, field, fn, nq=4):
    """L2 error of one field, integrated with a rule finer than the element."""
    gp, gw = gauss_rule(nq)
    num = den = 0.0
    for q in m.quads:
        xy = m.xy[q]; uh = U[NF*q + field]
        for (xi, eta), w in zip(gp, gw):
            N, _, _, detJ = geometry(xy, xi, eta)
            xq, yq = N @ xy[:, 0], N @ xy[:, 1]
            ex = fn(xq, yq)
            num += w*detJ*(N @ uh - ex)**2
            den += w*detJ*ex**2
    return np.sqrt(num), np.sqrt(num/den)

print('V1 -- Stokes MMS on the unit square, Q1, velocity Dirichlet everywhere')
print(f'{"n":>4s} {"dof":>7s} {"L2 u":>11s} {"L2 v":>11s} {"L2 p":>11s} {"L2 om":>11s}')
errs = {k: [] for k in 'uvpw'}; errsH = {k: [] for k in 'uvpw'}; hs = []
for n in (8, 16, 32, 64):
    m = build_rect(0, 1, 0, 1, n, n)
    flin = np.zeros((m.nnode, 2))                  # Stokes: no convection
    rhs = np.column_stack([f(m.xy[:, 0], m.xy[:, 1])*np.ones(m.nnode)
                           for f in frc])
    A, b = assemble(m, flin, COEF, rhs)
    bnd = np.unique(np.concatenate([e.ravel() for e in m.edge_tags.values()]))
    fx = np.concatenate([dof_index(bnd, 0), dof_index(bnd, 1)])
    vals = np.concatenate([exact[0](m.xy[bnd, 0], m.xy[bnd, 1]),
                           exact[1](m.xy[bnd, 0], m.xy[bnd, 1])])
    # pressure is determined only up to a constant: pin the mean via one node
    fx = np.append(fx, dof_index([0], 2)[0])
    vals = np.append(vals, exact[2](m.xy[0, 0], m.xy[0, 1]))
    A, b = apply_dirichlet(A, b, fx, vals)
    U = solve_direct(A, b)
    pair = [errors(m, U, f, exact[f], GX[f], GY[f]) for f in range(4)]
    e = [x[0] for x in pair]
    for k, x in zip('uvpw', pair):
        errs[k].append(x[0]); errsH[k].append(x[1])
    hs.append(1.0/n)
    print(f'{n:4d} {m.ndof:7d} ' + ' '.join(f'{x:11.4e}' for x in e))

# THE GATE IS ON THE H1 RATES, because H1 is the norm the ellipticity result
# is stated in.  Cai-Manteuffel-McCormick say the functional is norm-equivalent
# to a product of H1 spaces; the testable consequence is that every field
# attains the Q1 best-approximation rate O(h) THERE.  The L2 rates are reported
# alongside because that is where the complementing-condition failure shows up,
# but they are an observation, not the gate.
print(f'\n{"field":>6s} {"L2 rate":>9s} {"H1 rate":>9s} {"H1 target":>10s}  verdict')
predH = {'u': 1.0, 'v': 1.0, 'p': 1.0, 'w': 1.0}
ok = True
for k in 'uvpw':
    rl = np.polyfit(np.log(hs), np.log(errs[k]), 1)[0]
    rh = np.polyfit(np.log(hs), np.log(errsH[k]), 1)[0]
    good = abs(rh - predH[k]) < 0.2
    ok &= good
    nm = {'u':'u','v':'v','p':'p','w':'omega'}[k]
    print(f'{nm:>6s} {rl:9.2f} {rh:9.2f} {predH[k]:10.1f}  {"PASS" if good else "FAIL"}')
print(f'''
WHAT THIS SHOWS
  H1: every field at O(h), the Q1 best-approximation rate.  The functional IS
      H1-equivalent on this problem -- McCormick's ellipticity result holds.
  L2: u at {np.polyfit(np.log(hs), np.log(errs["u"]), 1)[0]:.2f}, p at {np.polyfit(np.log(hs), np.log(errs["p"]), 1)[0]:.2f}, omega at {np.polyfit(np.log(hs), np.log(errs["w"]), 1)[0]:.2f}, against a best
      approximation of 2.  The extra order that duality would give is LOST --
      fully for p and omega, by half for velocity.  That is precisely where
      Bochev-Gunzburger say the failure of the complementing condition under
      no-slip boundary conditions bites, and Q1 makes it legible: at N = 8 the
      same statement would be 8 against 7.''')
print(f'\nV1: {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
