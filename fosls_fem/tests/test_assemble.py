"""Gates G1 (mesh) and G4 (assembly).

G4'S CENTREPIECE is the exactly-representable solution, and it is worth more
than the rest of the gate combined.  Choose

    u = 1 + 2x - 3y      v = 4 - 5x - 2y      (so u_x + v_y = 0 exactly)
    omega = v_x - u_y = -2                    (so omega + u_y - v_x = 0)
    p = 0.7 + 1.3x - 0.4y

Every field is LINEAR, so it lies in the Q1 space and its interpolant is
itself.  With a linear linearisation (fu, fv) the momentum row
`a_mass u + a_flux (conv + p_x + nu omega_y)` is also linear, hence exactly
representable as nodal right-hand-side data.  Therefore `L U - f = 0`
POINTWISE, and so `L^T L U = L^T f` to round-off -- on ANY mesh, affine or
distorted, with ANY quadrature rule.

That independence is what makes it a good test: it isolates the element
matrices, the scatter, the dof numbering and the right-hand-side path from
every approximation in the method.  An index error, a sign error, a
transposed scatter or a missing term all fail it, and nothing else in the gate
can distinguish those.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
import scipy.sparse.linalg as spla
from fosls_fem.mesh import build_rect, NF
from fosls_fem.assemble import assemble, apply_dirichlet, dof_index
from fosls_fem.element_q1 import element_matrix

A_MASS, A_FLUX, W_CON, NU = 4.7434, 0.31623, 1.0, 0.01
COEF = (A_MASS, A_FLUX, W_CON, NU)
ok = True

def chk(name, val, tol):
    global ok
    p = bool(val < tol); ok &= p
    print(f'  {"PASS" if p else "FAIL"}  {name:56s} {val:9.2e} (tol {tol:.0e})')

def chkb(name, p, detail=''):
    global ok
    ok &= bool(p)
    print(f'  {"PASS" if p else "FAIL"}  {name:56s} {detail}')

print('G1 -- mesh')
for nm, m, area in (('4x3 rectangle', build_rect(0,2,0,1.5,4,3), 3.0),
                    ('6x6 distorted', build_rect(0,1,0,1,6,6,distort=0.25), 1.0)):
    good, items = m.check(exact_area=area)
    print(f'  {nm}: {m.nnode} nodes, {m.nelem} elements')
    for n, d, p in items:
        chkb(f'  {n}', p, d)

print('\nG4 -- assembly')

def exact_fields(xy):
    x, y = xy[:, 0], xy[:, 1]
    u = 1 + 2*x - 3*y
    v = 4 - 5*x - 2*y                      # u_x + v_y = 2 - 2 = 0
    om = np.full_like(x, -2.0)             # v_x - u_y = -5 + 3 = -2
    p = 0.7 + 1.3*x - 0.4*y
    return u, v, p, om

def exact_rhs(xy, flin):
    """The row right-hand side that makes L U_exact vanish, evaluated at nodes."""
    u, v, p, om = exact_fields(xy)
    ux, uy, vx, vy = 2.0, -3.0, -5.0, -2.0
    px, py = 1.3, -0.4
    fu, fv = flin[:, 0], flin[:, 1]
    fux, fuy = 0.6, -0.9                    # gradients of the linear flin below
    fvx, fvy = 1.1, 0.5
    r = np.zeros((len(u), NF))
    r[:, 0] = W_CON*(ux + vy)                                    # = 0
    r[:, 1] = om + uy - vx                                       # = 0
    r[:, 2] = A_MASS*u + A_FLUX*(fu*ux + fv*uy + u*fux + v*fuy + px + NU*0.0)
    r[:, 3] = A_MASS*v + A_FLUX*(fu*vx + fv*vy + u*fvx + v*fvy + py - NU*0.0)
    return r

for nm, m in (('1 element', build_rect(0,1,0,1,1,1)),
              ('4x3 affine', build_rect(0,2,0,1.5,4,3)),
              ('5x5 distorted', build_rect(0,1,0,1,5,5,distort=0.25))):
    x, y = m.xy[:, 0], m.xy[:, 1]
    flin = np.column_stack([0.3 + 0.6*x - 0.9*y, -0.2 + 1.1*x + 0.5*y])
    rhs = exact_rhs(m.xy, flin)
    A, b = assemble(m, flin, COEF, rhs)
    u, v, p, om = exact_fields(m.xy)
    U = np.empty(m.ndof)
    U[0::NF], U[1::NF], U[2::NF], U[3::NF] = u, v, p, om
    res = np.abs(A @ U - b).max()/max(np.abs(b).max(), 1.0)
    chk(f'G4 {nm}: ||L^T L U_exact - L^T f||_inf', res, 1e-12)
    chk(f'G4 {nm}: symmetry', abs(A - A.T).max()/abs(A).max(), 1e-14)

# one-element mesh: the global matrix IS the element matrix
m = build_rect(0,1,0,1,1,1)
flin = np.zeros((m.nnode, 2))
A, _ = assemble(m, flin, COEF)
Ae, _ = element_matrix(m.xy[m.quads[0]], flin[m.quads[0]], COEF)
g = (NF*m.quads[0][:, None] + np.arange(NF)[None, :]).ravel()
chk('G4 one element: global == element (permuted)',
    np.abs(A.toarray()[np.ix_(g, g)] - Ae).max()/np.abs(Ae).max(), 1e-15)

# THE ASSEMBLED NULL SPACE.  The element matrices are singular (gate G2) and
# the question that matters is whether those modes survive assembly and, if so,
# whether they are a CHECKERBOARD -- a count growing with the element count,
# which would make the method unusable -- or a fixed handful of global modes
# that boundary conditions remove.  Measured, it is the latter.
print('  ---- assembled null space vs mesh refinement')
print(f'        {"mesh":>6s} {"elems":>6s} {"dof":>5s} {"null<1e-10":>10s} '
      f'{"null<1e-6":>9s} {"cond after Dirichlet":>20s}')
counts, conds, hs = [], [], []
for n in (2, 4, 6, 8):
    m = build_rect(0,1,0,1,n,n)
    x, y = m.xy[:,0], m.xy[:,1]
    fl = np.column_stack([0.3+0.6*x-0.9*y, -0.2+1.1*x+0.5*y])
    A, b = assemble(m, fl, COEF, np.zeros((m.nnode, NF)))
    ev = np.sort(np.linalg.eigvalsh(A.toarray()))
    n10, n6 = int((ev < 1e-10*ev[-1]).sum()), int((ev < 1e-6*ev[-1]).sum())
    bnd = np.unique(np.concatenate([e.ravel() for e in m.edge_tags.values()]))
    fx = np.concatenate([dof_index(bnd,0), dof_index(bnd,1), [dof_index([0],2)[0]]])
    Ac, bc = apply_dirichlet(A, b, fx, np.zeros(len(fx)))
    evc = np.sort(np.linalg.eigvalsh(Ac.toarray()))
    cond = evc[-1]/evc[0]
    counts.append(n10); conds.append(cond); hs.append(1.0/n)
    print(f'        {n}x{n:<4d} {m.nelem:6d} {m.ndof:5d} {n10:10d} {n6:9d} '
          f'{cond:20.2e}')
    if n == 8:
        pc = np.zeros(m.ndof); pc[2::NF] = 1.0
        chk('assembled: pressure constant is still null',
            np.linalg.norm(A @ pc)/abs(A).max(), 1e-12)
        chk('after Dirichlet: symmetry preserved',
            abs(Ac - Ac.T).max()/abs(Ac).max(), 1e-14)
        chkb('after Dirichlet: nonsingular', evc[0] > 1e-10*evc[-1],
             f'[smallest {evc[0]:.2e}]')
        try:
            spla.splu(Ac.tocsc()); chkb('after Dirichlet: sparse LU succeeds', True)
        except Exception as e:
            chkb('after Dirichlet: sparse LU succeeds', False, str(e)[:50])

chkb('null count does NOT grow with the mesh (i.e. no checkerboard)',
     len(set(counts)) == 1, f'[{counts} over 4..64 elements]')
rate = np.polyfit(np.log(hs), np.log(conds), 1)[0]
chkb('conditioning after BCs scales no worse than h^-2',
     -rate < 2.3, f'[fitted cond ~ h^{rate:.2f}, textbook is -2]')

print(f'\nG1/G4: {"ALL PASS" if ok else "FAILURES PRESENT"}')
sys.exit(0 if ok else 1)
