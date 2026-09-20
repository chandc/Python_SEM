"""G7 -- the vectorised path must equal the reference path exactly.

The reference per-element routines are readable and gate-verified; the batched
ones are 35x faster and unreadable by comparison.  Keeping both, with this gate
between them, is the only way the speed is safe to use.

THE MESH IS DISTORTED ON PURPOSE.  On a rectangle the off-diagonal Jacobian
entries vanish, and a transposition in the inverse-Jacobian chain rule is
invisible.  That is not hypothetical -- it is the bug this gate caught on its
first run.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
from fosls_fem.mesh import build_rect, NF
from fosls_fem.assemble import assemble_newton, global_dofs
from fosls_fem.element_q1 import element_newton, element_newton_batch

COEF = (4.7434, 0.31623, 1.0, 0.01)
ok = True
print('G7 -- batched vs reference')
for dist in (0.0, 0.25):
    m = build_rect(0, 1, 0, 1, 6, 6, distort=dist)
    rng = np.random.default_rng(3)
    U = rng.normal(size=m.ndof)*0.2
    f = rng.normal(size=(m.nnode, NF))
    g = global_dofs(m)
    Ab, bb = element_newton_batch(m.xy[m.quads], U[g], COEF, f[m.quads])
    dA = dB = 0.0
    for e in range(m.nelem):
        Ae, be = element_newton(m.xy[m.quads[e]], U[g[e]], COEF, f[m.quads[e]])
        dA = max(dA, np.abs(Ae - Ab[e]).max()/np.abs(Ae).max())
        dB = max(dB, np.abs(be - bb[e]).max()/np.abs(be).max())
    p = dA < 1e-14 and dB < 1e-14; ok &= p
    print(f'  {"PASS" if p else "FAIL"}  element, distort {dist:4.2f}: '
          f'A {dA:.2e}  b {dB:.2e}')
    A1, b1 = assemble_newton(m, U, COEF, f, batched=True)
    A0, b0 = assemble_newton(m, U, COEF, f, batched=False)
    dA = abs(A1 - A0).max()/abs(A0).max(); dB = np.abs(b1-b0).max()/np.abs(b0).max()
    p = dA < 1e-14 and dB < 1e-14; ok &= p
    print(f'  {"PASS" if p else "FAIL"}  assembled, distort {dist:4.2f}: '
          f'A {dA:.2e}  b {dB:.2e}')
print(f'\nG7: {"PASS" if ok else "FAIL"}')
sys.exit(0 if ok else 1)
