"""Gates G2 and G3 for the Q1 element.

CRITERIA ARE MEASURED, NOT ASSUMED -- and the first two attempts were wrong.

Attempt 1 required "exactly one zero eigenvalue, the pressure constant" on a
single element.  Wrong twice over: an unconstrained element matrix is
legitimately singular (so is a Laplacian's), and the null space here is larger.

Attempt 2 required a fixed count -- 4 in the Stokes limit, 2 with convection.
Also wrong: measured over 20 random geometries the count DEPENDS ON THE
ELEMENT.  A rectangle in the Stokes limit gives 4, a general parallelogram 3, a
distorted quad 3; with convection, a parallelogram gives 2 and a distorted quad
usually 1.  Any fixed number is a property of the test element, not of the
formulation.

THE INVARIANT THAT IS ROBUST is that refining the quadrature must not change
the count.  A structural null mode is a property of the operator and survives
any rule; a mode created by under-integration disappears when the rule is
refined.  So the gate asserts

  (a) the pressure constant is a null mode exactly, on every element;
  (b) the count at the default 3x3 equals the count at 6x6 and at 8x8;
  (c) the deliberately under-integrated rules 1x1 and 2x2 give MORE,
      which proves the test can detect hourglassing at all.

WHY THE STRUCTURAL MODES EXIST.  Take a pure pressure perturbation p and set

    u = -(a_flux/a_mass) p_x ,   v = -(a_flux/a_mass) p_y ,   omega = 0.

The momentum rows vanish by construction; the continuity row gives
-(a_flux/a_mass) laplacian(p), zero for a bilinear p on an AFFINE element
because bilinear functions carry no x^2 or y^2 term; and the vorticity row
gives -(a_flux/a_mass)(p_yx - p_xy) = 0.  So on a rectangle every bilinear
pressure generates a null mode.  Distortion makes the physical basis
non-harmonic and convection breaks the construction, which is why the count
falls away from 4.  This is the FOSLS analogue of the checkerboard problem for
equal-order Q1-Q1, and whether it matters is settled at the ASSEMBLED level
(gate V2), not here.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
from fosls_fem.element_q1 import (geometry, element_matrix,
                                  element_matrix_affine, is_affine,
                                  gauss_rule, NN, NF)

COEF = (4.7434, 0.31623, 1.0, 0.01)
RECT = np.array([[0.,0.],[2.,0.],[2.,1.5],[0.,1.5]])
PARA = np.array([[0.,0.],[2.,0.],[2.7,1.5],[0.7,1.5]])
GEN  = np.array([[0.,0.],[2.,0.],[2.9,1.7],[0.3,1.2]])
rng = np.random.default_rng(7)
FLIN = rng.normal(size=(NN,2))
ZERO = np.zeros((NN,2))
ok = True

def chk(name, val, tol):
    global ok
    p = val < tol; ok &= p
    print(f'  {"PASS" if p else "FAIL"}  {name:56s} {val:9.2e} (tol {tol:.0e})')

def chkeq(name, got, want, extra=''):
    global ok
    p = got == want; ok &= p
    print(f'  {"PASS" if p else "FAIL"}  {name:56s} {got:9d} (want {want}) {extra}')

print('G2/G3 -- Q1 element')

e = max(max(abs(geometry(GEN,x,y)[0].sum()-1), abs(geometry(GEN,x,y)[1].sum()),
            abs(geometry(GEN,x,y)[2].sum())) for x,y in rng.uniform(-1,1,(20,2)))
chk('partition of unity: sum N = 1, sum grad N = 0', e, 1e-14)

f = 3*GEN[:,0] - 2*GEN[:,1]
e = max(max(abs(geometry(GEN,x,y)[1] @ f - 3), abs(geometry(GEN,x,y)[2] @ f + 2))
        for x,y in rng.uniform(-0.8,0.8,(10,2)))
chk('gradient of a linear field recovered exactly', e, 1e-11)

A,_ = element_matrix(GEN, FLIN, COEF)
chk('L^T L symmetric', np.abs(A-A.T).max()/np.abs(A).max(), 1e-15)

print('  --- G3: the Gauss path against the one case with ground truth')
for nm, xy in (('rectangle', RECT), ('parallelogram', PARA)):
    A1,_ = element_matrix(xy, FLIN, COEF)
    A2,_ = element_matrix_affine(xy, FLIN, COEF)
    chk(f'G3 {nm}: default 3x3 == closed form (5x5)',
        np.abs(A1-A2).max()/np.abs(A2).max(), 1e-14)
chk('G3 affine detection (general quad must be rejected)',
    0.0 if (is_affine(RECT) and is_affine(PARA) and not is_affine(GEN)) else 1.0, 0.5)

gp6, gw6 = gauss_rule(6)
for nm, xy, fl in (('affine, no convection', RECT, ZERO),
                   ('affine, with convection', RECT, FLIN),
                   ('general quad, with convection', GEN, FLIN)):
    ref,_ = element_matrix(xy, fl, COEF, None, gp6, gw6)
    gp2, gw2 = gauss_rule(2)
    a2,_ = element_matrix(xy, fl, COEF, None, gp2, gw2)
    a3,_ = element_matrix(xy, fl, COEF)
    print(f'        {nm:31s} 2x2 err {np.abs(a2-ref).max()/np.abs(ref).max():.1e}'
          f'   3x3 err {np.abs(a3-ref).max()/np.abs(ref).max():.1e}')
# A non-affine quad has a RATIONAL integrand -- J varies and J^-1 enters -- so
# NO finite rule is exact and the error only decays (measured: 6.7e-4, 4.1e-6,
# 2.5e-8, 1.5e-10 for 2x2..5x5).  The criterion is engineering accuracy, not
# machine precision, and it is the one place in this gate where that is true.
chk('G3 default 3x3 on a distorted quad (no rule is exact here)',
    np.abs(element_matrix(GEN,FLIN,COEF)[0]
           - element_matrix(GEN,FLIN,COEF,None,gp6,gw6)[0]).max()
    / np.abs(element_matrix(GEN,FLIN,COEF,None,gp6,gw6)[0]).max(), 1e-5)

print('  --- G2: the null space')
pc = np.zeros(NN*NF); pc[2::NF] = 1.0
gp8, gw8 = gauss_rule(8)
for nm, xy in (('rectangle', RECT), ('parallelogram', PARA), ('distorted', GEN)):
    for lbl, fl in (('Stokes', ZERO), ('convect', FLIN)):
        A3,_ = element_matrix(xy, fl, COEF)
        A6,_ = element_matrix(xy, fl, COEF, None, gp6, gw6)
        A8,_ = element_matrix(xy, fl, COEF, None, gp8, gw8)
        n3, n6, n8 = (int((np.sort(np.linalg.eigvalsh(M)) <
                           1e-8*np.linalg.eigvalsh(M).max()).sum())
                      for M in (A3, A6, A8))
        chkeq(f'G2 {nm:13s} {lbl:8s} count invariant under refinement',
              int(n3 == n6 == n8), 1, f'[3x3 {n3}, 6x6 {n6}, 8x8 {n8}]')
        chk(f'     pressure constant is null on {nm} / {lbl}',
            np.linalg.norm(A3 @ pc)/np.abs(A3).max(), 1e-14)

print('  --- G2 control: under-integration must ADD modes')
for n in (1, 2):
    gp, gw = gauss_rule(n)
    ev = np.sort(np.linalg.eigvalsh(element_matrix(RECT, FLIN, COEF, None, gp, gw)[0]))
    nlow = int((ev < 1e-8*ev[-1]).sum())
    ev3 = np.sort(np.linalg.eigvalsh(element_matrix(RECT, FLIN, COEF)[0]))
    n3 = int((ev3 < 1e-8*ev3[-1]).sum())
    chkeq(f'{n}x{n} Gauss adds spurious modes (vs {n3} at 3x3)',
          int(nlow > n3), 1, f'[{n}x{n} gives {nlow}]')

print(f'\nG2/G3: {"ALL PASS" if ok else "FAILURES PRESENT"}')
sys.exit(0 if ok else 1)
