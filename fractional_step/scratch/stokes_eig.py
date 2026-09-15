"""The Stokes decay eigenmode for Chan (1996) Fig. 1, solved before any CFD.

Chan reports a decay rate of 9.313316 for the channel Stokes problem, but does
NOT state nu or the streamwise wavenumber.  If the eigenproblem with the obvious
choices (nu = 1, alpha = 1 for a 0..2pi box) reproduces 9.313316, the setup is
confirmed; if it does not, my reading of the case is wrong and no amount of CFD
will fix that.

Formulation.  Stokes:  d(lap psi)/dt = nu lap^2 psi,  u = psi_y, v = -psi_x.
With psi = f(y) exp(i alpha x + s t):

    s (D^2 - a^2) f = nu (D^2 - a^2)^2 f

Put g = (D^2 - a^2) f (the vorticity amplitude).  Then g'' = (a^2 + s/nu) g, so
writing s = -nu (a^2 + b^2) gives g'' + b^2 g = 0 and

    f(y) = A cosh(a y) + B sinh(a y) + P cos(b y) + Q sin(b y)

the cosh/sinh part being the homogeneous solution of (D^2 - a^2) f = 0.

No-slip on both walls means u = f' = 0 AND v = -i a f = 0, i.e.

    f(0) = f'(0) = f(1) = f'(1) = 0

Four conditions, four constants -> a 4x4 determinant whose roots in b are the
eigenvalues.  Energy decays as exp(2 s t), and Chan's sigma is |s|.
"""
import numpy as np
from scipy.optimize import brentq

ALPHA = 1.0        # 2pi/Lx with Lx = 2pi -> fundamental streamwise mode
NU = 1.0
H = 1.0            # "four elements in the wall-to-wall direction, dimension one"
TARGET = 9.313316  # Chan (1996), Fig. 1 text


def M(b, a=ALPHA, h=H):
    """Rows: f(0), f'(0), f(h), f'(h).  Columns: A, B, P, Q."""
    return np.array([
        [1.0,               0.0,            1.0,              0.0],
        [0.0,               a,              0.0,              b],
        [np.cosh(a*h),      np.sinh(a*h),   np.cos(b*h),      np.sin(b*h)],
        [a*np.sinh(a*h),    a*np.cosh(a*h), -b*np.sin(b*h),   b*np.cos(b*h)],
    ])


def det(b, a=ALPHA, h=H):
    return np.linalg.det(M(b, a, h))


# scan for sign changes; the determinant is smooth in b away from b = 0
bs = np.linspace(1e-6, 40.0, 200001)
ds = np.array([det(b) for b in bs])
roots = []
for k in range(len(bs)-1):
    if ds[k] == 0.0:
        roots.append(bs[k])
    elif ds[k]*ds[k+1] < 0:
        roots.append(brentq(det, bs[k], bs[k+1], xtol=1e-14, rtol=1e-15))

# drop spurious near-zero roots (b -> 0 is the trivial/degenerate limit)
roots = [r for r in roots if r > 1e-3]

print(f"alpha = {ALPHA},  nu = {NU},  channel height = {H}")
print(f"Chan (1996) reports a decay rate of {TARGET}\n")
print(f"{'n':>3}{'beta':>14}{'s = -nu(a^2+b^2)':>20}{'|s| vs target':>16}")
for i, b in enumerate(roots[:8]):
    s = -NU*(ALPHA**2 + b**2)
    print(f"{i+1:>3}{b:>14.9f}{s:>20.9f}{abs(s)/TARGET:>15.6f}x")

best = min(roots, key=lambda b: abs(NU*(ALPHA**2 + b**2) - TARGET))
s_best = -NU*(ALPHA**2 + best**2)
print(f"\nclosest to the target: beta = {best:.9f}  ->  |s| = {abs(s_best):.9f}")
print(f"relative difference from {TARGET}: {abs(abs(s_best)-TARGET)/TARGET:.3e}")

# what nu would the SLOWEST mode need to hit the target?
b1 = roots[0]
nu_needed = TARGET/(ALPHA**2 + b1**2)
print(f"\nslowest mode beta_1 = {b1:.9f}")
print(f"nu that would make the SLOWEST mode decay at {TARGET}: {nu_needed:.9f}")

# eigenfunction for the slowest mode, for use as the IC
_, _, Vt = np.linalg.svd(M(b1))
c = Vt[-1]
print(f"null-vector residual |M c| = {np.linalg.norm(M(b1) @ c):.3e}")
print(f"coefficients (A, B, P, Q) = {np.array2string(c, precision=6)}")
