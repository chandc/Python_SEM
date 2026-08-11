"""Stokes decay eigenmode IC + energy harness for Chan (1996) Fig. 1.

Case identified by solving the eigenproblem first (stokes_eig.py / stokes_scan.py):

    x in [0, 2pi] periodic, alpha = 1;  y in [-1, 1] (HALF-height 1);  nu = 1
    slowest mode beta_1 = 2.8833558,  |s| = 9.313740  vs Chan's 9.313316
    nu required to hit the target = 0.999954, i.e. 1 -- so nu is not a free knob

The paper's "dimension of one" is the HALF-height.  With a full height of 1 the
rate is 38.6, a factor of 4.15 out, and the only way to reach 9.3133 would be to
tune nu -- which would make the validation circular.

psi = Re[ f(y) e^{i alpha x} ],   u = psi_y,  v = -psi_x,  omega = v_x - u_y
f(y) = A cosh(a y) + B sinh(a y) + P cos(b y) + Q sin(b y)

with (A,B,P,Q) the null vector of the four no-slip conditions f = f' = 0 at
y = -1 and y = +1.  Taking f real and psi = f(y) cos(alpha x) gives

    u =  f'(y) cos(a x)
    v =  a f(y) sin(a x)
    omega = v_x - u_y = a^2 f cos(a x) - f'' cos(a x) = (a^2 f - f'') cos(a x)

which is divergence-free by construction: u_x + v_y = -a f' sin + a f' sin = 0.
"""
import numpy as np
from scipy.optimize import brentq

ALPHA = 1.0
NU = 1.0
YLO, YHI = -1.0, 1.0        # half-height 1
LX = 2.0*np.pi


def _M(b, a=ALPHA):
    """No-slip conditions f=f'=0 at y=-1 and y=+1.  Columns (A,B,P,Q)."""
    rows = []
    for y in (YLO, YHI):
        rows.append([np.cosh(a*y), np.sinh(a*y), np.cos(b*y), np.sin(b*y)])
        rows.append([a*np.sinh(a*y), a*np.cosh(a*y), -b*np.sin(b*y), b*np.cos(b*y)])
    return np.array(rows)


def slowest_mode(a=ALPHA):
    """Smallest beta > 0 with det M = 0, and its null vector."""
    det = lambda b: np.linalg.det(_M(b, a))
    bs = np.linspace(1e-4, 20.0, 40001)
    ds = np.array([det(b) for b in bs])
    b1 = None
    for k in range(len(bs)-1):
        if bs[k] > 1e-3 and ds[k]*ds[k+1] < 0:
            b1 = brentq(det, bs[k], bs[k+1], xtol=1e-15, rtol=1e-15)
            break
    if b1 is None:
        raise RuntimeError("no eigenvalue found")
    _, sv, Vt = np.linalg.svd(_M(b1, a))
    c = Vt[-1]
    return b1, c, sv[-1]


def f_and_derivs(y, b, c, a=ALPHA):
    """f, f', f'' at y."""
    A, B, P, Q = c
    f = A*np.cosh(a*y) + B*np.sinh(a*y) + P*np.cos(b*y) + Q*np.sin(b*y)
    f1 = a*A*np.sinh(a*y) + a*B*np.cosh(a*y) - b*P*np.sin(b*y) + b*Q*np.cos(b*y)
    f2 = a*a*A*np.cosh(a*y) + a*a*B*np.sinh(a*y) - b*b*P*np.cos(b*y) - b*b*Q*np.sin(b*y)
    return f, f1, f2


def stokes_ic(mesh, amp=1.0e-3, a=ALPHA):
    """Eigenmode IC on the SEM grid, scaled so max|u| = amp.

    amp must be small: our solver always carries u.grad u, and the Stokes limit
    needs that term negligible.  Halving amp and re-measuring the rate is the
    check that it is (see the harness).
    """
    b1, c, resid = slowest_mode(a)
    n = mesh.N + 1
    U = np.zeros((mesh.nelem, n, n, 4))
    for e in range(mesh.nelem):
        for i in range(n):
            x = mesh.xnod[e, i]
            for j in range(n):
                y = mesh.ynod[e, j]
                f, f1, f2 = f_and_derivs(y, b1, c, a)
                U[e, i, j, 0] = f1*np.cos(a*x)                 # u  = psi_y
                U[e, i, j, 1] = a*f*np.sin(a*x)                # v  = -psi_x
                U[e, i, j, 3] = (a*a*f - f2)*np.cos(a*x)       # om = v_x - u_y
    scale = amp/np.abs(U[..., 0]).max()
    U *= scale
    U[..., 2] = 0.0                                            # pressure
    s = -NU*(a*a + b1*b1)
    return U, dict(beta=b1, s=s, sigma=abs(s), svd_resid=resid, coeff=c)


def energy(mesh, U):
    """E = 1/2 integral (u^2 + v^2) dx dy, using the SEM quadrature weights."""
    return 0.5*float(np.sum((U[..., 0]**2 + U[..., 1]**2)*mesh.wq))


if __name__ == '__main__':
    b1, c, resid = slowest_mode()
    s = -NU*(ALPHA**2 + b1**2)
    print(f"alpha = {ALPHA}, nu = {NU}, y in [{YLO}, {YHI}]")
    print(f"beta_1        = {b1:.10f}")
    print(f"s             = {s:.10f}")
    print(f"sigma = |s|   = {abs(s):.10f}   (Chan: 9.313316, "
          f"rel diff {abs(abs(s)-9.313316)/9.313316:.2e})")
    print(f"null-vector singular value = {resid:.3e}")
    print(f"(A,B,P,Q) = {np.array2string(c, precision=8)}")
    # no-slip check
    for y in (YLO, YHI):
        f, f1, _ = f_and_derivs(y, b1, c)
        print(f"  y = {y:+.1f}:  f = {f:+.3e}   f' = {f1:+.3e}")
