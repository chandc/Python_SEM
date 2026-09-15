"""Orr-Sommerfeld eigenmode for Chan (1996) Fig. 2, solved before any CFD.

Chan: Re = 7500, plane Poiseuille, and the Orr-Sommerfeld solution (via Streett)
"predicts a growth rate of 0.00223497 and a phase speed of 0.24989154".  Those
are hard published numbers, so the eigensolver can be validated on its own --
exactly as the Stokes case was, where checking first revealed the channel
half-height and saved a fitted nu.

    (U - c)(D^2 - a^2)phi - U'' phi = (D^2 - a^2)^2 phi / (i a Re)

on y in [-1, 1] with U = 1 - y^2, phi = phi' = 0 at both walls.  The classic
Re = 7500, alpha = 1 case has the well-known unstable mode
c ~ 0.24989154 + 0.00223497i, and omega = a*c so the growth rate is a*Im(c).

Chebyshev collocation + scipy.linalg.eig (generalised), with the two boundary
conditions imposed by row replacement.
"""
import numpy as np
from scipy.linalg import eig

RE = 7500.0
ALPHA = 1.0


def cheb(N):
    """Chebyshev differentiation matrix and Gauss-Lobatto nodes on [-1, 1]."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi*np.arange(N+1)/N)
    c = np.ones(N+1); c[0] = c[N] = 2.0
    c *= (-1.0)**np.arange(N+1)
    X = np.tile(x, (N+1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0/c)/(dX + np.eye(N+1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def solve(N=200, Re=RE, a=ALPHA):
    D, y = cheb(N)
    D2 = D @ D
    D4 = D2 @ D2
    I = np.eye(N+1)
    U = 1.0 - y**2
    Upp = -2.0*np.ones_like(y)

    L = (np.diag(U) @ (D2 - a*a*I) - np.diag(Upp)
         - (D4 - 2.0*a*a*D2 + a**4*I)/(1j*a*Re))
    M = D2 - a*a*I

    # phi = 0 and phi' = 0 at both ends, by row replacement
    for row, vec in ((0, I[0]), (N, I[N]), (1, D[0]), (N-1, D[N])):
        L[row, :] = vec
        M[row, :] = 0.0

    w, V = eig(L, M)
    ok = np.isfinite(w)
    w, V = w[ok], V[:, ok]
    # physical modes only: |c| bounded
    keep = np.abs(w) < 10.0
    w, V = w[keep], V[:, keep]
    i = np.argmax(w.imag)
    return w[i], V[:, i], y


if __name__ == '__main__':
    print(f"Orr-Sommerfeld, Re = {RE}, alpha = {ALPHA}, U = 1 - y^2")
    print("Chan (1996): growth rate 0.00223497, phase speed 0.24989154\n")
    print(f"{'N':>5}{'Re(c) phase speed':>22}{'Im(c)':>16}"
          f"{'growth = a*Im(c)':>20}")
    for N in (80, 120, 160, 200, 250):
        c, phi, y = solve(N)
        print(f"{N:>5}{c.real:>22.9f}{c.imag:>16.9f}{(ALPHA*c.imag):>20.9f}")

    c, phi, y = solve(250)
    print(f"\nphase speed  {c.real:.9f}   vs Chan 0.24989154   "
          f"rel {abs(c.real-0.24989154)/0.24989154:.2e}")
    print(f"growth rate  {ALPHA*c.imag:.9f}   vs Chan 0.00223497   "
          f"rel {abs(ALPHA*c.imag-0.00223497)/0.00223497:.2e}")
    print(f"wall values |phi(-1)| = {abs(phi[-1]):.2e}, |phi(+1)| = {abs(phi[0]):.2e}")
