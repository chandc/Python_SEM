"""Which (alpha, H, nu) gives Chan's 9.313316?  Identify, don't fit."""
import numpy as np
from scipy.optimize import brentq
TARGET = 9.313316

def beta1(a, h):
    M = lambda b: np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, a, 0.0, b],
        [np.cosh(a*h), np.sinh(a*h), np.cos(b*h), np.sin(b*h)],
        [a*np.sinh(a*h), a*np.cosh(a*h), -b*np.sin(b*h), b*np.cos(b*h)]])
    det = lambda b: np.linalg.det(M(b))
    bs = np.linspace(1e-4, 30.0/max(h,1e-9), 60001)
    ds = np.array([det(b) for b in bs])
    for k in range(len(bs)-1):
        if ds[k]*ds[k+1] < 0 and bs[k] > 1e-3:
            return brentq(det, bs[k], bs[k+1], xtol=1e-14)
    return np.nan

print(f"{'alpha':>7}{'H':>5}{'beta_1':>12}{'|s| at nu=1':>14}{'nu for target':>15}{'1/nu':>10}")
for h in (1.0, 2.0):
    for a in (0.5, 1.0, 2.0, 2*np.pi/ (2*np.pi)):
        b = beta1(a, h)
        if not np.isfinite(b): continue
        s1 = a**2 + b**2
        print(f"{a:>7.3f}{h:>5.1f}{b:>12.6f}{s1:>14.6f}{TARGET/s1:>15.6f}{s1/TARGET:>10.4f}")

# the alpha -> 0 limit: pure diffusion, s = -nu n^2 pi^2 / H^2
for h in (1.0, 2.0):
    for n in (1, 2):
        s1 = (n*np.pi/h)**2
        print(f"{0.0:>7.3f}{h:>5.1f}{'(a->0)':>12}{s1:>14.6f}{TARGET/s1:>15.6f}{s1/TARGET:>10.4f}")
