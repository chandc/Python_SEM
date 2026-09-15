#!/usr/bin/env python3
"""Plane-Poiseuille channel grid in the Fortran LSSEM grid format.

    python mesh_poiseuille_f90.py <L> <out.dat> <NX> <NY> <order>

A plain rectangle [0, L] x [0, 1] -- no step:

    west  (x = 0)   inlet, bc 3
    east  (x = L)   outlet, bc 4   (the FREEOUT driver leaves u,v,p,om free there)
    south (y = 0)   wall,  bc 1
    north (y = 1)   wall,  bc 1

The Fortran driver's inlet is  u = 6*eta*(1-eta),  eta = (y - ystep)/hinlet,
so running it with ystep = 0, hinlet = 1 gives

    u(y) = 6 y (1-y),   u_max = 1.5 at y = 0.5,   u_mean = 1

which is fully-developed plane Poiseuille for a channel of height 1.  The exact
solution is therefore imposed AT the inlet and should be reproduced everywhere:

    u = 6y(1-y)     v = 0     om = dv/dx - du/dy = 12y - 6
    dp/dx = nu * d2u/dy2 = -12/Re        =>   dp = 12 L / Re  over the domain

Every one of those is a check on the implementation, and all four are
representable exactly in the discrete space for any order >= 2, so a correct
implementation should return them to round-off.
"""
import sys
import numpy as np

L = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
OUT = sys.argv[2] if len(sys.argv) > 2 else 'poiseuille_grid.dat'
NX = int(sys.argv[3]) if len(sys.argv) > 3 else 4
NY = int(sys.argv[4]) if len(sys.argv) > 4 else 2
P = int(sys.argv[5]) if len(sys.argv) > 5 else 10


def gll_nodes(n):
    if n == 1:
        return np.array([-1.0, 1.0])
    x = np.cos(np.pi*np.arange(n+1)/n); xo = np.full_like(x, 2.0)
    Pm = np.zeros((n+1, n+1))
    while np.max(np.abs(x-xo)) > 1e-15:
        xo = x.copy(); Pm[:, 0] = 1.0; Pm[:, 1] = x
        for k in range(2, n+1):
            Pm[:, k] = ((2*k-1)*x*Pm[:, k-1]-(k-1)*Pm[:, k-2])/k
        x = xo - (x*Pm[:, n]-Pm[:, n-1])/((n+1)*Pm[:, n])
    return np.sort(x)


nterm = P+1
z01 = (gll_nodes(P)+1.0)/2.0
xb = np.linspace(0.0, L, NX+1)
yb = np.linspace(0.0, 1.0, NY+1)
nelem = NX*NY

wht = np.zeros(nelem); wid = np.zeros(nelem)
XP = np.zeros((nelem, nterm)); YP = np.zeros((nelem, nterm))
nbr = np.zeros((nelem, 4), dtype=int); bc = np.zeros((nelem, 4), dtype=int)


def me(col, row):
    return (row-1)*NX + col


for row in range(1, NY+1):
    y0, y1 = yb[row-1], yb[row]
    for col in range(1, NX+1):
        x0, x1 = xb[col-1], xb[col]
        e = me(col, row)-1
        wid[e] = x1-x0; wht[e] = y1-y0
        XP[e] = x0+z01*(x1-x0); YP[e] = y0+z01*(y1-y0)
        nbr[e, 0] = me(col-1, row) if col > 1 else 0
        nbr[e, 1] = me(col+1, row) if col < NX else 0
        nbr[e, 2] = me(col, row-1) if row > 1 else 0
        nbr[e, 3] = me(col, row+1) if row < NY else 0
        bc[e] = [3 if col == 1 else 0,          # west: inlet
                 4 if col == NX else 0,         # east: outlet
                 1 if row == 1 else 0,          # south: wall
                 1 if row == NY else 0]         # north: wall


def fmt(a):
    return " ".join(f"{v:.10f}" for v in a)


with open(OUT, "w") as f:
    f.write(f"{nelem} {nterm}\n")
    f.write(" ".join(f"{v:.10f}" for v in wht)+"\n")
    f.write(" ".join(f"{v:.10f}" for v in wid)+"\n")
    for e in range(nelem):
        f.write(fmt(XP[e])+" \n"); f.write(fmt(YP[e])+" \n")
        f.write(f"{nbr[e,0]} {nbr[e,1]} {nbr[e,2]} {nbr[e,3]} \n")
        f.write(f"{bc[e,0]} {bc[e,1]} {bc[e,2]} {bc[e,3]} \n")

print(f"wrote {OUT}: {nelem} elems = {NX} x {NY}, order {P} (nterm {nterm})")
print(f"  x in [0, {L}]   y in [0, 1]")
print(f"  inlet elems (bc 3 west): {[e+1 for e in range(nelem) if bc[e,0]==3]}")
print(f"  outlet elems (bc 4 east): {[e+1 for e in range(nelem) if bc[e,1]==4]}")
print(f"  run the Fortran with  ystep = 0.0, hinlet = 1.0  so that u = 6y(1-y)")
