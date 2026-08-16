#!/usr/bin/env python3
"""Gartling (1990) backward-facing-step grid, as used by Chan & Mittal fig. 3-6.

    python mesh_gartling.py <L> <out.dat> <NX> <NY> <order>

GEOMETRY (Chan & Mittal, CTR Proc. Summer Program 1996, p.353):
    "The rectangular flow domain is 17 units long and 1 unit high.  The flow
     enters the domain along the top half portion of the left boundary with a
     parabolic profile.  The Reynolds number based on the step height and mean
     velocity is 800.  Figure 4 shows the grid skeleton employed; there are 4
     elements in the vertical direction and 11 elements in the streamwise
     direction."

So unlike every other BFS grid in this repo there is NO upstream inlet channel:
the domain is the plain rectangle [0, L] x [-0.5, 0.5] and the inflow is imposed
AT the step plane.  The left boundary is split --

    x = 0, y in [ 0.0, 0.5]   parabolic inlet   (bc 3)
    x = 0, y in [-0.5, 0.0]   the step face, no-slip wall (bc 1)

which is why NY must be even: y = 0 has to fall on an element boundary.  Top and
bottom are no-slip (bc 1); the outlet at x = L is bc 4 (p = 0), which the driver
pairs with dw/dx = 0 to make the P+Z condition.

Element rows and columns are UNIFORM.  Chan gives element counts and nothing
about grading, so uniform is the faithful reading; grading near the step would
be a different grid from the one the figures were computed on.
"""
import sys
import numpy as np

L = float(sys.argv[1]) if len(sys.argv) > 1 else 17.0
OUT = sys.argv[2] if len(sys.argv) > 2 else 'gartling_nx11_N6_grid.dat'
NX = int(sys.argv[3]) if len(sys.argv) > 3 else 11
NY = int(sys.argv[4]) if len(sys.argv) > 4 else 4
P = int(sys.argv[5]) if len(sys.argv) > 5 else 6
GRADE = sys.argv[6] if len(sys.argv) > 6 else 'uniform'

assert NY % 2 == 0, "NY must be even so that y = 0 (the step lip) is an element edge"


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
if GRADE == 'chan':
    # Element boundaries measured off Chan & Mittal's own grid skeleton (top panel
    # of their fig. 5, p.354) at 600 dpi, using a darkness threshold low enough to
    # catch the faint lines in the right half of the scan:
    #
    #     x = 0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17
    #         |- five of width 1 -|-- six of width 2 --|
    #
    # 11 elements, matching the "11 elements in the streamwise direction" in his
    # text, graded exactly 2:1 with the fine half over the recirculation bubble.
    #
    # An earlier reading of this figure gave 13 elements at 0.89/1.78.  That was
    # wrong: a higher threshold missed the faint interior lines, and the axis
    # frame (which is inset from the grid block) was taken as the domain edge.
    xb = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0])
    NX = len(xb)-1
else:
    xb = np.linspace(0.0, L, NX+1)
yb = np.linspace(-0.5, 0.5, NY+1)
nelem = NX*NY
ROW_LIP = NY//2                      # rows 1..ROW_LIP are below y = 0 (step face)

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
        w = 0
        if col == 1:
            w = 3 if row > ROW_LIP else 1        # inlet above the lip, wall below
        ec = 4 if col == NX else 0               # outlet, p = 0
        s = 1 if row == 1 else 0                 # bottom wall
        n = 1 if row == NY else 0                # top wall
        bc[e] = [w, ec, s, n]


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

inl = [e+1 for e in range(nelem) if bc[e, 0] == 3]
stp = [e+1 for e in range(nelem) if bc[e, 0] == 1]
print(f"wrote {OUT}: {nelem} elems = {NX} x {NY}, order {P} (nterm {nterm})")
print(f"  x in [0, {L}]  uniform dx = {L/NX:.4f}")
print(f"  y in [-0.5, 0.5]  uniform dy = {1.0/NY:.4f}")
print(f"  inlet elements (bc 3): {inl}   y in [0, 0.5]")
print(f"  step-face elements (bc 1 west): {stp}   y in [-0.5, 0]")
print(f"  outlet elements (bc 4 east): {[e+1 for e in range(nelem) if bc[e,1]==4]}")
