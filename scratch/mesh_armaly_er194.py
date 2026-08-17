#!/usr/bin/env python3.11
"""Backward-facing-step mesh WITH an upstream inlet channel (Erturk geometry).

L-shaped domain (nondimensional, H=1, step S=0.5):
  * inlet channel:  x in [-LIN, 0], y in [0.5, 1]  (upper half only; the step
                    occupies x<0, y<0.5 as a solid block)
  * main domain:    x in [0, L],    y in [0, 1]

The parabolic inflow is imposed at x=-LIN (the inlet-channel entrance) and
develops down the channel before expanding over the step at x=0. Compare to the
existing 'bfs_grid.dat' (inflow AT the step, Gartling geometry).

Element numbering: main domain 1..80 (row-major bottom->top, as in mesh_bfs.py);
inlet-channel elements 81.. appended. Connectivity ties the inlet channel's east
edge to the main domain's upper-left elements (rows 3 & 4, col 1 = elems 41, 61).
"""
import numpy as np

# ---- geometry / resolution ----
P     = 10                       # polynomial order (nterm = P+1)
import sys as _s
L     = float(_s.argv[1]) if len(_s.argv)>1 else 17.0
STEP  = 0.94
XPOW  = 1.6                      # main x-grading (cluster near step)
YB    = [0.0, 0.47, 0.94, 1.44, 1.94]
LIN   = 2.0
OUT   = _s.argv[2] if len(_s.argv)>2 else "armaly_er194_long_grid.dat"
# Chan & Mittal (CTR Proc. Summer Program 1996, p.351) specify "72 elements for
# the long domain and 36 for the short domain".  NX=15, NXIN=6 -> 15*4 + 6*2 = 72.
# The short grid must therefore be built with NX=6 (6*4 + 6*2 = 36), NOT NX=15:
#   python mesh_armaly_er194.py 5.0 armaly_er194_short36_grid.dat 6
NX    = int(_s.argv[3]) if len(_s.argv)>3 else 15   # main-domain columns
NXIN  = int(_s.argv[4]) if len(_s.argv)>4 else 6    # inlet-channel columns
# -------------------------------

def gll_nodes(n):
    if n == 1: return np.array([-1.0, 1.0])
    x = np.cos(np.pi*np.arange(n+1)/n); xo = np.full_like(x, 2.0)
    Pm = np.zeros((n+1, n+1))
    while np.max(np.abs(x-xo)) > 1e-15:
        xo = x.copy(); Pm[:,0]=1.0; Pm[:,1]=x
        for k in range(2, n+1): Pm[:,k]=((2*k-1)*x*Pm[:,k-1]-(k-1)*Pm[:,k-2])/k
        x = xo - (x*Pm[:,n]-Pm[:,n-1])/((n+1)*Pm[:,n])
    return np.sort(x)

nterm = P+1
z01 = (gll_nodes(P)+1.0)/2.0
yb  = np.array(YB); ny = len(yb)-1
xb  = L*(np.arange(NX+1)/NX)**XPOW                 # main x-boundaries
# inlet-channel x-boundaries: -LIN -> 0, clustered near x=0 (the step)
xib = -LIN*(1.0-np.arange(NXIN+1)/NXIN)**1.5

NMAIN = NX*ny                                       # 80
NIC   = NXIN*2                                      # inlet channel: 2 rows (y in [0.5,1])
nelem = NMAIN + NIC
IROW_LOW, IROW_UP = 3, 4                            # main rows the inlet channel joins

def me(col,row): return (row-1)*NX + col            # main elem id (1-based)
def ie(col,row): return NMAIN + (row-1)*NXIN + col  # inlet-channel elem id (row=1 lower,2 upper)

wht=np.zeros(nelem); wid=np.zeros(nelem)
XP=np.zeros((nelem,nterm)); YP=np.zeros((nelem,nterm))
nbr=np.zeros((nelem,4),dtype=int); bc=np.zeros((nelem,4),dtype=int)   # W E S N

# ---- main domain (elems 1..80) ----
for row in range(1,ny+1):
    y0,y1=yb[row-1],yb[row]
    for col in range(1,NX+1):
        x0,x1=xb[col-1],xb[col]; e=me(col,row)-1
        wid[e]=x1-x0; wht[e]=y1-y0
        XP[e]=x0+z01*(x1-x0); YP[e]=y0+z01*(y1-y0)
        w=ec=s=n=0
        nbr[e,0]=me(col-1,row) if col>1 else 0
        nbr[e,1]=me(col+1,row) if col<NX else 0
        nbr[e,2]=me(col,row-1) if row>1 else 0
        nbr[e,3]=me(col,row+1) if row<ny else 0
        if col==1:
            if y0>=STEP-1e-9:
                # upper-left main elements now join the inlet channel (interior west)
                nbr[e,0]=ie(NXIN, IROW_LOW-2 if row==IROW_LOW else IROW_UP-2)  # -> inlet row 1 or 2
                w=0
            else:
                w=1                              # step face (wall) for lower rows
        if col==NX: ec=4                         # outlet
        if row==1:  s=1                          # bottom wall
        if row==ny: n=1                          # top wall NO-SLIP (Armaly fig 2b)
        bc[e]=[w,ec,s,n]

# ---- inlet channel (elems 81..92): 2 rows spanning y in [0.5,1] ----
ic_yb=[STEP,1.44,1.94]                             # y boundaries for the 2 inlet rows
for r in range(1,3):                             # r=1 lower [0.5,0.75], r=2 upper [0.75,1]
    y0,y1=ic_yb[r-1],ic_yb[r]
    for col in range(1,NXIN+1):
        x0,x1=xib[col-1],xib[col]; e=ie(col,r)-1
        wid[e]=x1-x0; wht[e]=y1-y0
        XP[e]=x0+z01*(x1-x0); YP[e]=y0+z01*(y1-y0)
        nbr[e,0]=ie(col-1,r) if col>1 else 0
        nbr[e,1]=ie(col+1,r) if col<NXIN else (me(1,IROW_LOW) if r==1 else me(1,IROW_UP))
        nbr[e,2]=ie(col,r-1) if r>1 else 0
        nbr[e,3]=ie(col,r+1) if r<2 else 0
        w=3 if col==1 else 0                     # parabolic inlet at x=-LIN
        s=1 if r==1 else 0                       # step-top wall (inlet channel bottom)
        n=1 if r==2 else 0                       # top wall NO-SLIP (Armaly fig 2b)
        bc[e]=[w,0,s,n]                          # east always interior (E=0)

def fmt(a): return " ".join(f"{v:.10f}" for v in a)
with open(OUT,"w") as f:
    f.write(f"{nelem} {nterm}\n")
    f.write(" ".join(f"{v:.10f}" for v in wht)+"\n")
    f.write(" ".join(f"{v:.10f}" for v in wid)+"\n")
    for e in range(nelem):
        f.write(fmt(XP[e])+" \n"); f.write(fmt(YP[e])+" \n")
        f.write(f"{nbr[e,0]} {nbr[e,1]} {nbr[e,2]} {nbr[e,3]} \n")
        f.write(f"{bc[e,0]} {bc[e,1]} {bc[e,2]} {bc[e,3]} \n")

print(f"wrote {OUT}: {nelem} elems ({NMAIN} main + {NIC} inlet-channel), order {P}")
print(f"  inlet channel: x in [{-LIN},0], y in [0.5,1], {NXIN} cols; inlet at x={-LIN}")
print(f"  inlet elems 81..{nelem}; parabolic-inlet elems (ibcw=3): {[e+1 for e in range(nelem) if bc[e,0]==3]}")
print(f"  main upper-left elems 41,61 west-neighbor now: {nbr[me(1,3)-1,0]}, {nbr[me(1,4)-1,0]} (inlet chan)")
print(f"  x range: [{XP.min():.2f}, {XP.max():.2f}], y range: [{YP.min():.2f}, {YP.max():.2f}]")
