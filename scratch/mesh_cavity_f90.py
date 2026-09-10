"""Unit lid-driven cavity grid in the Fortran LSSEM grid format.
    python mesh_cavity_f90.py <out.dat> <NX> <NY> <order>
bc codes: 1 wall, 2 moving lid (north), 0 interior."""
import sys, numpy as np
OUT, NX, NY, P = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def gll_nodes(n):
    x = np.cos(np.pi*np.arange(n+1)/n); xo = np.full_like(x, 2.0); Pm = np.zeros((n+1, n+1))
    while np.max(np.abs(x-xo)) > 1e-15:
        xo = x.copy(); Pm[:, 0] = 1.0; Pm[:, 1] = x
        for k in range(2, n+1): Pm[:, k] = ((2*k-1)*x*Pm[:, k-1]-(k-1)*Pm[:, k-2])/k
        x = xo - (x*Pm[:, n]-Pm[:, n-1])/((n+1)*Pm[:, n])
    return np.sort(x)
nterm = P+1; z01 = (gll_nodes(P)+1.0)/2.0; xb = np.linspace(0, 1, NX+1); yb = np.linspace(0, 1, NY+1); nelem = NX*NY
me = lambda col, row: (row-1)*NX + col
wht = np.zeros(nelem); wid = np.zeros(nelem); XP = np.zeros((nelem, nterm)); YP = np.zeros((nelem, nterm))
nbr = np.zeros((nelem, 4), int); bc = np.zeros((nelem, 4), int)
for row in range(1, NY+1):
    for col in range(1, NX+1):
        e = me(col, row)-1; x0, x1 = xb[col-1], xb[col]; y0, y1 = yb[row-1], yb[row]
        wid[e] = x1-x0; wht[e] = y1-y0; XP[e] = x0+z01*(x1-x0); YP[e] = y0+z01*(y1-y0)
        nbr[e] = [me(col-1, row) if col > 1 else 0, me(col+1, row) if col < NX else 0, me(col, row-1) if row > 1 else 0, me(col, row+1) if row < NY else 0]
        bc[e] = [1 if col == 1 else 0, 1 if col == NX else 0, 1 if row == 1 else 0, 2 if row == NY else 0]
fmt = lambda a: " ".join(f"{v:.10f}" for v in a)
with open(OUT, "w") as f:
    f.write(f"{nelem} {nterm}\n"); f.write(fmt(wht)+"\n"); f.write(fmt(wid)+"\n")
    for e in range(nelem):
        f.write(fmt(XP[e])+" \n"); f.write(fmt(YP[e])+" \n"); f.write(f"{nbr[e,0]} {nbr[e,1]} {nbr[e,2]} {nbr[e,3]} \n"); f.write(f"{bc[e,0]} {bc[e,1]} {bc[e,2]} {bc[e,3]} \n")
print(f"wrote {OUT}: {nelem} elems {NX}x{NY}, order {P}")
