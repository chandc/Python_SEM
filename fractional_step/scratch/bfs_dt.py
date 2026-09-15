"""Chan 1996 case: dt sweep to locate where accuracy degrades.
   Everything held fixed except dt: cnos_long_grid.dat, Re=389, order 10,
   nsub=1, tol=1e-6, cgsfac=1e-3, free outflow + SE-corner pin.
   usage: bfs_dt.py <dt> <nsteps>"""
import sys, os, time
SC='/private/tmp/claude-501/-Users-danielchan-Dropbox-F90-SEM/6eb12f11-0cab-40ba-b8d4-95d1b2eccac6/scratchpad'
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'); sys.path.insert(0,SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S

DT = float(sys.argv[1]); NST = int(sys.argv[2])
PINLOC = sys.argv[3] if len(sys.argv)>3 else 'outlet'
SNAP = 'snap' in sys.argv[4:]
PRE  = ([a.split('=')[1] for a in sys.argv[4:] if a.startswith('pre=')] or ['jacobi'])[0]
IC   = ('devc' if 'devc' in sys.argv[4:] else
        'dev'  if 'dev'  in sys.argv[4:] else 'zero')
MAXNEWTON = int(([a.split('=')[1] for a in sys.argv[4:] if a.startswith('newton=')] or [1])[0])
LBLEND = float(([a.split('=')[1] for a in sys.argv[4:] if a.startswith('lb=')] or [1.0])[0])
GRID = ([a.split('=')[1] for a in sys.argv[4:] if a.startswith('grid=')] or ['long'])[0]
SNAP_STEPS = {2,5,10,20,30,40,60,80,120,160,220,300,400,550,700}
RE, TOL, NITCGS, CGSFAC = 389.0, 1.0e-6, 40000, 1.0e-3
H = 0.5
mesh,_,_ = load(f'/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_{GRID}_grid.dat')
N = mesh.N; n = N+1
pin=None
if PINLOC == 'inlet':
    # top-left corner: u,v Dirichlet from BOTH the inlet plane and the top wall,
    # 9.5 units from the outflow plane, in fully-developed flow.
    for e in range(mesh.nelem):
        if mesh.bc[e,0]==3 and mesh.bc[e,3]==1: pin=(e,0,n-1); break
elif PINLOC == 'inletll':
    # LOWER-left corner of the inlet plane: west inlet BC meets the wall that
    # forms the top of the step.  (x=-1, y=0.5) on the Chan grids.
    for e in range(mesh.nelem):
        if mesh.bc[e,0]==3 and mesh.bc[e,2]==1: pin=(e,0,0); break
else:
    for e in range(mesh.nelem):
        if mesh.bc[e,1]==4 and mesh.bc[e,2]==1: pin=(e,n-1,0); break
outl=[e for e in range(mesh.nelem) if mesh.bc[e,1]==4]
for e in outl: mesh.bc[e,1]=0
st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
U0 = np.zeros((mesh.nelem,n,n,4))
if IC == 'dev':
    # ORIGINAL (kept for reproducibility -- has a defect, see IC == 'devc').
    # The branch test is on the NODE x, so the node column at x=0 takes the
    # downstream profile even inside elements lying wholly in the inlet channel:
    # 10 of 11 nodes get u=1.5 and the 11th gets 0.5625, a single-node spike
    # inside one spectral element.  max|div| = 778 at t=0.
    for e in range(mesh.nelem):
        for i in range(n):
            xx = mesh.xnod[e,i]
            for j in range(n):
                yy = mesh.ynod[e,j]
                if xx < -1e-9:                       # inlet channel, y in [0.5,1]
                    eta = (yy-0.5)/0.5
                    U0[e,i,j,0] = 6.0*eta*(1.0-eta)
                    U0[e,i,j,3] = -12.0*(1.0-2.0*eta)
                else:                                 # expanded channel, y in [0,1]
                    U0[e,i,j,0] = 3.0*yy*(1.0-yy)
                    U0[e,i,j,3] = -3.0*(1.0-2.0*yy)
elif IC == 'devc':
    # CONTINUOUS developed IC, divergence-free by construction.
    #
    # Branching by element instead of by node would only MOVE the jump: the two
    # elements share the x=0 nodes, so the field would be multi-valued there.
    # A C0 field cannot represent a jump at a node at all.  Instead, blend from
    # the step-exit profile to the developed profile over a length LBLEND:
    #
    #   u(x,y) = (1-s) u_step(y) + s u_dev(y),   s = smoothstep(x/LBLEND)
    #   u_step = inlet profile above y=0.5, ZERO below  (satisfies step-face no-slip)
    #   u_dev  = 3y(1-y), the fully developed expanded profile
    #
    # Both carry flux 0.5, so every convex combination does too -- mass is
    # conserved at every station, not just at the two ends.  v then follows from
    # continuity, v = -s'(x) G(y) with G(y) = int_0^y (u_dev - u_step) dy',
    # and G(0) = G(1) = 0 gives v = 0 on both walls automatically.
    # smoothstep has s'(0) = s'(LBLEND) = 0, so v -> 0 at both ends of the blend.
    Lb = LBLEND
    def u_dev(y):   return 3.0*y*(1.0-y)
    def du_dev(y):  return 3.0 - 6.0*y
    def u_step(y):
        if y <= 0.5: return 0.0
        eta = 2.0*y - 1.0
        return 6.0*eta*(1.0-eta)
    def du_step(y):
        if y <= 0.5: return 0.0
        eta = 2.0*y - 1.0
        return 12.0*(1.0 - 2.0*eta)
    def G(y):                       # int_0^y (u_dev - u_step)
        I = 1.5*y*y - y**3
        if y > 0.5:
            eta = 2.0*y - 1.0
            I -= 0.5*(3.0*eta*eta - 2.0*eta**3)
        return I
    for e in range(mesh.nelem):
        for i in range(n):
            xx = mesh.xnod[e,i]
            if xx <= 0.0:
                t, sp, spp = 0.0, 0.0, 0.0
            else:
                t = min(xx/Lb, 1.0)
                sp  = (6.0*t - 6.0*t*t)/Lb          # s'(x)
                spp = (6.0 - 12.0*t)/Lb**2          # s''(x)
                if xx >= Lb: sp = spp = 0.0
            sv = 3.0*t*t - 2.0*t**3                  # smoothstep
            for j in range(n):
                yy = mesh.ynod[e,j]
                if xx < 0.0:                         # inlet channel: unchanged
                    eta = (yy-0.5)/0.5
                    U0[e,i,j,0] = 6.0*eta*(1.0-eta)
                    U0[e,i,j,3] = -12.0*(1.0-2.0*eta)
                else:
                    U0[e,i,j,0] = (1.0-sv)*u_step(yy) + sv*u_dev(yy)
                    U0[e,i,j,1] = -sp*G(yy)
                    U0[e,i,j,3] = (-spp*G(yy)
                                   - ((1.0-sv)*du_step(yy) + sv*du_dev(yy)))
hist=[U0]
w=lgl_weights(N); D=diff_matrix(N)
inlet=lambda x,y,t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
fl=lambda U,e,i: np.sum(w*U[e,i,:,0])*(mesh.hy[e]/2)
INL=[e for e in range(mesh.nelem) if mesh.bc[e,0]==3]
nit=[0]; _p=S.pcg_solve
from lssem2d import precond as _P
# pc = p/2 with deg=4 is the setting that actually wins; pc=2,deg=2 cuts
# iterations 7.6x but each V-cycle is too cheap to pay for itself and comes
# out SLOWER than plain Jacobi (measured on this case: 0.91x vs 1.20x).
_PKW={'jacobi':{}, 'cheb':dict(deg=6),
      'pmg':dict(pc=max(2,N//2),deg=4,coarse_deg=10)}
def pcg(state,b,fu,fv,M,mw,pin_p=False,max_iter=NITCGS,tol=None,cgsfac=None):
    pre=None
    if PRE!='jacobi':
        pre=_P.make({'cheb':'chebyshev4','pmg':'pmg2'}[PRE],state,fu,fv,M,pin_p,**_PKW[PRE])
    x,it=_p(state,b,fu,fv,M,mw,pin_p=pin_p,max_iter=NITCGS,tol=TOL,cgsfac=CGSFAC,precond=pre)
    nit[0]+=it; return x,it
S.pcg_solve=pcg
def reatt(U):
    bot=[e for e in range(mesh.nelem) if mesh.y0[e]<0.25 and mesh.x0[e]>=-1e-9]
    xs,tw=[],[]
    for e in bot:
        for i in range(n):
            xs.append(mesh.xnod[e,i]); tw.append(np.dot(D[0,:],U[e,i,:,0])*(2.0/mesh.hy[e]))
    o=np.argsort(xs); xs=np.array(xs)[o]; tw=np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k]<0 and tw[k+1]>0 and xs[k]>0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return np.nan
TAG=(f"dt{DT:g}".replace('.','p'))+("_inpin" if PINLOC=="inlet" else ("_llpin" if PINLOC=="inletll" else ""))+(("_"+IC) if IC!="zero" else "")+(("_"+PRE) if PRE!="jacobi" else "")+(("_"+GRID) if GRID!="long" else "")
log=open(f'{SC}/dt_{TAG}.log','w',buffering=1)
def P(s):
    print(s); log.write(s+'\n')
P("="*78)
P("INPUT PARAMETERS")
P("="*78)
P(f"  grid file        cnos_{GRID}_grid.dat")
P(f"  elements         {mesh.nelem}          polynomial order   {N}   ({mesh.nelem*(N+1)**2*4} dof)")
P(f"  domain           x in [{mesh.xnod.min():.2f}, {mesh.xnod.max():.2f}]  ->  L/h = {mesh.xnod.max()/H:.1f}"
  f"   (step height h = {H})")
P(f"  Reynolds number  {RE:g}        nu = {1.0/RE:.6e}")
P(f"  time step  dt    {DT:g}          steps {NST}   ->  t_end = {DT*NST:g}")
P(f"  BDF order        1 on step 1, then 2 (fac1 = 1.5)     nsub = 1, max_newton = {MAXNEWTON}")
P(f"  initial cond.    {IC}" + (f"   (blend length Lb = {LBLEND:g})" if IC=="devc" else ""))
P(f"  inlet BC         parabolic, u = 6*eta*(1-eta), eta = (y-0.5)/0.5, peak 1.5, v = 0")
P(f"  outlet BC        FREE OUTFLOW - nothing imposed on u, v, p, omega (bc 4 -> 0)")
P(f"  pressure pin     single node, {PINLOC} -> elem {pin[0]} node {pin[1:]}"
  f" (x={mesh.xnod[pin[0],pin[1]]:.4f}, y={mesh.ynod[pin[0],pin[2]]:.4f})")
P(f"  walls            no-slip u = v = 0 (top, bottom, step face)")
P(f"  CG tolerance     tol = {TOL:g}   cgsfac = {CGSFAC:g}   cg_max_iter = {NITCGS}")
P(f"  preconditioner   {PRE}" + (f"   {_PKW[PRE]}" if PRE!="jacobi" else ""))
P("="*78)


P(f"{'step':>6}{'t':>8}{'diff':>11}{'Qout/Qin':>10}{'rms div':>11}{'x_r/h':>9}{'max|u|':>8}{'CGit':>8}{'min/1k':>8}")
t0=time.time()
for s in range(NST):
    nit[0]=0
    U=S.step_bdf(st,hist,time=s*DT,max_newton=MAXNEWTON,newton_tol=1e-10,newton_factor=0.0,
                 custom_inlet=inlet,pin_p=pin,cgsfac=CGSFAC,cg_max_iter=NITCGS,verbose=False)
    if not np.all(np.isfinite(U)): P(f"  NON-FINITE at step {s}"); break
    if s%int(max(NST//30,1))==0 or s==NST-1:
        ux=dUdx(np.ascontiguousarray(U[...,0]),D,mesh.facx)
        vy=dUdy(np.ascontiguousarray(U[...,1]),D,mesh.facy)
        dv=ux+vy
        d=np.max(np.abs(hist[0]-hist[1])) if len(hist)>1 else np.nan
        P(f"{s:>6}{s*DT:>8.1f}{d:>11.2e}{sum(fl(U,e,-1) for e in outl)/sum(fl(U,e,0) for e in INL):>10.4f}"
          f"{np.sqrt((dv**2).mean()):>11.3e}{reatt(U)/H:>9.3f}{np.max(np.abs(U[...,0])):>8.3f}"
          f"{nit[0]:>8}{(time.time()-t0)/(s+1)*1000/60:>8.1f}")
    if s%200==0 or s==NST-1:
        np.savez_compressed(f'{SC}/dt_{TAG}_state.npz',U=U,step=s,dt=DT,xnod=mesh.xnod,
            ynod=mesh.ynod,y0=mesh.y0,hy=mesh.hy,x0=mesh.x0,hx=mesh.hx)
    if SNAP and (s in SNAP_STEPS or s==NST-1):
        np.savez_compressed(f'{SC}/dt_{TAG}_snap{s:05d}.npz',U=U,step=s,dt=DT,xnod=mesh.xnod,
            ynod=mesh.ynod,y0=mesh.y0,hy=mesh.hy,x0=mesh.x0,hx=mesh.hx)
P(f"DONE dt={DT}: x_r/h={reatt(U)/H:.3f}  {(time.time()-t0)/60:.1f} min")
log.close()
