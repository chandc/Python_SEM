"""Python (Armaly spec, P+Z) vs the Fortran reference, normalised by step height.

There is NO Fortran P+Z run: the F90 has FREEOUT / PMASK / TRACTION but nothing
imposing a vorticity condition.  run_armaly_p0_long is p=0 only AND uses the
symmetry-top grid AND sits at Re=778 AND never converged (resid 7.6e-03 at
t=48.5).  So the Fortran reference here is run_chan389_long: correct Re, no-slip
walls, converged to 7.7e-07, FREE outflow, expansion ratio 2.0.

Geometries differ (ER 1.94 vs 2.0), so everything is normalised:
    x/S, y/S            S = step height (0.94 armaly, 0.50 cnos)
    u, v                already scaled by u_avg,inlet = 1 in both
    p                   re-referenced to the inlet-plane mean
    omega * S / U       vorticity carries 1/length, so scale by S
"""
import os, sys
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'); sys.path.insert(0,SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix, lgl_weights
from fsol import load_solution
P='/Users/danielchan/Dropbox/F90_SEM/pmg_clean'

def pack(U,xn,yn,hy,S,lab,col,sty):
    U=U.copy(); n=U.shape[1]; wq=lgl_weights(n-1); xmin=xn.min()
    tot=a=0.0
    for e in range(U.shape[0]):
        if abs(xn[e,0]-xmin)<1e-9:
            tot+=np.sum(wq*U[e,0,:,2])*(hy[e]/2); a+=hy[e]
    U[...,2]-=tot/a                      # common pressure datum
    U[...,3]*=S                          # omega * S / U
    px,py,q=[],[],[[],[],[],[]]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e,i]/S); py.append(yn[e,j]/S)   # normalise by S
                for k in range(4): q[k].append(U[e,i,j,k])
    px,py=np.array(px),np.array(py)
    tri=Triangulation(px,py); cx=px[tri.triangles].mean(1); cy=py[tri.triangles].mean(1)
    tri.set_mask((cx<0)&(cy<1.0))        # step block: x<0, y/S<1
    return dict(lab=lab,col=col,sty=sty,U=U,xn=xn/S,yn=yn/S,hy=hy/S,n=n,S=S,
                xmin=px.min(),xmax=px.max(),ytop=py.max(),
                f=[LinearTriInterpolator(tri,np.array(q[k])) for k in range(4)])

C=[]
d=np.load(f'{SC}/armaly_long_pz.npz')
C.append(pack(d['U'],d['xnod'],d['ynod'],d['hy'],0.94,
              'PY Armaly ER1.94 / P+Z','tab:green',dict(ls='-',lw=2.3)))
s=load_solution(f'{P}/run_chan389_long/chan389_long.dat')
hy=np.array([s['ynod'][e,-1]-s['ynod'][e,0] for e in range(s['nelem'])])
C.append(pack(s['U'],s['xnod'],s['ynod'],hy,0.50,
              'FORT cnos ER2.0 / free','k',dict(ls='--',lw=1.9)))
d=np.load(f'{SC}/bfs_long_pz.npz')
C.append(pack(d['U'],d['xnod'],d['ynod'],d['hy'],0.50,
              'PY cnos ER2.0 / P+Z','tab:blue',
              dict(ls='none',marker='o',ms=4.4,mfc='none',mew=1.2)))

def reatt(c):
    n=c['n']; D=diff_matrix(n-1); xs,tw=[],[]
    for e in range(c['U'].shape[0]):
        if c['yn'][e,0]>0.02 or c['xn'][e,0]<-1e-9: continue
        for i in range(n):
            xs.append(c['xn'][e,i]); tw.append(np.dot(D[0,:],c['U'][e,i,:,0])*(2.0/c['hy'][e]))
    o=np.argsort(xs); xs,tw=np.array(xs)[o],np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k]<0 and tw[k+1]>0 and xs[k]>0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return float('nan')

# ---- streamlines
fig,axs=plt.subplots(len(C),1,figsize=(14.5,3.0*len(C)))
for ax,c in zip(axs,C):
    gx=np.linspace(c['xmin'],c['xmax'],1200); gy=np.linspace(0,c['ytop'],260)
    GX,GY=np.meshgrid(gx,gy)
    ui=np.array(c['f'][0](GX,GY).filled(np.nan)); vi=np.array(c['f'][1](GX,GY).filled(np.nan))
    ax.contourf(GX,GY,np.ma.masked_invalid(np.hypot(ui,vi)),levels=40,cmap='viridis',vmin=0,vmax=1.55)
    ax.contourf(GX,GY,np.ma.masked_where(np.nan_to_num(ui)>=0,np.ones_like(GX)),
                levels=[.5,1.5],colors=['red'],alpha=.28)
    ax.streamplot(gx,gy,np.nan_to_num(ui),np.nan_to_num(vi),density=2.4,color="w",linewidth=.6,arrowsize=.65)
    ax.add_patch(plt.Rectangle((c['xmin'],0),-c['xmin'],1.0,fc='0.85',ec='k',lw=1.1,zorder=5))
    xr=reatt(c)
    if np.isfinite(xr): ax.plot([xr],[0],'r^',ms=11,zorder=7,clip_on=False)
    ax.axvspan(8.05-0.7,8.05+0.7,color='gold',alpha=.22,zorder=1)
    ax.axvline(8.05,color='goldenrod',lw=2.0,ls='--',zorder=6)
    ax.axvline(c['xmax'],color='yellow',lw=3,zorder=6)
    ax.set_xlim(-2.5,18); ax.set_ylim(0,2.1); ax.set_ylabel('y/S')
    ax.set_title(f"{c['lab']}   |   x_r/S = {xr:.3f}   (Armaly measured 8.05)",fontsize=10)
axs[-1].set_xlabel('x/S')
fig.suptitle('BFS Re = 389, normalised by STEP height.  Gold = Armaly measured reattachment.',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(f'{SC}/../figs/armaly_vs_fortran_streamlines.png',dpi=120,bbox_inches='tight')
print('figs/armaly_vs_fortran_streamlines.png')

# ---- u, v, p, omega profiles
ST=[1.0,2.0,4.0,6.0,8.0,12.0]
NM=['u','v','p - p_inlet',r'$\omega\,S/U$']
yy=np.linspace(0.005,1.995,420)
fig,axs=plt.subplots(4,len(ST),figsize=(3.0*len(ST),13.2),sharey=True)
for r in range(4):
    for k,x in enumerate(ST):
        ax=axs[r,k]
        for c in C:
            if x>c['xmax']+1e-9: continue
            yv=yy[yy<=c['ytop']]
            v=np.array(c['f'][r](np.full_like(yv,x),yv).filled(np.nan))
            if c['sty'].get('marker'): ax.plot(v[::18],yv[::18],color=c['col'],label=c['lab'],**c['sty'])
            else: ax.plot(v,yv,color=c['col'],label=c['lab'],**c['sty'])
        ax.axvline(0,color='k',lw=.7,ls=':'); ax.axhline(1.0,color='0.6',lw=.7,ls=':')
        ax.grid(alpha=.3)
        if r==0: ax.set_title(f'x/S = {x:g}',fontsize=10)
        if r==3: ax.set_xlabel('value')
        if k==0: ax.set_ylabel(f'{NM[r]}\n\ny/S')
axs[0,0].legend(fontsize=7.5,loc='upper left')
fig.suptitle('BFS Re = 389 -- u, v, p, omega at stations in step heights.  '
             'Dotted line = step height y/S = 1.\n'
             'BLACK vs BLUE = same geometry (ER 2.0), different code and outflow BC '
             '--> this is the VALIDATION.\n'
             'GREEN = a DIFFERENT GEOMETRY (ER 1.94, Armaly\'s actual rig). It is a '
             'different flow, not an error;\nit has no code-to-code reference, only '
             'Armaly\'s measured reattachment (matched to 1.2%).',fontsize=11)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(f'{SC}/../figs/armaly_vs_fortran_profiles.png',dpi=120,bbox_inches='tight')
print('figs/armaly_vs_fortran_profiles.png')

# Two DIFFERENT questions, reported separately.
FORT, PYc, PYa = C[1], C[2], C[0]
print("\nA. VALIDATION -- same geometry (ER 2.0), different code AND outflow BC")
print("   PY cnos / P+Z  vs  FORT cnos / free")
print(f"{'x/S':>6}{'u':>12}{'v':>12}{'p':>12}{'omega*S':>12}")
yv=yy[yy<=min(c['ytop'] for c in C)]
for x in ST:
    row=""
    for r in range(4):
        a=np.array(FORT['f'][r](np.full_like(yv,x),yv).filled(np.nan))
        b=np.array(PYc['f'][r](np.full_like(yv,x),yv).filled(np.nan))
        m=np.isfinite(a)&np.isfinite(b); row+=f"{np.abs(a[m]-b[m]).max():>12.3e}"
    print(f"{x:>6g}"+row)
print("\nB. GEOMETRY SENSITIVITY -- same code AND same BC (Python, P+Z), ER 2.0 vs 1.94")
print("   NOT an error: these are two different rigs.  Armaly's is the ER 1.94 one.")
print(f"{'x/S':>6}{'u':>12}{'v':>12}{'p':>12}{'omega*S':>12}")
for x in ST:
    row=""
    for r in range(4):
        a=np.array(PYc['f'][r](np.full_like(yv,x),yv).filled(np.nan))
        b=np.array(PYa['f'][r](np.full_like(yv,x),yv).filled(np.nan))
        m=np.isfinite(a)&np.isfinite(b); row+=f"{np.abs(a[m]-b[m]).max():>12.3e}"
    print(f"{x:>6g}"+row)
print("\nThere is NO Fortran solution at ER 1.94: the armaly_* grids have a symmetry")
print("top wall, sit at Re=778, and that run never converged (resid 7.6e-03 at t=48.5).")
