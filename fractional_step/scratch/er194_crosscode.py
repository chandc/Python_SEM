"""Cross-code at ARMALY'S ACTUAL GEOMETRY (ER 1.94, no-slip top, Re=389).

All four solutions are on the SAME grid now, so differences are code + outflow BC
only -- no geometry contamination.  This is what the earlier figure could not do,
because no Fortran solution existed at ER 1.94.

  FORT / free    SEM_2D_BFS_FREEOUT, re=194.5 (= nu 2/389), dt=0.5, nitcgs=500, p-MG
  FORT / p=0     SEM_2D_BFS_PMASK,   same settings.  NOTE: p=0 only -- the Fortran
                 has no vorticity condition, so this is the "P" of P+Z, not the pair.
  PY   / P+Z     p=0 AND d(omega)/dx=0, dt=1, fully converged solve
  PY   / free    dt=0.5 legacy weighting, fully converged solve
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
P='/Users/danielchan/Dropbox/F90_SEM/pmg_clean'; S=0.94

def pack(U,xn,yn,hy,lab,col,sty):
    U=U.copy(); n=U.shape[1]; wq=lgl_weights(n-1); xmin=xn.min()
    tot=a=0.0
    for e in range(U.shape[0]):
        if abs(xn[e,0]-xmin)<1e-9:
            tot+=np.sum(wq*U[e,0,:,2])*(hy[e]/2); a+=hy[e]
    U[...,2]-=tot/a; U[...,3]*=S
    px,py,q=[],[],[[],[],[],[]]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e,i]/S); py.append(yn[e,j]/S)
                for k in range(4): q[k].append(U[e,i,j,k])
    px,py=np.array(px),np.array(py)
    tri=Triangulation(px,py); cx=px[tri.triangles].mean(1); cy=py[tri.triangles].mean(1)
    tri.set_mask((cx<0)&(cy<1.0))
    return dict(lab=lab,col=col,sty=sty,U=U,xn=xn/S,yn=yn/S,hy=hy/S,n=n,
                xmin=px.min(),xmax=px.max(),ytop=py.max(),
                f=[LinearTriInterpolator(tri,np.array(q[k])) for k in range(4)])

C=[]
for tag,sub,lab,col,sty in (
    ('free','run_armaly_er194_free/armaly_er194_free.dat','FORT ER1.94 / free','k',dict(ls='-',lw=2.0)),
    ('p0','run_armaly_er194_p0/armaly_er194_p0.dat','FORT ER1.94 / p=0','tab:orange',dict(ls='--',lw=1.9))):
    s=load_solution(f'{P}/{sub}')
    hy=np.array([s['ynod'][e,-1]-s['ynod'][e,0] for e in range(s['nelem'])])
    C.append(pack(s['U'],s['xnod'],s['ynod'],hy,lab,col,sty))
for f,lab,col,sty in (
    ('armaly_long_pz.npz','PY ER1.94 / P+Z','tab:green',dict(ls='none',marker='o',ms=4.6,mfc='none',mew=1.3)),
    ('armaly_long_free_dt0.5_leg_nit200000.npz','PY ER1.94 / free','tab:red',dict(ls=':',lw=1.9))):
    d=np.load(f'{SC}/{f}')
    C.append(pack(d['U'],d['xnod'],d['ynod'],d['hy'],lab,col,sty))

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

print(f"{'case':>22}{'max|u|':>10}{'x_r/S':>9}{'vs Armaly 8.05':>17}")
for c in C:
    xr=reatt(c); c['xr']=xr
    print(f"{c['lab']:>22}{np.abs(c['U'][...,0]).max():>10.4f}{xr:>9.3f}{(xr-8.05)/8.05*100:>16.1f}%")

ST=[1.0,2.0,4.0,6.0,8.0,12.0]; yy=np.linspace(0.005,2.05,420)
ref=C[0]
for r,nm in ((0,'u'),(1,'v'),(2,'p'),(3,'omega*S')):
    print(f"\n=== max |{nm} - FORT/free|   (same grid, so this is code+BC only) ===")
    print(f"{'x/S':>6}"+"".join(f"{c['lab']:>22}" for c in C[1:]))
    for x in ST:
        yv=yy[yy<=min(c['ytop'] for c in C)]
        a=np.array(ref['f'][r](np.full_like(yv,x),yv).filled(np.nan)); row=""
        for c in C[1:]:
            b=np.array(c['f'][r](np.full_like(yv,x),yv).filled(np.nan))
            m=np.isfinite(a)&np.isfinite(b); row+=f"{np.abs(a[m]-b[m]).max():>22.3e}"
        print(f"{x:>6g}"+row)

fig,axs=plt.subplots(4,len(ST),figsize=(3.0*len(ST),13.2),sharey=True)
NM=['u','v','p - p_inlet',r'$\omega\,S/U$']
for r in range(4):
    for k,x in enumerate(ST):
        ax=axs[r,k]
        for c in C:
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
fig.suptitle("ARMALY GEOMETRY (ER 1.94, no-slip, Re=389) -- ALL FOUR ON THE SAME GRID.\n"
             "Differences are code + outflow BC only.  Fortran 'p=0' has no vorticity "
             "condition, so it is the P of P+Z, not the pair.",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(f'{SC}/../figs/armaly_vs_fortran_profiles.png',dpi=120,bbox_inches='tight')
print('\nfigs/armaly_vs_fortran_profiles.png  (replaces the cross-GEOMETRY version)')

# ---- streamlines, same four cases, same grid ----
fig,axs=plt.subplots(len(C),1,figsize=(14.5,2.8*len(C)))
for ax,c in zip(axs,C):
    gx=np.linspace(c['xmin'],c['xmax'],1200); gy=np.linspace(0,c['ytop'],260)
    GX,GY=np.meshgrid(gx,gy)
    ui=np.array(c['f'][0](GX,GY).filled(np.nan)); vi=np.array(c['f'][1](GX,GY).filled(np.nan))
    ax.contourf(GX,GY,np.ma.masked_invalid(np.hypot(ui,vi)),levels=40,cmap='viridis',vmin=0,vmax=1.55)
    ax.contourf(GX,GY,np.ma.masked_where(np.nan_to_num(ui)>=0,np.ones_like(GX)),
                levels=[.5,1.5],colors=['red'],alpha=.28)
    ax.streamplot(gx,gy,np.nan_to_num(ui),np.nan_to_num(vi),density=2.4,color="w",
                  linewidth=.6,arrowsize=.65)
    ax.add_patch(plt.Rectangle((c['xmin'],0),-c['xmin'],1.0,fc='0.85',ec='k',lw=1.1,zorder=5))
    if np.isfinite(c['xr']): ax.plot([c['xr']],[0],'r^',ms=11,zorder=7,clip_on=False)
    ax.axvspan(8.05-0.7,8.05+0.7,color='gold',alpha=.22,zorder=1)
    ax.axvline(8.05,color='goldenrod',lw=2.0,ls='--',zorder=6)
    ax.axvline(c['xmax'],color='yellow',lw=3,zorder=6)
    ax.set_xlim(-2.5,18.5); ax.set_ylim(0,2.15); ax.set_ylabel('y/S')
    ax.set_title(f"{c['lab']}   |   x_r/S = {c['xr']:.3f}   "
                 f"({(c['xr']-8.05)/8.05*100:+.1f}% vs Armaly 8.05)",fontsize=10)
axs[-1].set_xlabel('x/S')
fig.suptitle('ARMALY GEOMETRY (ER 1.94, no-slip, Re = 389) -- all four on the SAME grid.\n'
             'Gold dashed = Armaly measured reattachment, band = digitisation +/- 0.7.',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig(f'{SC}/../figs/armaly_vs_fortran_streamlines.png',dpi=120,bbox_inches='tight')
print('figs/armaly_vs_fortran_streamlines.png  (replaces the cross-GEOMETRY version)')
