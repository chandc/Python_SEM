"""Short-domain Armaly BFS under P+Z: streamlines, with the long-domain solution
above it on the same scale for reference."""
import os, sys
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'); sys.path.insert(0,SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix
S=0.94
def pack(f,lab):
    d=np.load(f'{SC}/{f}'); U,xn,yn,hy=d['U'],d['xnod'],d['ynod'],d['hy']; n=U.shape[1]
    px,py,pu,pv=[],[],[],[]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e,i]/S); py.append(yn[e,j]/S)
                pu.append(U[e,i,j,0]); pv.append(U[e,i,j,1])
    px,py=np.array(px),np.array(py)
    tri=Triangulation(px,py); cx=px[tri.triangles].mean(1); cy=py[tri.triangles].mean(1)
    tri.set_mask((cx<0)&(cy<1.0))
    return dict(lab=lab,U=U,xn=xn/S,yn=yn/S,hy=hy/S,n=n,xmin=px.min(),xmax=px.max(),
                ytop=py.max(),fu=LinearTriInterpolator(tri,np.array(pu)),
                fv=LinearTriInterpolator(tri,np.array(pv)),status=str(d['status']))
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
C=[pack('armaly_short_pz.npz','SHORT / P+Z   (outlet x/S = 5.3)'),
   pack('armaly_long_pz.npz','LONG / P+Z  -- reference, same grid family')]
fig,axs=plt.subplots(2,1,figsize=(13.5,6.4))
for ax,c in zip(axs,C):
    gx=np.linspace(c['xmin'],c['xmax'],1200); gy=np.linspace(0,c['ytop'],280)
    GX,GY=np.meshgrid(gx,gy)
    ui=np.array(c['fu'](GX,GY).filled(np.nan)); vi=np.array(c['fv'](GX,GY).filled(np.nan))
    cf=ax.contourf(GX,GY,np.ma.masked_invalid(np.hypot(ui,vi)),levels=40,cmap='viridis',vmin=0,vmax=1.55)
    ax.contourf(GX,GY,np.ma.masked_where(np.nan_to_num(ui)>=0,np.ones_like(GX)),
                levels=[.5,1.5],colors=['red'],alpha=.28)
    ax.streamplot(gx,gy,np.nan_to_num(ui),np.nan_to_num(vi),density=2.8,color="w",
                  linewidth=.7,arrowsize=.75)
    ax.add_patch(plt.Rectangle((c['xmin'],0),-c['xmin'],1.0,fc='0.85',ec='k',lw=1.2,zorder=5))
    xr=reatt(c)
    if np.isfinite(xr): ax.plot([xr],[0],'r^',ms=12,zorder=7,clip_on=False)
    ax.axvspan(8.05-0.7,8.05+0.7,color='gold',alpha=.22,zorder=1)
    ax.axvline(8.05,color='goldenrod',lw=2.2,ls='--',zorder=6)
    ax.axvline(c['xmax'],color='yellow',lw=4,zorder=6)
    ax.set_xlim(-2.5,18.5); ax.set_ylim(0,2.15); ax.set_ylabel('y/S')
    t=f"x_r/S = {xr:.3f}  ({(xr-8.05)/8.05*100:+.1f}% vs Armaly 8.05)" if np.isfinite(xr) else "no reattachment"
    ax.set_title(f"{c['lab']}   |   max|u| = {np.abs(c['U'][...,0]).max():.4f}   |   {t}",fontsize=10)
    plt.colorbar(cf,ax=ax,pad=.01,fraction=.021,label='|u|')
axs[-1].set_xlabel('x/S')
fig.suptitle('Armaly BFS, ER 1.94, Re = 389 — SHORT domain under P+Z vs the LONG reference.\n'
             'Yellow = that case\'s outlet;  gold dashed = Armaly measured reattachment '
             '(band = digitisation ±0.7);  red = reversed flow.',fontsize=11.5)
fig.tight_layout(rect=[0,0,1,0.89])
fig.savefig(f'{SC}/../figs/armaly_short_pz_streamlines.png',dpi=130,bbox_inches='tight')
print('figs/armaly_short_pz_streamlines.png')
for c in C: print(f"  {c['lab']:>44}  x_r/S = {reatt(c):.3f}")
