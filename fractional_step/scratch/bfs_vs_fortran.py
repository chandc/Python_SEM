"""Fortran reference vs our Python runs, short and long domain."""
import os, sys
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'); sys.path.insert(0,SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix
from fsol import load_solution
P='/Users/danielchan/Dropbox/F90_SEM/pmg_clean'; H=0.5

def pack(U,xn,yn,hy):
    n=U.shape[1]; px,py,q=[],[],[[],[],[]]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e,i]); py.append(yn[e,j])
                for k in range(3): q[k].append(U[e,i,j,k])
    px,py=np.array(px),np.array(py)
    tri=Triangulation(px,py); cx=px[tri.triangles].mean(1); cy=py[tri.triangles].mean(1)
    tri.set_mask((cx<0)&(cy<0.5))
    return dict(U=U,xn=xn,yn=yn,hy=hy,n=n,xmax=px.max(),xmin=px.min(),
                f=[LinearTriInterpolator(tri,np.array(q[k])) for k in range(3)])

def reatt(c):
    n=c['n']; D=diff_matrix(n-1); xs,tw=[],[]
    for e in range(c['U'].shape[0]):
        if c['yn'][e,0]>0.01 or c['xn'][e,0]<-1e-9: continue
        for i in range(n):
            xs.append(c['xn'][e,i]); tw.append(np.dot(D[0,:],c['U'][e,i,:,0])*(2.0/c['hy'][e]))
    o=np.argsort(xs); xs,tw=np.array(xs)[o],np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k]<0 and tw[k+1]>0 and xs[k]>0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return float('nan')

C={}
for tag,sol in (('FORT short / free','run_chan389_short/chan389_short.dat'),
                ('FORT long / free','run_chan389_long/chan389_long.dat')):
    s=load_solution(f'{P}/{sol}'); C[tag]=pack(s['U'],s['xnod'],s['ynod'],
        np.array([s['ynod'][e,-1]-s['ynod'][e,0] for e in range(s['nelem'])]))
for tag,f in (('PY short / P+Z','bfs_pz_state.npz'),('PY long / P+Z','bfs_long_pz.npz'),
              ('PY long / free','bfs_long_free.npz')):
    d=np.load(f'{SC}/{f}'); C[tag]=pack(d['U'],d['xnod'],d['ynod'],d['hy'])

print(f"{'case':>20}{'max|u|':>10}{'x_r':>10}{'x_r/h':>9}")
for k,c in C.items():
    xr=reatt(c)
    print(f"{k:>20}{np.abs(c['U'][...,0]).max():>10.4f}"
          f"{(f'{xr:.4f}' if np.isfinite(xr) else 'none'):>10}"
          f"{(f'{xr/H:.3f}' if np.isfinite(xr) else '--'):>9}")

yy=np.linspace(0.002,0.998,400)
ST=[0.5,1.0,2.0,2.4]
print(f"\nmax |u - FORT long| at overlap stations (the closest thing to a reference)")
print(f"{'x':>6}" + "".join(f"{k:>20}" for k in C if k!='FORT long / free'))
for x in ST:
    ref=np.array(C['FORT long / free']['f'][0](np.full_like(yy,x),yy).filled(np.nan))
    row=""
    for k,c in C.items():
        if k=='FORT long / free': continue
        if x>c['xmax']+1e-9: row+=f"{'--':>20}"; continue
        v=np.array(c['f'][0](np.full_like(yy,x),yy).filled(np.nan))
        m=np.isfinite(ref)&np.isfinite(v); row+=f"{np.abs(ref[m]-v[m]).max():>20.3e}"
    print(f"{x:>6g}"+row)

fig,axs=plt.subplots(1,4,figsize=(16,4.4),sharey=True)
STY={'FORT short / free':('tab:orange','-',2.4),'FORT long / free':('k','-',2.0),
     'PY short / P+Z':('tab:blue','none',0),'PY long / P+Z':('tab:green','--',1.8),
     'PY long / free':('tab:red',':',1.8)}
for a,x in zip(axs,ST):
    for k,c in C.items():
        if x>c['xmax']+1e-9: continue
        col,ls,lw=STY[k]
        v=np.array(c['f'][0](np.full_like(yy,x),yy).filled(np.nan))
        if ls=='none': a.plot(v[::16],yy[::16],'o',color=col,ms=4.5,mfc='none',mew=1.3,label=k)
        else: a.plot(v,yy,ls,color=col,lw=lw,label=k)
    a.axvline(0,color='k',lw=.7,ls=':'); a.axhline(0.5,color='0.6',lw=.7,ls=':')
    a.set_title(f'x = {x:g}  (x/h = {x/H:g})',fontsize=10); a.grid(alpha=.3); a.set_xlabel('u')
axs[0].set_ylabel('y'); axs[0].legend(fontsize=7.5,loc='upper left')
fig.suptitle('BFS Re=389: Fortran reference (free outflow) vs Python.  '
             'Fortran SHORT overshoots to max|u| = 1.736; every other case is 1.500.',fontsize=12)
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig(f'{SC}/../figs/bfs_vs_fortran.png',dpi=125,bbox_inches='tight')
print('\nfigs/bfs_vs_fortran.png')
