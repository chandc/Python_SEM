"""Reader for the Fortran LSSEM restart/solution files (iform=1 ASCII).

Layout, reverse-engineered and validated against cnos_short_grid.dat:

    header:  time  Re
             nelem  nnode(=nterm^2)  nterm  nfld  per(=nfld*nnode)
    then per element, stride 11 + 11 + 4*per + 8 = 1966 for nterm = 11:
      [   0:  11]  x nodes            -- matches mesh xnod[e] exactly
      [  11:  22]  y nodes            -- matches mesh ynod[e] exactly
      [  22: +per]  block 0: (u,v,p,om) INTERLEAVED per node, j fastest
      ... three further blocks of `per` (BDF history levels)
      [ last 8  ]  4 neighbours + 4 bc codes

Only block 0 is returned -- that is the current solution.

Validated by: coordinates reproduce the grid to 0.0; wall nodes carry u = v = 0;
the inlet plane reproduces the analytic parabola.
"""
import numpy as np


def load_solution(path, nterm_expect=None):
    t = open(path).read().split()
    time = float(t[0]); re = float(t[1])
    nelem, nnode, nterm, nfld, per = (int(t[i]) for i in range(2, 7))
    if nterm_expect is not None and nterm != nterm_expect:
        raise ValueError(f"nterm {nterm} != expected {nterm_expect}")
    d = np.array([float(x) for x in t[7:]])
    stride = 2*nterm + 4*per + 8
    if d.size != nelem*stride:
        raise ValueError(f"expected {nelem*stride} data tokens, got {d.size}")
    d = d.reshape(nelem, stride)
    xn = d[:, 0:nterm].copy()
    yn = d[:, nterm:2*nterm].copy()
    blk = d[:, 2*nterm:2*nterm+per]                     # block 0 = current
    U = blk.reshape(nelem, nterm, nterm, nfld).copy()   # (e, i, j, field), j fastest
    nb = d[:, -8:-4].astype(int); bc = d[:, -4:].astype(int)
    return dict(time=time, re=re, U=U, xnod=xn, ynod=yn, nb=nb, bc=bc,
                nelem=nelem, N=nterm-1)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
    from fgrid import load
    P = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean'
    for tag, grid, sol in (('short', 'cnos_short_grid.dat',
                            'run_chan389_short/chan389_short.dat'),
                           ('long', 'cnos_long_grid.dat',
                            'run_chan389_long/chan389_long.dat')):
        m, _, _ = load(f'{P}/{grid}')
        s = load_solution(f'{P}/{sol}')
        n = s['N']+1
        dx = np.abs(s['xnod']-m.xnod).max(); dy = np.abs(s['ynod']-m.ynod).max()
        # wall check: bottom-wall elements, j = 0 row
        wu = 0.0
        for e in range(s['nelem']):
            if m.bc[e, 2] == 1:
                wu = max(wu, np.abs(s['U'][e, :, 0, 0:2]).max())
            if m.bc[e, 3] == 1:
                wu = max(wu, np.abs(s['U'][e, :, -1, 0:2]).max())
        # inlet check against 6*eta*(1-eta), eta = (y-0.5)/0.5
        ie = [e for e in range(s['nelem']) if m.bc[e, 0] == 3]
        er = 0.0
        for e in ie:
            eta = (m.ynod[e, :]-0.5)/0.5
            er = max(er, np.abs(s['U'][e, 0, :, 0] - 6.0*eta*(1.0-eta)).max())
        print(f"{tag:>6}: t={s['time']:.1f} Re={s['re']:.0f} nelem={s['nelem']} N={s['N']}"
              f"  coord err {max(dx,dy):.1e}  max|u,v| on walls {wu:.2e}"
              f"  inlet vs parabola {er:.2e}  max|u| {np.abs(s['U'][...,0]).max():.4f}")
