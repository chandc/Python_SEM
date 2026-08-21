"""Export TGV simulation frames to VTK for ParaView.

    uv run --quiet python scratch/tgv_to_vtk.py re100            # all frames
    uv run --quiet python scratch/tgv_to_vtk.py re400 0 10       # frame range

Reads scratch/tgv_frames_<tag>/frame_####.npz (complex64 mode-space state
saved by tgv3d.py) and writes

    scratch/tgv_vtk_<tag>/tgv_<tag>_####.vtr    XML RectilinearGrid, one per frame
    scratch/tgv_vtk_<tag>/tgv_<tag>.pvd         time-series collection (open THIS)

Open the .pvd in ParaView: the time slider then carries the true simulation
times, not frame indices.

THE GRID.  The (x, y) plane is SEM: a tensor product of per-element GLL nodes,
stored with interface nodes duplicated across neighbouring elements.  Because
`build_channel` is a tensor-product mesh, the GLOBAL unique coordinates form a
rectilinear grid (non-uniform spacing in x and y, uniform in z) -- so the data
maps onto a VTK RectilinearGrid EXACTLY, with no interpolation: duplicates are
collapsed (their values agree -- the field is C0), and z gets one wrap plane
appended (value at z = Lz copies z = 0) so the periodic volume renders closed.

FIELDS (point data):
    velocity   (3-vector)   u, v, w
    vorticity  (3-vector)   the solver's own omega fields
    pressure   (scalar)
    omega_mag  (scalar)     |omega|
    Q          (scalar)     Q-criterion, 0.5*(|W|^2 - |S|^2), computed
                            spectrally in z and by the SEM derivative in x, y
                            -- the standard isosurface for TGV visualisation

Q is computed from the velocity gradient tensor BEFORE deduplication, on the
element-local representation, so the derivatives are the solver's own.
"""
import os, sys, glob
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import tgv3d as G
from lssem3d import operator as OP, deriv as DV, fourier as FR

TOL = 1e-9


def global_axes(m):
    """Unique sorted x and y coordinates, and per-(element, local-index) maps."""
    def axis(nodvals):
        vals = np.unique(np.round(nodvals.ravel()/TOL).astype(np.int64))*TOL
        idx = {int(round(v/TOL)): k for k, v in enumerate(vals)}
        return vals, idx
    xs, xmap = axis(m.xnod)
    ys, ymap = axis(m.ynod)
    ix = np.array([[xmap[int(round(x/TOL))] for x in m.xnod[e]]
                   for e in range(m.nelem)])
    iy = np.array([[ymap[int(round(y/TOL))] for y in m.ynod[e]]
                   for e in range(m.nelem)])
    return xs, ys, ix, iy


def scatter_to_grid(field, ix, iy, nx, ny):
    """(nelem, n, n, nz) local -> (nx, ny, nz) global (duplicates agree)."""
    out = np.empty((nx, ny, field.shape[-1]))
    for e in range(field.shape[0]):
        out[ix[e][:, None], iy[e][None, :], :] = field[e]
    return out


def q_criterion(s, Uc):
    """Q = 0.5(|W|^2 - |S|^2) from the velocity-gradient tensor, mode space."""
    D, fx, fy, kz = s['D'], s['m'].facx, s['m'].facy, s['kz']
    u, v, w = (Uc[..., f, :] for f in (OP.U_, OP.V_, OP.W_))
    gx = [DV.ddx(f, D, fx) for f in (u, v, w)]
    gy = [DV.ddy(f, D, fy) for f in (u, v, w)]
    gz = [FR.ddz(f, kz) for f in (u, v, w)]
    A = [[FR.to_physical(g[i], s['nz']) for g in (gx, gy, gz)]
         for i in range(3)]                      # A[i][j] = du_i/dx_j, physical
    Q = np.zeros(A[0][0].shape)
    for i in range(3):
        for j in range(3):
            S = 0.5*(A[i][j] + A[j][i])
            W = 0.5*(A[i][j] - A[j][i])
            Q += 0.5*(W**2 - S**2)
    return Q


def _arr(name, data, ncomp=1):
    flat = np.asarray(data, dtype=np.float32).reshape(-1)
    body = '\n'.join(' '.join(f'{v:.7g}' for v in flat[k:k+6])
                     for k in range(0, len(flat), 6))
    return (f'<DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="{ncomp}" format="ascii">\n{body}\n'
            f'</DataArray>\n')


def write_vtr(path, xs, ys, zs, fields):
    """fields: list of (name, ncomp, array) with array shape (nx,ny,nz[,ncomp]).
    VTK expects x fastest -- arrays are transposed to (z, y, x[, comp])."""
    nx, ny, nz = len(xs), len(ys), len(zs)
    with open(path, 'w') as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="RectilinearGrid" version="0.1" '
                'byte_order="LittleEndian">\n'
                f'<RectilinearGrid WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n'
                f'<Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n')
        f.write('<Coordinates>\n')
        for nm, cs in (('x', xs), ('y', ys), ('z', zs)):
            f.write(_arr(nm, cs))
        f.write('</Coordinates>\n<PointData>\n')
        for name, ncomp, a in fields:
            if ncomp == 1:
                f.write(_arr(name, a.transpose(2, 1, 0), 1))
            else:
                f.write(_arr(name, a.transpose(2, 1, 0, 3), ncomp))
        f.write('</PointData>\n</Piece>\n</RectilinearGrid>\n</VTKFile>\n')


def wrapz(a):
    """Append the periodic z = Lz plane (copy of z = 0) along axis 2."""
    return np.concatenate([a, a[:, :, :1]], axis=2)


def convert(tag, lo=0, hi=None):
    cfg = G.CASES[tag]
    s = G.setup(N=cfg['N'], ex=cfg['ex'], ey=cfg['ey'], nz=cfg['nz'],
                nu=cfg['nu'])
    m = s['m']
    xs, ys, ix, iy = global_axes(m)
    nx, ny = len(xs), len(ys)
    zs = np.concatenate([s['zpl'], [s['lz']]])       # wrap plane appended
    frames = sorted(glob.glob(f'{SC}/tgv_frames_{tag}/frame_*.npz'))[lo:hi]
    outdir = f'{SC}/tgv_vtk_{tag}'
    os.makedirs(outdir, exist_ok=True)
    entries = []
    for f in frames:
        z = np.load(f)
        Uc = z['U'].astype(np.complex128)
        t = float(z['t'])
        P = FR.to_physical(Uc, s['nz'])               # (nelem,n,n,7,nzp)
        Q = q_criterion(s, Uc)

        def grid(local):                              # local (nelem,n,n,nzp)
            return wrapz(scatter_to_grid(local, ix, iy, nx, ny))

        vel = np.stack([grid(P[..., f_, :]) for f_ in
                        (OP.U_, OP.V_, OP.W_)], axis=-1)
        vor = np.stack([grid(P[..., f_, :]) for f_ in
                        (OP.OX_, OP.OY_, OP.OZ_)], axis=-1)
        fields = [('velocity', 3, vel),
                  ('vorticity', 3, vor),
                  ('pressure', 1, grid(P[..., OP.P_, :])),
                  ('omega_mag', 1, np.sqrt((vor**2).sum(-1))),
                  ('Q', 1, grid(Q))]
        idx = int(os.path.basename(f).split('_')[1].split('.')[0])
        name = f'tgv_{tag}_{idx:04d}.vtr'
        write_vtr(f'{outdir}/{name}', xs, ys, zs, fields)
        entries.append((t, name))
        print(f'{name}  t = {t:.3f}  grid {nx}x{ny}x{len(zs)}', flush=True)
    # .pvd collection: true time values on the ParaView slider
    with open(f'{outdir}/tgv_{tag}.pvd', 'w') as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" '
                'version="0.1" byte_order="LittleEndian">\n<Collection>\n')
        for t, name in entries:
            f.write(f'<DataSet timestep="{t:.6f}" group="" part="0" '
                    f'file="{name}"/>\n')
        f.write('</Collection>\n</VTKFile>\n')
    print(f'\nwrote {len(entries)} .vtr files + tgv_{tag}.pvd in {outdir}')
    print(f'Open {outdir}/tgv_{tag}.pvd in ParaView; isosurface Q or '
          f'omega_mag, colour by velocity magnitude.')


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 're100'
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3]) + 1 if len(sys.argv) > 3 else None
    convert(tag, lo, hi)
