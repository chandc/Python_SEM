import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='2'
sys.path.insert(0,'.')
import numpy as np


def main():
    import lssem3d
    from lssem3d import deriv as DV
    from lssem2d.lgl import diff_matrix
    from lssem2d.mesh import build_bfs
    chk, out, er = sys.argv[1], sys.argv[2], float(sys.argv[3])
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 7
    lout = float(sys.argv[5]) if len(sys.argv) > 5 else 18.0
    H_in = 1.0; S_h = H_in*(er - 1.0)
    m = build_bfs(N, E_in_x=3, E_out_x=16, E_y=3, L_in=2.0, L_out=lout,
                  H_in=H_in, H_step=S_h, xpow=1.6)
    D = diff_matrix(N)
    X = np.array([m.xnod[e] for e in range(m.nelem)])
    last_m = 0
    f = open(out, 'a')
    while True:
        try:
            mt = os.path.getmtime(chk)
        except FileNotFoundError:
            time.sleep(60); continue
        if mt > last_m:
            time.sleep(3)
            try:
                d = np.load(chk)
                U, t = d['U'], float(d['t'])
            except Exception:
                time.sleep(20); continue
            last_m = mt
            try:
                du = DV.ddy(U[..., 0:1, :], D, m.facy)[..., 0, 0].real/4
            except Exception as ex:
                f.write(f'# error {ex}\n'); f.flush()
                time.sleep(120); continue
            xs, tw = [], []
            for e in range(m.nelem):
                if m.bc[e, 2] == 1 and abs(m.y0[e]) < 1e-12 and m.x0[e] >= -1e-12:
                    for i in range(N + 1):
                        xs.append(X[e][i]); tw.append(du[e, i, 0])
            xs, tw = np.array(xs), np.array(tw)
            o = np.argsort(xs); xs, tw = xs[o], tw[o]
            sgn = np.sign(tw)
            cr = [f'{xs[i]/S_h:.2f}' for i in
                  np.flatnonzero(sgn[:-1]*sgn[1:] < 0) if xs[i] > 0.3*S_h]
            f.write(f'{t:.2f},' + ';'.join(cr) + '\n'); f.flush()
        time.sleep(120)


if __name__ == '__main__':
    main()
