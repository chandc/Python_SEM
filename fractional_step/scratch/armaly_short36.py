"""The short domain AS CHAN & MITTAL SPECIFY IT: 36 elements, not 72.

Chan & Mittal, "Large-eddy simulation of a backward facing step flow using a
least-squares spectral element method", CTR Proc. Summer Program 1996, p.351:

    "The total number of elements is 72 for the long domain and 36 for the
     short domain."
    "For the short domain case, having a reverse flow on part of the outflow
     boundary does not present numerical convergence problem."

Our armaly_er194_short_grid.dat was built with NX=15 -> 72 elements, i.e. the
LONG domain's element count squeezed into the short extent.  That is 2.7x finer
per step height than the long grid and about 2x Chan's short grid.  It reports a
reattachment at x_r/S = 5.174 -- a fabricated closure just inside the outlet,
where Chan reports reverse flow still crossing the outflow plane.

This runs the 36-element grid (NX=6, NXIN=6) at otherwise identical settings.
Saved to armaly_short36_{pz,free}.npz.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import armaly_run as A

A.GRIDS['short36'] = 'grids/armaly_er194_short36_grid.dat'

if __name__ == '__main__':
    print(f"CHAN-SPEC short domain: 36 elements, x in [-2,5], ER 1.94, "
          f"nu = {A.NU:.6e} (Armaly D=2h, Re=389)")
    print("reference: 72-elem short grid gave x_r/S = 5.174, du/dy_outlet = +1.411")
    print("           long grid gave x_r/S = 8.145;  Armaly measured 8.05\n")
    for pz in (True, False):
        A.run('short36', pz)
