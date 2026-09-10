import glob, numpy as np, sys
rows = []
for f in sorted(glob.glob('scratch/hm1/*.npz')):
    z = np.load(f, allow_pickle=True); u = z['u']; y = z['y']; sel = (y > 0.75) & (y < 0.95); d = np.diff(u[sel])
    rows.append((f.split('/')[-1][:-4], int(z['N']), str(z['kind']), float(z['dt']), float(z['beta_h2']) if str(z['kind']) == 'hm1' else np.nan, float(z['t']), float(z['rms_u']), float(z['rms_v']), float(np.abs(z['U'][..., 1]).max()), float(np.abs(d).mean()), int((np.sign(d[1:]) != np.sign(d[:-1])).sum()), d.size-1))
print(f'{"run":36s} {"N":>3} {"kind":>6} {"dt":>5} {"beta":>7} {"t_end":>6} {"rms u":>7} {"rms v":>7} {"max|v|":>7} {"zigzag":>7} {"signchg":>8}')
for r in sorted(rows, key=lambda r: (r[1], r[2], -r[3])):
    print(f'{r[0]:36s} {r[1]:3d} {r[2]:>6} {r[3]:5g} {r[4]:7g} {r[5]:6.1f} {r[6]:7.4f} {r[7]:7.4f} {r[8]:7.4f} {r[9]:7.4f} {r[10]:3d}/{r[11]:<3d}')
