#!/bin/bash
while pgrep -f fs_minchan_stats >/dev/null; do sleep 60; done
sleep 10
cd /Users/danielchan/sem_fs_wt
nohup /Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/.venv/bin/python -u \
  scratch/fs_km_econ.py > scratch/_km_econ/console.log 2>&1 &
echo "KM+E TGV launched pid $!"
