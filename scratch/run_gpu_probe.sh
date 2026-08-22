#!/bin/bash
# Deploy and run the FP64 probe on the DGX Spark (or any ssh host).
#
#   bash scratch/run_gpu_probe.sh [host]
#
# Requires working ssh auth.  As of 2026-08-22 `ssh spark-b85b` returns
# "Permission denied (publickey,password)" -- the NVIDIA Sync pairing has
# lapsed or the key rotated.  Re-pair in NVIDIA Sync, then run this.
set -euo pipefail
HOST="${1:-spark-b85b}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== checking auth to ${HOST}"
ssh -o BatchMode=yes -o ConnectTimeout=10 "${HOST}" true || {
  echo "ssh auth to ${HOST} failed -- re-pair in NVIDIA Sync, or add a key with ssh-add"
  exit 1
}
echo "== copying probe"
scp -q "${HERE}/gpu_fp64_probe.py" "${HOST}:/tmp/gpu_fp64_probe.py"
echo "== running (needs numpy; uses torch or cupy if present)"
ssh "${HOST}" 'python3 /tmp/gpu_fp64_probe.py' | tee "${HERE}/gpu_fp64_probe_${HOST}.log"
echo "== saved to scratch/gpu_fp64_probe_${HOST}.log"
