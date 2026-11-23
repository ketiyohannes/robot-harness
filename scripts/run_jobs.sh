#!/usr/bin/env bash
# Run one shell command per line of a job file, at most N at a time; each job logs to results/logs/.
# Usage: scripts/run_jobs.sh jobs.txt [N]
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/logs
N=${2:-8}
i=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  i=$((i + 1))
  log="results/logs/$(echo "$cmd" | tr -c 'A-Za-z0-9_' '_' | cut -c1-120).log"
  echo "[$(date +%H:%M:%S)] start $i: $cmd -> $log"
  bash -c "$cmd" > "$log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 5; done
done < "$1"
wait
echo "[$(date +%H:%M:%S)] all jobs done"
