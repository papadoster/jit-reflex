#!/usr/bin/env bash
# E1 kill-test, fully offline: all 12 levels, writes results/probe/{summary.csv,verdict.json,log.txt}.
# Keep the Mac on power with the lid open; caffeinate stops idle sleep for the duration of the run.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_OFFLINE=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
mkdir -p results/probe
start=$(date +%s)
caffeinate -i uv run src/probe.py run "$@" 2>&1 | tee results/probe/log.txt
echo "E1 done in $(( ($(date +%s) - start) / 60 )) min"
