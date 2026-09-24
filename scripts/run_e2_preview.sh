#!/usr/bin/env bash
# Mini-E2 on the Mac CPU, fully offline (~1 h): 12 levels, 64 episodes, seed 0, d=2, s=6; naive / realtime / pred / reflex.
# A free early signal before renting a GPU; NOT the Gate 2 grid (that is Task 12). Keep the Mac on power, lid open.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_OFFLINE=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
OUT=${OUT:-results/e2_preview}
mkdir -p "$OUT"
start=$(date +%s)
caffeinate -i uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 64 --seeds 0 \
  --methods naive realtime pred reflex --delays 2 --horizons 6 --output-dir "$OUT" "$@" 2>&1 \
  | grep --line-buffered -v "execute_horizon=.*prefix_attention_horizon" | tee "$OUT/log.txt"
uv run src/plot.py table --results-glob "$OUT/results.csv" 2>&1 | tail -8 | tee -a "$OUT/log.txt"
echo "E2 preview done in $(( ($(date +%s) - start) / 60 )) min"
