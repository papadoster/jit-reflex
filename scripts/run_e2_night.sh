#!/usr/bin/env bash
# E2-night on the Mac CPU, fully offline (~4 h, spec section 5): early signal before the GPU run, not Gate 2.
#   1. v_med calibration  2. kicks c=1 at d=2 s=6  3. no kicks at d=4 s=4  4. rtc_reflex at d=2 s=6 (vs the preview)
# 12 levels x 64 episodes, seed 0. Each step saves after every method; a failed step doesn't stop the next ones.
# Keep the Mac on power with the lid open.
set -uo pipefail
cd "$(dirname "$0")/.."
export UV_OFFLINE=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
OUT=results/e2_night
mkdir -p "$OUT"
start=$(date +%s)
log() { echo "[$(( ($(date +%s) - start) / 60 )) min] $*" | tee -a "$OUT/log.txt"; }
eval_run() {  # eval_run <dir> <eval_flow args...>
  local dir=$1; shift
  log "start $dir"
  caffeinate -i uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 64 --seeds 0 \
    --output-dir "$OUT/$dir" "$@" 2>&1 | grep --line-buffered -v "prefix_attention_horizon" | tee -a "$OUT/log.txt" \
    || log "FAILED $dir"
}

log "1/4 v_med calibration"
V=$(caffeinate -i uv run src/probe.py kick-speed 2>&1 | awk '/v_med/ {print $3}')
if [ -z "$V" ]; then log "FAILED v_med, skipping kicks"; else
  echo "v_med=$V" | tee "$OUT/v_med.txt"
  log "2/4 kicks c=1 (kick_std=$V), d=2 s=6"
  eval_run kick_d2s6 --config.kick-prob 0.02 --config.kick-std "$V" \
    --methods naive realtime pred reflex rtc_reflex --delays 2 --horizons 6
fi
log "3/4 no kicks, d=4 s=4"
eval_run d4s4 --methods naive realtime reflex rtc_reflex --delays 4 --horizons 4
log "4/4 rtc_reflex, d=2 s=6"
eval_run rtc_d2s6 --methods rtc_reflex --delays 2 --horizons 6

uv run src/plot.py table --results-glob "results/e2_*/**/results.csv" 2>&1 | tail -25 | tee -a "$OUT/log.txt"
uv run src/plot.py table --results-glob "results/e2_preview/results.csv" 2>&1 | tail -6 | tee -a "$OUT/log.txt"
log "E2-night done"
