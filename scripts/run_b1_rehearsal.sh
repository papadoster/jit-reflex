#!/usr/bin/env bash
# B1-P + B1-R on the Mac CPU, fully offline (~2.5 h): world models, the prediction-error table with the calibration
# decision, and a tiny closed-loop grid on dev seed 99 (NOT the B1 seeds 10-12). The closed-loop numbers are
# exploratory: they check the pipeline and decide nothing (spec B1 4.2-4.3). Keep the Mac on power, lid open.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_OFFLINE=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
B=results/b1
OUT=$B/rehearsal
mkdir -p "$OUT"
start=$(date +%s)
caffeinate -i uv run src/predictors.py train --out-dir $B/world_models 2>&1 | tee "$OUT/train.txt"
caffeinate -i uv run src/predictors.py errors --world-model-dir $B/world_models --out $B/errors.csv 2>&1 \
  | tee "$OUT/errors.txt"
eval_run() {  # eval_run <dir> <eval_flow args...>
  local dir=$1; shift
  caffeinate -i uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 8 --seeds 99 \
    --world-model-dir $B/world_models --output-dir "$OUT/$dir" "$@" 2>&1 \
    | grep --line-buffered -v prefix_attention_horizon | tee "$OUT/$dir.log"
}
eval_run eval_d3 --methods naive realtime pred reflex rtc_reflex --predictors oracle phys0.2 learned --delays 3 --horizons 5
eval_run eval_d1 --methods naive realtime pred reflex --predictors oracle phys0.2 learned --delays 1 --horizons 1 7
eval_run eval_d1_rtc --methods rtc_reflex --predictors oracle phys0.2 learned --delays 1 --horizons 7
uv run src/plot.py b1 --results-glob "$OUT/eval*/results.csv" --errors-csv $B/errors.csv --out-dir "$OUT" \
  --p-mid phys0.2 2>&1 | tee "$OUT/b1.txt"
echo "B1 rehearsal done in $(( ($(date +%s) - start) / 60 )) min"
