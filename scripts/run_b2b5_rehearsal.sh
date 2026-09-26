#!/usr/bin/env bash
# B2+B5 rehearsal on the Mac CPU (spec §11): every stage of gpu_b2b5.sh at toy scale, seed 99, fully offline. The
# numbers mean nothing: this checks the pipeline end to end. Keep the Mac on power, lid open. Needs the committed
# results/b2b5/e_std.npz and lock.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_OFFLINE=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
O=results/b2b5
R=$O/rehearsal
mkdir -p $R
start=$(date +%s)
uv run src/b2b5.py lock --check
caffeinate -i uv run src/b2b5.py latency --out-dir $R --repeats 5 --warmup 1 2>&1 | tee $R/latency.txt
uv run src/b2b5.py place --out-dir $R | tee -a $R/latency.txt
L=worlds/l/catapult.json
caffeinate -i uv run src/a2c2.py synth --level-path $L --out-dir $R/expert
caffeinate -i uv run src/a2c2.py expert --data-dir $R/expert --level-paths $L --out-dir $R/a2c2 --num-epochs 2 2>&1 \
  | tee $R/a2c2_train.txt
caffeinate -i uv run src/a2c2.py distill --level-paths $L --out-dir $R/a2c2_distill --num-envs 16 --num-chunks 80 \
  2>&1 | tee $R/distill_train.txt
EVAL="src/eval_flow.py --run-path checkpoints/bc --config.num-evals 8 --seeds 99
  --world-model-dir results/b1/world_models --e-std $O/e_std.npz"
ev() {  # the status is eval_flow's (pipefail); grep filtering every line exits 1, hence `|| true`
  caffeinate -i uv run $EVAL "$@" 2>&1 | { grep --line-buffered -v prefix_attention_horizon || true; } \
    | tee -a $R/eval.log
}
ev --methods naive realtime realtime10 --cells 3,5 --output-dir $R/eval_A
# the heads exist for catapult only: their own dir, as eval_flow's resume keeps only the configs with as many level
# rows as this call has levels (1 here), so sharing eval_A would drop the 12-level rows
ev --methods a2c2 a2c2_distill --cells 3,5 --heads-root $R --level-paths $L --output-dir $R/eval_heads
ev --methods rtc_reflex --predictors learned --cells 3,5 --output-dir $R/eval_A
ev --methods pred reflex t3 m3 --predictors learned --cells 3,5 --output-dir $R/eval_B
ev --methods t3 m3 --predictors learned --cells 1,7 --output-dir $R/eval_B
ev --methods late1 --predictors learned --cells 4,4 --output-dir $R/eval_B
uv run src/b2b5.py forecast --out-dir $R
uv run src/b2b5.py summarize --out-dir $R --no-strict --seeds 99 2>&1 | tee $R/b2b5.txt
echo "B2+B5 rehearsal done in $(( ($(date +%s) - start) / 60 )) min"
