#!/usr/bin/env bash
# B1 on a Linux NVIDIA GPU host (spec 2026-09-25-b1-staleness-predictors-design.md, section 4.4). Run from the unpacked
# repo root: ./scripts/gpu_b1.sh. The host downloads Kinetix, packages and checkpoints itself and trains its own world
# models (same seeds as on the Mac). Writes results/b1/gpu/** and packs it (without model weights) into b1_results.tgz.
# PHYS = the three calibrated phys levels, middle one = p_mid (spec B1 4.2); PB = the Jacobian batch (4 if tight).
set -uo pipefail
cd "$(dirname "$0")/.."
KINETIX_SHA=cf7453ea103fa0b77348af1a39f689c658161613
if [ ! -d third_party/kinetix/kinetix ]; then
  rm -rf third_party/kinetix
  git clone -q https://github.com/FLAIROx/Kinetix.git third_party/kinetix && git -C third_party/kinetix checkout -q $KINETIX_SHA
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
uv sync || exit 1
mkdir -p checkpoints/bc/31/policies
for L in grasp_easy catapult cartpole_thrust hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker h17_unicycle chain_lander catcher_v3 trampoline car_launch; do
  f=checkpoints/bc/31/policies/worlds_l_$L.pkl
  [ -f "$f" ] || curl -fsSL -o "$f" "https://storage.googleapis.com/rtc-assets/bc/31/policies/worlds_l_$L.pkl"
done
uv run python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'no GPU visible'" || exit 1
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1
G=results/b1/gpu
PB=${PB:-16}
PHYS=${PHYS:?set PHYS to the calibrated phys levels, e.g. PHYS="0.1 0.2 0.3" ./scripts/gpu_b1.sh}
PMID=$(echo $PHYS | awk '{print $2}')
PREDS="oracle $(for p in $PHYS; do printf 'phys%s ' "$p"; done)learned"
MID="oracle phys$PMID learned"
echo "PHYS=$PHYS PMID=$PMID PREDS=$PREDS"
mkdir -p $G
echo "[$(date +%T)] world models"
uv run src/predictors.py train --out-dir $G/world_models 2>&1 | tee $G/train.txt || exit 1
echo "[$(date +%T)] prediction errors"
uv run src/predictors.py errors --phys $PHYS --world-model-dir $G/world_models --out $G/errors.csv 2>&1 \
  | tee $G/errors.txt || exit 1
run() {  # run <dir> <eval_flow args...>; on failure (e.g. out of memory) retry once with a smaller Jacobian batch
  local dir=$1; shift
  echo "[$(date +%T)] $dir"
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 10 11 12 --package-batch $PB \
    --world-model-dir $G/world_models --output-dir $G/$dir "$@" 2>&1 | grep --line-buffered -v prefix_attention_horizon \
    | tee -a $G/$dir.log && return
  echo "[$(date +%T)] $dir failed, retrying with --package-batch 4"
  mv $G/$dir/results.csv $G/$dir/results.try1.csv 2>/dev/null
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 10 11 12 --package-batch 4 \
    --world-model-dir $G/world_models --output-dir $G/$dir "$@" 2>&1 | grep --line-buffered -v prefix_attention_horizon \
    | tee -a $G/$dir.log || { echo "FAILED $dir"; failed=1; }
}
failed=0
run eval_d3 --methods naive realtime pred reflex rtc_reflex --predictors $MID --delays 3 --horizons 5
run eval_d1 --methods naive realtime pred reflex --predictors $PREDS --delays 1
run eval_d1_rtc --methods rtc_reflex --predictors $MID --delays 1
for s in 1 4 7; do
  uv run src/probe.py cost --batch 1 --delay 1 --horizon $s --out $G/cost_b1_s$s.csv 2>&1 | tee -a $G/cost.txt
done
if [ $failed = 1 ]; then
  echo "!!! some eval runs FAILED — the verdict is not computed; rerun the failed dirs" | tee $G/b1.txt
else
  uv run src/plot.py b1 --results-glob "$G/eval*/results.csv" --errors-csv $G/errors.csv --out-dir $G --p-mid phys$PMID \
    2>&1 | tee $G/b1.txt
fi
tar czf b1_results.tgz --exclude='*.pkl' $G && echo "[$(date +%T)] done: b1_results.tgz"
