#!/usr/bin/env bash
# E2 on a Linux NVIDIA GPU host (plan Task 12 + E2b, spec section 5). Run from the unpacked repo root: ./scripts/gpu_e2.sh
# The host downloads Kinetix, packages and checkpoints itself; the Mac only uploads the ~1 MB code archive.
# Writes results/eval/** and packs them into e2_results.tgz. Override the Jacobian batch with PB=4 if memory is tight.
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
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
E=results/eval
PB=${PB:-16}
mkdir -p $E
run() {  # run <dir> <eval_flow args...>; on failure (e.g. out of memory) retry once with a smaller Jacobian batch
  local dir=$1; shift
  echo "[$(date +%T)] $dir"
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 0 1 2 --package-batch $PB \
    --output-dir $E/$dir "$@" 2>&1 | grep --line-buffered -v prefix_attention_horizon && return
  echo "[$(date +%T)] $dir failed, retrying with --package-batch 4"
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 0 1 2 --package-batch 4 \
    --output-dir $E/$dir "$@" 2>&1 | grep --line-buffered -v prefix_attention_horizon || echo "FAILED $dir"
}
METHODS="naive realtime pred reflex reflex_chunk rtc_reflex"
run main --methods $METHODS --delays 1 2 3 4 --minmax
run oracle --methods naive --delays 0 --horizons 1
for mc in 0.1 0.3 1.0; do run mc$mc --methods reflex --delays 2 --horizons 6 --max-correction $mc; done
V=$(uv run src/probe.py kick-speed | awk '/v_med/ {print $3}')
echo "v_med=$V" | tee $E/v_med.txt
for c in 0.5 1 2; do
  STD=$(python3 -c "print($c * $V)")
  run kick$c --config.kick-prob 0.02 --config.kick-std "$STD" --methods $METHODS --delays 2 --horizons 2 6
  run kick$c-oracle --config.kick-prob 0.02 --config.kick-std "$STD" --methods naive --delays 0 --horizons 1
done
uv run src/probe.py cost --batch 1 --out $E/cost_b1.csv | tee $E/cost.txt
uv run src/probe.py cost --batch 256 --out $E/cost_b256.csv | tee -a $E/cost.txt
tar czf e2_results.tgz results/eval && echo "[$(date +%T)] done: e2_results.tgz"
