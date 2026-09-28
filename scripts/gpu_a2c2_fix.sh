#!/usr/bin/env bash
# A2C2 fix rerun on a Linux NVIDIA GPU host (spec docs/superpowers/specs/2026-09-28-a2c2-fix-design.md, §5). Run it
# inside tmux: a lost SSH session kills the run.
#   tmux new -s a2c2fix
#   git clone https://github.com/papadoster/jit-reflex.git && cd jit-reflex && git checkout <commit> && ./scripts/gpu_a2c2_fix.sh
# Writes results/b2b5_fix/** and packs it into a2c2_fix_results.tgz WITHOUT the head weights (~7 MB each): their
# SHA-256 are in heads.sha256; fetch them separately only if needed. Resume: rerun it, finished stages are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -n "${TMUX:-}" ] || { echo "!!! not inside tmux: a lost SSH session kills the run (tmux new -s a2c2fix)"; sleep 10; }
O=results/b2b5_fix
mkdir -p $O
HEAD_SHA=$(git rev-parse HEAD)
if [ -f $O/COMMIT ] && [ "$(cat $O/COMMIT)" != "$HEAD_SHA" ]; then
  echo "!!! $O/COMMIT is $(cat $O/COMMIT) but HEAD is $HEAD_SHA: never mix code versions on resume"; exit 1
fi
echo $HEAD_SHA > $O/COMMIT
KINETIX_SHA=cf7453ea103fa0b77348af1a39f689c658161613
if [ ! -d third_party/kinetix/kinetix ]; then
  rm -rf third_party/kinetix
  git clone -q https://github.com/FLAIROx/Kinetix.git third_party/kinetix && git -C third_party/kinetix checkout -q $KINETIX_SHA
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
uv sync || exit 1
LEVELS="grasp_easy catapult cartpole_thrust hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker h17_unicycle
  chain_lander catcher_v3 trampoline car_launch"
mkdir -p checkpoints/bc/31/policies checkpoints/expert
for L in $LEVELS; do
  f=checkpoints/bc/31/policies/worlds_l_$L.pkl
  [ -f "$f" ] || curl -fsSL -o "$f" "https://storage.googleapis.com/rtc-assets/bc/31/policies/worlds_l_$L.pkl"
done
sha256sum checkpoints/bc/31/policies/*.pkl > $O/policies.sha256
JAX_PLATFORMS=cpu uv run src/a2c2_fix.py experts | while read -r L path url; do
  [ -f "$path" ] || curl -fsSL -o "$path" "$url" || { echo "!!! expert for $L failed"; exit 1; }
done || exit 1
JAX_PLATFORMS=cpu uv run src/a2c2_fix.py experts | awk '{print $2}' | xargs sha256sum > $O/experts.sha256
uv run python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'no GPU visible'" || exit 1
export PYTHONUNBUFFERED=1
[ -f $O/head_latency.json ] || uv run src/a2c2_fix.py latency --out-dir $O || exit 1
echo "[$(date +%T)] relabel (12 levels)"
JAX_PLATFORMS=cpu uv run src/a2c2_fix.py experts | while read -r L path url; do
  [ -f $O/a2c2_paper/worlds_l_$L.pkl ] && continue
  uv run src/a2c2.py relabel --level-path worlds/l/$L.json --experts "$path" --out-dir $O/a2c2_paper --paper \
    --num-chunks 256 </dev/null 2>&1 | tee -a $O/relabel_train.txt
  [ -f $O/a2c2_paper/worlds_l_$L.pkl ] && sha256sum $O/a2c2_paper/worlds_l_$L.pkl >> $O/heads.sha256 \
    || echo "!!! relabel for $L failed: its configs will be missing" | tee -a $O/relabel_train.txt
done
echo "[$(date +%T)] grid: a2c2_paper, 16 cells x seeds 20-22"
CELLS="1,1 1,2 1,3 1,4 1,5 1,6 1,7 2,2 2,3 2,4 2,5 2,6 3,3 3,4 3,5 4,4"
EVAL="src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --methods a2c2_paper --cells $CELLS
  --seeds 20 21 22 --heads-root $O --output-dir $O/eval"
uv run $EVAL 2>&1 | grep --line-buffered -v prefix_attention_horizon >> $O/eval.log \
  || { echo "[$(date +%T)] retry" >> $O/eval.log; uv run $EVAL 2>&1 | grep --line-buffered -v prefix_attention_horizon >> $O/eval.log; }
echo "[$(date +%T)] summary"
JAX_PLATFORMS=cpu uv run src/a2c2_fix.py summarize --new-dir $O 2>&1 | tee $O/fix_summary.txt \
  || { echo "!!! strict summary FAILED (see above)" | tee -a $O/fix_summary.txt;
       JAX_PLATFORMS=cpu uv run src/a2c2_fix.py summarize --new-dir $O --no-strict 2>&1 | tee -a $O/fix_summary.txt; }
(cd $O && find . -type f ! -name 'MANIFEST.*' ! -name '*.pkl' -printf '%s %p\n' | sort -k2 > MANIFEST.sizes \
  && find . -type f ! -name MANIFEST.sha256 ! -name '*.pkl' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
tar czf a2c2_fix_results.tgz --exclude='*.pkl' $O && echo "[$(date +%T)] done: a2c2_fix_results.tgz ($(du -h a2c2_fix_results.tgz | cut -f1))"
echo "On the Mac: tar xzf a2c2_fix_results.tgz && (cd $O && shasum -a 256 -c MANIFEST.sha256); only then stop the pod"
