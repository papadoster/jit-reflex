#!/usr/bin/env bash
# B2+B5 on a Linux NVIDIA GPU host (spec docs/superpowers/specs/2026-09-26-b2b5-gpu-design.md, §10). The pod clones the
# public repo at the run's commit, so the Mac uploads nothing. Run it inside tmux: a lost SSH session kills the workers.
#   tmux new -s b2b5   (after a lost SSH session: tmux attach -t b2b5)
#   git clone https://github.com/papadoster/jit-reflex.git && cd jit-reflex && git checkout <commit> && ./scripts/gpu_b2b5.sh
# Writes results/b2b5/** and packs it (with the trained heads, without the expert data) into b2b5_results.tgz.
# Resume: after a crash just rerun it; latency, trained heads and finished eval configs are skipped. PB = the Jacobian
# batch (16; a failed block is retried with 4). Progress and hours left:
#   JAX_PLATFORMS=cpu uv run src/b2b5.py forecast --cut $CUT
# Spec §10 fuse: if forecast > 20 h: stop the workers (Ctrl-C), then rerun with CUT=1..4:
#   CUT=1 ./scripts/gpu_b2b5.sh
# The spec's cut steps 1..CUT leave the commands (the lock and the grid stay as they are); summarize lists them as
# "cut". Service calls of b2b5.py (all but latency) run on the CPU: no GPU memory next to the workers.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -n "${TMUX:-}" ] || { echo "!!! not inside tmux: a lost SSH session kills the workers (tmux new -s b2b5)"; sleep 10; }
O=results/b2b5
CUT=${CUT:-0}
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
b2b5() { JAX_PLATFORMS=cpu uv run src/b2b5.py "$@"; }  # service calls: on the CPU, no GPU memory next to the workers
LEVELS="grasp_easy catapult cartpole_thrust hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker h17_unicycle
  chain_lander catcher_v3 trampoline car_launch"
mkdir -p checkpoints/bc/31/policies
for L in $LEVELS; do
  f=checkpoints/bc/31/policies/worlds_l_$L.pkl
  [ -f "$f" ] || curl -fsSL -o "$f" "https://storage.googleapis.com/rtc-assets/bc/31/policies/worlds_l_$L.pkl"
done
sha256sum checkpoints/bc/31/policies/*.pkl > $O/policies.sha256  # the Mac compares them with its local copies
uv run python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'no GPU visible'" || exit 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1
b2b5 lock --check || exit 1
[ -f $O/latency.json ] || { uv run src/b2b5.py latency 2>&1 | tee $O/latency.txt || exit 1; }
b2b5 place 2>&1 | tee -a $O/latency.txt || exit 1
# Spec §14: an A2C2 failure does not stop the pod. eval_flow drops a head method whose heads are missing and runs the
# rest; strict summarize then lists the missing configs. Rerun the script to retry a failed level: the finished levels
# are skipped.
echo "[$(date +%T)] A2C2 (expert data, one level at a time)"
mkdir -p $O/expert
for L in $LEVELS; do
  [ -f $O/a2c2/worlds_l_$L.pkl ] && continue
  f=$O/expert/worlds_l_$L.npz
  curl -fsSL --retry 3 -o $f.part "https://storage.googleapis.com/rtc-assets/expert/data/worlds_l_$L.npz" \
    && mv $f.part $f && sha256sum $f | tee -a $O/expert_data.sha256
  [ -f $f ] && uv run src/a2c2.py expert --data-dir $O/expert --level-paths worlds/l/$L.json 2>&1 \
    | tee -a $O/a2c2_train.txt
  st=$?
  rm -f $f $f.part  # after every attempt, success or failure: repeated failures cannot fill the disk
  if [ $st = 0 ]; then
    b2b5 lock --add $O/a2c2 || exit 1
  else
    echo "!!! A2C2 for $L failed: the a2c2 configs will be skipped, the rest runs" | tee -a $O/a2c2_train.txt
  fi
done
echo "[$(date +%T)] A2C2-distill"
for L in $LEVELS; do
  [ -f $O/a2c2_distill/worlds_l_$L.pkl ] && continue
  if uv run src/a2c2.py distill --level-paths worlds/l/$L.json 2>&1 | tee -a $O/distill_train.txt; then
    b2b5 lock --add $O/a2c2_distill || exit 1
  else
    echo "!!! A2C2-distill for $L failed: the a2c2_distill configs will be skipped, the rest runs" \
      | tee -a $O/distill_train.txt
  fi
done
b2b5 lock --add $O/a2c2 $O/a2c2_distill || exit 1  # a head saved right before a crash, unrecorded
EVAL="src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --world-model-dir results/b1/world_models
  --e-std $O/e_std.npz --heads-root $O"
# run_line <worker> <eval_flow block args>; a failure (out of memory) -> one retry with a small Jacobian batch. The
# pipeline's status is eval_flow's: grep filtering every line exits 1, hence `|| true` (a spurious retry would only
# cost time, as eval_flow skips the finished configs). uv gets /dev/null so it cannot eat the callers' `while read`.
run_line() {
  local w=$1; shift
  JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_$w uv run $EVAL --package-batch ${PB:-16} "$@" </dev/null 2>&1 \
    | { grep --line-buffered -v prefix_attention_horizon || true; } >> $O/eval_$w.log && return
  echo "[$(date +%T)] retry with --package-batch 4: $*" >> $O/eval_$w.log
  JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_$w uv run $EVAL --package-batch 4 "$@" </dev/null 2>&1 \
    | { grep --line-buffered -v prefix_attention_horizon || true; } >> $O/eval_$w.log
}
worker() {  # every block of worker $1; finished configs are skipped (resume)
  b2b5 commands --worker $1 --cut $CUT </dev/null | while read -r args; do run_line $1 $args; done
}
echo "[$(date +%T)] grid (CUT=$CUT): two workers, logs $O/eval_A.log and eval_B.log"
echo "  hours left: JAX_PLATFORMS=cpu uv run src/b2b5.py forecast --cut $CUT   (> 20 h: Ctrl-C, rerun with CUT=1..4)"
(export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45; worker A) & (export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45; worker B) & wait
echo "[$(date +%T)] second pass, one process: fills whatever failed"
worker A; worker B
b2b5 lock --check || exit 1
analyze() {  # strict; if configs are missing (e.g. a failed A2C2 head, spec §14) b2b5.txt lists them and the analysis
  # runs again without strict (they give MISSING), so the rules and the GRAY extension still come
  b2b5 summarize --cut $CUT 2>&1 | tee $O/b2b5.txt && return
  echo "!!! strict summarize FAILED (see above): rerun the script, finished configs are skipped" | tee -a $O/b2b5.txt
  b2b5 summarize --no-strict --cut $CUT 2>&1 | tee -a $O/b2b5.txt
}
analyze
if [ -s $O/extension.txt ]; then  # spec §6.5: a GRAY rule gets seeds 23-25 for its configs, then the verdict again
  echo "[$(date +%T)] GRAY extension"
  while read -r w args; do run_line $w $args; done < $O/extension.txt  # EVAL, worker cache, </dev/null
  analyze
fi
rm -rf $O/expert
(cd $O && find . -type f ! -name 'MANIFEST.*' -printf '%s %p\n' | sort -k2 > MANIFEST.sizes \
  && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
tar czf b2b5_results.tgz $O && echo "[$(date +%T)] done: b2b5_results.tgz ($(du -h b2b5_results.tgz | cut -f1))"
echo "On the Mac: tar xzf b2b5_results.tgz && (cd $O && shasum -a 256 -c MANIFEST.sha256) and compare"
echo "  $O/policies.sha256 with: shasum -a 256 checkpoints/bc/31/policies/*.pkl; only then stop the pod"
