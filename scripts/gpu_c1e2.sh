#!/usr/bin/env bash
# C1-E2 on one Linux NVIDIA GPU host (RTX 4090), spec docs/superpowers/specs/2026-09-28-c1-e2-object-reflex-design.md
# sections 6, 7, 11. The pod clones the public repo, so the Mac uploads nothing. Pod: >= 16 vCPUs (the grid runs up to 16
# env worker processes), 24 GB GPU, >= 40 GB disk. Run it inside tmux: a lost SSH session kills the workers.
#   tmux new -s c1e2   (after a lost SSH session: tmux attach -t c1e2)
#   git clone https://github.com/papadoster/jit-reflex.git && cd jit-reflex && git checkout c1-e2
# Stages, each run separately after the owner's go-ahead, in this order:
#   ./scripts/gpu_c1e2.sh smoke      a few minutes: 6 episodes on the debug inits 48-49 (never read), GJ on CUDA included
#   ./scripts/gpu_c1e2.sh base       spec 11(a): stock lerobot-eval, 26 tasks x inits 0-9, s=1 and s=10 -> baseline.json
#   ./scripts/gpu_c1e2.sh pilot      spec 11(b), section 7: T_ramp x K_p on inits 44-47; prints the chosen pair
#   on the Mac: journal the printed line "C1-E2 coefficients: T_ramp=.. K_p=.." in the spec, commit, push; here: git pull
#   ./scripts/gpu_c1e2.sh grid TR KP spec 11(c), section 6: 14456 episodes into results/c1-e2/grid.jsonl
# Every stage first sets up (idempotent: venv/ with pinned versions, LIBERO config, CUDA and EGL checks, pod_env.txt) and
# ends by refreshing results/c1-e2/MANIFEST.sha256 and packing results/c1-e2 into c1e2_results.tgz. Logs:
# results/c1-e2/logs/. Resume: rerun the same stage; the runner skips finished episodes and retries failed batches.
# Download (e.g. scp -P <port> root@<pod ip>:jit-reflex/c1e2_results.tgz . or runpodctl send), then on the Mac:
#   tar xzf c1e2_results.tgz && (cd results/c1-e2 && shasum -a 256 -c MANIFEST.sha256)
# Spec 11 fuse: if the forecast printed by the pilot is > 10 h, the owner cuts lines in grid below (first G+J, then
# control T0/PPC, then smooth and G-post). Cut lines fail the summary's completeness gate, which then needs an
# --override-gate journaled in the spec before reading.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -n "${TMUX:-}" ] || { echo "!!! not inside tmux: a lost SSH session kills the workers (tmux new -s c1e2)"; sleep 10; }
O=results/c1-e2
LOG=$O/logs
P=venv/bin/python
SPEC=docs/superpowers/specs/2026-09-28-c1-e2-object-reflex-design.md
mkdir -p $LOG
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} PYTHONUNBUFFERED=1

setup() {
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
  if [ ! -f venv/c1e2-installed ]; then  # the Mac env (docs/c1/scouting.md section 10) with torch for CUDA 12.6
    # torch 2.11.0 on PyPI is built for CUDA 13 (driver >= 580); RunPod's 4090 drivers are 550-570: cu126 runs there
    W=https://download.pytorch.org/whl/cu126
    PIN=("torch @ $W/torch-2.11.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      "torchvision @ $W/torchvision-0.26.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      transformers==5.5.4 numpy==2.2.6 gymnasium==1.3.0 mujoco==3.3.2)
    [ -x $P ] || uv venv --python 3.12 venv || exit 1
    uv pip install --python $P "lerobot[smolvla]==0.6.1" "${PIN[@]}" || exit 1
    # egl_probe (needs cmake) stays out: only robomimic's own env wrappers import it, LIBERO and lerobot do not
    uv pip install --python $P --no-deps hf-libero==0.1.4 robomimic==0.2.0 || exit 1
    uv pip install --python $P "hydra-core>=1.2,<1.4" robosuite==1.4.0 bddl==1.0.1 easydict einops thop matplotlib \
      cloudpickle opencv-python future "lerobot[dataset,scipy-dep]==0.6.1" "${PIN[@]}" || exit 1
    touch venv/c1e2-installed
  fi
  # hf-libero asks for a dataset path on its first import (answer N) and fetches its assets on first use: both here,
  # once, not in 16 env workers at a time
  echo N | $P -c "import os, libero.libero as L; b = L.get_libero_path('bddl_files')
assert os.path.isdir(b), f'stale ~/.libero/config.yaml: {b}'; print('LIBERO assets:', L.get_assets_path())" || exit 1
  $P -c "import torch; assert torch.cuda.is_available(), 'CUDA is not visible to torch'
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))" || exit 1
  local probe='import os, cv2, mujoco
mujoco.GLContext(64, 64).make_current()
try:  # the renderer name only informs (and catches a CPU renderer); the context above is the check
    from OpenGL import GL; r = GL.glGetString(GL.GL_RENDERER).decode()
except Exception as e:
    r = f"unknown ({e!r})"
print("GL renderer:", r)
assert os.environ["MUJOCO_GL"] != "egl" or not any(s in r.lower() for s in ("llvmpipe", "softpipe", "swrast")), \
    "EGL renders on the CPU"'
  if ! $P -c "$probe" </dev/null; then
    echo "offscreen rendering failed: installing the GL libraries and retrying"
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libopengl0 libglib2.0-0 \
      </dev/null
    $P -c "$probe" </dev/null || {
      echo "!!! offscreen $MUJOCO_GL rendering does not work on this pod (see above). Often the pod exposes only the"
      echo "    driver's compute capability (NVIDIA_DRIVER_CAPABILITIES without graphics). The owner decides: another"
      echo "    pod or template, or the much slower CPU fallback osmesa:"
      echo "    apt-get install -y libosmesa6 && MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ./scripts/gpu_c1e2.sh <stage>"
      exit 1
    }
  fi
  { date -u; git rev-parse HEAD; echo "vCPUs: $(nproc)"; nvidia-smi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    $P -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"; uv pip freeze --python $P
  } > $O/pod_env.txt 2>&1
  grep -A1 driver_version $O/pod_env.txt
}

pack() {
  (cd $O && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  tar czf c1e2_results.tgz $O && echo "[$(date +%T)] packed c1e2_results.tgz ($(du -h c1e2_results.tgz | cut -f1))"
}

# The code that makes the episodes (not docs, the summary or this script, so a journal commit and the fuse's cuts pass).
code_hash() { git rev-parse HEAD:src HEAD:scripts/c1e2_run.py HEAD:scripts/c1e2_j.py 2>/dev/null; }
check_code() {  # check_code record|require
  git diff --quiet HEAD -- src scripts/c1e2_run.py scripts/c1e2_j.py \
    || { echo "!!! uncommitted changes in the runner code: commit them on the Mac and pull, or reset them"; exit 1; }
  if [ ! -f $O/CODE ]; then
    [ "$1" = record ] || { echo "!!! no $O/CODE: run the pilot first (pilot and grid must run the same code)"; exit 1; }
    code_hash > $O/CODE
  fi
  [ "$(cat $O/CODE)" = "$(code_hash)" ] || {
    echo "!!! the runner code differs from $O/CODE (the pilot's): pilot and grid must run the same code."
    echo "    Rerun the pilot (delete $O/CODE and $O/pilot_*.jsonl), or delete $O/CODE deliberately."; exit 1; }
}

run2() {  # run2 <log> <runner args>: every line twice; the second pass retries failed batches (the runner resumes)
  local log=$LOG/$1.log pass; shift
  for pass in 1 2; do
    echo "[$(date +%T)] pass $pass: $*" | tee -a "$log"
    $P scripts/c1e2_run.py --vector async --device cuda "$@" </dev/null 2>&1 | tee -a "$log" \
      || echo "!!! runner exited with an error (pass $pass)" | tee -a "$log"
  done
}

rate() { grep -o '[0-9.]* s/episode' "$1" | tail -1; }

case "${1:-}" in
  smoke | base | pilot | grid) ;;
  *) echo "usage: $0 smoke | base | pilot | grid T_RAMP K_P"; exit 1 ;;
esac
setup 2>&1 | tee -a $LOG/setup.log
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
case "${1:-}" in
  smoke)
    rm -f $O/smoke.jsonl $LOG/smoke.log
    $P scripts/c1e2_run.py --tasks libero_spatial:0 --cells A --methods none,G,GJ --kinds step --inits 48-49 \
      --n-envs 2 --vector async --device cuda --out $O/smoke.jsonl </dev/null 2>&1 | tee -a $LOG/smoke.log
    n=$(cat $O/smoke.jsonl 2>/dev/null | wc -l); f=$(grep -c FAILED $LOG/smoke.log)
    echo "[$(date +%T)] smoke: $n of 6 records, $f FAILED; runner $(rate $LOG/smoke.log) (2 envs, GJ included)"
    pack
    [ "$n" -eq 6 ] && [ "$f" -eq 0 ] || { echo "!!! smoke failed: see $LOG/smoke.log"; exit 1; }
    ;;
  base)  # stock lerobot-eval: n_episodes is per task; in a batch of 10 async envs env j starts from init state j
    # (LiberoEnv episode_index), so every task gets inits 0-9. Its videos stay out of the results (eval_output/).
    mkdir -p $O/base
    for s in 1 10; do
      for suite in libero_spatial libero_object libero_goal; do
        d=s${s}_$suite
        [ -f $O/base/$d/eval_info.json ] && continue
        ids=$([ $suite = libero_goal ] && echo "[1,2,4,6,8,9]" || echo "[0,1,2,3,4,5,6,7,8,9]")
        rm -rf eval_output/c1e2/$d
        echo "[$(date +%T)] lerobot-eval $d" | tee -a $LOG/base.log
        venv/bin/lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --policy.device=cuda \
          --policy.n_action_steps=$s --env.type=libero --env.task=$suite --env.task_ids="$ids" \
          --eval.n_episodes=10 --eval.batch_size=10 --output_dir=eval_output/c1e2/$d </dev/null 2>&1 \
          | tee -a $LOG/base.log || { echo "!!! lerobot-eval $d failed: rerun base"; exit 1; }
        mkdir -p $O/base/$d && cp eval_output/c1e2/$d/eval_info.json $O/base/$d/ || exit 1
      done
    done
    $P scripts/c1e2_pod.py baseline $O/base/s*_* > $O/baseline.json || exit 1
    cat $O/baseline.json
    pack
    ;;
  pilot)
    check_code record
    for tr in 1 5 10; do
      for kp in 0.3 1.0; do
        run2 pilot --cells C --methods G --kinds step --inits 44-47 --t-ramp $tr --k-p $kp --n-envs 4 \
          --out $O/pilot_tr${tr}_kp${kp}.jsonl
      done
    done
    $P scripts/c1e2_pod.py pilot $O/pilot_tr*_kp*.jsonl | tee $O/pilot_choice.txt
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    r=$(rate $LOG/pilot.log); r=${r% s/episode}
    h=$(awk -v r="${r:-0}" 'BEGIN { printf "%.1f", r * 14456 / 3600 }')
    echo "Rough grid forecast: ${r:-?} s/episode x 14456 = $h h (rough: the pilot batches 4 envs, the grid 10-16;" \
      "the G+J line is slower, spec: about +0.5 h). Spec 11 fuse: > 10 h -> the owner cuts lines." \
      | tee -a $O/pilot_choice.txt
    pack
    ;;
  grid)
    TR=${2:?usage: grid T_RAMP K_P}; KP=${3:?usage: grid T_RAMP K_P}
    # spec section 7: the chosen coefficients are journaled and pushed before the grid; the last marker in the spec counts
    m=$(git show HEAD:$SPEC | grep -oE 'C1-E2 coefficients: T_ramp=[0-9]+ K_p=[0-9.]+' | tail -1)
    [ "$m" = "C1-E2 coefficients: T_ramp=$TR K_p=$KP" ] || {
      echo "!!! the spec at HEAD journals '${m:-no coefficients}', not 'C1-E2 coefficients: T_ramp=$TR K_p=$KP'."
      echo "    Journal the pilot's line in the spec on the Mac, commit, push, then here: git pull"; exit 1; }
    check_code require
    # --n-envs: the largest divisor <= 16 of the line's group (methods x inits per task, cell and kind): no idle envs
    CO=(--t-ramp "$TR" --k-p "$KP" --out $O/grid.jsonl)
    run2 grid --kinds step --cells A,B,C,D,E,F,Gp --methods none,T0,G,GT,PPC --inits 0-9 --headroom --n-envs 10 "${CO[@]}"
    run2 grid --kinds step --cells A --methods none --inits 10-39 --n-envs 15 "${CO[@]}"
    run2 grid --kinds step --cells C --methods GT --inits 10-39 --n-envs 15 "${CO[@]}"
    run2 grid --kinds control --cells A,B,C,D,E,F,Gp --methods none,G,T0,PPC --inits 40-43 --n-envs 16 "${CO[@]}"
    run2 grid --kinds step --cells C --methods Gpost --inits 0-9 --n-envs 10 "${CO[@]}"
    run2 grid --kinds smooth --cells C --methods none,G,T0,PPC,PPC9 --inits 0-3 --n-envs 10 "${CO[@]}"
    run2 grid --kinds step --cells A --methods GJ --inits 0-3 --n-envs 4 "${CO[@]}"
    echo "[$(date +%T)] grid.jsonl: $(cat $O/grid.jsonl 2>/dev/null | wc -l) of 14456 records (the summary checks each)"
    for f in $LOG/*.log; do echo "  $f: $(grep -c FAILED "$f") FAILED lines (first-pass failures included)"; done
    echo "On the Mac: python scripts/c1e2_summary.py $O/grid.jsonl --baseline $O/baseline.json"
    pack
    ;;
esac
