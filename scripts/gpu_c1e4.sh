#!/usr/bin/env bash
# C1-E4 part 1 on one Linux NVIDIA GPU host (RTX 4090), spec docs/superpowers/specs/2026-10-03-c1-e4-design.md sections
# 4.1, 6, 7, 8, 12. Pod and setup as scripts/gpu_c1e3.sh: >= 16 vCPUs, 24 GB GPU, >= 60 GB on /workspace; tmux; a
# Hugging Face token with access to the gated google/paligemma-3b-pt-224 (huggingface-cli login).
#   tmux new -s c1e4   (after a lost SSH session: tmux attach -t c1e4)
#   cd /workspace && git clone https://github.com/papadoster/jit-reflex.git && cd jit-reflex && git checkout c1-e4
# Stages, each run separately after the owner's go-ahead, in this order:
#   ./scripts/gpu_c1e4.sh smoke   first the GPU peak and tempo of pi0.5 in fp32 (6 envs, then 5, 4 while the peak is
#                                 > 23 GB) and in bf16 one row per call (10 envs, repeat check) on full batches, and
#                                 SmolVLA's tempo; then 58 debug episodes (inits 48-49, never read) over every arm, cell
#                                 and code path; the §4.1 rule -> precision_choice.txt; the environment check against
#                                 C1-E2's pod
#   ./scripts/gpu_c1e4.sh base    pi0.5's stock lerobot-eval (n_action_steps 10, 26 tasks x 10) -> baseline_pi05.json;
#                                 SmolVLA's too if env_check.txt is not SAME -> baseline.json
#   ./scripts/gpu_c1e4.sh trial   spec §7: P1-P3 in the chosen precision, the offline choices and the forecast for an
#                                 RTX 4090 and a 48 GB card -> trial_choice.txt
#   on the Mac: journal the printed line "C1-E4 trial: precision=.. ... pad=0" in the spec, commit, push; if the 48 GB
#   card is cheaper, the owner decides first (spec §4.1); here: git pull
#   ./scripts/gpu_c1e4.sh grid    spec §6, §8, §12: the phases of scripts/c1e4_summary.py --plan into grid_pi05.jsonl,
#                                 grid_smolvla.jsonl, grid_smolvla_b.jsonl, joined into grid_all.jsonl. Refuses unless
#                                 the spec at HEAD and trial_choice.txt carry the same trial line, HEAD is pushed, the pod
#                                 has no local edits, a SmolVLA baseline applies and the runner code is the trial's
#                                 (results/c1-e4/pod/CODE). Phase "control" ends with the pi control check (fails: the
#                                 grid stops, spec §8), phase "q0" with the Q0 gate (fails or cannot run: the delay-axis
#                                 ids are held and results/c1-e4/pod/q0_held is written; later grid runs keep holding
#                                 them, spec §8; Q0_HOLD=run runs them, report only).
# One stage at a time on a pod (flock on .git/c1e4-pod.lock). Trial and grid run with HF_HUB_OFFLINE=1.
# Spec §12 fuse: the forecast goes to forecast.txt after each brain's first line; above 32 h the owner cuts report rows,
# in this order: 12 16 11 15 10 9 14 13 17. Rule rows are never cut. To cut: Ctrl-C, then CUT="12 16" ./scripts/gpu_c1e4.sh
# grid (resume skips finished episodes). Cut rows fail the summary's completeness gate, which then needs an
# --override-gate journaled in the spec before reading.
# Every exit appends the environment to results/c1-e4/pod/pod_env_<stage>.txt, refreshes MANIFEST.sha256 and packs
# results/c1-e4/pod into c1e4_results.tgz. Watchdog, two passes and resume as scripts/gpu_c1e3.sh (STALL, default 1200 s).
set -uo pipefail
cd "$(dirname "$0")/.."
[ -n "${TMUX:-}" ] || { echo "!!! not inside tmux: a lost SSH session kills the workers (tmux new -s c1e4)"; sleep 10; }
O=results/c1-e4/pod
LOG=$O/logs
P=venv/bin/python
SPEC=docs/superpowers/specs/2026-10-03-c1-e4-design.md
STALL=${STALL:-1200}
PI_REPO=lerobot/pi05_libero_finetuned_v044
PI_REV=8e174154ef5f6c60a8da12ae99c303d8963138c1  # spec §4.1, with the SHA-256 of model.safetensors
PI_SHA=877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae
PEAK_MAX=23552  # MiB: spec §4.1, 23 GB
mkdir -p $LOG
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
if [ -d /workspace ]; then
  export HF_HOME=/workspace/.cache/huggingface UV_CACHE_DIR=/workspace/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/workspace/.cache/uv-python
fi

setup() {  # as scripts/gpu_c1e3.sh (same pins: the read gate compares with C1-E2's pod); pi0.5 needs nothing more
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
  if [ ! -f venv/c1e4-installed ] || ! $P -V >/dev/null 2>&1; then
    W=https://download.pytorch.org/whl/cu126
    PIN=("torch @ $W/torch-2.11.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      "torchvision @ $W/torchvision-0.26.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      transformers==5.5.4 numpy==2.2.6 gymnasium==1.3.0 mujoco==3.3.2)
    uv venv --clear --python 3.12 venv || exit 1
    uv pip install --python $P "lerobot[smolvla]==0.6.1" "${PIN[@]}" || exit 1
    uv pip install --python $P --no-deps hf-libero==0.1.4 robomimic==0.2.0 || exit 1
    uv pip install --python $P "hydra-core>=1.2,<1.4" robosuite==1.4.0 bddl==1.0.1 easydict einops thop matplotlib \
      cloudpickle opencv-python future "lerobot[dataset,scipy-dep]==0.6.1" "${PIN[@]}" || exit 1
    touch venv/c1e4-installed
  fi
  echo N | $P -c "import os, libero.libero as L; b = L.get_libero_path('bddl_files')
assert os.path.isdir(b), f'stale ~/.libero/config.yaml: {b}'; a = L.get_assets_path()
assert os.path.isdir(a), f'no LIBERO assets at {a}'; print('LIBERO assets:', a)" || exit 1
  $P -c "import torch; assert torch.cuda.is_available(), 'CUDA is not visible to torch'
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))" || exit 1
  local probe='import os, cv2, mujoco
mujoco.GLContext(64, 64).make_current()
try:
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
      echo "!!! offscreen $MUJOCO_GL rendering does not work on this pod (see above): the owner decides (another pod"
      echo "    or template, or MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa after apt-get install -y libosmesa6)"
      exit 1
    }
  fi
  # spec §4.1: the gated PaliGemma tokenizer, then pi0.5's pinned weights under their SHA-256 (checked once)
  $P -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')" >/dev/null || {
    echo "!!! no access to google/paligemma-3b-pt-224 (pi0.5's tokenizer): huggingface-cli login with a token that"
    echo "    has accepted the Gemma terms"; exit 1; }
  SNAP=$($P -c "from huggingface_hub import snapshot_download as d; print(d('$PI_REPO', revision='$PI_REV'))" | tail -1) || exit 1
  if [ "$(cat venv/pi05-sha-ok 2>/dev/null)" != "$SNAP" ]; then
    echo "$PI_SHA  $SNAP/model.safetensors" | sha256sum -c - || { echo "!!! pi0.5 weights do not match the spec's SHA-256"
      exit 1; }
    echo "$SNAP" > venv/pi05-sha-ok
  fi
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
}

record() {  # record <stage>: appends this run's environment to $O/pod_env_<stage>.txt
  { echo "=== $(date -u) $1, HEAD $(git rev-parse HEAD), MUJOCO_GL=$MUJOCO_GL, vCPUs $(nproc)"
    echo "code (src, c1e4_run.py):" $(code_hash)
    echo "HuggingFaceVLA/smolvla_libero snapshots:" \
      $(ls "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--HuggingFaceVLA--smolvla_libero/snapshots)
    echo "pi0.5 revision $PI_REV, weights SHA-256 checked: $(cat venv/pi05-sha-ok 2>/dev/null || echo no)"
    timeout 60 nvidia-smi; $P -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
    uv pip freeze --python $P; } >> $O/pod_env_$1.txt 2>&1
}

# what sets the numbers: setup's pins plus robosuite and libero (a transitive update elsewhere does not cost a base run)
KEY_PKGS='torch|torchvision|numpy|mujoco|robosuite|hf-libero|lerobot|transformers|gymnasium|robomimic|bddl'
env_key() {  # as scripts/gpu_c1e3.sh: GPU model, SmolVLA snapshot, python and the KEY_PKGS pins of the file's first record
  local p=$KEY_PKGS; [ "${2:-}" = freeze ] && p='[A-Za-z0-9_.-]+'
  awk '/^=== /{n++} n==1' "$1" | grep -oE \
    "NVIDIA GeForce RTX [0-9]+( Ti)?|snapshots: [0-9a-f]+|^Using Python [0-9.]+|^($p)(==[^ ]+| @ [^ ]+)\$" | sort -u
}

tgz() {  # tgz <archive> <paths..>: to a .tmp, then renamed, so an interrupted tar never leaves a truncated archive
  tar czf "$1.tmp" "${@:2}" && mv -f "$1.tmp" "$1" && echo "[$(date +%T)] packed $1 ($(du -h "$1" | cut -f1))"
}
pack() {
  (cd $O && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  tgz c1e4_results.tgz $O
}

CODE_PATHS="src scripts/c1e4_run.py"
code_hash() {
  local f
  for f in $CODE_PATHS; do git rev-parse -q --verify "HEAD:$f" || { echo "!!! $f is not in HEAD" >&2; return 1; }; done
}
check_code() {  # check_code record|require: trial and grid run the same code, recorded in $O/CODE by the trial
  local h old why= d=$O/old_$(date -u +%Y%m%d-%H%M%S)
  h=$(code_hash) || exit 1
  [ -z "$(git status --porcelain -- $CODE_PATHS)" ] || { git status --short -- $CODE_PATHS
    echo "!!! uncommitted or untracked files in the runner code (above): commit them on the Mac and pull, or remove them"
    exit 1; }
  old=$(ls -d $O/trial*.jsonl $O/grid*.jsonl $O/CODE 2>/dev/null | tr '\n' ' ')
  if [ -f $O/CODE ]; then
    [ "$(cat $O/CODE)" = "$h" ] || why="the runner code changed since the trial ($O/CODE)"
  elif [ "$1" = record ] && [ -z "$old" ]; then
    echo "$h" > $O/CODE
  else
    why="no $O/CODE (the trial's code)"
  fi
  [ -z "$why" ] && return
  echo "!!! $why: trial and grid must run the same code. Archive the old results, then rerun the trial:"
  [ -z "$old" ] || echo "    mkdir $d && mv $old$d/"
  echo "    ./scripts/gpu_c1e4.sh trial"
  exit 1
}

wd() {  # as scripts/gpu_c1e3.sh: the command in its own process group, killed after $STALL s without log growth
  local log=$1 pid n m t=$SECONDS rc; shift
  : >> "$log"; n=$(wc -c < "$log")
  setsid bash -o pipefail -c '"${@:2}" </dev/null 2>&1 | tee -a "$1"' wd "$log" "$@" &
  pid=$!
  trap "kill -KILL -- -$pid 2>/dev/null; exit 130" INT TERM HUP
  while kill -0 $pid 2>/dev/null; do
    sleep 5
    kill -0 $pid 2>/dev/null || break
    m=$(wc -c < "$log")
    [ "$m" = "$n" ] || { n=$m; t=$SECONDS; }
    if [ $((SECONDS - t)) -ge "$STALL" ]; then
      kill -KILL -- -$pid 2>/dev/null; wait $pid 2>/dev/null; trap - INT TERM HUP
      echo "!!! stalled for $((SECONDS - t)) s, killed: $*" | tee -a "$log"
      return 124
    fi
  done
  wait $pid; rc=$?
  trap - INT TERM HUP
  return $rc
}

run2() {  # as scripts/gpu_c1e3.sh: pass 1, then pass 2 (resume, retries failed batches) unless pass 1 was clean
  local log=$LOG/$1.log rates=$LOG/$1.rates n rc f r small=; shift
  : >> "$log"; n=$(wc -c < "$log")
  echo "[$(date +%T)] pass 1: $*" | tee -a "$log"
  wd "$log" $P scripts/c1e4_run.py --vector async --device cuda "$@"; rc=$?
  f=$(tail -c +$((n + 1)) "$log" | grep -c FAILED)
  r=$(tail -c +$((n + 1)) "$log" | rate -); [ -z "$r" ] || echo "${r% s/episode}" >> "$rates"
  [ $rc = 0 ] || echo "!!! runner exited with an error (pass 1)" | tee -a "$log"
  if [ $rc = 0 ] && [ "$f" = 0 ]; then
    echo "[$(date +%T)] pass 2 skipped: pass 1 exited cleanly, no failed batch" | tee -a "$log"; return 0
  fi
  [ $rc = 0 ] && [ "$f" -le 3 ] && small="--n-envs 2"  # fp32 and one-row bf16 chunks do not depend on the batch
  echo "[$(date +%T)] pass 2 ($f failed batches in pass 1, exit $rc): $* $small" | tee -a "$log"
  wd "$log" $P scripts/c1e4_run.py --vector async --device cuda "$@" $small; rc=$?
  [ $rc = 0 ] || echo "!!! runner exited with an error (pass 2)" | tee -a "$log"
  return $rc  # the last pass's exit status
}

rate() { grep -o '[0-9.]* s/episode' "$1" | tail -1; }
full_rate() {  # seconds per episode over the full batches of a runner log ("batch <task> <cell,kind> real=N/N X s")
  awk '/^batch / { split($4, r, "[=/]"); if (r[2] == r[3]) { s += $5; n += r[2] } } END { if (n) printf "%.3f", s / n }' "$1"
}
peak_run() {  # peak_run <peak file> <command..>: the command while nvidia-smi samples the GPU memory (MiB) every second
  local f=$1 s rc; shift
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$f.samples" 2> "$f.err" & s=$!
  "$@"; rc=$?
  kill $s 2>/dev/null; wait $s 2>/dev/null
  sort -n "$f.samples" | tail -1 > "$f"
  return $rc
}
choice() { grep -oE "$1" "$2" 2>/dev/null | tail -1; }

CUT=${CUT:-}

case "${1:-}" in
  smoke | base | trial | grid) ;;
  *) echo "usage: $0 smoke | base | trial | grid"; exit 1 ;;
esac
# one stage at a time: a second instance would run the same episodes into the same files (the lock is outside results)
if command -v flock >/dev/null; then
  exec 9>.git/c1e4-pod.lock  # in .git: setup's "uv venv --clear" would delete a lock under venv/
  flock -n 9 || { echo "!!! a stage already runs on this pod (it holds .git/c1e4-pod.lock): tmux attach -t c1e4"; exit 1; }
else
  echo "!!! no flock here: nothing stops a second instance"
fi
[ -z "$(git ls-files $O)" ] || { echo "!!! git tracks results of an earlier run under $O: the grid would resume to"
  echo "    0 episodes. Run from a commit without them, or move them away, as the owner decides."; exit 1; }
setup 2>&1 | tee -a $LOG/setup.log
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
SNAP=$(cat venv/pi05-sha-ok)
summ() { $P scripts/c1e4_summary.py "$@"; }
summ --plan --cut "$CUT" >/dev/null || exit 1
STAGE=$1
trap 'trap "" INT TERM HUP; record $STAGE; pack' EXIT
case "$STAGE" in
  smoke)
    rm -f $O/smoke*.jsonl $O/tempo*.jsonl $O/pod_env_check.txt $O/precision_choice.txt $O/env_check.txt $O/tempo.txt \
      $LOG/smoke.log $LOG/tempo*.log
    # spec §4.1, first: GPU peak and tempo on full batches; 5 arms x 2 inits = 10 episodes per task fill a batch of 10
    # (and one of 6, 5); Gcal puts the shadow calls (and bf16-row's shadow path) into the peak
    T=(--tasks libero_spatial:1,libero_object:1,libero_goal:2 --cells A --methods none,G,Gcal,Gk0,Gauto --inits 48-49
      --vector async --device cuda)
    for e in 6 5 4; do
      rm -f $O/tempo_fp32.jsonl
      peak_run venv/peak_fp32 wd $LOG/tempo_fp32_$e.log $P scripts/c1e4_run.py --brain pi05 "${T[@]}" --kbar 0.355 \
        --precision fp32 --n-envs $e --out $O/tempo_fp32.jsonl || { echo "!!! pi0.5 fp32 tempo run failed"; exit 1; }
      grep -qE '^[0-9]+$' venv/peak_fp32 || { echo "!!! no GPU memory samples (venv/peak_fp32.err)"; exit 1; }
      FE=$e
      echo "pi0.5 fp32, $e envs: GPU peak $(cat venv/peak_fp32) MiB (max $PEAK_MAX)" | tee -a $O/tempo.txt
      [ "$(cat venv/peak_fp32)" -le $PEAK_MAX ] && break
      [ $e = 4 ] && echo "!!! the peak is above $PEAK_MAX MiB at 4 envs too: going on with 4" | tee -a $O/tempo.txt
    done
    grep -q "All keys loaded successfully" $LOG/tempo_fp32_$FE.log || {
      echo "!!! pi0.5 did not report all keys loaded (a silent partial load)"; exit 1; }
    BS=-
    if peak_run venv/peak_bf16 wd $LOG/tempo_bf16.log $P scripts/c1e4_run.py --brain pi05 "${T[@]}" --kbar 0.355 \
      --precision bf16-row --n-envs 10 --repeat-check --out $O/tempo_bf16.jsonl; then BS=$(full_rate $LOG/tempo_bf16.log)
    else echo "!!! pi0.5 bf16-row tempo run failed: fp32 stays (spec §4.1)" | tee -a $O/tempo.txt; fi
    if grep -qE '^[0-9]+$' venv/peak_bf16; then FITS=$([ "$(cat venv/peak_bf16)" -le $PEAK_MAX ] && echo 1 || echo 0)
    else FITS=0; echo "!!! no GPU memory samples for bf16-row (venv/peak_bf16.err): it does not fit" | tee -a $O/tempo.txt
    fi
    wd $LOG/tempo_s.log $P scripts/c1e4_run.py --brain smolvla "${T[@]}" --kbar 0.04 --n-envs 10 --out $O/tempo_s.jsonl \
      || { echo "!!! SmolVLA tempo run failed"; exit 1; }
    FS=$(full_rate $LOG/tempo_fp32_$FE.log); SS=$(full_rate $LOG/tempo_s.log)
    BIT=$(grep -oE 'C1-E4 repeat: .* bitwise=[01]' $LOG/tempo_bf16.log | head -1 | grep -oE '[01]$')
    echo "pi0.5 bf16-row, 10 envs: GPU peak $(cat venv/peak_bf16) MiB; $(grep -m1 'C1-E4 repeat' $LOG/tempo_bf16.log)" \
      | tee -a $O/tempo.txt
    echo "tempo (s/episode, full batches): pi_fp32=${FS:--} pi_bf16=${BS:--} smolvla=${SS:--}" | tee -a $O/tempo.txt
    [ -n "$FS" ] && [ -n "$SS" ] || { echo "!!! no full batch in the fp32 or SmolVLA tempo run"; exit 1; }
    # then every arm, cell and code path (pi0.5 at the chosen fp32 envs; init 48 of libero_object:0 is class 2)
    sm() { echo "[$(date +%T)] smoke: $*" | tee -a $LOG/smoke.log
      wd $LOG/smoke.log $P scripts/c1e4_run.py "$@" --vector async --device cuda \
        || echo "!!! runner exited with an error" | tee -a $LOG/smoke.log; }
    sm --brain pi05 --tasks libero_spatial:0 --cells A --kbar 0.355 --inits 48-49 --n-envs $FE --out $O/smoke_pi.jsonl \
      --methods none,T0,PPC,G,GT,Gkeep,GR,Gk0,Gk0T,Gauto,GautoT,Gcal,G@glr,G@ema
    sm --brain pi05 --tasks libero_object:0 --cells A10,A20,A40,C --kbar 0.355 --methods none,G,T0 --inits 48 \
      --classes 2 --n-envs $FE --out $O/smoke_pi.jsonl
    sm --brain pi05 --tasks libero_goal:1 --cells A --kinds control --methods none --shadow-pseudo --trace --inits 48-49 \
      --n-envs 2 --out $O/smoke_p1.jsonl
    sm --brain smolvla --tasks libero_spatial:0 --cells A,A40 --kbar 0.04 --methods none,Gauto,Gcal,GautoT --inits 48-49 \
      --n-envs 10 --out $O/smoke_s.jsonl
    n=$(cat $O/smoke*.jsonl 2>/dev/null | wc -l); f=$(grep -c FAILED $LOG/smoke.log)
    echo "[$(date +%T)] smoke: $n of 58 records (28 + 12 + 2 + 16), $f FAILED"
    [ "$n" -eq 58 ] && [ "$f" -eq 0 ] || { echo "!!! smoke failed: see $LOG/smoke.log"; exit 1; }
    summ --precision "$FS" "$FE" "$BS" "$FITS" "${BIT:-0}" | tee $O/precision_choice.txt
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    # spec §7 item 4: C1-E2's SmolVLA baseline holds on the same GPU model, model snapshot, python and key packages
    record check
    if d=$(diff <(env_key results/c1-e2/pod_env_grid.txt) <(env_key $O/pod_env_check.txt)); then
      echo "C1-E4 env: SAME as C1-E2's grid pod (GPU model, model snapshot, python, $KEY_PKGS)" | tee $O/env_check.txt
    else
      { echo "C1-E4 env: DIFFERENT from C1-E2's grid pod (< C1-E2, > here): run base for SmolVLA too"; echo "$d"; } \
        | tee $O/env_check.txt
    fi
    { echo "for the record only, the whole package freeze (< C1-E2, > here):"
      diff <(env_key results/c1-e2/pod_env_grid.txt freeze) <(env_key $O/pod_env_check.txt freeze) || true; } \
      >> $O/env_check.txt  # a differing freeze is for the record only, not a smoke failure
    ;;
  base)  # spec §7 item 4: stock lerobot-eval, s = 10, the 26 tasks x 10
    mkdir -p $O/base
    brains=(pi05); grep -q '^C1-E4 env: SAME' $O/env_check.txt 2>/dev/null || brains+=(smolvla)
    for b in "${brains[@]}"; do
      if [ $b = pi05 ]; then pol=(--policy.path=$SNAP --policy.compile_model=false); bs=5
      else pol=(--policy.path=$($P -c "import sys; sys.path.insert(0, 'src'); import gauto
from huggingface_hub import snapshot_download as d; r, v = gauto.BRAINS['smolvla']; print(d(r, revision=v))" | tail -1))
        bs=10; fi  # the pinned SmolVLA snapshot
      for suite in libero_spatial libero_object libero_goal; do
        d=${b}_$suite
        [ -f $O/base/$d/eval_info.json ] && continue
        ids=$([ $suite = libero_goal ] && echo "[1,2,4,6,8,9]" || echo "[0,1,2,3,4,5,6,7,8,9]")
        rm -rf eval_output/c1e4/$d
        echo "[$(date +%T)] lerobot-eval $d" | tee -a $LOG/base.log
        wd $LOG/base.log venv/bin/lerobot-eval "${pol[@]}" --policy.device=cuda --policy.n_action_steps=10 \
          --env.type=libero --env.task=$suite --env.task_ids="$ids" --eval.n_episodes=10 --eval.batch_size=$bs \
          --output_dir=eval_output/c1e4/$d || { echo "!!! lerobot-eval $d failed: rerun base"; exit 1; }
        mkdir -p $O/base/$d && cp eval_output/c1e4/$d/eval_info.json $O/base/$d/ || exit 1
      done
      $P scripts/c1e3_pilot.py baseline $O/base/${b}_* > $O/$([ $b = pi05 ] && echo baseline_pi05 || echo baseline).json \
        || exit 1
    done
    cat $O/baseline*.json
    ;;
  trial)
    export HF_HUB_OFFLINE=1  # everything is cached by now: no silent weight change, no Hub outage mid-run
    m=$(choice 'C1-E4 precision: precision=(fp32|bf16-row) envs=[0-9]+' $O/precision_choice.txt)
    [ -n "$m" ] || { echo "!!! no precision choice ($O/precision_choice.txt): run smoke first"; exit 1; }
    PREC=$(echo "$m" | grep -oE 'precision=[a-z0-9-]+' | cut -d= -f2); ENVS=$(echo "$m" | grep -oE '[0-9]+$')
    check_code record
    run2 trial_p1 --brain pi05 --precision $PREC --n-envs $ENVS --cells A --kinds control --methods none --shadow-pseudo \
      --trace --inits 44-47 --out $O/trial_p1.jsonl || exit 1
    run2 trial_p2 --brain pi05 --precision $PREC --n-envs $ENVS --cells A --methods Gcal --inits 40-43 \
      --out $O/trial_p2.jsonl || exit 1
    run2 trial_p3 --brain smolvla --n-envs 10 --cells A --methods Gcal --inits 40-43 --out $O/trial_p3.jsonl || exit 1
    summ --trial $O/trial_p1.jsonl $O/trial_p2.jsonl $O/trial_p3.jsonl --choice $O/precision_choice.txt 2>&1 \
      | tee $O/trial_choice.txt
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    t=$(choice 'tempo \(s/episode, full batches\): .*' $O/tempo.txt)
    pis=$(echo "$t" | grep -oE "pi_$([ $PREC = fp32 ] && echo fp32 || echo bf16)=[0-9.]+" | cut -d= -f2)
    ss=$(echo "$t" | grep -oE 'smolvla=[0-9.]+' | cut -d= -f2)
    { echo "forecast (smoke tempo: pi0.5 $PREC ${pis:-?} s/episode one lane, SmolVLA ${ss:-?} s/episode per lane, two lanes):"
      summ --forecast "${pis:-0}" "${ss:-0}" 2>&1
      echo "if card48_cheaper is true: tell the owner before the grid (spec §4.1); fuse_32h: report rows to cut (§12)"; } \
      | tee -a $O/trial_choice.txt
    ;;
  grid)
    export HF_HUB_OFFLINE=1  # everything is cached by now: no silent weight change, no Hub outage mid-run
    # the plan comes from the working tree's scripts/c1e4_summary.py: it, this script and the runner run as committed
    F="scripts/c1e4_summary.py scripts/gpu_c1e4.sh $CODE_PATHS"
    git diff --quiet HEAD -- $F || { git status --short -- $F
      echo "!!! local edits on the pod (above): the grid reads its plan and runs its code as committed. Undo them"
      echo "    (git checkout -- <file>), or commit them on the Mac, push, then here: git pull"; exit 1; }
    mark='C1-E4 trial: precision=(fp32|bf16-row) tau_k=[0-9.]+ kbar_pi05=-?[0-9.]+ kbar_smolvla=-?[0-9.]+ N=[0-9/]+ envs=[0-9]+ pad=0'
    m=$(git show HEAD:$SPEC | grep -oE "$mark" | tail -1)
    c=$(choice "$mark" $O/trial_choice.txt)
    [ -n "$m" ] && [ "$m" = "$c" ] || { echo "!!! the spec at HEAD journals '${m:-no trial line}', the trial on this pod"
      echo "    chose '${c:-nothing}' ($O/trial_choice.txt). Journal the trial's line in the spec on the Mac, commit,"
      echo "    push, then here: git pull"; exit 1; }
    git rev-parse -q --verify @{u} >/dev/null 2>&1 || { echo "!!! no upstream branch: git checkout c1-e4 && git pull"
      exit 1; }
    git fetch -q && git merge-base --is-ancestor HEAD @{u} || {
      echo "!!! HEAD is not pushed (not in its upstream): push the journal commit from the Mac, then here: git pull"; exit 1; }
    if grep -q '^C1-E4 env: SAME' $O/env_check.txt 2>/dev/null; then BASE=results/c1-e2/baseline.json
    elif [ -s $O/baseline.json ]; then BASE=$O/baseline.json
    else echo "!!! the environment differs from C1-E2's pod ($O/env_check.txt) and there is no $O/baseline.json:"
      echo "    ./scripts/gpu_c1e4.sh base (spec §8)"; exit 1; fi
    check_code require
    v() { echo "$m" | grep -oE " $1=-?[0-9a-z.-]+" | cut -d= -f2; }
    CO=(--t-ramp 5 --k-p 1.0 --tau-k "$(v tau_k)")
    lane_args() {  # pi0.5 on lane p, SmolVLA on lanes a and b (spec §12)
      if [ $1 = p ]; then echo "--precision $(v precision) --n-envs $(v envs) --kbar $(v kbar_pi05)"
      else echo "--precision fp32 --n-envs 10 --kbar $(v kbar_smolvla)"; fi; }
    out() { case $1 in p) echo $O/grid_pi05.jsonl ;; a) echo $O/grid_smolvla.jsonl ;; b) echo $O/grid_smolvla_b.jsonl ;; esac; }
    echo "[$(date +%T)] grid with ${CO[*]}; lanes: p $(lane_args p), a/b $(lane_args a); baseline $BASE; cut: ${CUT:-none}" \
      | tee -a $O/forecast.txt
    run_lane() {  # run_lane <lane> <plan lines of that lane>: in the lane's own shell, so Ctrl-C reaches wd's trap
      local ph ln id n args
      while IFS=$'\t' read -r ph ln id n args; do
        [ -n "$id" ] || continue
        echo "[$(date +%T)] phase $ph, lane $ln, id $id ($n episodes): $args" | tee -a $LOG/grid_$ln.log
        # shellcheck disable=SC2086
        run2 grid_$ln $args $(lane_args $ln) "${CO[@]}" --out "$(out $ln)" </dev/null
      done <<< "$2"
    }
    run_phase() {  # run_phase <phase> <plan>: the phase's lanes in parallel
      local ln pids=() lines
      trap 'kill -TERM ${pids[*]:-} 2>/dev/null; wait; exit 130' INT TERM HUP  # before the fork: no lane escapes
      for ln in p a b; do
        lines=$(awk -F'\t' -v ph="$1" -v ln=$ln '$1 == ph && $2 == ln' <<< "$2")
        [ -n "$lines" ] || continue
        run_lane $ln "$lines" & pids+=($!)
      done
      wait "${pids[@]}"
      trap - INT TERM HUP
    }
    fcast() {  # fcast <why>: the §12 fuse from the latest pass-1 rates of lanes p and a (else the smoke's tempo)
      local pr sr t
      t=$(choice 'tempo \(s/episode, full batches\): .*' $O/tempo.txt)
      pr=$(tail -1 $LOG/grid_p.rates 2>/dev/null); sr=$(tail -1 $LOG/grid_a.rates 2>/dev/null)
      [ -n "$pr" ] || pr=$(echo "$t" | grep -oE "pi_$([ "$(v precision)" = fp32 ] && echo fp32 || echo bf16)=[0-9.]+" | cut -d= -f2)
      [ -n "$sr" ] || sr=$(echo "$t" | grep -oE 'smolvla=[0-9.]+' | cut -d= -f2)
      echo "[$(date +%T)] forecast after $1 (pi0.5 ${pr:-?}, SmolVLA ${sr:-?} s/episode per lane; whole plan):" \
        "$(summ --forecast "${pr:-0}" "${sr:-0}" --cut "$CUT" $HOLD 2>&1)" | tee -a $O/forecast.txt
      summ --forecast "${pr:-0}" "${sr:-0}" --cut "$CUT" $HOLD 2>/dev/null | grep -q '"fuse_32h": true' && {
        echo "!!! forecast > 32 h (spec §12 fuse): the owner cuts report rows in the order 12 16 11 15 10 9 14 13 17:"
        echo "    Ctrl-C, then CUT=\"12 16 ...\" ./scripts/gpu_c1e4.sh grid. The run goes on until then."; } \
        | tee -a $O/forecast.txt
    }
    HOLD=
    PLAN=$(summ --plan --cut "$CUT") || exit 1
    run_phase control "$PLAN"
    fcast "the pi control rows"
    echo "=== $(date -u) pi control check" >> $O/gates.txt
    summ --control $(out p) 2>&1 | tee -a $O/gates.txt; q=${PIPESTATUS[0]}
    [ $q = 0 ] || { echo "!!! the pi control check did not pass (exit $q): the grid stops; Q0 is not computed;" \
      "the owner decides after the analysis (spec §8)" | tee -a $O/gates.txt; exit 1; }
    run_phase q0 "$PLAN"
    echo "=== $(date -u) Q0 gate" >> $O/gates.txt
    summ --q0 $(out p) 2>&1 | tee -a $O/gates.txt; q=${PIPESTATUS[0]}
    case $q in  # the summary exits 3 if Q0 does not pass; any other error is the gate not running
      0) echo "Q0 passed: the delay-axis ids run" | tee -a $O/gates.txt ;;
      *) why=$([ $q = 3 ] && echo "Q0 did not pass" || echo "the Q0 gate failed to run (exit $q)")
        if [ "${Q0_HOLD:-}" = run ]; then
          echo "!!! $why; Q0_HOLD=run: the delay-axis ids run, report only (spec §8)" | tee -a $O/gates.txt
        else HOLD=--hold; echo "$(date -u): $why" >> $O/q0_held
          { echo "!!! $why: the delay-axis ids are held (spec §8). Later, with the owner's decision:"
          echo "    Q0_HOLD=run ${CUT:+CUT=\"$CUT\" }./scripts/gpu_c1e4.sh grid (report only; resume skips finished episodes)"
          } | tee -a $O/gates.txt; fi ;;
    esac
    if [ -f $O/q0_held ] && [ -z "$HOLD" ] && [ "${Q0_HOLD:-}" != run ]; then  # sticky: spec §8
      HOLD=--hold; { echo "!!! an earlier run held the delay-axis ids ($O/q0_held; spec §8: later runs are report only):"
        echo "    they stay held; Q0_HOLD=run ${CUT:+CUT=\"$CUT\" }./scripts/gpu_c1e4.sh grid runs them as report rows"
      } | tee -a $O/gates.txt
    fi
    PLAN=$(summ --plan --cut "$CUT" $HOLD) || exit 1
    for ph in $(cut -f1 <<< "$PLAN" | uniq); do
      [ "$ph" = control ] || [ "$ph" = q0 ] || run_phase "$ph" "$PLAN"
      [ "$ph" = smolvla ] && fcast "the SmolVLA rule rows"
    done
    cat $(out p) $(out a) $(out b) 2>/dev/null > $O/grid_all.jsonl
    echo "[$(date +%T)] grid_all.jsonl: $(wc -l < $O/grid_all.jsonl) records (the summary checks each id)"
    for f in $LOG/grid_*.log; do echo "  $f: $(grep -c FAILED "$f") FAILED lines (first-pass failures included)"; done
    echo "On the Mac: python scripts/c1e4_summary.py $O/grid_all.jsonl --baseline $BASE$([ -f $O/q0_held ] && echo " --q0-held")"
    ;;
esac
