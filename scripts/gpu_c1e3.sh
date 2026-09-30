#!/usr/bin/env bash
# C1-E3 on one Linux NVIDIA GPU host (RTX 4090), spec docs/superpowers/specs/2026-09-29-c1-e3-handoff-reflex-design.md
# sections 6, 7, 8, 11. Pod and setup as scripts/gpu_c1e2.sh: >= 16 vCPUs, 24 GB GPU, >= 40 GB on /workspace; tmux.
#   tmux new -s c1e3   (after a lost SSH session: tmux attach -t c1e3)
#   cd /workspace && git clone https://github.com/papadoster/jit-reflex.git && cd jit-reflex && git checkout c1-e3
# Stages, each run separately after the owner's go-ahead, in this order:
#   ./scripts/gpu_c1e3.sh smoke   minutes: 40 debug episodes (inits 48-49, never read) over every arm, kind and alpha
#                                 path; the environment check against C1-E2's pod (env_check.txt) and the constant
#                                 batch test (spec §7 item 6, pad_choice.txt)
#   ./scripts/gpu_c1e3.sh base    only if env_check.txt is not SAME: stock lerobot-eval s = 10 -> baseline.json
#   ./scripts/gpu_c1e3.sh pilot   spec §7: P1-P6 (inits 44-47) in two lanes, the offline choices -> pilot_choice.txt
#   on the Mac: journal the printed line "C1-E3 pilot: K=.. ... pad=.." in the spec, commit, push; here: git pull
#   ./scripts/gpu_c1e3.sh grid    spec §6: 31150 episodes in two lanes into results/c1-e3/grid.jsonl and grid_b.jsonl,
#                                 joined into grid_all.jsonl. Refuses unless the spec at HEAD and pilot_choice.txt
#                                 carry the same pilot line, HEAD is pushed, the pod has no local edits, a baseline
#                                 applies and the runner code is the pilot's (results/c1-e3/CODE); one grid at a time
#                                 (flock). Rule rows run first in each lane, report rows last.
#                                 Lane B starts with row 10 and checks Q3a (scripts/c1e3_summary.py --q3a): if it fails
#                                 or cannot run, the other alpha-brain rows are held for the owner (spec §8;
#                                 ALPHA_ROWS=run overrides).
# Spec §11 fuse: after each lane's first line the forecast goes to forecast.txt; above 16 h the owner cuts report rows,
# in this order: 26 22 24 23 15 19 17 9 8 4 25x (row 25's inits 10-19), then 14 25. Rule rows are never cut. To cut:
# Ctrl-C, then CUT="26 22" ./scripts/gpu_c1e3.sh grid (resume skips finished episodes). Cut rows fail the summary's
# completeness gate, which then needs an --override-gate journaled in the spec before reading.
# Every exit appends the environment to results/c1-e3/pod_env_<stage>.txt, refreshes MANIFEST.sha256 and packs
# results/c1-e3 into c1e3_results.tgz. The pilot's brain images (eval_output/c1e3_images) go into c1e3_images.tgz: for
# the owner's eyes only, never committed. Watchdog, two passes and resume as scripts/gpu_c1e2.sh (STALL, default 1200 s).
set -uo pipefail
cd "$(dirname "$0")/.."
[ -n "${TMUX:-}" ] || { echo "!!! not inside tmux: a lost SSH session kills the workers (tmux new -s c1e3)"; sleep 10; }
O=results/c1-e3
LOG=$O/logs
P=venv/bin/python
SPEC=docs/superpowers/specs/2026-09-29-c1-e3-handoff-reflex-design.md
STALL=${STALL:-1200}
mkdir -p $LOG
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
if [ -d /workspace ]; then
  export HF_HOME=/workspace/.cache/huggingface UV_CACHE_DIR=/workspace/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/workspace/.cache/uv-python
fi

setup() {  # as scripts/gpu_c1e2.sh (same pins: the read gate compares with C1-E2's pod)
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
  if [ ! -f venv/c1e3-installed ] || ! $P -V >/dev/null 2>&1; then
    W=https://download.pytorch.org/whl/cu126
    PIN=("torch @ $W/torch-2.11.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      "torchvision @ $W/torchvision-0.26.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      transformers==5.5.4 numpy==2.2.6 gymnasium==1.3.0 mujoco==3.3.2)
    uv venv --clear --python 3.12 venv || exit 1
    uv pip install --python $P "lerobot[smolvla]==0.6.1" "${PIN[@]}" || exit 1
    uv pip install --python $P --no-deps hf-libero==0.1.4 robomimic==0.2.0 || exit 1
    uv pip install --python $P "hydra-core>=1.2,<1.4" robosuite==1.4.0 bddl==1.0.1 easydict einops thop matplotlib \
      cloudpickle opencv-python future "lerobot[dataset,scipy-dep]==0.6.1" "${PIN[@]}" || exit 1
    touch venv/c1e3-installed
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
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
}

record() {  # record <stage>: appends this run's environment to $O/pod_env_<stage>.txt
  { echo "=== $(date -u) $1, HEAD $(git rev-parse HEAD), MUJOCO_GL=$MUJOCO_GL, vCPUs $(nproc)"
    echo "code (src, c1e3_run.py):" $(code_hash)
    echo "HuggingFaceVLA/smolvla_libero snapshots:" \
      $(ls "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--HuggingFaceVLA--smolvla_libero/snapshots)
    timeout 60 nvidia-smi; $P -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
    uv pip freeze --python $P; } >> $O/pod_env_$1.txt 2>&1
}

# what sets the numbers: setup's pins plus robosuite and libero (a transitive update elsewhere does not cost a base run)
KEY_PKGS='torch|torchvision|numpy|mujoco|robosuite|hf-libero|lerobot|transformers|gymnasium|robomimic|bddl'
env_key() {  # env_key <pod_env file> [freeze]: GPU model, model snapshot, python and the KEY_PKGS pins of the file's
  # first record; with freeze, every package pin (for the record only)
  local p=$KEY_PKGS; [ "${2:-}" = freeze ] && p='[A-Za-z0-9_.-]+'
  awk '/^=== /{n++} n==1' "$1" | grep -oE \
    "NVIDIA GeForce RTX [0-9]+( Ti)?|snapshots: [0-9a-f]+|^Using Python [0-9.]+|^($p)(==[^ ]+| @ [^ ]+)\$" | sort -u
}

tgz() {  # tgz <archive> <paths..>: to a .tmp, then renamed, so an interrupted tar never leaves a truncated archive
  tar czf "$1.tmp" "${@:2}" && mv -f "$1.tmp" "$1" && echo "[$(date +%T)] packed $1 ($(du -h "$1" | cut -f1))"
}
pack() {
  (cd $O && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  tgz c1e3_results.tgz $O
}
pack_images() {  # the pilot's brain images (spec §7 item 5), for the owner's eyes only, never committed
  if [ -d eval_output/c1e3_images ]; then
    tgz c1e3_images.tgz eval_output/c1e3_images && echo "brain images for the owner (never committed): c1e3_images.tgz"
  else
    echo "!!! no brain images (eval_output/c1e3_images is missing): P1's --save-images 10 saved nothing"
  fi
}

CODE_PATHS="src scripts/c1e3_run.py"
code_hash() {
  local f
  for f in $CODE_PATHS; do git rev-parse -q --verify "HEAD:$f" || { echo "!!! $f is not in HEAD" >&2; return 1; }; done
}
check_code() {  # check_code record|require: pilot and grid run the same code, recorded in $O/CODE by the pilot
  local h old why= d=$O/old_$(date -u +%Y%m%d-%H%M%S)
  h=$(code_hash) || exit 1
  [ -z "$(git status --porcelain -- $CODE_PATHS)" ] || { git status --short -- $CODE_PATHS
    echo "!!! uncommitted or untracked files in the runner code (above): commit them on the Mac and pull, or remove them"
    exit 1; }
  old=$(ls -d $O/pilot*.jsonl $O/grid*.jsonl $O/CODE 2>/dev/null | tr '\n' ' ')
  if [ -f $O/CODE ]; then
    [ "$(cat $O/CODE)" = "$h" ] || why="the runner code changed since the pilot ($O/CODE)"
  elif [ "$1" = record ] && [ -z "$old" ]; then
    echo "$h" > $O/CODE
  else
    why="no $O/CODE (the pilot's code)"
  fi
  [ -z "$why" ] && return
  echo "!!! $why: pilot and grid must run the same code. Archive the old results, then rerun the pilot:"
  [ -z "$old" ] || echo "    mkdir $d && mv $old$d/"
  echo "    ./scripts/gpu_c1e3.sh pilot"
  exit 1
}

wd() {  # as scripts/gpu_c1e2.sh: the command in its own process group, killed after $STALL s without log growth
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

run2() {  # as scripts/gpu_c1e2.sh: pass 1, then pass 2 (resume, retries failed batches) unless pass 1 was clean
  local log=$LOG/$1.log rates=$LOG/$1.rates n rc f r small=; shift
  : >> "$log"; n=$(wc -c < "$log")
  echo "[$(date +%T)] pass 1: $*" | tee -a "$log"
  wd "$log" $P scripts/c1e3_run.py --vector async --device cuda "$@"; rc=$?
  f=$(tail -c +$((n + 1)) "$log" | grep -c FAILED)
  r=$(tail -c +$((n + 1)) "$log" | rate -); [ -z "$r" ] || echo "${r% s/episode}" >> "$rates"
  [ $rc = 0 ] || echo "!!! runner exited with an error (pass 1)" | tee -a "$log"
  if [ $rc = 0 ] && [ "$f" = 0 ]; then
    echo "[$(date +%T)] pass 2 skipped: pass 1 exited cleanly, no failed batch" | tee -a "$log"; return
  fi
  # not with the constant brain batch: 2 envs would change the retried episodes' brain batch (spec §4, journal item 11)
  [ $rc = 0 ] && [ "$f" -le 3 ] && [[ " $* " != *" --pad-brain 1 "* ]] && small="--n-envs 2"
  echo "[$(date +%T)] pass 2 ($f failed batches in pass 1, exit $rc): $* $small" | tee -a "$log"
  wd "$log" $P scripts/c1e3_run.py --vector async --device cuda "$@" $small \
    || echo "!!! runner exited with an error (pass 2)" | tee -a "$log"
}

rate() { grep -o '[0-9.]* s/episode' "$1" | tail -1; }

CUT=${CUT:-}

case "${1:-}" in
  smoke | base | pilot | grid) ;;
  *) echo "usage: $0 smoke | base | pilot | grid"; exit 1 ;;
esac
# one grid at a time: a second instance would run the same episodes into the same files (no flock on the Mac)
if [ "$1" = grid ]; then
  if command -v flock >/dev/null; then
    exec 9>$O/grid.lock
    flock -n 9 || { echo "!!! a grid already runs on this pod (it holds $O/grid.lock): tmux attach -t c1e3"; exit 1; }
  else
    echo "!!! no flock here: nothing stops a second grid instance"
  fi
fi
[ -z "$(git ls-files $O)" ] || { echo "!!! git tracks results of an earlier run under $O: the grid would resume to"
  echo "    0 episodes. Run from a commit without them, or move them away, as the owner decides."; exit 1; }
setup 2>&1 | tee -a $LOG/setup.log
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
# the grid's runner lines, per lane: row, episodes, arguments (spec §6 from ROWS; refuses to cut a rule row)
plan() { $P scripts/c1e3_summary.py --plan "$@" --cut "$CUT"; }
plan A >/dev/null || exit 1
STAGE=$1
# a second Ctrl-C would run wd's or a lane's trap (exit 130) inside this one and cut the packing short
trap 'trap "" INT TERM HUP; record $STAGE; pack; [ $STAGE != pilot ] || pack_images' EXIT
case "$STAGE" in
  smoke)
    rm -f $O/smoke.jsonl $O/pad1.jsonl $O/pad0.jsonl $O/pod_env_check.txt $LOG/smoke.log $LOG/pad*.log
    S=(--tasks libero_spatial:0 --pad-brain 1)
    for a in "--cells A --methods none,T0,G,GT,Gkeep,GkeepT,GR,GRT,GRret,PPC --inits 48-49 --n-envs 10 --save-images 2" \
      "--cells C --kinds control,close --methods none,GR+surr,GR+contact,GR@real,PPC@real,GR@s0.5 --inits 48 --n-envs 6 --trace" \
      "--cells F --alpha 1 --methods none,T0,GRT --inits 48-49 --n-envs 6" \
      "--tasks libero_10:0 --cells A --methods none,GR --inits 48 --n-envs 2"; do
      echo "[$(date +%T)] smoke: $a" | tee -a $LOG/smoke.log
      wd $LOG/smoke.log $P scripts/c1e3_run.py "${S[@]}" $a --vector async --device cuda --out $O/smoke.jsonl \
        || echo "!!! runner exited with an error" | tee -a $LOG/smoke.log
    done
    n=$(cat $O/smoke.jsonl 2>/dev/null | wc -l); f=$(grep -c FAILED $LOG/smoke.log)
    echo "[$(date +%T)] smoke: $n of 40 records (20 + 12 + 6 + 2), $f FAILED"
    [ "$n" -eq 40 ] && [ "$f" -eq 0 ] || { echo "!!! smoke failed: see $LOG/smoke.log"; exit 1; }
    # spec §8: C1-E2's baseline holds on the same GPU model, model snapshot, python and key package versions
    record check
    if d=$(diff <(env_key results/c1-e2/pod_env_grid.txt) <(env_key $O/pod_env_check.txt)); then
      echo "C1-E3 env: SAME as C1-E2's grid pod (GPU model, model snapshot, python, $KEY_PKGS)" | tee $O/env_check.txt
    else
      { echo "C1-E3 env: DIFFERENT from C1-E2's grid pod (< C1-E2, > here): run base before the grid"; echo "$d"; } \
        | tee $O/env_check.txt
    fi
    { echo "for the record only, the whole package freeze (< C1-E2, > here):"
      diff <(env_key results/c1-e2/pod_env_grid.txt freeze) <(env_key $O/pod_env_check.txt freeze); } >> $O/env_check.txt
    # spec §7 item 6: 8 control episodes as none and GR, padded and not. Unpadded first: the first run pays the cold
    # disk cache, which would count against padding. 3 envs: a task's 8 episodes (none 48-51, GR 48-51) split 3 + 3 + 2,
    # so none and GR of an init get other batch neighbours and, unpadded, other batch sizes (with 4 the test is vacuous)
    for pd in 0 1; do
      wd $LOG/pad$pd.log $P scripts/c1e3_run.py --tasks libero_spatial:0,libero_object:0 --cells A --kinds control \
        --methods none,GR --inits 48-51 --n-envs 3 --pad-brain $pd --vector async --device cuda --out $O/pad$pd.jsonl \
        || { echo "!!! padding run pad=$pd failed"; exit 1; }
    done
    $P scripts/c1e3_pilot.py pad $O/pad1.jsonl $O/pad0.jsonl "$(rate $LOG/pad1.log | cut -d' ' -f1)" \
      "$(rate $LOG/pad0.log | cut -d' ' -f1)" | tee $O/pad_choice.txt
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    rm -rf eval_output/c1e3_images  # the smoke's images only tested the path; the pilot archive holds P1's alone
    ;;
  base)  # as C1-E2's base, s = 10 only (the read gate's number)
    mkdir -p $O/base
    for suite in libero_spatial libero_object libero_goal; do
      d=s10_$suite
      [ -f $O/base/$d/eval_info.json ] && continue
      ids=$([ $suite = libero_goal ] && echo "[1,2,4,6,8,9]" || echo "[0,1,2,3,4,5,6,7,8,9]")
      rm -rf eval_output/c1e3/$d
      echo "[$(date +%T)] lerobot-eval $d" | tee -a $LOG/base.log
      wd $LOG/base.log venv/bin/lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --policy.device=cuda \
        --policy.n_action_steps=10 --env.type=libero --env.task=$suite --env.task_ids="$ids" \
        --eval.n_episodes=10 --eval.batch_size=10 --output_dir=eval_output/c1e3/$d \
        || { echo "!!! lerobot-eval $d failed: rerun base"; exit 1; }
      mkdir -p $O/base/$d && cp eval_output/c1e3/$d/eval_info.json $O/base/$d/ || exit 1
    done
    $P scripts/c1e3_pilot.py baseline $O/base/s10_* > $O/baseline.json || exit 1
    cat $O/baseline.json
    ;;
  pilot)
    PAD=$(grep -oE 'C1-E3 padding: pad=[01]' $O/pad_choice.txt 2>/dev/null | grep -oE '[01]$')
    [ -n "$PAD" ] || { echo "!!! no padding choice ($O/pad_choice.txt): run smoke first"; exit 1; }
    check_code record
    # temporary K = 20 and tau_k = 1 cm (spec §7); P1: 10 episodes' brain images at the shift
    C=(--inits 44-47 --n-envs 4 --pad-brain "$PAD")
    pa() { run2 pilot --cells A --methods GR --save-images 10 "${C[@]}" --out $O/pilot.jsonl
      run2 pilot --cells A --kinds control --methods none --trace "${C[@]}" --out $O/pilot.jsonl
      run2 pilot --cells C --methods none --trace "${C[@]}" --out $O/pilot.jsonl; }
    pb() { run2 pilot_b --cells A --methods GR --alpha 1 "${C[@]}" --out $O/pilot_b.jsonl
      run2 pilot_b --cells A --kinds close --methods none --trace "${C[@]}" --out $O/pilot_b.jsonl
      run2 pilot_b --cells C --kinds control --methods none --trace "${C[@]}" --out $O/pilot_b.jsonl; }
    a= b=
    trap 'kill -TERM $a $b 2>/dev/null; wait; exit 130' INT TERM HUP
    pa & a=$!
    pb & b=$!
    wait $a; wait $b
    trap - INT TERM HUP
    # the brain images go into c1e3_images.tgz on exit (pack_images in the EXIT trap)
    $P scripts/c1e3_pilot.py choose "$PAD" $O/pilot.jsonl $O/pilot_b.jsonl 2>&1 | tee $O/pilot_choice.txt
    [ "${PIPESTATUS[0]}" = 0 ] || exit 1
    r=$(cat $LOG/pilot.rates $LOG/pilot_b.rates 2>/dev/null | awk '{ s += $1; n++ } END { if (n) printf "%.2f", s / n }')
    echo "Rough grid forecast: ${r:-?} s/episode per lane (pilot: 4 envs a lane, grid: 10) x 31150 / 2 lanes =" \
      "$(awk -v r="${r:-0}" 'BEGIN { printf "%.1f", r * 31150 / 2 / 3600 }') h. Spec §11 fuse: > 16 h." \
      | tee -a $O/pilot_choice.txt
    ;;
  grid)
    # the rows come from the working tree's scripts/c1e3_summary.py: it, this script and the runner run as committed
    F="scripts/c1e3_summary.py scripts/gpu_c1e3.sh $CODE_PATHS"
    git diff --quiet HEAD -- $F || { git status --short -- $F
      echo "!!! local edits on the pod (above): the grid reads its rows and runs its code as committed. Undo them"
      echo "    (git checkout -- <file>), or commit them on the Mac, push, then here: git pull"; exit 1; }
    mark='C1-E3 pilot: K=[0-9]+ tau_k=[0-9.]+ rc=[0-9.]+ dir=[01] beta=[0-9.]+ eps_noise=[0-9.]+ vmin_noise=[0-9.]+ pad=[01]'
    m=$(git show HEAD:$SPEC | grep -oE "$mark" | tail -1)
    c=$(grep -oE "$mark" $O/pilot_choice.txt 2>/dev/null | tail -1)
    [ -n "$m" ] && [ "$m" = "$c" ] || { echo "!!! the spec at HEAD journals '${m:-no pilot line}', the pilot on this"
      echo "    pod chose '${c:-nothing}' ($O/pilot_choice.txt). Journal the pilot's line in the spec on the Mac, commit,"
      echo "    push, then here: git pull"; exit 1; }
    git rev-parse -q --verify @{u} >/dev/null 2>&1 || { echo "!!! no upstream branch: git checkout c1-e3 && git pull"
      exit 1; }
    git fetch -q && git merge-base --is-ancestor HEAD @{u} || {
      echo "!!! HEAD is not pushed (not in its upstream): push the journal commit from the Mac, then here: git pull"; exit 1; }
    if grep -q '^C1-E3 env: SAME' $O/env_check.txt 2>/dev/null; then BASE=results/c1-e2/baseline.json
    elif [ -s $O/baseline.json ]; then BASE=$O/baseline.json
    else echo "!!! the environment differs from C1-E2's pod ($O/env_check.txt) and there is no $O/baseline.json:"
      echo "    ./scripts/gpu_c1e3.sh base (spec §8)"; exit 1; fi
    check_code require
    v() { echo "$m" | grep -oE " $1=[0-9.]+" | cut -d= -f2; }
    CO=(--t-ramp 5 --k-p 1.0 --tau-k "$(v tau_k)" --K "$(v K)" --rc "$(v rc)" --dir "$(v dir)" --beta "$(v beta)"
      --eps-noise "$(v eps_noise)" --vmin-noise "$(v vmin_noise)" --pad-brain "$(v pad)" --n-envs 10)
    echo "[$(date +%T)] grid with ${CO[*]}; baseline $BASE; cut: ${CUT:-none}" | tee -a $O/forecast.txt
    { plan A; plan B; } | awk -F'\t' '{ s += $2 } END { print "planned episodes: " s " of 31150" }' | tee -a $O/forecast.txt
    # run_plan <log name> <out> <plan lines>: in the lane's own shell, not a pipeline, so the lane's TERM (Ctrl-C) reaches
    # wd's trap, which kills the runner's process group
    run_plan() {
      local row n args
      while IFS=$'\t' read -r row n args; do
        [ -n "$row" ] || continue
        echo "[$(date +%T)] row $row ($n episodes): $args" | tee -a $LOG/$1.log
        # shellcheck disable=SC2086
        run2 "$1" $args "${CO[@]}" --out "$2" </dev/null
      done <<< "$3"
    }
    forecast() {  # forecast <lane> <log name> <plan args> <lines in the rates file before the lane's first line>: that
      # line's pass-1 rate x the lane's planned episodes
      local r n h
      r=$(tail -n +$(($4 + 1)) $LOG/$2.rates 2>/dev/null | tail -1)
      n=$(plan $3 | awk -F'\t' '{ s += $2 } END { print s + 0 }')
      [ -n "$r" ] || { echo "[$(date +%T)] !!! no rate for the forecast: lane $1's first line ran no episode here (done" \
        "before a restart, or failed); $n episodes planned, the owner checks the rate in $LOG/$2.log (spec §11 fuse)" \
        | tee -a $O/forecast.txt; return; }
      h=$(awk -v r="$r" -v n="$n" 'BEGIN { printf "%.1f", r * n / 3600 }')
      echo "[$(date +%T)] lane $1 forecast: $r s/episode x $n episodes = $h h" | tee -a $O/forecast.txt
      awk -v h="$h" 'BEGIN { exit !(h > 16) }' && { echo "!!! lane $1 forecast > 16 h (spec §11 fuse): the owner cuts"
        echo "    report rows in the order 26 22 24 23 15 19 17 9 8 4 25x, then 14 25: Ctrl-C, then"
        echo "    CUT=\"26 22 ...\" ./scripts/gpu_c1e3.sh grid. The run goes on until then."; } | tee -a $O/forecast.txt
    }
    lane_a() {
      local p k
      p=$(plan A) || return
      k=$(cat $LOG/grid.rates 2>/dev/null | wc -l)
      run_plan grid $O/grid.jsonl "$(head -1 <<< "$p")"
      forecast A grid A "$k"
      run_plan grid $O/grid.jsonl "$(tail -n +2 <<< "$p")"
    }
    lane_b() {
      local p q k why hold=
      p=$(plan B) || return
      k=$(cat $LOG/grid_b.rates 2>/dev/null | wc -l)
      run_plan grid_b $O/grid_b.jsonl "$(head -1 <<< "$p")"  # row 10: the Q3a gate
      echo "=== $(date -u) Q3a gate" >> $O/q3a.txt
      $P scripts/c1e3_summary.py --q3a $O/grid_b.jsonl 2>&1 | tee -a $O/q3a.txt; q=${PIPESTATUS[0]}
      case $q in  # the summary exits 3 if Q3a does not pass; any other error is the gate not running
        0) why= ;;
        3) why="Q3a did not pass" ;;
        *) why="Q3a gate failed to run (exit $q)" ;;
      esac
      if [ -z "$why" ]; then echo "Q3a passed: the alpha-brain rows run" | tee -a $O/q3a.txt
      elif [ "${ALPHA_ROWS:-}" = run ]; then echo "!!! $why; ALPHA_ROWS=run: the alpha-brain rows run" | tee -a $O/q3a.txt
      else hold=--hold; { echo "!!! $why: the other alpha-brain rows are held for the owner (spec §8). To run them:"
        echo "    Ctrl-C (or wait for the end), then ALPHA_ROWS=run ${CUT:+CUT=\"$CUT\" }./scripts/gpu_c1e3.sh grid"
        echo "    (resume skips finished episodes)"; } | tee -a $O/q3a.txt; fi
      forecast B grid_b "B $hold" "$k"
      p=$(plan B $hold) || return
      run_plan grid_b $O/grid_b.jsonl "$(tail -n +2 <<< "$p")"
    }
    pa= pb=
    trap 'kill -TERM $pa $pb 2>/dev/null; wait; exit 130' INT TERM HUP
    lane_a & pa=$!
    lane_b & pb=$!
    wait $pa; wait $pb
    trap - INT TERM HUP
    cat $O/grid.jsonl $O/grid_b.jsonl > $O/grid_all.jsonl
    echo "[$(date +%T)] grid_all.jsonl: $(wc -l < $O/grid_all.jsonl) of 31150 records (the summary checks each)"
    for f in $LOG/grid*.log; do echo "  $f: $(grep -c FAILED "$f") FAILED lines (first-pass failures included)"; done
    echo "On the Mac: python scripts/c1e3_summary.py $O/grid_all.jsonl --baseline $BASE"
    ;;
esac
