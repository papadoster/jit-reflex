#!/usr/bin/env bash
# C1-E4 part 2 on one Linux NVIDIA GPU host, spec docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md sections 3-7,
# 9, 13. Pod: RunPod Community RTX 4090 (~$0.35/h), >= 16 vCPUs, EGL, >= 120 GB on /workspace (LIBERO-plus assets 6.4 GB
# zipped, 9.4 GB unpacked; OFT and OFT+ 15 GB each; two venvs), root (apt-get), tmux; a Hugging Face token with access
# to the gated google/paligemma-3b-pt-224 (pi0.5's tokenizer; huggingface-cli login).
#   tmux new -s c1e4m   (after a lost SSH session: tmux attach -t c1e4m)
#   cd /workspace && git clone https://github.com/papadoster/jit-reflex.git && cd jit-reflex && git checkout c1-e4
# Stages, each run separately after the owner's go-ahead, in this order:
#   ./scripts/gpu_c1e4m.sh setup    libmagickwand-dev; venv (LeRobot, part 1's pins + the scene packages) and
#                                   $DEPS/venv-oft (openvla-oft + the transformers fork, pinned commits, + the scene
#                                   packages); LIBERO-MAX, LIBERO-plus, LIBERO-PRO at their pins; PRO data and the two
#                                   scene configs; LIBERO-plus assets, pi0.5 and OFT / OFT+ weights under their SHA-256
#   ./scripts/gpu_c1e4m.sh smoke    HF offline, never read: first 2 motion-blur cases through real Wand (SmolVLA, Base);
#                                   OFT / OFT+ self-tests and TF's GPU memory; latency, one row per call; per brain the
#                                   GPU peak (pi0.5: 6 envs, then 5, 4 while > 23 GB), tempo, repeat check and pair
#                                   validity on 20 calibration debug cases (EGL leak watch on pi0.5's); a bench move
#                                   smoke; crash and resume; async against sync; the OFT venv checks -> tempo.txt and
#                                   forecast.txt. The owner picks the fuse cut (spec §13) from forecast.txt and journals
#                                   it; M2 rides on M1's first pi0.5 line, so its cut is decided before the grid
#   ./scripts/gpu_c1e4m.sh connect  spec §9: Base on the 300 connect cases, 4 brains, then the connection gate. Fails:
#                                   results/c1-e4/part2/pod/connect_fail, everything stops for the analysis
#   ./scripts/gpu_c1e4m.sh calib    spec §5: M0 (4 brains, Gcal on the 60 calibration cases) and B-cal (bench, 60
#                                   pairs) -> the line "C1-E4 part 2 calib: ..." in calib.txt. N < 45 for a brain: the
#                                   stage stops with the extra cases needed; the owner freezes and journals the new split
#                                   first (spec §5), git pull, calib again (M0 then reruns into files of the new split)
#   on the Mac: journal the calib line in the spec, commit, push; here: git pull
#   CUT="D M2" ./scripts/gpu_c1e4m.sh grid   spec §6, §7, §13: the plan's grid lines by priority (M1 pi0.5 -> M1 OFT+
#                                   -> M4 SmolVLA -> B oracle -> M4 OFT -> B noisy -> D; M2 with M1's first 300),
#                                   one runner process on the GPU at a time, the forecast after each line, then
#                                   grid_all.jsonl and the summary. Refuses unless the spec at HEAD journals the pod's
#                                   calib line, HEAD is pushed, the tree is clean, connect passed and the runner code is
#                                   connect's (results/c1-e4/part2/pod/CODE). Fuse order D M2 Bnoisy M4oft B59 (report
#                                   blocks only): Ctrl-C, then CUT="..." ./scripts/gpu_c1e4m.sh grid (resume)
# One stage at a time on a pod (flock). Each runner line has its own --out (block, brain, set, source, d), runs twice
# (pass 2 resumes and retries failed groups, scripts/gpu_c1e3.sh's watchdog STALL, default 1200 s) and a stage fails if a
# runner still exits non-zero after pass 2. Every exit appends the environment to pod_env_<stage>.txt, refreshes
# MANIFEST.sha256 and packs results/c1-e4/part2/pod into c1e4m_results.tgz.
# DRY=1 [DRY_PY=python-with-numpy] [KBAR_LINE="C1-E4 part 2 calib: ..."]: print every command instead of running it
# (setup included), files of this script to a temporary dir: a dry run on the Mac.
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=${DRY:-}
dry() { [ -n "$DRY" ]; }
dry || [ -n "${TMUX:-}" ] || { echo "!!! not inside tmux: a lost SSH session kills the workers (tmux new -s c1e4m)"
  sleep 10; }
O=results/c1-e4/part2/pod  # the runners' records
W=$O  # this script's own files; a dry run writes them to a temporary dir, never into the repository
dry && W=$(mktemp -d)
LOG=$W/logs
P=venv/bin/python
SP=$P  # the summary's python (numpy, hf-libero); a dry run on the Mac: DRY_PY
dry && SP=${DRY_PY:-python3}
SPEC=docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md
DEPS=${DEPS:-/workspace/lmax-deps}  # outside the repository: no LIBERO-MAX file in it (it has no license, spec §3)
LMAX=$DEPS/LIBERO-MAX
VO=$DEPS/venv-oft
PO=$VO/bin/python
VID=${VID:-/workspace/c1e4m_videos}  # outside the repository
STALL=${STALL:-1200}
PEAK_MAX=23552  # MiB: part 1's 23 GB
PI_REPO=lerobot/pi05_libero_finetuned_v044
PI_REV=8e174154ef5f6c60a8da12ae99c303d8963138c1  # part 1 (spec 2026-10-03 §4.1), with the SHA-256 of model.safetensors
PI_SHA=877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae
LMAX_REV=a1e3cef258b3e00487db5282aef4e36010aaf401  # spec §3
PLUS_REV=4976dc30028e805ff8094b55501d532c48fec182
PRO_REV=2b910b5b5f53016bef9907632f6f840f1ce2229c
PRO_DATA_REV=c86fc3b8293185a6f373677018ff3e37f8391602  # HF dataset zhouxueyang/LIBERO-Pro
ASSETS_REV=dd2bd61b7d9a6fef1abc52d606e983b41886a149  # HF dataset Sylvest/LIBERO-plus, assets.zip (its LFS SHA-256)
ASSETS_SHA=96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
OFT_GIT=e4287e94541f459edc4feabc4e181f537cd569a8  # github.com/moojink/openvla-oft main, 2026-10-05 (git ls-remote)
TF_GIT=bc339d9ad707454c0c115970db43c260067c61ab  # moojink/transformers-openvla-oft, tag v4.40.1-openvla-oft
DLIMP_GIT=040105d256bd28866cc6620621a3d5f7b6b91b46  # moojink/dlimp_openvla main (openvla-oft's unpinned dependency)
# the weight files of OFT / OFT+ at the revisions of scripts/c1e4m_brains.py OFT, LFS SHA-256 from the HF API (metadata
# only); config.json is not checked: openvla-oft's get_vla rewrites it on load
OFT_SUMS="oft 8b30a7951e68703e1731957190d9f1d6e1eaa82f05b53909608eb0510875b11d model-00001-of-00004.safetensors
oft af4773166950ddda1da6b3c5367796a52e7d5e7216f97041d9ad0721a25e53fe model-00002-of-00004.safetensors
oft 5a62791dd46ec4a85b353920aa07eff3582ceb5e9b6b853ad0f38de36fcab75d model-00003-of-00004.safetensors
oft a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60 model-00004-of-00004.safetensors
oft f9647226ca9a1ee64ff8ed1ec380c89110afc6fc2bf6d0ae463cc4023165b4dc action_head--300000_checkpoint.pt
oft 1792c6e19381d3c7c814e2d49836a855d3a38b0693919416f52b3c5af39505c8 proprio_projector--300000_checkpoint.pt
oftplus 45f48b9bc6d7535f5498e826b2b0d9da4aae03853bd4681a10decb480de55fd3 model-00001-of-00004.safetensors
oftplus 47500ca78f427bfb675879d93bea6f514480f460740610458addef1921c4522b model-00002-of-00004.safetensors
oftplus 17cc127c4ed727708a6138103392b79290d45bb8c9edac949674625b0d2d0bf5 model-00003-of-00004.safetensors
oftplus a877e3fece1feafb80f59f91585ce04379ee39e2bf9a25cb7b4acf237e896e60 model-00004-of-00004.safetensors
oftplus adc70a2da8cadd90b5f5cdec1771058130e62e48ede4c8cd351a8c35ee740330 action_head--150000_checkpoint.pt
oftplus 491e30d2aaa1d9ff33521a93a511b1828f8deb33515dd2b43b470ade74569221 proprio_projector--150000_checkpoint.pt"
LINE='C1-E4 part 2 calib: kbar_pi05=-?[0-9]+\.[0-9]+ \[-?[0-9]+\.[0-9]+, -?[0-9]+\.[0-9]+\] N=[0-9]+( · (bench: )?kbar_[a-z0-9]+=-?[0-9]+\.[0-9]+ \[-?[0-9]+\.[0-9]+, -?[0-9]+\.[0-9]+\] N=[0-9]+){5}'
mkdir -p $LOG
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}  # part 1's journal "trial"
export OPENVLA_OFT_ROOT=$DEPS/openvla-oft C1E4_OFT_CKPT=${C1E4_OFT_CKPT:-/workspace/oft-ckpt} \
  TF_FORCE_GPU_ALLOW_GROWTH=true  # openvla-oft imports TensorFlow (image resize): it must not take the GPU's memory
if [ -d /workspace ]; then
  export HF_HOME=/workspace/.cache/huggingface UV_CACHE_DIR=/workspace/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/workspace/.cache/uv-python
fi

x() { if dry; then echo "+ $*"; else "$@"; fi; }  # a command with side effects: printed in a dry run
wr() { local f=$1; shift; if dry; then echo "+ write $f: $*"; else printf '%s\n' "$@" > "$f"; fi; }
fail() { echo "!!! $*"; exit 1; }
refuse() { echo "!!! $*"; dry && echo "    (DRY: going on)" || exit 1; }

setup() {
  # LIBERO-plus's motion blur (19 sensor-noise cases) needs ImageMagick's MagickWand (Wand); spec §3
  if dry || ! dpkg -s libmagickwand-dev >/dev/null 2>&1; then
    dry || [ "$(id -u)" = 0 ] || fail "not root: run  apt-get update && apt-get install -y libmagickwand-dev  first"
    x apt-get update -qq && x env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libmagickwand-dev </dev/null \
      || exit 1
  fi
  command -v uv >/dev/null || x sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' || exit 1
  # the scene packages at the Mac's versions (~/Desktop/M2R-lmax-deps/pylib); Wand: the pod only
  SCENE=(scikit-image==0.26.0 imageio==2.38.0 tifffile==2026.9.20 lazy_loader==0.6 Wand==0.6.13)
  if dry || [ ! -f venv/c1e4m-installed ] || ! $P -V >/dev/null 2>&1; then  # part 1's venv (scripts/gpu_c1e4.sh)
    local WH=https://download.pytorch.org/whl/cu126
    PIN=("torch @ $WH/torch-2.11.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      "torchvision @ $WH/torchvision-0.26.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
      transformers==5.5.4 numpy==2.2.6 gymnasium==1.3.0 mujoco==3.3.2)
    x uv venv --clear --python 3.12 venv || exit 1
    x uv pip install --python $P "lerobot[smolvla]==0.6.1" "${PIN[@]}" || exit 1
    x uv pip install --python $P --no-deps hf-libero==0.1.4 robomimic==0.2.0 || exit 1
    x uv pip install --python $P "hydra-core>=1.2,<1.4" robosuite==1.4.0 bddl==1.0.1 easydict einops thop matplotlib \
      cloudpickle opencv-python future "lerobot[dataset,scipy-dep]==0.6.1" "${PIN[@]}" "${SCENE[@]}" || exit 1
    x touch venv/c1e4m-installed
  fi
  x mkdir -p $DEPS
  clone() {  # clone <url> <dir> <commit>
    { [ -d "$2/.git" ] || x git clone -q "$1" "$2"; } && x git -C "$2" checkout -q "$3" || exit 1
  }
  clone https://github.com/liberomax/LIBERO-MAX.git $LMAX $LMAX_REV
  clone https://github.com/sylvestf/LIBERO-plus.git $DEPS/LIBERO-plus $PLUS_REV
  clone https://github.com/Zxy-MLlab/LIBERO-PRO.git $DEPS/LIBERO-PRO $PRO_REV  # its assets are in its git tree
  clone https://github.com/moojink/openvla-oft.git $DEPS/openvla-oft $OFT_GIT
  # openvla-oft's venv (python 3.10, torch 2.2.0 as its pyproject) with the transformers fork: plain transformers
  # silently gives wrong chunks (parallel decoding needs the fork's bidirectional attention); plus the scenes
  if dry || [ ! -f $VO/c1e4m-installed ] || ! $PO -V >/dev/null 2>&1; then
    x uv venv --clear --python 3.10 $VO || exit 1
    wr $DEPS/oft-overrides.txt "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@$TF_GIT" \
      "dlimp @ git+https://github.com/moojink/dlimp_openvla.git@$DLIMP_GIT"
    x uv pip install --python $PO --override $DEPS/oft-overrides.txt -e $DEPS/openvla-oft robosuite==1.4.0 mujoco==3.3.2 \
      bddl==1.0.1 "hydra-core>=1.2,<1.4" easydict thop cloudpickle opencv-python future gymnasium==1.3.0 \
      scikit-image==0.25.2 imageio==2.38.0 lazy_loader==0.6 Wand==0.6.13 || exit 1  # 0.26: python >= 3.11
    x uv pip install --python $PO --no-deps hf-libero==0.1.4 robomimic==0.2.0 || exit 1
    x touch $VO/c1e4m-installed
  fi
  # LIBERO-PRO's data and config through LIBERO-MAX's own setup script (it checks LIBERO-PRO's commit); it calls hf
  x env PATH="$PWD/venv/bin:$PATH" $P $LMAX/scripts/setup_libero_pro_substrate.py --libero-pro-root $DEPS/LIBERO-PRO \
    --dataset-root $DEPS/libero-pro-data --config-dir $DEPS/libero-pro-config --revision $PRO_DATA_REV \
    --runtime-revision $PRO_REV || exit 1
  local PL=$DEPS/LIBERO-plus/libero/libero  # as LIBERO-MAX's docs/RUNTIME_INTEGRATION.md (JSON is YAML)
  x mkdir -p $DEPS/libero-plus-config
  wr $DEPS/libero-plus-config/config.yaml "{\"benchmark_root\": \"$PL\", \"bddl_files\": \"$PL/bddl_files\", \
\"init_states\": \"$PL/init_files\", \"assets\": \"$PL/assets\", \"datasets\": \"$DEPS/libero-plus-data\"}"
  if dry || [ "$(cat $DEPS/plus-assets-ok 2>/dev/null)" != $ASSETS_SHA ]; then  # LIBERO-plus's README: unzip there
    x $P -c "from huggingface_hub import hf_hub_download as d
d('Sylvest/LIBERO-plus', 'assets.zip', repo_type='dataset', revision='$ASSETS_REV', local_dir='$DEPS/dl')" || exit 1
    x sh -c "echo '$ASSETS_SHA  $DEPS/dl/assets.zip' | sha256sum -c -" || fail "LIBERO-plus assets.zip: SHA-256 mismatch"
    x $P -c "import zipfile; zipfile.ZipFile('$DEPS/dl/assets.zip').extractall('$PL')" || exit 1
    x rm -f $DEPS/dl/assets.zip
    wr $DEPS/plus-assets-ok $ASSETS_SHA
  fi
  # standard LIBERO (hf-libero) for the bench, the self-tests and the latency: assets and ~/.libero (both venvs)
  for py in $P $PO; do
    x sh -c "echo N | $py -c \"import os, libero.libero as L; a = L.get_assets_path()
assert os.path.isdir(a), f'no LIBERO assets at {a}'; print('LIBERO assets:', a)\"" || exit 1
    x $py -c "import wand.image; wand.image.Image(width=8, height=8); print('Wand: MagickWand loads')" || exit 1
  done
  x $P -c "import torch; assert torch.cuda.is_available(), 'CUDA is not visible to torch'
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))" || exit 1
  local probe='import os, mujoco
mujoco.GLContext(64, 64).make_current()
from OpenGL import GL; r = GL.glGetString(GL.GL_RENDERER).decode()
print("GL renderer:", r)
assert not any(s in r.lower() for s in ("llvmpipe", "softpipe", "swrast")), "EGL renders on the CPU"'
  if ! dry && ! $P -c "$probe" </dev/null; then  # as scripts/gpu_c1e4.sh
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libopengl0 libglib2.0-0 \
      </dev/null
    $P -c "$probe" </dev/null || fail "offscreen EGL rendering does not work on this pod: the owner decides"
  fi
  # weights: pi0.5 and SmolVLA as part 1 (pinned revisions, pi0.5's SHA-256); building both once on the CPU caches
  # every file they read (PaliGemma's gated tokenizer, SmolVLM's processor), so later stages run HF offline
  x $P -c "import sys; sys.path.insert(0, 'scripts'); import c1e4m_brains as cb
for b in ('pi05', 'smolvla'): cb.make_brain(b, 'cpu', 'fp32'); print(b, 'cached')" \
    || fail "a LeRobot brain does not build (pi0.5: huggingface-cli login with a token that accepted the Gemma terms)"
  x $P -c "from huggingface_hub import snapshot_download as d
p = d('$PI_REPO', revision='$PI_REV'); assert '$PI_REV' in p; open('venv/pi05-snapshot', 'w').write(p)" || exit 1
  if dry || [ "$(cat venv/pi05-sha-ok 2>/dev/null)" != "$(cat venv/pi05-snapshot)" ]; then
    x sh -c "echo '$PI_SHA  $(cat venv/pi05-snapshot 2>/dev/null)/model.safetensors' | sha256sum -c -" \
      || fail "pi0.5 weights do not match the spec's SHA-256"
    x cp venv/pi05-snapshot venv/pi05-sha-ok
  fi
  # OFT / OFT+: the local_dir copy scripts/c1e4m_brains.py OFTBrain loads (same path, no lora_adapter/), then the SHA-256
  # of its weight files (checked once)
  local b dir get
  for b in oft oftplus; do
    get="import sys; sys.path.insert(0, 'scripts'); import c1e4m_brains as cb
from huggingface_hub import snapshot_download as d; r, v = cb.OFT['$b']
print(d(r, revision=v, local_dir='$C1E4_OFT_CKPT/' + r.split('/')[1] + '-' + v[:10], ignore_patterns=['lora_adapter/*']))"
    dry && { echo "+ $PO -c \"$get\""; echo "+ sha256sum -c: the 6 weight files of $b (OFT_SUMS)"; continue; }
    dir=$($PO -c "$get" | tail -1) || exit 1
    [ "$(cat $C1E4_OFT_CKPT/$b-sha-ok 2>/dev/null)" = "$dir" ] && continue
    awk -v b=$b '$1 == b { print $2 "  " $3 }' <<< "$OFT_SUMS" | (cd "$dir" && sha256sum -c -) \
      || fail "$b weights do not match their SHA-256"
    echo "$dir" > $C1E4_OFT_CKPT/$b-sha-ok
  done
  x touch $DEPS/c1e4m-setup-ok
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
}

record() {  # record <stage>: appends this run's environment to $O/pod_env_<stage>.txt
  { echo "=== $(date -u) $1, HEAD $(git rev-parse HEAD), MUJOCO_GL=$MUJOCO_GL, vCPUs $(nproc)"
    echo "code ($CODE_PATHS):" $(code_hash)
    for d in $LMAX $DEPS/LIBERO-plus $DEPS/LIBERO-PRO $DEPS/openvla-oft; do echo "$d: $(git -C $d rev-parse HEAD)"; done
    cat $DEPS/libero-pro-config/source_lock.json
    echo "LIBERO-plus assets SHA-256: $(cat $DEPS/plus-assets-ok 2>/dev/null || echo no)"
    echo "pi0.5 weights SHA-256 checked: $(cat venv/pi05-sha-ok 2>/dev/null || echo no)"
    for b in oft oftplus; do echo "$b weights SHA-256 checked: $(cat $C1E4_OFT_CKPT/$b-sha-ok 2>/dev/null || echo no)"; done
    echo "HuggingFaceVLA/smolvla_libero snapshots:" \
      $(ls "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--HuggingFaceVLA--smolvla_libero/snapshots)
    timeout 60 nvidia-smi
    for py in $P $PO; do echo "--- $py"; $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
      uv pip freeze --python $py; done; } >> $O/pod_env_$1.txt 2>&1
}

tgz() {  # tgz <archive> <paths..>: to a .tmp, then renamed, so an interrupted tar never leaves a truncated archive
  tar czf "$1.tmp" "${@:2}" && mv -f "$1.tmp" "$1" && echo "[$(date +%T)] packed $1 ($(du -h "$1" | cut -f1))"
}
pack() {
  (cd $O && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  tgz c1e4m_results.tgz $O
}

CODE_PATHS="src scripts/c1e4m_run.py scripts/c1e4m_brains.py scripts/c1e4_run.py"
code_hash() {
  local f
  for f in $CODE_PATHS; do git rev-parse -q --verify "HEAD:$f" || { echo "!!! $f is not in HEAD" >&2; return 1; }; done
}
check_code() {  # check_code record|require: connect, calib and grid run the same runner code (recorded by connect)
  local h
  dry && { echo "+ check_code $1 ($CODE_PATHS at HEAD vs $O/CODE)"; return; }
  h=$(code_hash) || exit 1
  [ -z "$(git status --porcelain -- $CODE_PATHS)" ] || { git status --short -- $CODE_PATHS
    fail "uncommitted or untracked files in the runner code (above): commit them on the Mac and pull, or remove them"; }
  if [ "$1" = record ] && [ ! -f $O/CODE ]; then echo "$h" > $O/CODE; return; fi
  [ "$(cat $O/CODE 2>/dev/null)" = "$h" ] || fail "the runner code is not connect's ($O/CODE): the owner decides (archive
    the results under $O and rerun from connect, or check out connect's commit)"
}

wd() {  # as scripts/gpu_c1e3.sh: the command in its own process group, killed after $STALL s without log growth
  local log=$1 pid n m t=$SECONDS rc; shift
  dry && { echo "+ $*"; return 0; }
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

run2() {  # run2 <log name> <runner command..>: pass 1, then pass 2 (resume retries failed groups) unless pass 1 was clean
  local log=$LOG/$1.log n rc f small=; shift
  : >> "$log"; n=$(wc -c < "$log")
  echo "[$(date +%T)] pass 1: $*" | tee -a "$log"
  wd "$log" "$@"; rc=$?
  f=$(tail -c +$((n + 1)) "$log" | grep -c FAILED)
  [ $rc = 0 ] || echo "!!! runner exited with an error (pass 1)" | tee -a "$log"
  if [ $rc = 0 ] && [ "$f" = 0 ]; then
    echo "[$(date +%T)] pass 2 skipped: pass 1 exited cleanly, no failed group" | tee -a "$log"; return 0
  fi
  [ "$f" -le 3 ] && small="--n-envs 2"  # the last --n-envs wins; fp32 and one-row chunks do not depend on the batch
  echo "[$(date +%T)] pass 2 ($f failed groups in pass 1, exit $rc): $* $small" | tee -a "$log"
  wd "$log" "$@" $small; rc=$?
  [ $rc = 0 ] || echo "!!! runner exited with an error (pass 2)" | tee -a "$log"
  return $rc
}

peak_run() {  # peak_run <peak file> <command..>: the command while nvidia-smi samples the GPU memory (MiB) every second
  local f=$1 s rc; shift
  dry && { "$@"; echo 0 > "$f"; return; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$f.samples" 2> "$f.err" & s=$!
  "$@"; rc=$?
  kill $s 2>/dev/null; wait $s 2>/dev/null
  sort -n "$f.samples" | tail -1 > "$f"
  return $rc
}

mrun() {  # mrun <source> <brain>: the words of scripts/c1e4m_run.py in the brain's venv with the source's scene env
  local r=LIBERO-plus py=$P
  [ "$1" = pro ] && r=LIBERO-PRO
  case $2 in oft | oftplus) py=$PO ;; esac
  echo env LIBERO_SOURCE_PACKAGE_ROOT=$DEPS/$r/libero LIBERO_CONFIG_PATH=$DEPS/libero-$1-config \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=$LMAX/scripts/libero_source_overlay:$LMAX/src LIBERO_MAX_ROOT=$LMAX \
    $py scripts/c1e4m_run.py --device cuda
}
BRUN="$P scripts/c1e4_run.py --vector async --device cuda"  # the bench: standard LIBERO, no scene env
# seconds per episode: LIBERO-MAX "group G: N cases M episodes X s"; the bench's full batches as scripts/gpu_c1e4.sh
gtempo() { awk '$1 == "group" && $4 == "cases" { s += $7; n += $5 } END { if (n) printf "%.3f", s / n }' "$@" 2>/dev/null; }
full_rate() {
  awk '/^batch / { split($4, r, "[=/]"); if (r[2] == r[3]) { s += $5; n += r[2] } } END { if (n) printf "%.3f", s / n }' \
    "$@" 2>/dev/null
}
envs() {  # envs <runner c1e4m|c1e4> <brain>: LIBERO-MAX the smoke's choice; the bench part 1's (pi0.5 6, SmolVLA 10)
  local e
  [ $1 = c1e4 ] && { [ $2 = pi05 ] && echo 6 || echo 10; return; }
  e=$(grep -oE "C1-E4m envs: .*" $W/tempo.txt 2>/dev/null | tail -1 | grep -oE " $2=[0-9]+" | cut -d= -f2)
  echo "${e:-$([ $2 = smolvla ] && echo 8 || echo 6)}"  # the default: a dry run only (the stages need the smoke's)
}
dbg() {  # dbg <source> <k>: the first k calibration cases of that source (debug cases, comma-separated)
  $SP -c "import json; c = json.load(open('results/c1-e4/part2/split.json'))['calib']
print(','.join([i for i in c if i.startswith('pro-') == ('$1' == 'pro')][:$2]))"
}
summ() { $SP scripts/c1e4m_summary.py "$@"; }
oft_numpy() {  # spec §4: openvla-oft's import makes json.dumps pass numpy silently; the runner writes strict JSON
  dry && return 0
  ! grep -l '"__numpy__"' $O/*oft*.jsonl 2>/dev/null || fail "\"__numpy__\" in the OFT records above"
}

run_phase() {  # run_phase <phase> <plan>: the phase's lines one after another (one runner on the GPU at a time); 1 when
  local ph rn b src id n args bad=  # a runner still exits non-zero after its pass 2 (after running the others)
  while IFS=$'\t' read -r ph rn b src id n args; do
    [ "$ph" = "$1" ] || continue
    local out=$O/$id.jsonl
    # M0 per split: an extended split (spec §5) changes the runner's conf, its resume key
    [ "$id" != "${id#M0_}" ] && out=$O/${id}_$(summ_sha).jsonl
    echo "[$(date +%T)] phase $ph, $id ($n episodes): $args" | tee -a $LOG/$ph.log
    if [ $rn = c1e4m ]; then
      [[ "$args " == *"--source $src "* ]] || fail "$id: --source is not its scene env $src"
      # shellcheck disable=SC2086
      run2 $id $(mrun $src $b) $args --n-envs $(envs c1e4m $b) --out $out </dev/null || bad+=" $id"
    else
      # shellcheck disable=SC2086
      run2 $id $BRUN $args --n-envs $(envs c1e4 $b) --out $out </dev/null || bad+=" $id"
    fi
    [ "$1" = grid ] && fcast $id $([ $rn = c1e4m ] && echo lmax || echo bench)_$b
  done <<< "$2"
  [ -z "$bad" ] || { echo "!!! still failing after pass 2:$bad" | tee -a $LOG/$1.log; return 1; }
}
summ_sha() { $SP -c "import json; print(json.load(open('results/c1-e4/part2/split.json'))['sha256'][:12])"; }
fcast() {  # fcast <line id> <tempo key>: this line's tempo replaces the smoke's for that brain, then the forecast again
  local r t
  dry && { echo "+ summ --forecast $W/tempo.txt --cut \"$CUT\" (after $1)"; return; }
  r=$([ "${2%%_*}" = lmax ] && gtempo $LOG/$1.log || full_rate $LOG/$1.log)
  t=$(grep -E '^tempo \(s/episode\): ' $W/tempo.txt | tail -1)
  [ -n "$r" ] && t=$(sed -E "s/ $2=[0-9.]+/ $2=$r/" <<< "$t") && echo "$t" >> $W/tempo.txt
  { echo "[$(date +%T)] forecast after $1 ($2 ${r:-?} s/episode)"; summ --forecast $W/tempo.txt --cut "$CUT"; } \
    | tee -a $W/forecast.txt | grep -E '"(total|planned)"' -A2
  echo "    fuse (spec §13, owner's target 32-40 h): cut in the order D M2 Bnoisy M4oft B59 by Ctrl-C, then" \
    "CUT=\"...\" ./scripts/gpu_c1e4m.sh grid (resume)"
}

CUT=${CUT:-}
case "${1:-}" in
  setup | smoke | connect | calib | grid) ;;
  *) echo "usage: $0 setup | smoke | connect | calib | grid"; exit 1 ;;
esac
STAGE=$1
# one stage at a time: a second instance would run the same episodes into the same files (the lock is outside results)
if command -v flock >/dev/null; then
  exec 9>"$(git rev-parse --git-dir)/c1e4m-pod.lock"  # in .git: not under a venv that setup clears
  flock -n 9 || fail "a stage already runs on this pod (it holds the lock in .git): tmux attach -t c1e4m"
else
  echo "!!! no flock here: nothing stops a second instance"
fi
[ -z "$(git ls-files $O)" ] || fail "git tracks results of an earlier run under $O: resume would see them as done.
    Run from a commit without them, or move them away, as the owner decides."
[ $STAGE = setup ] || summ --plan --cut "$CUT" >/dev/null || exit 1
dry || trap 'trap "" INT TERM HUP; record $STAGE; pack' EXIT
if [ $STAGE = setup ]; then
  setup 2>&1 | tee -a $LOG/setup.log
  exit "${PIPESTATUS[0]}"
fi
dry || [ -f $DEPS/c1e4m-setup-ok ] || fail "no $DEPS/c1e4m-setup-ok: ./scripts/gpu_c1e4m.sh setup first"
[ $STAGE = smoke ] || dry || grep -q '^C1-E4m envs:' $W/tempo.txt 2>/dev/null \
  || fail "no smoke tempo and envs ($W/tempo.txt): ./scripts/gpu_c1e4m.sh smoke first"
export HF_HUB_OFFLINE=1  # everything is cached by setup: no silent weight change, no Hub outage mid-run
case "$STAGE" in
  smoke)
    dry || rm -f $O/smoke_*.jsonl $W/latency.jsonl $W/tempo.txt $W/leak.txt $LOG/smoke*.log $LOG/tempo*.log
    # 1. in the first minutes, LIBERO-plus's motion blur through real Wand (ImageMagick): SmolVLA, Base, 2 of the 5
    #    motion-blur cases of connect (calibration has none), with videos to look at
    BLUR=libero_goal-t1584-target_relocation-d1-p195,libero_object-t1399-distractor_burst-d1-p195
    wd $LOG/smoke_blur.log $(mrun plus smolvla) --brain smolvla --source plus --set connect --cases $BLUR --arms= \
      --n-envs 2 --video-cases $BLUR --video-dir $VID --out $O/smoke_blur.jsonl || fail "the motion-blur smoke failed"
    dry || { grep -q 'Wand real' $LOG/smoke_blur.log && [ "$(grep -c '"skipped_reason": null' $O/smoke_blur.jsonl)" = 2 ] \
      || fail "motion blur did not run through real Wand ($LOG/smoke_blur.log)"; }
    # 2. OFT and OFT+ in their venv, from the repository (not openvla-oft's root): shape, gripper, repeat bitwise;
    #    gymnasium there; TensorFlow holds no GPU memory after the first call (the GPU's used memory, nothing else on
    #    it, against torch's reserve: containers often hide the per-process list)
    x $PO -c "import gymnasium, tensorflow as tf
print('venv-oft: gymnasium', gymnasium.__version__, '| TF sees', tf.config.list_physical_devices('GPU'))" \
      || fail "the OFT venv lacks gymnasium or TensorFlow"
    for b in oft oftplus; do
      wd $LOG/smoke_selftest_$b.log $PO scripts/c1e4m_brains.py --selftest-oft $b --device cuda || fail "$b self-test"
      dry || grep -q 'repeat_bitwise=1' $LOG/smoke_selftest_$b.log || fail "$b: the repeated call is not bitwise"
    done
    wd $LOG/smoke_tfmem.log $PO -c "import subprocess, sys, torch
sys.path.insert(0, 'scripts'); import c1e4m_brains as cb
b = cb.make_brain('oft', 'cuda', 'bf16'); raw, env, text = cb.raw_libero_obs()
b([raw], [env], [text], [0], ['libero_spatial'])
used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.split()[0])
res = torch.cuda.memory_reserved() // 2**20
print(f'C1-E4m tf_mem: GPU {used} MiB, torch reserved {res} MiB, other {used - res} MiB')
assert used - res < 1536, 'something besides torch (TensorFlow?) holds GPU memory'" || fail "TF's GPU memory check"
    # 3. latency, one row per call after warm-up (spec §10, report only)
    for bp in pi05:fp32 smolvla:fp32 oft:bf16 oftplus:bf16; do
      b=${bp%:*}; py=$P; [ $b = pi05 ] || [ $b = smolvla ] || py=$PO
      wd $LOG/smoke_latency_$b.log $py scripts/c1e4_latency.py --brain $b --precision ${bp#*:} --device cuda \
        || fail "$b latency"
      dry || grep -h '^{' $LOG/smoke_latency_$b.log | tail -1 >> $W/latency.jsonl
    done
    # 4. per brain, calibration debug cases (never read), arms none, Gauto (placeholder kbar) and Gcal (the shadow
    #    pairs in the peak): GPU peak, tempo and repeat check on the first 3 of each source (a full Dynamic batch of 6),
    #    then the first 10 of each source into the same files (resume) for the pair validity
    tempo_run() {  # tempo_run <brain> <envs> <cases per source> <log> [runner args..]
      local b=$1 e=$2 k=$3 log=$4 s rc=0; shift 4
      for s in plus pro; do
        wd $log $(mrun $s $b) --brain $b --source $s --set calib --cases $(dbg $s $k) --arms none,Gauto,Gcal --kbar 0.2 \
          --n-envs $e --out $O/smoke_${b}_$s.jsonl "$@" || rc=1
      done
      return $rc
    }
    leak_watch() {  # every 20 s: groups done, GPU memory, each env worker's RSS (EGL contexts are rebuilt per reset)
      while sleep 20; do
        echo "$(date +%T) groups=$(grep -c '^group ' "$1") gpu_mib=$(nvidia-smi --query-gpu=memory.used \
          --format=csv,noheader,nounits | head -1) worker_rss_mib=$(ps -eo rss=,args= \
          | awk '/multiprocessing.spawn/ && !/awk/ { printf "%d ", $1 / 1024 }')"
      done >> $W/leak.txt
    }
    for b in pi05 oftplus oft smolvla; do
      case $b in pi05) try="6 5 4" ;; smolvla) try=8 ;; *) try=6 ;; esac
      for e in $try; do
        dry || rm -f $O/smoke_${b}_*.jsonl
        peak_run $W/peak_$b tempo_run $b $e 3 $LOG/tempo_${b}_e$e.log --repeat-check || fail "$b tempo run ($e envs)"
        dry || grep -qE '^[0-9]+$' $W/peak_$b || fail "no GPU memory samples ($W/peak_$b.err)"
        eval "E_$b=$e"
        echo "$b, $e envs: GPU peak $(cat $W/peak_$b) MiB (max $PEAK_MAX);" \
          "$(grep -h 'C1-E4m repeat' $LOG/tempo_${b}_e$e.log 2>/dev/null | tr '\n' ' ')" | tee -a $W/tempo.txt
        [ "$(cat $W/peak_$b)" -le $PEAK_MAX ] && break
        [ $e = 4 ] && echo "!!! the peak is above $PEAK_MAX MiB at 4 envs too: going on with 4" | tee -a $W/tempo.txt
      done
      rep=$(grep -h 'C1-E4m repeat' $LOG/tempo_${b}_e$e.log 2>/dev/null)
      dry || { [ -n "$rep" ] && ! grep -qv 'bitwise=1' <<< "$rep"; } \
        || echo "!!! $b: the repeat check is missing or not bitwise (report, not a stop)" | tee -a $W/tempo.txt
      lw=; [ $b = pi05 ] && ! dry && { leak_watch $LOG/tempo_${b}_e$e.log & lw=$!; }  # >= 50 starts: 56 new ones
      tempo_run $b $e 10 $LOG/tempo_${b}_e$e.log || fail "$b validity run"
      [ -z "$lw" ] || kill $lw
    done
    x summ --smoke $O/smoke_pi05_*.jsonl $O/smoke_oftplus_*.jsonl $O/smoke_oft_*.jsonl $O/smoke_smolvla_*.jsonl \
      | tee $W/smoke_validity.txt
    [ "${PIPESTATUS[0]}" = 0 ] || fail "pair validity, check (g) or skipped cases in the smoke (above)"
    # 5. the bench, kind move: Glead and the noisy eyes, 2 tasks x 2 inits x 2 cells x 3 arms = 24 per brain
    for b in pi05 smolvla; do
      wd $LOG/smoke_bench_$b.log $BRUN --brain $b --tasks libero_spatial:0,libero_object:0 --kinds move --cells A,A20 \
        --methods none,Glead,Gauto@cv --inits 48-49 --kbar 0.2 --n-envs 6 --out $O/smoke_bench_$b.jsonl \
        || fail "$b bench move smoke"
      dry || [ "$(wc -l < $O/smoke_bench_$b.jsonl)" = 24 ] || fail "$b bench move smoke: not 24 records"
    done
    # 6. crash and resume (SmolVLA, Base, 6 cases in groups of 2): a killed env worker fails its group and the run
    #    exits non-zero; a runner killed -9 with its last line cut in half (a kill inside a write) resumes with no
    #    duplicate line
    CR=(--brain smolvla --source plus --set calib --cases $(dbg plus 6) --arms= --n-envs 2 --out $O/smoke_crash.jsonl)
    if dry; then
      echo "+ $(mrun plus smolvla) ${CR[*]} &   # kill -9 one env worker after 'group 0:', expect exit 1"
      echo "+ $(mrun plus smolvla) ${CR[*]} &   # kill -9 the runner after 'group 0:', cut its last line in half"
    else
      upto() { local i; for i in $(seq 900); do grep -q "$1" "$2" && return 0; kill -0 $3 2>/dev/null || return 1
        sleep 1; done; return 1; }
      $(mrun plus smolvla) "${CR[@]}" > $LOG/smoke_crash_a.log 2>&1 & pid=$!
      upto 'group 0:' $LOG/smoke_crash_a.log $pid && kill -9 "$(pgrep -P $pid -f multiprocessing.spawn | head -1)"
      wait $pid && fail "a failed group did not make the runner exit non-zero ($LOG/smoke_crash_a.log)"
      grep -q 'FAILED' $LOG/smoke_crash_a.log || fail "the killed worker did not fail its group"
      $(mrun plus smolvla) "${CR[@]}" > $LOG/smoke_crash_b.log 2>&1 & pid=$!
      upto 'group 0:' $LOG/smoke_crash_b.log $pid; pkill -9 -P $pid; kill -9 $pid 2>/dev/null; wait $pid
      $SP -c "import os; p = '$O/smoke_crash.jsonl'; b = open(p, 'rb').read(); k = b.rstrip(b'\n').rfind(b'\n') + 1
os.truncate(p, k + (len(b) - k) // 2); print('cut', (len(b) - k) - (len(b) - k) // 2, 'bytes of the last line')"
    fi
    wd $LOG/smoke_crash_c.log $(mrun plus smolvla) "${CR[@]}" || fail "the resumed crash run did not end cleanly"
    dry || $SP -c "import json; rs = [json.loads(l) for l in open('$O/smoke_crash.jsonl')]
keys = [(r['case_id'], r['kind'], r['arm']) for r in rs]
assert len(keys) == len(set(keys)) == 6 and all(k == 'base' for _, k, _ in keys), keys
print('crash and resume: 6 cases, one Base line each, no duplicate')" || fail "crash and resume"
    # 7. async workers against the main process, bitwise, one case at 360 px
    for v in async sync; do
      wd $LOG/smoke_$v.log $(mrun plus smolvla) --brain smolvla --source plus --set calib --cases $(dbg plus 1) \
        --arms none,Gauto --kbar 0.2 --n-envs 2 --vector $v --out $O/smoke_$v.jsonl || fail "the $v run"
    done
    dry || $SP -c "import json; f = lambda v: {(r['kind'], r['arm']): (r['act_hash'], r['success'], r['e'])
    for r in map(json.loads, open(f'$O/smoke_{v}.jsonl'))}
a, s = f('async'), f('sync'); assert a == s and len(a) == 3, (a, s); print('async == sync, bitwise:', a)" \
      || fail "async and sync differ"
    # 8. the OFT venv: s + d <= H = 8 refused before the model loads; no "__numpy__" in any OFT record
    if dry; then echo "+ $(mrun plus oft) --brain oft --source plus --set calib --arms=none --d 8 ...  # must refuse"
    elif $(mrun plus oft) --brain oft --source plus --set calib --cases $(dbg plus 1) --arms=none --d 8 \
      --out $W/never.jsonl > $LOG/smoke_oft_d8.log 2>&1 || ! grep -q 'chunk H = 8' $LOG/smoke_oft_d8.log; then
      fail "OFT took --d 8 (or failed otherwise: $LOG/smoke_oft_d8.log)"
    fi
    oft_numpy
    dry && E_pi05=6 E_oftplus=6 E_oft=6 E_smolvla=8
    t="lmax_pi05=$(gtempo $LOG/tempo_pi05_e$E_pi05.log) lmax_oftplus=$(gtempo $LOG/tempo_oftplus_e6.log)"
    t+=" lmax_oft=$(gtempo $LOG/tempo_oft_e6.log) lmax_smolvla=$(gtempo $LOG/tempo_smolvla_e8.log)"
    t+=" bench_pi05=$(full_rate $LOG/smoke_bench_pi05.log) bench_smolvla=$(full_rate $LOG/smoke_bench_smolvla.log)"
    { echo "tempo (s/episode): $t"; echo "C1-E4m envs: pi05=$E_pi05 oftplus=6 oft=6 smolvla=8"; } | tee -a $W/tempo.txt
    dry || ! grep -qE '=( |$)' <<< "$t" || fail "a tempo is missing (no group / full batch in its log)"
    x summ --forecast $W/tempo.txt | tee $W/forecast.txt
    echo "[$(date +%T)] smoke done: the owner picks the fuse cut from $W/forecast.txt (spec §13) and journals it"
    ;;
  connect)
    check_code record
    run_phase connect "$(summ --plan)" || fail "connect: a runner line failed (above); the gate is not computed"
    oft_numpy
    echo "=== $(date -u) connect gate" >> $W/gates.txt
    x summ --connect $O/connect_*.jsonl | tee -a $W/gates.txt; q=${PIPESTATUS[0]}
    [ $q = 0 ] || { [ $q = 3 ] && x touch $W/connect_fail; fail "connect did not pass (exit $q): everything stops for
    the analysis (spec §9)"; }
    ;;
  calib)
    [ ! -f $W/connect_fail ] || refuse "connect failed ($W/connect_fail): no calibration"
    check_code require
    x summ --connect $O/connect_*.jsonl >/dev/null || refuse "the connect gate has not passed: ./scripts/gpu_c1e4m.sh connect"
    run_phase calib "$(summ --plan)" || fail "calib: a runner line failed (above)"
    oft_numpy
    x summ --calib $O/M0_*.jsonl $O/Bcal_*.jsonl | tee $W/calib.txt; q=${PIPESTATUS[0]}
    [ $q = 4 ] && fail "N < 45 for a brain (above): the owner freezes the split with the extra cases and journals it before any evaluation is read (spec §5); then git pull and
    ./scripts/gpu_c1e4m.sh calib again"
    [ $q = 0 ] || fail "the calibration summary failed (exit $q)"
    echo "On the Mac: journal the line above in the spec ($SPEC), commit, push; here: git pull, then the grid"
    ;;
  grid)
    # the plan comes from the working tree's summary: it, this script and the runners run as committed and pushed
    git diff --quiet HEAD -- . && [ -z "$(git status --porcelain --untracked-files=all -- src scripts)" ] || {
      git status --short -- src scripts; refuse "local edits on the pod (above): undo them, or commit on the Mac, push,
    then here: git pull"; }
    m=$(git show HEAD:$SPEC 2>/dev/null | grep -oE "$LINE" | tail -1)
    dry && [ -z "$m" ] && m=${KBAR_LINE:-}
    c=$(grep -oE "$LINE" $W/calib.txt 2>/dev/null | tail -1)
    [ -n "$m" ] || fail "the spec at HEAD journals no calibration line: journal calib.txt's line on the Mac, push, pull"
    [ "$m" = "$c" ] || refuse "the spec at HEAD journals '$m', this pod's calibration printed '${c:-nothing}'
    ($W/calib.txt)"
    { git rev-parse -q --verify @{u} >/dev/null 2>&1 && git fetch -q && git merge-base --is-ancestor HEAD @{u}; } \
      || refuse "HEAD is not pushed (not in its upstream): push from the Mac, then here: git pull"
    [ ! -f $W/connect_fail ] || refuse "connect failed ($W/connect_fail): no grid"
    check_code require
    PLAN=$(summ --plan --cut "$CUT" --kbar-line "$m") || exit 1
    echo "[$(date +%T)] grid, cut: ${CUT:-none}; $m" | tee -a $W/forecast.txt
    run_phase grid "$PLAN"; rc=$?
    oft_numpy
    if dry; then echo "+ cat <every plan line's --out> > $O/grid_all.jsonl"
    else for id in $(cut -f5 <<< "$PLAN"); do cat $O/$id.jsonl $O/${id}_*.jsonl 2>/dev/null; done > $O/grid_all.jsonl
      echo "[$(date +%T)] grid_all.jsonl: $(wc -l < $O/grid_all.jsonl) records (the summary checks each block)"; fi
    if dry; then echo "+ summ $O/grid_all.jsonl --cut \"$CUT\" --fig $O/fig_target_relocation.png"
    else summ $O/grid_all.jsonl --cut "$CUT" --fig $O/fig_target_relocation.png > $LOG/summary.log
      grep -A4 '"verdicts"' $LOG/summary.log || grep -m1 -A3 '"read_gate"' $LOG/summary.log; fi
    echo "On the Mac: python scripts/c1e4m_summary.py $O/grid_all.jsonl --cut \"$CUT\" --fig fig.png"
    [ $rc = 0 ] || fail "grid lines still failing after pass 2 (above): the summary's completeness gate shows them"
    ;;
esac
