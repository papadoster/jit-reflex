#!/usr/bin/env bash
# C1-E4 part 2 on two Linux NVIDIA GPU hosts in parallel, split by brain (spec docs/superpowers/specs/
# 2026-10-05-c1-e4-part2-design.md sections 3-7, 9, 13; owner 2026-10-06: nothing is cut). Every arm of a brain runs on
# one pod, so pairs never mix machines:
#   POD=A  pi0.5: connect, M0, B-cal, M1 pi0.5 (+ M2), block B pi0.5, D. A Hugging Face token with access to the gated
#          google/paligemma-3b-pt-224 (pi0.5's tokenizer; huggingface-cli login). >= 16 vCPUs, 48 GB RAM, 80 GB disk
#   POD=B  OFT+, OFT, SmolVLA: connect, M0, B-cal, M1 OFT+, M4 S, block B SmolVLA, M4 OFT. No token (open models; one
#          present is ignored). >= 16 vCPUs, 64 GB RAM, 120 GB disk (LIBERO-plus assets 9.4 GB, OFT and OFT+ 15 GB each)
#   POD=C  the controls (docs/superpowers/specs/2026-10-20-c1-e4-controls-design.md, "controls spec" §7, §12): pi0.5 +
#          OFT+ (OFT with TAIL=1) on one pod, results under results/c1-e4/controls/pod_C; its stages are below. A token
#          as pod A. >= 16 vCPUs, 64 GB RAM, 120 GB disk
# Each pod: RunPod Community RTX 4090 (~$0.34/h), EGL, root (apt-get), tmux, /workspace.
#   tmux new -s c1e4m   (after a lost SSH session: tmux attach -t c1e4m)
#   cd /workspace && git clone https://github.com/papadoster/jit-reflex.git && cd jit-reflex && git checkout c1-e4
# Stages, each run separately after the owner's go-ahead, in this order (POD=A or POD=B before each):
#   setup       libmagickwand-dev; venv (LeRobot, part 1's pins + the scene packages); B also $DEPS/venv-oft
#               (openvla-oft + the transformers fork, pinned commits, + the scene packages); LIBERO-MAX, LIBERO-plus,
#               LIBERO-PRO at their pins, PRO data and the two scene configs, LIBERO-plus assets under their SHA-256;
#               A: the token check, pi0.5's weights by SHA-256; B: SmolVLA, OFT / OFT+ weights and dataset statistics
#   smoke       HF offline, never read: the first minutes 1-2 motion-blur cases through real Wand (A pi0.5, B SmolVLA
#               and one OFT+ in venv-oft: its own Wand), Base; B: OFT / OFT+ self-tests and TF's GPU memory; latency,
#               one row per call; per brain the GPU peak (n-envs down while > 23 GB), tempo, repeat check and pair
#               validity on calibration debug cases with all its grid arms (EGL leak watch); crash and resume; async
#               against sync; a bench move smoke -> tempo_smoke.txt and forecast.txt. A failed rerun keeps the last
#               passed smoke's
#   connect     spec §9: Base on the 300 connect cases for the pod's brains, then its connection gate. Fails:
#               $O/connect_fail, the pod stops and the owner is asked. A brain the owner dropped after its failed
#               connect (the summary's DROPPED) runs no LIBERO-MAX line after connect; the gate is the others'
#   calib       spec §5: M0 (Gcal on the 60 calibration cases) and B-cal (bench, 60 pairs) -> the pod's line
#               "C1-E4 part 2 calib A|B: ..." in calib.txt. N < 45 for a brain (exit 4):
#                 1. calib exits 4 and names the brains ("short");
#                 2. [K=20] ./scripts/gpu_c1e4m.sh calib_more: Gcal on the next K evaluation cases in order (the bench:
#                    its spare pairs) for those brains; M0 never reruns (connect, calib and calib_more run on the split
#                    as first frozen, $O/split0.json, their files keyed by its SHA-256);
#                 3. calib_more prints the line and "extra" (calib.txt);
#                 4. on the Mac: refreeze the split with extra = the maximum of both pods' (maxwrap.freeze_split),
#                    journal it with the line, push; 5. here: git pull; 6. grid (its lines use the refrozen split)
#   on the Mac: journal the pod's calib line in the spec, commit, push; here: git pull. Money: the owner adds both pods'
#               post-smoke forecasts when journaling; above $30 the owner is asked before the grid (spec §13). This
#               script only prints its own forecast at the end of smoke and after each grid line
#   grid        spec §6, §7, §13: the pod's grid lines by priority (A: M1 pi0.5 with M2 -> M1 pi0.5 rest -> B oracle ->
#               B noisy -> D; B: M1 OFT+ -> M4 SmolVLA -> B oracle -> B noisy -> M4 OFT), one runner process on the GPU
#               at a time, the forecast after each line, then grid_all_$POD.jsonl. Refuses unless the spec at HEAD
#               journals this pod's calib line (equal to calib.txt) and the other pod's (a refreeze moves both pods'
#               evaluation cases), the split has calib's extra, HEAD is pushed, the tree is clean, connect passed and
#               the runner code is connect's ($O/CODE). ONLY=<id regex>: only those lines (bench lines alone need no
#               line of the other pod). The reading: on the Mac with both pods' grid_all files
# POD=C pi0.5 + OFT+ (OFT with TAIL=1), the controls (controls spec): setup | cprobe | csmoke | cgrid | ctail | vgrid |
# vtgrid, each
# after the previous one passed; every runner line at n-envs 6 (pass 2 too), part 2's batches (§7):
#   setup       A's and B's without SmolVLA: the token check and pi0.5's SHA-256, venv-oft and OFT+'s weights (OFT's
#               with TAIL=1), pytest
#   cprobe      §7.1: part 2's pieces repeated as they ran (scripts/c1e4c_summary.py --probe-plan), act_hash against the
#               saved records -> mode.json (P or S per probe); the machine -> PLATFORM (card, driver, torch and CUDA of
#               both venvs): csmoke, cgrid and ctail refuse another one (mode П holds where the probe ran)
#   csmoke      §12: the C1-E4 tests on the pod; PPCL and VS0.3 on 6 pi0.5 cases (prefix_ok >= 5 of 6 each, else stop)
#               and on the bench -> the GPU peak, tempo_smoke.txt (removed first: a failed rerun leaves none),
#               forecast.txt: the owner decides
#   cgrid       [CLINES="1 2 3"] the lines CLINES (default 1-7) by priority (--plan --line N, from mode.json and
#               k_select.json), the k selection after lines 1 and 4, the forecast after each runner line; with a mode S,
#               PPC on PPCL's latched cases; grid_all_C.jsonl of the lines run so far. §12: above the $13 cap the lines
#               that do not fit wait for the owner; CLINES in increasing order, 2, 3, 6, 7 with 1 (k_π), 5 with 1 and 4.
#               A later cgrid with more lines resumes (the runners skip done cases)
#   ctail       TAIL=1, after cgrid and the owner's go: line 8 (OFT), grid_all_C.jsonl again
#   vgrid       [VLINES="V3d10 V3d20 Nglr V2eps"] [NGLR_ARMS=PPCL@glr,PPC@glr | PPCL@glr]
#               PBVS (docs/superpowers/specs/2026-10-20-c1-e4-pbvs-design.md) and the controls
#               journal's noisy cell, after a finished cgrid (grid_all_C.jsonl newer than every C*.jsonl; refused while a
#               cgrid / ctail stage or a runner is live, or before controls lines 2 and 3 are complete and the frozen
#               file's Base cases are V2's first 108 Plus: V2 / V3 / Nglr / V2eps pair with them). The platform, the C1-E4
#               tests with test_vservo and test_c1e4v_summary, both summaries' selftests, then its own code record
#               $O/CODE.v (CODE_PATHS + the PBVS summary; $O/CODE stays the controls'). Then V0 -> band.json -> vsmoke
#               (PBVS0.3 on 6 cases; its Base e must be line 2's) -> the forecast at the controls' tempo -> V1 ->
#               pbvs_k.json -> V2 -> VLINES (any of V3d10 V3d20 Nglr V2eps, each once, in the order given; V3 = V3d10
#               V3d20; default all, in this order; V2eps runs only when band.json's delta > 15 mm), each by
#               scripts/c1e4v_summary.py --plan --line L (V3d10, V3d20: one d of V3 each; Nglr with NGLR_ARMS, the
#               cheap cell PPCL@glr alone and PPC@glr part 2's: unset, the pod's first ($O/NGLR_ARMS) or the full
#               cell; another than the pod's first refused). A forecast of V1 + V2 + V3
#               above $4 holds VLINES and the reading for the owner (a clean exit; after the ok: V3_OK=1, each line alone
#               with VLINES). band.json / pbvs_k.json are recomputed only when a V0 / V1 file is newer. Then
#               grid_all_V.jsonl and the reading (summary_V.json). A rerun resumes. Note: CODE_PATHS holds src, so once
#               the PBVS code is pulled ctail's check against $O/CODE refuses: intended, the owner decides on the tail
#   vtgrid      VS-T, the teach-by-showing servo (the PBVS spec's journal "2026-10-21, 11:00 UTC — ветка VS-T
#               гоняется"), after vgrid's clean end (its reading PASS, the last line of reading_V.txt, newer than
#               grid_all_V.jsonl, itself newer than every V line's file; refused while a cgrid / ctail / vgrid stage
#               or a runner is live, or before controls lines 2 and 3 are complete). The platform, the C1-E4 tests with test_vservo,
#               test_vst and both summaries' tests, the three summaries' selftests, then its own code record $O/CODE.t
#               (CODE_PATHS + the PBVS and VS-T summaries). Then VT1 -> vst_k.json -> VT2 -> VT3d10 -> VT3d20, each by
#               scripts/c1e4t_summary.py --plan --line L, every Base full (s* is its first close); vst_k.json is
#               recomputed only when a VT1 file is newer. Then grid_all_T.jsonl and the reading (summary_T.json). A
#               rerun resumes. Once the VS-T code is pulled, vgrid's check against $O/CODE.v refuses (src changed)
# One stage at a time on a pod (flock). Each runner line has its own --out (block, brain, set, source, d), runs twice
# (pass 2 resumes and retries failed groups, scripts/gpu_c1e3.sh's watchdog STALL, default 7200 s) and a stage fails if a
# runner still exits non-zero after pass 2. Every exit appends the environment to $O/pod_env_<stage>.txt, refreshes
# $O/MANIFEST.sha256 and packs $O into c1e4m_results_$POD.tgz.
# DRY=1 [DRY_PY=python-with-numpy] [KBAR_LINE="C1-E4 part 2 calib A: ..."] [SHORT="m/pi05"]: print every command instead
# of running it (setup included), no network, files of this script to a temporary dir: a dry run on the Mac.
set -uo pipefail
cd "$(dirname "$0")/.."
DRY=${DRY:-}
dry() { [ -n "$DRY" ]; }
POD=${POD:-}
TAIL=${TAIL:-}  # POD=C: 1 adds OFT (controls spec §12, line 8)
case $POD in  # the pod's LIBERO-MAX brains, its bench brain (spec §13) and its brains in venv-oft
  A) BRAINS=pi05; BB=pi05; OFTB= ;;
  B) BRAINS="oftplus oft smolvla"; BB=smolvla; OFTB="oft oftplus" ;;
  C) OFTB=oftplus; [ "$TAIL" = 1 ] && OFTB="oft oftplus"; BRAINS="pi05 $OFTB"; BB=pi05 ;;
  *) echo "!!! POD=A (pi0.5), POD=B (OFT+, OFT, SmolVLA) or POD=C (the controls) is required (spec §13)"; exit 1 ;;
esac
has_pi() { [ $POD = A ] || [ $POD = C ]; }  # pi0.5: its gated tokenizer (a token), its weights' SHA-256
has_oft() { [ -n "$OFTB" ]; }  # venv-oft, openvla-oft
dry || [ -n "${TMUX:-}" ] || { echo "!!! not inside tmux: a lost SSH session kills the workers (tmux new -s c1e4m)"
  sleep 10; }
O=results/c1-e4/part2/pod_$POD  # the runners' records
[ $POD = C ] && O=results/c1-e4/controls/pod_C
W=$O  # this script's own files; a dry run writes them to a temporary dir, never into the repository
dry && W=$(mktemp -d)
LOG=$W/logs
SM=$O/smoke  # the smoke's records (never read): a smoke rerun does not touch the grid's
P=venv/bin/python
SP=$P  # the summary's python (numpy, hf-libero); a dry run on the Mac: DRY_PY
dry && SP=${DRY_PY:-python3}
SPEC=docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md
SPLIT=results/c1-e4/part2/split.json
DEPS=${DEPS:-/workspace/lmax-deps}  # outside the repository: no LIBERO-MAX file in it (it has no license, spec §3)
LMAX=$DEPS/LIBERO-MAX
VO=$DEPS/venv-oft
PO=$VO/bin/python
PYS=$P; has_oft && PYS="$P $PO"  # the pod's venvs
VID=${VID:-/workspace/c1e4m_videos}  # outside the repository
STALL=${STALL:-7200}  # an M1/M2 pi0.5 group (6 cases x 8 episodes) runs 15-30 min without output
K=${K:-20}  # calib_more: the next K evaluation cases (bench: spare pairs) in order, spec §5
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
# their dataset_statistics.json (the action un-normalization; a plain git file, not LFS): git blob ids from the HF API
OFT_STATS="oft abde7987899224c1931d3c456d9729a469d72952
oftplus f50b07c5f671b079b3409361318d22f2ee9f1ee7"
NUM='-?[0-9]+\.[0-9]+ \[-?[0-9]+\.[0-9]+, -?[0-9]+\.[0-9]+\] N=[0-9]+'
# the pod's calibration line, loosely (the summary's --plan --kbar-line checks it exactly)
line_re() { echo "C1-E4 part 2 calib $1: kbar_[a-z0-9]+=$NUM( · (bench: )?kbar_[a-z0-9]+=$NUM)+"; }
mkdir -p $LOG
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl} PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
# a container can show the host's CPUs (nproc 256) under a cgroup quota of ~30: per-process BLAS / torch pools of nproc
# threads in every env worker would oversubscribe it. The same on both pods, so pairs never mix settings
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}  # part 1's journal "trial"
export OPENVLA_OFT_ROOT=$DEPS/openvla-oft C1E4_OFT_CKPT=${C1E4_OFT_CKPT:-/workspace/oft-ckpt} \
  TF_FORCE_GPU_ALLOW_GROWTH=true  # openvla-oft imports TensorFlow (image resize): it must not take the GPU's memory
if [ -d /workspace ]; then
  export HF_HOME=/workspace/.cache/huggingface UV_CACHE_DIR=/workspace/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/workspace/.cache/uv-python
fi
[ $POD = B ] && { unset HF_TOKEN HUGGING_FACE_HUB_TOKEN; export HF_HUB_DISABLE_IMPLICIT_TOKEN=1; }  # open models only

x() { if dry; then echo "+ $*"; else "$@"; fi; }  # a command with side effects: printed in a dry run
wr() { local f=$1; shift; if dry; then echo "+ write $f: $*"; else printf '%s\n' "$@" > "$f"; fi; }
fail() { echo "!!! $*"; exit 1; }
refuse() { echo "!!! $*"; dry && echo "    (DRY: going on)" || exit 1; }

setup() {
  # a host whose CUDA does not start fails in seconds, not after the installs (part 2's first two pods, driver
  # 580.126.09: /dev/nvidia-uvm gives EIO, cuInit 999, nvidia-smi fine)
  dry || python3 -c "import ctypes; r = ctypes.CDLL('libcuda.so.1').cuInit(0); assert r == 0, f'cuInit {r}'" \
    || fail "CUDA does not start on this host (driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)): a new pod"
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
  [ $POD != C ] || x uv pip install --python $P pytest==9.1.1 || exit 1  # csmoke runs the C1-E4 tests (§12)
  # pods A, C: pi0.5's tokenizer is gated; a missing token or unaccepted terms stop here, before anything big is fetched
  has_pi && { x $P -c "from huggingface_hub import model_info, whoami
print('HF user:', whoami()['name']); model_info('google/paligemma-3b-pt-224'); print('paligemma: access ok')" \
    || fail "pod $POD needs a Hugging Face token with access to the gated google/paligemma-3b-pt-224 (pi0.5's tokenizer):
    accept its terms on the model page, then huggingface-cli login (or: hf auth login), then setup again"; }
  x mkdir -p $DEPS
  clone() {  # clone <url> <dir> <commit>
    { [ -d "$2/.git" ] || x git clone -q "$1" "$2"; } && x git -C "$2" checkout -q "$3" || exit 1
  }
  clone https://github.com/liberomax/LIBERO-MAX.git $LMAX $LMAX_REV
  clone https://github.com/sylvestf/LIBERO-plus.git $DEPS/LIBERO-plus $PLUS_REV
  clone https://github.com/Zxy-MLlab/LIBERO-PRO.git $DEPS/LIBERO-PRO $PRO_REV  # its assets are in its git tree
  # pod B: openvla-oft's venv (python 3.10, torch 2.2.0 as its pyproject) with the transformers fork: plain transformers
  # silently gives wrong chunks (parallel decoding needs the fork's bidirectional attention); plus the scenes
  has_oft && clone https://github.com/moojink/openvla-oft.git $DEPS/openvla-oft $OFT_GIT
  # tensorflow-metadata >= 1.17 is generated for protobuf 5 (TF 2.15 keeps 4.25); 1.16.1 asks protobuf < 4.21 on python
  # 3.10, which breaks wandb: both forced; wandb >= 0.29 has protos for protobuf 5+ only: 0.28.0 (imports checked on the
  # pod, 2026-10-17). Marker 3: venvs built before rebuild
  if has_oft && { dry || [ "$(cat $VO/c1e4m-installed 2>/dev/null)" != 3 ] || ! $PO -V >/dev/null 2>&1; }; then
    x uv venv --clear --python 3.10 $VO || exit 1
    wr $DEPS/oft-overrides.txt "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@$TF_GIT" \
      "dlimp @ git+https://github.com/moojink/dlimp_openvla.git@$DLIMP_GIT" tensorflow-metadata==1.16.1 protobuf==4.25.9 \
      wandb==0.28.0
    x uv pip install --python $PO --override $DEPS/oft-overrides.txt -e $DEPS/openvla-oft robosuite==1.4.0 mujoco==3.3.2 \
      bddl==1.0.1 "hydra-core>=1.2,<1.4" easydict thop cloudpickle opencv-python future gymnasium==1.3.0 \
      scikit-image==0.25.2 imageio==2.38.0 lazy_loader==0.6 Wand==0.6.13 || exit 1  # 0.26: python >= 3.11
    x uv pip install --python $PO --no-deps hf-libero==0.1.4 robomimic==0.2.0 || exit 1
    x $PO -c "import sys; sys.path.insert(0, '$DEPS/openvla-oft'); import tensorflow_datasets, wandb
import experiments.robot.openvla_utils" >/dev/null || fail "venv-oft: openvla-oft does not import"
    wr $VO/c1e4m-installed 3
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
  # the zip keeps its authors' path (inspire/hdd/.../LIBERO-plus-0/assets): that assets dir goes to $PL/assets
  if ! dry && [ ! -d $PL/assets ]; then
    a=$(find $PL/inspire -type d -path '*/LIBERO-plus-0/assets' 2>/dev/null | head -1)
    [ -n "$a" ] && mv "$a" $PL/assets && rm -rf $PL/inspire
  fi
  dry || [ -f $PL/assets/scenes/libero_floor_base_style.xml ] || fail "LIBERO-plus assets are not at $PL/assets"
  # standard LIBERO (hf-libero) for the bench, the self-tests and the latency: assets and ~/.libero (the pod's venvs)
  for py in $PYS; do
    x sh -c "echo N | $py -c \"import os, libero.libero as L; a = L.get_assets_path()
assert os.path.isdir(a), f'no LIBERO assets at {a}'; print('LIBERO assets:', a)\"" || exit 1
    x $py -c "import wand.image; wand.image.Image(width=8, height=8); print('Wand: MagickWand loads')" || exit 1
  done
  x $P -c "import torch; assert torch.cuda.is_available(), 'CUDA is not visible to torch'
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))" || exit 1
  # the context is kept (c): an unreferenced one can be freed at once, and GL_RENDERER then comes back empty although
  # MuJoCo renders on the GPU (pod A, 2026-10-17); the EGL vendor of that context is checked too
  local probe='import os, mujoco
from mujoco.egl import egl_ext as EGL
c = mujoco.GLContext(64, 64); c.make_current()
v = EGL.eglQueryString(EGL.eglGetCurrentDisplay(), EGL.EGL_VENDOR).decode()
from OpenGL import GL; r = (GL.glGetString(GL.GL_RENDERER) or b"?").decode()
print("EGL vendor:", v, "GL renderer:", r)
assert v == "NVIDIA" and not any(s in r.lower() for s in ("llvmpipe", "softpipe", "swrast")), "EGL renders on the CPU"'
  if ! dry && ! $P -c "$probe" </dev/null; then  # as scripts/gpu_c1e4.sh
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libegl1 libgl1 libopengl0 libglib2.0-0 \
      </dev/null
    $P -c "$probe" </dev/null || fail "offscreen EGL rendering does not work on this pod: the owner decides"
  fi
  # weights: the pod's LeRobot brain as part 1 (pinned revision); building it once on the CPU caches every file it reads
  # (PaliGemma's gated tokenizer, SmolVLM's processor), so later stages run HF offline
  x $P -c "import sys; sys.path.insert(0, 'scripts'); import c1e4m_brains as cb
cb.make_brain('$BB', 'cpu', 'fp32'); print('$BB cached')" || fail "$BB does not build"
  if has_pi; then  # pi0.5's SHA-256 (part 1)
    x $P -c "from huggingface_hub import snapshot_download as d
p = d('$PI_REPO', revision='$PI_REV'); assert '$PI_REV' in p; open('venv/pi05-snapshot', 'w').write(p)" || exit 1
    if dry || [ "$(cat venv/pi05-sha-ok 2>/dev/null)" != "$(cat venv/pi05-snapshot)" ]; then
      x sh -c "echo '$PI_SHA  $(cat venv/pi05-snapshot 2>/dev/null)/model.safetensors' | sha256sum -c -" \
        || fail "pi0.5 weights do not match the spec's SHA-256"
      x cp venv/pi05-snapshot venv/pi05-sha-ok
    fi
  fi
  # pods B, C, OFT / OFT+: the local_dir copy scripts/c1e4m_brains.py OFTBrain loads (same path, no lora_adapter/), then
  # the SHA-256 of its weight files and the git blob id of its dataset statistics (checked once)
  local b dir get
  for b in $OFTB; do
    get="import sys; sys.path.insert(0, 'scripts'); import c1e4m_brains as cb
from huggingface_hub import snapshot_download as d; r, v = cb.OFT['$b']
print(d(r, revision=v, local_dir='$C1E4_OFT_CKPT/' + r.split('/')[1] + '-' + v[:10], ignore_patterns=['lora_adapter/*']))"
    dry && { echo "+ $PO -c \"$get\""; echo "+ sha256sum -c: the 6 weight files of $b (OFT_SUMS);" \
      "git hash-object dataset_statistics.json (OFT_STATS)"; continue; }
    dir=$($PO -c "$get" | tail -1) || exit 1
    [ "$(cat $C1E4_OFT_CKPT/$b-sha-ok 2>/dev/null)" = "$dir" ] && continue
    awk -v b=$b '$1 == b { print $2 "  " $3 }' <<< "$OFT_SUMS" | (cd "$dir" && sha256sum -c -) \
      || fail "$b weights do not match their SHA-256"
    [ "$(git hash-object "$dir/dataset_statistics.json")" = "$(awk -v b=$b '$1 == b { print $2 }' <<< "$OFT_STATS")" ] \
      || fail "$b dataset_statistics.json does not match its git blob id"
    echo "$dir" > $C1E4_OFT_CKPT/$b-sha-ok
  done
  x touch $DEPS/c1e4m-setup-ok
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || true
}

record() {  # record <stage>: appends this run's environment to $O/pod_env_<stage>.txt
  { echo "=== $(date -u) pod $POD $1, HEAD $(git rev-parse HEAD), MUJOCO_GL=$MUJOCO_GL, vCPUs $(nproc)"
    echo "code ($CODE_PATHS):" $(code_hash)
    for d in $LMAX $DEPS/LIBERO-plus $DEPS/LIBERO-PRO $(has_oft && echo $DEPS/openvla-oft); do
      echo "$d: $(git -C $d rev-parse HEAD)"; done
    cat $DEPS/libero-pro-config/source_lock.json
    echo "LIBERO-plus assets SHA-256: $(cat $DEPS/plus-assets-ok 2>/dev/null || echo no)"
    has_pi && echo "pi0.5 weights SHA-256 checked: $(cat venv/pi05-sha-ok 2>/dev/null || echo no)"
    for b in $OFTB; do
      echo "$b weights SHA-256 checked: $(cat $C1E4_OFT_CKPT/$b-sha-ok 2>/dev/null || echo no)"; done
    [ $POD = B ] && echo "HuggingFaceVLA/smolvla_libero snapshots:" \
      $(ls "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--HuggingFaceVLA--smolvla_libero/snapshots)
    timeout 60 nvidia-smi
    for py in $PYS; do echo "--- $py"
      $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
      echo "scene packages:" $(uv pip freeze --python $py \
        | grep -iE '^(scikit-image|imageio|tifffile|lazy.loader|wand|mujoco|robosuite|bddl|hf-libero)( |=)')
      uv pip freeze --python $py; done; } >> $O/pod_env_$1.txt 2>&1
}

tgz() {  # tgz <archive> <paths..>: to a .tmp, then renamed, so an interrupted tar never leaves a truncated archive
  tar czf "$1.tmp" "${@:2}" && mv -f "$1.tmp" "$1" && echo "[$(date +%T)] packed $1 ($(du -h "$1" | cut -f1))"
}
pack() {
  (cd $O && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  tgz c1e4m_results_$POD.tgz $O
}

CODE_PATHS="src scripts/c1e4m_run.py scripts/c1e4m_brains.py scripts/c1e4_run.py"
# POD=C: the controls summary's plan, mode and k selection too (and part 2's summary, which it imports)
[ $POD = C ] && CODE_PATHS+=" scripts/c1e4c_summary.py scripts/c1e4m_summary.py"
code_hash() {
  local f
  for f in $CODE_PATHS; do git rev-parse -q --verify "HEAD:$f" || { echo "!!! $f is not in HEAD" >&2; return 1; }; done
}
check_code() {  # check_code record|require: connect, calib, calib_more and grid run the same runner code (connect's)
  local h
  dry && { echo "+ check_code $1 ($CODE_PATHS at HEAD vs $O/CODE)"; return; }
  h=$(code_hash) || exit 1
  [ -z "$(git status --porcelain -- $CODE_PATHS)" ] || { git status --short -- $CODE_PATHS
    fail "uncommitted or untracked files in the runner code (above): commit them on the Mac and pull, or remove them"; }
  if [ "$1" = record ] && [ ! -f $O/CODE ]; then echo "$h" > $O/CODE; return; fi
  [ "$(cat $O/CODE 2>/dev/null)" = "$h" ] || fail "the runner code is not connect's ($O/CODE): the owner decides (archive
    the results under $O and rerun from connect, or check out connect's commit)"
}
check_code_v() {  # check_code_v check|record [t], vgrid (PBVS spec §5) / vtgrid (t: the VS-T journal): the stage's own
  local h f=$O/CODE.${2:-v}  # record $O/CODE.v / CODE.t of CODE_PATHS (its summaries added), written by record once its
  # tests passed (its first line starts); then both require it. $O/CODE: the controls'
  dry && { echo "+ check_code_${2:-v} $1 ($CODE_PATHS at HEAD vs $f)"; return; }
  h=$(code_hash) || exit 1
  [ -z "$(git status --porcelain -- $CODE_PATHS)" ] || { git status --short -- $CODE_PATHS
    fail "uncommitted or untracked files in the code (above): commit them on the Mac and pull, or remove them"; }
  if [ ! -f $f ]; then [ "$1" != record ] || echo "$h" > $f; return; fi
  [ "$(cat $f)" = "$h" ] || fail "the code is not the first $STAGE's ($f): the owner decides"
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
  local log=$LOG/$1.log n rc f k small=; shift
  : >> "$log"; n=$(wc -c < "$log")
  echo "[$(date +%T)] pass 1: $*" | tee -a "$log"
  wd "$log" "$@"; rc=$?
  f=$(tail -c +$((n + 1)) "$log" | grep -c FAILED)
  # LIBERO-MAX's last line "done N of M cases, failed K"; none after a watchdog kill (124), an OOM kill (137), a crash
  k=$(tail -c +$((n + 1)) "$log" | grep -oE '^done [0-9]+ of [0-9]+ cases, failed [0-9]+' | tail -1)
  k=${k##* }
  [ $rc = 0 ] || echo "!!! runner exited with an error (pass 1)" | tee -a "$log"
  if [ $rc = 0 ] && [ "$f" = 0 ]; then
    echo "[$(date +%T)] pass 2 skipped: pass 1 exited cleanly, no failed group" | tee -a "$log"; return 0
  fi
  # 2 envs only after a run that ended by itself with few failed groups (part 1: the bench, exit 0); otherwise the same
  # envs. The last --n-envs wins; fp32 and one-row chunks do not depend on the batch
  { [ -n "$k" ] && [ "$k" -le 3 ]; } || { [ $rc = 0 ] && [ "$f" -le 3 ]; } && small="--n-envs 2"
  [ $POD = C ] && small=  # controls spec §7: part 2's batches of 6 (pi0.5's fp32 chunks depend on the batch)
  echo "[$(date +%T)] pass 2 ($f failed groups in pass 1, exit $rc): $* $small" | tee -a "$log"
  wd "$log" "$@" $small; rc=$?
  [ $rc = 0 ] || echo "!!! runner exited with an error (pass 2)" | tee -a "$log"
  return $rc
}

peak_run() {  # peak_run <peak file> <command..>: the command while nvidia-smi samples the GPU memory (MiB) every second
  local f=$1 s rc; shift
  dry && { echo 0 > "$f"; "$@"; return; }  # the command's exit: a broken plan fails a dry run too
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
  e=$(grep -oE "C1-E4m envs: .*" $W/tempo_smoke.txt 2>/dev/null | tail -1 | grep -oE " $2=[0-9]+" | cut -d= -f2)
  echo "${e:-$([ $2 = smolvla ] && echo 8 || echo 6)}"  # default: a dry run; POD=C (part 2's 6, controls spec §7)
}
dbg() {  # dbg <source> <k>: the first k calibration cases of that source (debug cases, comma-separated)
  $SP -c "import json; c = json.load(open('$SPLIT'))['calib']
print(','.join([i for i in c if i.startswith('pro-') == ('$1' == 'pro')][:$2]))"
}
summ() { $SP scripts/c1e4m_summary.py "$@"; }
cc() { $SP scripts/c1e4c_summary.py "$@"; }  # POD=C: the controls summary (controls spec §5, §7, §12)
cv() { $SP scripts/c1e4v_summary.py "$@"; }  # POD=C vgrid: the PBVS summary (PBVS spec §3-§8)
ct() { $SP scripts/c1e4t_summary.py "$@"; }  # POD=C vtgrid: the VS-T summary (the PBVS spec's VS-T journal)
SUMM=cc  # the summary cphase plans with (vgrid: cv, vtgrid: ct); not CC, the C compiler's variable
cphase() {  # cphase <phase> <cc args..>: the summary's plan, run by run_phase. A dry run where the summary cannot plan
  local ph=$1 p q; shift  # yet (exit 3, on the Mac: no mode.json or k_select.json) prints the call instead
  if dry; then p=$($SUMM "$@" 2>$W/cc.err); q=$?; [ $q = 3 ] && { echo "+ $SUMM $*"; return 0; }; cat $W/cc.err
  else p=$($SUMM "$@"); q=$?; fi
  [ $q = 0 ] || { echo "!!! $SUMM $* failed (exit $q)"; return 1; }
  # run_phase skips other phases' lines silently: a line the summary names otherwise would never run
  [ -z "$p" ] || [ "$(cut -f1 <<< "$p" | sort -u)" = "$ph" ] || { echo "!!! $SUMM $*: lines of a phase other than $ph"
    return 1; }
  run_phase $ph "$p"
}
cfc() {  # cfc <tempo file>: the controls forecast (controls spec §12), mode.json and k_select.json if there
  local d  # the grid's records only: the probe's share keys with line 2's (Base, G-auto in mode S)
  dry && { echo "+ cc --forecast $1 --done <the grid's records, $O/C[0-9]*.jsonl>"; return; }
  d=$(ls $O/C[0-9]*.jsonl 2>/dev/null)
  # shellcheck disable=SC2086
  cc --forecast "$1" ${d:+--done $d}
}
vfc() {  # vfc <tempo file>: the PBVS forecast (spec §8, the journals: each line's cost, exit 3 when V1 + V2 + V3 > $4)
  local d  # the V lines' records only (the smoke's PBVS0.3 records share their keys with V2's first 6)
  dry && { echo "+ cv --forecast $1 --nglr-arms $NGLR_ARMS --done <the V lines' records, $O/V[0-3]_*.jsonl" \
    "$O/Nglr_*.jsonl $O/V2eps_*.jsonl>"; return; }
  d=$(ls $O/V[0-3]_*.jsonl $O/Nglr_*.jsonl $O/V2eps_*.jsonl 2>/dev/null)
  # shellcheck disable=SC2086
  cv --forecast "$1" --nglr-arms $NGLR_ARMS ${d:+--done $d}
}
TV='import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda)'
platform() {  # POD=C: the machine mode П is decided on (cprobe): the card and its driver, torch and its CUDA per venv
  local py
  if dry; then echo "+ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
    for py in $P $PO; do echo "+ $py -c '$TV'"; done; return; fi
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || return 1
  for py in $P $PO; do $py -c "$TV" || return 1; done
}
check_platform() {  # csmoke, cgrid, ctail: cprobe's machine ($O/PLATFORM)
  local p
  dry && { echo "+ check_platform: the output of these against $O/PLATFORM"; platform; return; }
  p=$(platform) && [ "$p" = "$(cat $O/PLATFORM 2>/dev/null)" ] || { echo "$p"; fail "this machine (above) is not \
cprobe's ($O/PLATFORM): mode П was decided on another machine; rerun cprobe here, or ask the owner"; }
}
oft_numpy() {  # spec §4: openvla-oft's import makes json.dumps pass numpy silently; the runner writes strict JSON
  dry || [ $POD = A ] && return 0
  ! grep -l '"__numpy__"' $O/*oft*.jsonl $SM/*oft*.jsonl 2>/dev/null || fail "\"__numpy__\" in the OFT records above"
}

run_phase() {  # run_phase <phase> <plan>: the phase's lines one after another (one runner on the GPU at a time); 1 when
  local ph rn b src id n args bad=  # a runner still exits non-zero after its pass 2 (after running the others)
  while IFS=$'\t' read -r ph rn b src id n args; do
    [ "$ph" = "$1" ] || continue
    local out=$O/$id.jsonl
    # M0 and calib_more: keyed by split0's SHA-256 (their --split), which a refrozen split does not change (spec §5)
    case $id in M0*) out=$O/${id}_${S0:0:12}.jsonl ;; esac
    echo "[$(date +%T)] phase $ph, $id ($n episodes): $args" | tee -a $LOG/$ph.log
    if [ $rn = c1e4m ]; then
      [[ "$args " == *"--source $src "* ]] || fail "$id: --source is not its scene env $src"
      # shellcheck disable=SC2086
      run2 $id $(mrun $src $b) $args --n-envs $(envs c1e4m $b) --out $out </dev/null || bad+=" $id"
    else
      # shellcheck disable=SC2086
      run2 $id $BRUN $args --n-envs $(envs c1e4 $b) --out $out </dev/null || bad+=" $id"
    fi
    case $1 in grid | c[1-8] | cc5) fcast $id $([ $rn = c1e4m ] && echo lmax || echo bench)_$b ;; esac
  done <<< "$2"
  [ -z "$bad" ] || { echo "!!! still failing after pass 2:$bad" | tee -a $LOG/$1.log; return 1; }
}
fcast() {  # fcast <line id> <tempo key>: this line's tempo replaces the last one for that brain, then the forecast
  local r t
  dry && { [ $POD = C ] && { cfc $W/tempo_grid.txt; return; }
    echo "+ summ <the pod's records> --forecast $W/tempo_grid.txt --pod $POD (after $1)"; return; }
  r=$([ "${2%%_*}" = lmax ] && gtempo $LOG/$1.log || full_rate $LOG/$1.log)
  t=$(cat $W/tempo_smoke.txt $W/tempo_grid.txt 2>/dev/null | grep -E '^tempo \(s/episode\): ' | tail -1)
  [ -n "$r" ] && t=$(sed -E "s/ $2=[0-9.]+/ $2=$r/" <<< "$t")
  echo "$t" >> $W/tempo_grid.txt
  # shellcheck disable=SC2046
  { echo "[$(date +%T)] forecast after $1 ($2 ${r:-?} s/episode)"
    if [ $POD = C ]; then cfc $W/tempo_grid.txt
    else summ $(ls $O/*.jsonl 2>/dev/null | grep -v /grid_all_) --forecast $W/tempo_grid.txt --pod $POD; fi; } \
    | tee -a $W/forecast.txt | if [ $POD = C ]; then cat; else grep -E '"(total|left)"' -A3; fi
}
calib_line() {  # the pod's line from M0, B-cal and calib_more's records -> calib.txt; N < 45: calib_more first
  local q
  x summ --calib $O/M0*.jsonl $O/Bcal*.jsonl --pod $POD | tee $W/calib.txt; q=${PIPESTATUS[0]}
  [ $q = 4 ] && fail "N < 45 for$(grep -oE "^C1-E4 part 2 short $POD: .*" $W/calib.txt | cut -d: -f2-) (above): [K=20]
    POD=$POD ./scripts/gpu_c1e4m.sh calib_more, Gcal on the next cases in order; M0 does not rerun (spec §5)"
  [ $q = 0 ] || fail "the calibration summary failed (exit $q)"
  ! grep -q "^C1-E4 part 2 extra $POD:" $W/calib.txt || echo "!!! extra (above): on the Mac refreeze the split with" \
    "extra = the maximum of both pods' m (maxwrap.freeze_split) and journal it with the line before either grid"
  echo "On the Mac: journal the line above in the spec ($SPEC), commit, push; the owner adds both pods' forecasts" \
    "(over ~\$27: told before the grid, spec §13). Here: git pull, then POD=$POD ./scripts/gpu_c1e4m.sh grid"
}

case "$POD:${1:-}" in
  [ABC]:setup | [AB]:smoke | [AB]:connect | [AB]:calib | [AB]:calib_more | [AB]:grid) ;;
  C:cprobe | C:csmoke | C:cgrid | C:ctail | C:vgrid | C:vtgrid) ;;
  *) echo "usage: POD=A|B $0 setup | smoke | connect | calib | calib_more | grid"
     echo "       POD=C [TAIL=1] $0 setup | cprobe | csmoke | cgrid | ctail | vgrid | vtgrid"; exit 1 ;;
esac
STAGE=$1
# one stage at a time: a second instance would run the same episodes into the same files (the lock is outside results)
if command -v flock >/dev/null; then
  exec 9>"$(git rev-parse --git-dir)/c1e4m-pod-$POD.lock"  # in .git: not under a venv that setup clears
  flock -n 9 || fail "a stage already runs on this pod (it holds the lock in .git): tmux attach -t c1e4m"
else
  echo "!!! no flock here: nothing stops a second instance"
fi
[ -z "$(git ls-files $O)" ] || fail "git tracks results of an earlier run under $O: resume would see them as done.
    Run from a commit without them, or move them away, as the owner decides."
dry || trap 'trap "" INT TERM HUP; record $STAGE; pack' EXIT
if [ $STAGE = setup ]; then
  setup 2>&1 | tee -a $LOG/setup.log
  exit "${PIPESTATUS[0]}"
fi
[ $POD = C ] || summ --plan --pod $POD >/dev/null || exit 1
[ $POD = C ] || S0=$(summ --split0 $W/split0.json) || exit 1  # the split as first frozen: connect, calib, calib_more's
dry || [ -f $DEPS/c1e4m-setup-ok ] || fail "no $DEPS/c1e4m-setup-ok: POD=$POD ./scripts/gpu_c1e4m.sh setup first"
[ $STAGE = smoke ] || [ $POD = C ] || dry || grep -q '^C1-E4m envs:' $W/tempo_smoke.txt 2>/dev/null \
  || fail "no passed smoke ($W/tempo_smoke.txt): POD=$POD ./scripts/gpu_c1e4m.sh smoke first"
export HF_HUB_OFFLINE=1  # everything is cached by setup: no silent weight change, no Hub outage mid-run
case "$STAGE" in
  smoke)
    dry || { rm -rf $SM $W/latency.jsonl $W/leak.txt $W/tempo_run.txt $W/peak_* $LOG/smoke*.log $LOG/tempo*.log
      mkdir -p $SM; }
    # 1. in the first minutes, LIBERO-plus's motion blur through real Wand (ImageMagick), Base, 2 of the 5 motion-blur
    #    cases of connect (calibration has none), with videos to look at: every venv of the pod has its own Wand
    BLUR=libero_goal-t1584-target_relocation-d1-p195,libero_object-t1399-distractor_burst-d1-p195
    blur() {  # blur <brain> <cases>
      wd $LOG/smoke_blur_$1.log $(mrun plus $1) --brain $1 --source plus --set connect --cases $2 --arms= --n-envs 2 \
        --video-cases $2 --video-dir $VID --out $SM/blur_$1.jsonl || fail "the $1 motion-blur smoke failed"
      dry || { grep -q 'Wand real' $LOG/smoke_blur_$1.log && [ "$(grep -c '"skipped_reason": null' $SM/blur_$1.jsonl)" \
        = $(($(tr -cd , <<< "$2" | wc -c) + 1)) ] || fail "$1: motion blur did not run through real Wand"; }
    }
    blur $BB $BLUR
    [ $POD = B ] && blur oftplus ${BLUR%%,*}
    if [ $POD = B ]; then
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
    fi
    # 3. latency of the pod's brains, one row per call after warm-up (spec §10, report only)
    for b in $BRAINS; do
      pr=fp32; py=$P; case $b in oft | oftplus) pr=bf16; py=$PO ;; esac
      wd $LOG/smoke_latency_$b.log $py scripts/c1e4_latency.py --brain $b --precision $pr --device cuda \
        || fail "$b latency"
      dry || grep -h '^{' $LOG/smoke_latency_$b.log | tail -1 >> $W/latency.jsonl
    done
    # 4. per brain, calibration debug cases (never read), all its grid arms and Gcal (the shadow pairs in the peak):
    #    GPU peak, tempo and repeat check on the first 3 of each source, n-envs down while the peak is above 23 GB; then
    #    the first 10 of each source into the same files (resume) for the pair validity, with the EGL leak watch
    tempo_run() {  # tempo_run <brain> <envs> <cases per source> <log> [runner args..]
      local b=$1 e=$2 k=$3 log=$4 s rc=0 arms=none,G,Gauto,PPC,Gcal; shift 4
      [ $b = pi05 ] && arms=none,G,Gauto,PPC,T0,G@glr,PPC@glr,Gcal  # M2 rides on M1's pi0.5 calls
      for s in plus pro; do
        wd $log $(mrun $s $b) --brain $b --source $s --set calib --cases $(dbg $s $k) --arms $arms --kbar 0.2 \
          --n-envs $e --out $SM/${b}_$s.jsonl "$@" || rc=1
      done
      return $rc
    }
    leak_watch() {  # leak_watch <log> <brain>: every 20 s groups done, GPU memory, each env worker's RSS (EGL contexts)
      while sleep 20; do
        echo "$(date +%T) $2 groups=$(grep -c '^group ' "$1") gpu_mib=$(nvidia-smi --query-gpu=memory.used \
          --format=csv,noheader,nounits | head -1) worker_rss_mib=$(ps -eo rss=,args= \
          | awk '/multiprocessing.spawn/ && !/awk/ { printf "%d ", $1 / 1024 }')"
      done >> $W/leak.txt
    }
    for b in $BRAINS; do
      for e in $(seq $([ $b = smolvla ] && echo 8 || echo 6) -1 4); do
        dry || rm -f $SM/${b}_*.jsonl
        peak_run $W/peak_$b tempo_run $b $e 3 $LOG/tempo_${b}_e$e.log --repeat-check || fail "$b tempo run ($e envs)"
        dry || grep -qE '^[0-9]+$' $W/peak_$b || fail "no GPU memory samples ($W/peak_$b.err)"
        eval "E_$b=$e"
        echo "$b, $e envs: GPU peak $(cat $W/peak_$b) MiB (max $PEAK_MAX);" \
          "$(grep -h 'C1-E4m repeat' $LOG/tempo_${b}_e$e.log 2>/dev/null | tr '\n' ' ')" | tee -a $W/tempo_run.txt
        [ "$(cat $W/peak_$b)" -le $PEAK_MAX ] && break
        [ $e = 4 ] && echo "!!! $b: the peak is above $PEAK_MAX MiB at 4 envs too: going on with 4" \
          | tee -a $W/tempo_run.txt
      done
      rep=$(grep -h 'C1-E4m repeat' $LOG/tempo_${b}_e$e.log 2>/dev/null)
      dry || { [ -n "$rep" ] && ! grep -qv 'bitwise=1' <<< "$rep"; } \
        || echo "!!! $b: the repeat check is missing or not bitwise (report, not a stop)" | tee -a $W/tempo_run.txt
      lw=; dry || { leak_watch $LOG/tempo_${b}_e$e.log $b & lw=$!; }  # >= 50 env starts per brain
      tempo_run $b $e 10 $LOG/tempo_${b}_e$e.log || fail "$b validity run"
      [ -z "$lw" ] || kill $lw
    done
    # shellcheck disable=SC2046
    x summ --smoke $(for b in $BRAINS; do echo $SM/${b}_plus.jsonl $SM/${b}_pro.jsonl; done) --pod $POD \
      | tee $W/smoke_validity.txt
    [ "${PIPESTATUS[0]}" = 0 ] || fail "pair validity, check (g) or skipped cases in the smoke (above)"
    # 5. the pod's bench brain, kind move: Glead and the noisy eyes, 2 tasks x 2 inits x 2 cells x 3 arms = 24
    wd $LOG/smoke_bench_$BB.log $BRUN --brain $BB --tasks libero_spatial:0,libero_object:0 --kinds move --cells A,A20 \
      --methods none,Glead,Gauto@cv --inits 48-49 --kbar 0.2 --n-envs 6 --out $SM/bench_$BB.jsonl \
      || fail "$BB bench move smoke"
    dry || [ "$(wc -l < $SM/bench_$BB.jsonl)" = 24 ] || fail "$BB bench move smoke: not 24 records"
    # 6. crash and resume (the bench brain, Base, 6 cases in groups of 2): a killed env worker fails its group and the
    #    run exits non-zero; a runner killed -9 with its last line cut in half (a kill inside a write) resumes with no
    #    duplicate line
    CR=(--brain $BB --source plus --set calib --cases $(dbg plus 6) --arms= --n-envs 2 --out $SM/crash.jsonl)
    if dry; then
      echo "+ $(mrun plus $BB) ${CR[*]} &   # kill -9 one env worker after 'group 0:', expect exit 1"
      echo "+ $(mrun plus $BB) ${CR[*]} &   # kill -9 the runner after 'group 0:', cut its last line in half"
    else
      upto() { local i; for i in $(seq 900); do grep -q "$1" "$2" && return 0; kill -0 $3 2>/dev/null || return 1
        sleep 1; done; return 1; }
      $(mrun plus $BB) "${CR[@]}" > $LOG/smoke_crash_a.log 2>&1 & pid=$!
      upto 'group 0:' $LOG/smoke_crash_a.log $pid && kill -9 "$(pgrep -P $pid -f multiprocessing.spawn | head -1)"
      wait $pid && fail "a failed group did not make the runner exit non-zero ($LOG/smoke_crash_a.log)"
      grep -q 'FAILED' $LOG/smoke_crash_a.log || fail "the killed worker did not fail its group"
      $(mrun plus $BB) "${CR[@]}" > $LOG/smoke_crash_b.log 2>&1 & pid=$!
      upto 'group 0:' $LOG/smoke_crash_b.log $pid; pkill -9 -P $pid; kill -9 $pid 2>/dev/null; wait $pid
      $SP -c "import os; p = '$SM/crash.jsonl'; b = open(p, 'rb').read(); k = b.rstrip(b'\n').rfind(b'\n') + 1
os.truncate(p, k + (len(b) - k) // 2); print('cut', (len(b) - k) - (len(b) - k) // 2, 'bytes of the last line')"
    fi
    wd $LOG/smoke_crash_c.log $(mrun plus $BB) "${CR[@]}" || fail "the resumed crash run did not end cleanly"
    dry || $SP -c "import json; rs = [json.loads(l) for l in open('$SM/crash.jsonl')]
keys = [(r['case_id'], r['kind'], r['arm']) for r in rs]
assert len(keys) == len(set(keys)) == 6 and all(k == 'base' for _, k, _ in keys), keys
print('crash and resume: 6 cases, one Base line each, no duplicate')" || fail "crash and resume"
    # 7. async workers against the main process, bitwise, one case
    for v in async sync; do
      wd $LOG/smoke_$v.log $(mrun plus $BB) --brain $BB --source plus --set calib --cases $(dbg plus 1) \
        --arms none,Gauto --kbar 0.2 --n-envs 2 --vector $v --out $SM/$v.jsonl || fail "the $v run"
    done
    dry || $SP -c "import json; f = lambda v: {(r['kind'], r['arm']): (r['act_hash'], r['success'], r['e'])
    for r in map(json.loads, open(f'$SM/{v}.jsonl'))}
a, s = f('async'), f('sync'); assert a == s and len(a) == 3, (a, s); print('async == sync, bitwise:', a)" \
      || fail "async and sync differ"
    # 8. pod B, the OFT venv: s + d <= H = 8 refused before the model loads; no "__numpy__" in any OFT record
    if [ $POD = B ]; then
      if dry; then echo "+ $(mrun plus oft) --brain oft --source plus --set calib --arms=none --d 8 ...  # must refuse"
      elif $(mrun plus oft) --brain oft --source plus --set calib --cases $(dbg plus 1) --arms=none --d 8 \
        --out $W/never.jsonl > $LOG/smoke_oft_d8.log 2>&1 || ! grep -q 'chunk H = 8' $LOG/smoke_oft_d8.log; then
        fail "OFT took --d 8 (or failed otherwise: $LOG/smoke_oft_d8.log)"
      fi
    fi
    oft_numpy
    if dry; then for b in $BRAINS; do eval "E_$b=\$(envs c1e4m $b)"; done; fi
    t=; en=
    for b in $BRAINS; do v=E_$b; t+=" lmax_$b=$(gtempo $LOG/tempo_${b}_e${!v}.log)"; en+=" $b=${!v}"; done
    t+=" bench_$BB=$(full_rate $LOG/smoke_bench_$BB.log)"
    dry || ! grep -qE '=( |$)' <<< "$t" || fail "a tempo is missing (no group / full batch in its log)"
    { echo "tempo (s/episode):$t"; echo "C1-E4m envs:$en"; } | tee -a $W/tempo_run.txt
    x cp $W/tempo_run.txt $W/tempo_smoke.txt  # the stages read the last passed smoke's
    x summ --forecast $W/tempo_smoke.txt --pod $POD | tee $W/forecast.txt
    echo "[$(date +%T)] smoke done. Pod $POD's forecast is above ($W/forecast.txt): the owner adds the other pod's" \
      "when journaling the calib line; over ~\$27 together the owner is told before the grid (spec §13)"
    ;;
  connect)
    check_code record
    run_phase connect "$(summ --plan --pod $POD)" || fail "connect: a runner line failed (above); no gate computed"
    oft_numpy
    echo "=== $(date -u) connect gate, pod $POD" >> $W/gates.txt
    x summ --connect $O/connect_*.jsonl --pod $POD | tee -a $W/gates.txt; q=${PIPESTATUS[0]}
    [ $q = 0 ] || { [ $q = 3 ] && x touch $W/connect_fail; fail "connect did not pass (exit $q): pod $POD stops, the
    owner is asked (spec §9, §13)"; }
    x rm -f $W/connect_fail  # an earlier gate's (a brain since dropped: the summary's DROPPED, spec journal 2026-10-17)
    ;;
  calib | calib_more)
    [ ! -f $W/connect_fail ] || refuse "connect failed ($W/connect_fail): no calibration"
    check_code require
    x summ --connect $O/connect_*.jsonl --pod $POD >/dev/null \
      || refuse "the connect gate has not passed: POD=$POD ./scripts/gpu_c1e4m.sh connect"
    if [ $STAGE = calib ]; then
      run_phase calib "$(summ --plan --pod $POD)" || fail "calib: a runner line failed (above)"
    else  # spec §5: Gcal on the next K cases in order for calib's brains with N < 45 (calib.txt's "short" line)
      short=$(grep -oE "^C1-E4 part 2 short $POD: .*" $W/calib.txt 2>/dev/null | tail -1 | cut -d: -f2-)
      dry && [ -z "$short" ] && short=${SHORT:-"$(for b in $BRAINS; do printf 'm/%s ' $b; done)b/$BB"}
      [ -n "$short" ] || fail "calib.txt names no brain with N < 45: POD=$POD ./scripts/gpu_c1e4m.sh calib first"
      [[ "$short" != *b/* ]] || x $SP -c "import json, sys; sys.path.insert(0, 'src'); import maxwrap
more = maxwrap.load_split('results/c1-e4/part2/bench_calib_pairs.json')['more']
json.dump(more[:$K], open('$O/bench_more.json', 'w'))" || exit 1  # a list: the runner reads a frozen file's calib only
      run_phase more "$(summ --plan --pod $POD --more "$short" --k $K)" || fail "calib_more: a runner line failed"
    fi
    oft_numpy
    calib_line
    ;;
  grid)
    # the plan comes from the working tree's summary: it, this script and the runners run as committed and pushed
    git diff --quiet HEAD -- . && [ -z "$(git status --porcelain --untracked-files=all -- src scripts)" ] || {
      git status --short -- src scripts; refuse "local edits on the pod (above): undo them, or commit on the Mac, push,
    then here: git pull"; }
    spec=$(git show HEAD:$SPEC 2>/dev/null)
    m=$(grep -oE "$(line_re $POD)" <<< "$spec" | tail -1)
    dry && [ -z "$m" ] && m=${KBAR_LINE:-}
    c=$(grep -oE "$(line_re $POD)" $W/calib.txt 2>/dev/null | tail -1)
    [ -n "$m" ] || fail "the spec at HEAD journals no calibration line of pod $POD: journal calib.txt's line on the Mac,
    push, pull"
    [ "$m" = "$c" ] || refuse "the spec at HEAD journals '$m', this pod's calibration printed '${c:-nothing}'
    ($W/calib.txt)"
    # a refreeze (spec §5: N < 45 on either pod) moves both pods' evaluation cases: no grid before both lines
    other=$([ $POD = A ] && echo B || echo A)
    # ONLY (an id regex, owner 2026-10-17): only those plan lines. Bench lines alone (runner c1e4: tasks x inits, the
    # pod's own bench kbar) never use the split, so they need no line of the other pod
    if [ -n "${ONLY:-}" ] && ! summ --plan --pod $POD --kbar-line "$m" | awk -F'\t' -v re="^($ONLY)\$" '$5 ~ re' \
      | cut -f2 | grep -qx c1e4m; then
      echo "ONLY=$ONLY: bench lines only (no split): no calibration line of pod $other needed"
    else
      grep -qE "$(line_re $other)" <<< "$spec" || refuse "the spec at HEAD has no calibration line of pod $other yet:
    its calib may still refreeze the split (spec §5); journal it on the Mac, push, pull"
    fi
    ex=$(grep -oE "^C1-E4 part 2 extra $POD: m=[0-9]+" $W/calib.txt 2>/dev/null | grep -oE '[0-9]+$')
    [ -z "$ex" ] || [ "$($SP -c "import json; print(json.load(open('$SPLIT'))['extra'])")" -ge "$ex" ] \
      || refuse "calib used $ex further cases (calib.txt) but the split at HEAD has a smaller extra: refreeze on the
    Mac (spec §5), push, pull"
    dry || { git rev-parse -q --verify @{u} >/dev/null 2>&1 && git fetch -q \
      && git merge-base --is-ancestor HEAD @{u}; } \
      || refuse "HEAD is not pushed (not in its upstream): push from the Mac, then here: git pull"
    [ ! -f $W/connect_fail ] || refuse "connect failed ($W/connect_fail): no grid"
    check_code require
    PLAN=$(summ --plan --pod $POD --kbar-line "$m") || exit 1
    [ -z "${ONLY:-}" ] || PLAN=$(awk -F'\t' -v re="^($ONLY)\$" '$5 ~ re' <<< "$PLAN")
    [ -n "$PLAN" ] || fail "ONLY=${ONLY:-} matches no line of the plan"
    echo "[$(date +%T)] grid, pod $POD; $m" | tee -a $W/forecast.txt
    run_phase grid "$PLAN"; rc=$?
    oft_numpy
    if dry; then echo "+ cat <every plan line's --out> > $O/grid_all_$POD.jsonl"
    else GA=$O/grid_all_$POD.jsonl
      for id in $(cut -f5 <<< "$PLAN"); do cat $O/$id.jsonl $O/${id}_*.jsonl 2>/dev/null; done > $GA
      echo "[$(date +%T)] $GA: $(wc -l < $GA) records (the summary checks each block)"
    fi
    echo "On the Mac, with both pods' results: python scripts/c1e4m_summary.py" \
      "results/c1-e4/part2/pod_A/grid_all_A.jsonl results/c1-e4/part2/pod_B/grid_all_B.jsonl --fig fig.png"
    [ $rc = 0 ] || fail "grid lines still failing after pass 2 (above): the summary's completeness gate shows them"
    ;;
  cprobe)  # controls spec §7.1: part 2's pieces repeated as they ran, act_hash against the saved -> P or S per probe
    check_code record
    cphase cprobe --probe-plan || fail "cprobe: a probe line failed (above); no mode computed"
    oft_numpy
    x cc --probe $O/P1_*.jsonl $O/P2_*.jsonl $O/P3_*.jsonl | tee $W/mode.txt  # -> $O/mode.json
    [ "${PIPESTATUS[0]}" = 0 ] || fail "the probe summary failed (above): no mode"
    if dry; then platform; echo "+ write $O/PLATFORM: the output of these"
    else p=$(platform) || fail "nvidia-smi or torch failed: no $O/PLATFORM"; echo "$p" > $O/PLATFORM; fi
    ;;
  csmoke)  # controls spec §12 (outcomes never read): the tests, PPCL and VS0.3 with the Base prefix, tempo, forecast
    dry || [ -f $O/mode.json ] || fail "no $O/mode.json: POD=C ./scripts/gpu_c1e4m.sh cprobe first"
    check_code require
    check_platform
    # tempo_smoke.txt too: a failed rerun must not leave an earlier pass's, which opens cgrid's gate
    dry || rm -f $O/Csmoke_*.jsonl $LOG/Csmoke_*.log $LOG/csmoke*.log $W/tempo_run.txt $W/peak_csmoke* \
      $W/tempo_smoke.txt
    # the C1-E4 tests only: other modules of tests/ import jax / flax, which the LeRobot venv has not; no Hub needed
    x $P -m pytest -q tests/test_objreflex.py tests/test_gauto.py tests/test_handoff.py tests/test_lead.py \
      tests/test_maxwrap.py tests/test_c1e4m_run.py tests/test_c1e4c_summary.py tests/test_c1e4m_brains.py 2>&1 \
      | tee $LOG/csmoke_tests.log
    [ "${PIPESTATUS[0]}" = 0 ] || fail "the C1-E4 tests fail on the pod (above): the owner decides"
    peak_run $W/peak_csmoke cphase csmoke --plan --line smoke || fail "csmoke: a runner line failed (above)"
    dry || grep -qE '^[0-9]+$' $W/peak_csmoke || fail "no GPU memory samples ($W/peak_csmoke.err)"
    x cc --smoke-check $O/Csmoke_pi05*.jsonl \
      || fail "PPCL or VS0.3 has prefix_ok < 5 of 6 (above): stop, the owner decides (controls spec §12)"
    # until a grid line measures them, results/c1-e4/part2/pod_B/tempo_grid.txt: OFT+ part 2's last measured M1 OFT+
    # tempo, OFT part 2's last measured M4 OFT tempo
    t=" lmax_pi05=$(gtempo $LOG/Csmoke_pi05*.log) lmax_oftplus=12.497 lmax_oft=12.993"
    t+=" bench_pi05=$(full_rate $LOG/Csmoke_bench*.log)"
    dry || ! grep -qE '=( |$)' <<< "$t" || fail "a tempo is missing (no group / full batch in its log)"
    { echo "csmoke: GPU peak $(cat $W/peak_csmoke) MiB (max $PEAK_MAX)"; echo "tempo (s/episode):$t"; } \
      | tee $W/tempo_run.txt
    x cp $W/tempo_run.txt $W/tempo_smoke.txt  # cgrid's gate and first tempo: the last passed csmoke's
    cfc $W/tempo_smoke.txt | tee $W/forecast.txt
    echo "[$(date +%T)] csmoke done. The forecast is above ($W/forecast.txt), the modes in mode.txt: the owner decides"
    ;;
  cgrid)  # controls spec §12: lines 1-7 strictly by priority, k after lines 1 and 4 (§5), PPC on the latched (§7.2)
    CLINES=${CLINES:-1 2 3 4 5 6 7}  # §12: above the $13 cap the lines that do not fit wait for the owner
    p=0
    for L in $CLINES; do  # by priority, each line with the selection lines its k needs
      [[ $L =~ ^[1-7]$ ]] && [ $L -gt $p ] || fail "CLINES='$CLINES': lines of 1-7 in increasing (priority) order"
      [[ $L =~ ^[14]$ ]] || [[ " $CLINES " == *" 1 "* ]] || fail "CLINES='$CLINES': line $L needs line 1 (k_π)"
      [ $L != 5 ] || [[ " $CLINES " == *" 4 "* ]] || fail "CLINES='$CLINES': line 5 needs line 4 (k_OFT+)"
      p=$L
    done
    dry || grep -q '^tempo (s/episode): ' $W/tempo_smoke.txt 2>/dev/null \
      || fail "no passed csmoke ($W/tempo_smoke.txt): POD=C ./scripts/gpu_c1e4m.sh csmoke first"
    check_code require
    check_platform
    echo "[$(date +%T)] cgrid, pod C, lines $CLINES" | tee -a $W/forecast.txt
    for L in $CLINES; do
      cphase c$L --plan --line $L || fail "cgrid: line $L still fails after pass 2 (above)"
      b=; [ $L = 1 ] && b=pi05; [ $L = 4 ] && b=oftplus
      [ -z "$b" ] && continue
      x cc --select $b $O/C${L}_*.jsonl | tee -a $W/k_select.txt  # -> $O/k_select.json
      [ "${PIPESTATUS[0]}" = 0 ] || fail "the k selection of $b failed (above)"
    done
    C5="--plan --line c5 --records"  # the records of lines 2, 3, 7 run so far (this cgrid's or an earlier one's)
    if dry; then echo "+ cc $C5 <those of $O/C2_*.jsonl $O/C3_*.jsonl $O/C7_*.jsonl there>   # with a mode S"
    elif grep -qE '"mode": *"S"' $O/mode.json && rs=$(ls $O/C[237]_*.jsonl 2>/dev/null); then
      # shellcheck disable=SC2086
      cphase cc5 $C5 $rs || fail "cgrid: the PPC pass on the latched cases still fails after pass 2 (above)"
    fi
    oft_numpy
    x sh -c "cat $O/C[1-8]*.jsonl > $O/grid_all_C.jsonl"  # the grid lines' records, not the probe's or the smoke's
    echo "On the Mac: python scripts/c1e4c_summary.py $O/grid_all_C.jsonl (mode.json, k_select.json beside it)"
    ;;
  ctail)  # controls spec §12 line 8: OFT, VS(k_pi) and G-auto, always self-contained; after cgrid, on the owner's go
    [ "$TAIL" = 1 ] || fail "ctail runs with TAIL=1 (after POD=C TAIL=1 ./scripts/gpu_c1e4m.sh setup: OFT's weights)"
    dry || [ -f $O/grid_all_C.jsonl ] || fail "no finished cgrid (no $O/grid_all_C.jsonl): cgrid first"
    dry || [ -f $C1E4_OFT_CKPT/oft-sha-ok ] || fail "no OFT weights: POD=C TAIL=1 ./scripts/gpu_c1e4m.sh setup first"
    check_code require
    check_platform
    cphase c8 --plan --line 8 || fail "ctail: line 8 still fails after pass 2 (above)"
    oft_numpy
    x sh -c "cat $O/C[1-8]*.jsonl > $O/grid_all_C.jsonl"
    ;;
  vgrid)  # PBVS spec §3-§8 and the journals: V0, band, vsmoke, forecast, V1, select, V2, VLINES, the reading
    # the lines after V2, run in the order given (owner 2026-10-21: V3 one line per d, so the last line is one d and
    # cuttable): known tokens only, V3 = V3d10 V3d20, none twice after that
    VL=
    for L in ${VLINES:-V3d10 V3d20 Nglr V2eps}; do
      case $L in V3) VL+=" V3d10 V3d20" ;; V3d10 | V3d20 | Nglr | V2eps) VL+=" $L" ;;
        *) fail "VLINES='$VLINES': $L is not a line (V3d10 V3d20 Nglr V2eps, V3 = V3d10 V3d20)" ;; esac
    done
    [ -z "$(tr ' ' '\n' <<< "$VL" | sort | uniq -d)" ] || fail "VLINES='$VLINES': a line twice (V3 = V3d10 V3d20)"
    VLINES=${VL# }
    # the noisy cell's arms (owner 2026-10-21): the full cell, or PPCL@glr alone (PPC@glr part 2's, through the pod);
    # unset: the pod's first choice, once its Nglr ran ($O/NGLR_ARMS), else the full cell; never two on one pod
    [ -n "${NGLR_ARMS:-}" ] || NGLR_ARMS=$(cat $O/NGLR_ARMS 2>/dev/null) || NGLR_ARMS=PPCL@glr,PPC@glr
    case $NGLR_ARMS in PPCL@glr,PPC@glr | PPCL@glr) ;;
      *) fail "NGLR_ARMS='$NGLR_ARMS': PPCL@glr,PPC@glr (the full cell, default) or PPCL@glr (the cheap one)" ;; esac
    [ ! -f $O/NGLR_ARMS ] || [ "$(cat $O/NGLR_ARMS)" = "$NGLR_ARMS" ] \
      || fail "NGLR_ARMS=$NGLR_ARMS: this pod's Nglr ran with $(cat $O/NGLR_ARMS) ($O/NGLR_ARMS): the owner decides"
    # a finished cgrid (lines 2, 3, 7 and the C5 pass: grid_all_C.jsonl is written last) and no stage or runner live (one
    # runner on the GPU; the brackets keep pgrep from matching its own command line)
    dry || { [ -f $O/grid_all_C.jsonl ] && [ -z "$(find $O -maxdepth 1 -name 'C*.jsonl' -newer $O/grid_all_C.jsonl)" ]; } \
      || fail "cgrid has not finished ($O/grid_all_C.jsonl missing or older than a C*.jsonl): vgrid starts after it"
    if ! dry; then
      live=$(pgrep -af 'gpu_c1e4[m]\.sh +(cgrid|ctail)|c1e4m_ru[n]\.py|c1e4_ru[n]\.py')
      [ -z "$live" ] || fail "a controls stage or a runner is live (below): vgrid starts after it ends
$live"
    fi
    # V2, Nglr, V2eps pair with controls line 2's records and Base batches (its frozen file's 108 Plus cases first, on
    # disk), V3 with line 3's
    x cv --controls-check $O/C2_*.jsonl $O/C3_*.jsonl || refuse "controls lines 2 and 3 are not complete, or the frozen \
file's Base cases are not V2's first 108 Plus (above): V2 and V3 pair with them"
    CODE_PATHS+=" scripts/c1e4v_summary.py"
    check_code_v check
    check_platform
    x $P -m pytest -q tests/test_objreflex.py tests/test_gauto.py tests/test_handoff.py tests/test_lead.py \
      tests/test_maxwrap.py tests/test_c1e4m_run.py tests/test_c1e4c_summary.py tests/test_c1e4m_brains.py \
      tests/test_vservo.py tests/test_c1e4v_summary.py 2>&1 | tee $LOG/vgrid_tests.log
    [ "${PIPESTATUS[0]}" = 0 ] || fail "the C1-E4 tests fail on the pod (above): the owner decides"
    x cc --selftest && x cv --selftest || fail "a summary's selftest fails (above): the owner decides"
    check_code_v record  # the first vgrid's, once its tests passed: later vgrids require it
    SUMM=cv  # cphase plans with the PBVS summary; its lines take no per-line forecast (tempo_grid.txt stays the controls')
    echo "[$(date +%T)] vgrid, pod C, then $VLINES (Nglr: $NGLR_ARMS)" | tee -a $W/forecast.txt
    cphase v0 --plan --line V0 || fail "vgrid: V0 still fails after pass 2 (above)"
    # spec §3: the arm model and its band from V0 (and band_eps.json above 15 mm), again only when a V0 file is newer
    if dry || [ ! -f $O/band.json ] || [ -n "$(find $O -maxdepth 1 -name 'V0_*.jsonl' -newer $O/band.json)" ]; then
      x cv --band $O/V0_*.jsonl | tee -a $W/band.txt
      [ "${PIPESTATUS[0]}" = 0 ] || fail "the band of V0 failed (above): V0 incomplete?"
    fi
    # the smoke afresh until V1 starts (a stale smoke's records would pass its gate on resume)
    dry || ls $O/V1_*.jsonl >/dev/null 2>&1 || rm -f $O/Vsmoke_*.jsonl $LOG/Vsmoke_*.log
    cphase vsmoke --plan --line vsmoke || fail "vgrid: the smoke still fails after pass 2 (above)"
    x cv --smoke-check $O/Vsmoke_pi05*.jsonl --controls $O/C2_pi05_eval300_plus_d0.jsonl || fail "PBVS0.3 has \
prefix_ok < 5 of 6, a case short, no counters, or a Base e not line 2's (above): stop, the owner decides"
    # spec §8 at the controls grid's tempo (its latest lmax_pi05, lines 2 / 3): V1 + V2 + V3 above $4 holds VLINES
    hold=
    if dry; then vfc $W/tempo_grid.txt
    else vfc $W/tempo_grid.txt | tee -a $W/forecast.txt; q=${PIPESTATUS[0]}
      { [ $q = 0 ] || [ $q = 3 ]; } || fail "the PBVS forecast failed (exit $q, above)"
      [ $q = 3 ] && { [ "${V3_OK:-}" = 1 ] && echo "V3_OK=1: the owner's ok, $VLINES run" || hold=1; }
    fi
    cphase v1 --plan --line V1 || fail "vgrid: V1 still fails after pass 2 (above)"
    if dry || [ ! -f $O/pbvs_k.json ] || [ -n "$(find $O -maxdepth 1 -name 'V1_*.jsonl' -newer $O/pbvs_k.json)" ]; then
      x cv --select $O/V1_*.jsonl | tee -a $W/pbvs_k.txt  # spec §4 -> $O/pbvs_k.json
      [ "${PIPESTATUS[0]}" = 0 ] || fail "the lambda selection failed (above): V1 incomplete?"
    fi
    cphase v2 --plan --line V2 || fail "vgrid: V2 still fails after pass 2 (above)"
    if [ -n "$hold" ]; then  # the owner's go / no-go stays blind: nothing is read
      echo "!!! $VLINES and the reading wait for the owner's ok: the PBVS forecast of V1 + V2 + V3 is above \$4" \
        "(above, PBVS spec §8). After the ok: V3_OK=1 [VLINES=...] POD=C ./scripts/gpu_c1e4m.sh vgrid" \
        | tee -a $W/forecast.txt
      exit 0
    fi
    for L in $VLINES; do  # V2eps: an empty plan unless band.json's delta > 15 mm
      a=
      [ $L = Nglr ] && { a="--nglr-arms $NGLR_ARMS"; wr $O/NGLR_ARMS $NGLR_ARMS; }
      # shellcheck disable=SC2086
      cphase "$(tr A-Z a-z <<< $L)" --plan --line $L $a || fail "vgrid: $L still fails after pass 2 (above)"
    done
    x sh -c "cat $O/V[0-3]_*.jsonl $O/Nglr_*.jsonl $O/V2eps_*.jsonl 2>/dev/null > $O/grid_all_V.jsonl"  # not the smoke
    x cv $O/grid_all_V.jsonl --controls $O/C2_*.jsonl $O/C3_*.jsonl --nglr-arms $NGLR_ARMS --out $O/summary_V.json \
      > $W/reading_V.txt; q=$?
    tail -1 $W/reading_V.txt
    [ $q = 0 ] || fail "the reading's checks before reading failed (exit $q, $W/reading_V.txt): the owner decides"
    echo "The reading: $O/summary_V.json; on the Mac: python scripts/c1e4v_summary.py $O/grid_all_V.jsonl" \
      "--controls $O/C2_*.jsonl $O/C3_*.jsonl --nglr-arms $NGLR_ARMS (band.json, pbvs_k.json, mode.json," \
      "k_select.json beside it)"
    ;;
  vtgrid)  # the VS-T journal: VT1, vst_k.json, VT2, VT3d10, VT3d20, the reading
    # vgrid's clean end: its reading's last line PASS (reading_V.txt, written after grid_all_V.jsonl, which is newer
    # than every V line's file) and no stage or runner live (the brackets keep pgrep from matching its own command line)
    dry || { [ -f $O/grid_all_V.jsonl ] && [ $W/reading_V.txt -nt $O/grid_all_V.jsonl ] \
      && [ -z "$(find $O -maxdepth 1 \( -name 'V[0-3]_*.jsonl' -o -name 'Nglr_*.jsonl' -o -name 'V2eps_*.jsonl' \) \
        -newer $O/grid_all_V.jsonl)" ] \
      && [ "$(tail -1 $W/reading_V.txt)" = "C1-E4 PBVS checks before reading: PASS" ]; } \
      || fail "vgrid has not ended cleanly (grid_all_V.jsonl missing or older than a V line's file, or its reading \
$W/reading_V.txt older or not PASS): vtgrid starts after it"
    if ! dry; then
      live=$(pgrep -af 'gpu_c1e4[m]\.sh +(cgrid|ctail|vgrid)|c1e4m_ru[n]\.py|c1e4_ru[n]\.py')
      [ -z "$live" ] || fail "a stage or a runner is live (below): vtgrid starts after it ends
$live"
    fi
    # VT2 and VT3 pair with controls lines 2 and 3 (their cases, order and Base groups; e as their Bases')
    x cv --controls-check $O/C2_*.jsonl $O/C3_*.jsonl || refuse "controls lines 2 and 3 are not complete (above): VT2 \
and VT3 pair with them"
    CODE_PATHS+=" scripts/c1e4v_summary.py scripts/c1e4t_summary.py"
    check_code_v check t
    check_platform
    x $P -m pytest -q tests/test_objreflex.py tests/test_gauto.py tests/test_handoff.py tests/test_lead.py \
      tests/test_maxwrap.py tests/test_c1e4m_run.py tests/test_c1e4c_summary.py tests/test_c1e4m_brains.py \
      tests/test_vservo.py tests/test_c1e4v_summary.py tests/test_vst.py tests/test_c1e4t_summary.py 2>&1 \
      | tee $LOG/vtgrid_tests.log
    [ "${PIPESTATUS[0]}" = 0 ] || fail "the C1-E4 tests fail on the pod (above): the owner decides"
    x cc --selftest && x cv --selftest && x ct --selftest \
      || fail "a summary's selftest fails (above): the owner decides"
    check_code_v record t  # the first vtgrid's, once its tests passed: later vtgrids require it
    SUMM=ct  # cphase plans with the VS-T summary
    echo "[$(date +%T)] vtgrid, pod C: VT1, vst_k.json, VT2, VT3d10, VT3d20" | tee -a $W/forecast.txt
    cphase vt1 --plan --line VT1 || fail "vtgrid: VT1 still fails after pass 2 (above)"
    # the journal: lambda by §4's rule from VT1 (vst_k.json), again only when a VT1 file is newer
    if dry || [ ! -f $O/vst_k.json ] || [ -n "$(find $O -maxdepth 1 -name 'VT1_*.jsonl' -newer $O/vst_k.json)" ]; then
      x ct --select-vst $O/VT1_*.jsonl | tee -a $W/vst_k.txt
      [ "${PIPESTATUS[0]}" = 0 ] || fail "the VS-T lambda selection failed (above): VT1 incomplete?"
    fi
    for L in VT2 VT3d10 VT3d20; do
      cphase "$(tr A-Z a-z <<< $L)" --plan --line $L || fail "vtgrid: $L still fails after pass 2 (above)"
    done
    x sh -c "cat $O/VT[1-3]_*.jsonl > $O/grid_all_T.jsonl"
    x ct $O/grid_all_T.jsonl --controls $O/C2_*.jsonl $O/C3_*.jsonl --v $O/grid_all_V.jsonl --out $O/summary_T.json \
      > $W/reading_T.txt; q=$?
    tail -1 $W/reading_T.txt
    [ $q = 0 ] || fail "the VS-T reading's checks before reading failed (exit $q, $W/reading_T.txt): the owner decides"
    echo "The reading: $O/summary_T.json; on the Mac: python scripts/c1e4t_summary.py $O/grid_all_T.jsonl" \
      "--controls $O/C2_*.jsonl $O/C3_*.jsonl --v $O/grid_all_V.jsonl (vst_k.json, band.json, pbvs_k.json," \
      "mode.json, k_select.json beside it)"
    ;;
esac
