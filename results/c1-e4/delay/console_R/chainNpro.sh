#!/bin/bash
# Descriptive 'none' branch (owner 2026-11-19, delay spec journal): R0's/R20's exact runner command with
# --arms=none --base-prefix, on the same pod and code, no git pull. Pass 2 (resume) on a non-clean pass 1.
cd /workspace/jit-reflex
O=results/c1-e4/delay/pod_R; L=$O/logs
export LIBERO_SOURCE_PACKAGE_ROOT=/workspace/lmax-deps/LIBERO-plus/libero LIBERO_CONFIG_PATH=/workspace/lmax-deps/libero-plus-config \
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=/workspace/lmax-deps/LIBERO-MAX/scripts/libero_source_overlay:/workspace/lmax-deps/LIBERO-MAX/src \
  LIBERO_MAX_ROOT=/workspace/lmax-deps/LIBERO-MAX
# the stage script's own exports (scripts/gpu_c1e4m.sh lines 234-243, 697), as R0 and R20 ran
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OPENVLA_OFT_ROOT=/workspace/lmax-deps/openvla-oft C1E4_OFT_CKPT=/workspace/oft-ckpt TF_FORCE_GPU_ALLOW_GROWTH=true \
  HF_HOME=/workspace/.cache/huggingface UV_CACHE_DIR=/workspace/.cache/uv UV_PYTHON_INSTALL_DIR=/workspace/.cache/uv-python \
  HF_HUB_OFFLINE=1
[ "$(git rev-parse HEAD)" = "$(cat /workspace/N_HEAD)" ] || { echo "chainR: N HEAD changed, stop $(date -u +%FT%T)" >> /workspace/chainR.log; exit 1; }
# fix (journal 2026-11-20): the scene roots per source, as scripts/gpu_c1e4m.sh's mrun: pro -> LIBERO-PRO and
# libero-pro-config. chainN.sh set LIBERO-plus for all four lines, so its two pro lines failed with no record.
until grep -q "^chainR: N done, packed" /workspace/chainR.log; do sleep 60; done  # chainN.sh has ended
for d in 0 20; do f=$L/N${d}_pi05_evalrest_pro_d${d}.log; [ -f $f ] && mv $f $L/N${d}_pi05_evalrest_pro_d${d}.wrong_root.log; done
export LIBERO_SOURCE_PACKAGE_ROOT=/workspace/lmax-deps/LIBERO-PRO/libero LIBERO_CONFIG_PATH=/workspace/lmax-deps/libero-pro-config
env | grep -E "^LIBERO_(SOURCE|CONFIG)" | sed 's/^/pro lines: /' >> $O/pod_env_N.txt
for d in 0 20; do src=pro
  id=N${d}_pi05_evalrest_${src}_d${d}
  cmd="venv/bin/python scripts/c1e4m_run.py --device cuda --brain pi05 --source $src --set evalrest --arms=none --base-prefix --d $d --kbar 0.306 --band results/c1-e4/controls/pod_C/band_eps.json --n-envs 6 --out $O/$id.jsonl"
  for pass in 1 2; do
    echo "[$(date -u +%T)] pass $pass: $cmd" >> $L/$id.log
    $cmd >> $L/$id.log 2>&1 </dev/null; rc=$?
    tail -3 $L/$id.log | grep -qE "failed 0" && [ $rc = 0 ] && break
  done
  echo "chainR: $id exit $rc $(date -u +%FT%T)" >> /workspace/chainR.log
done
(cd $O && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
tar czf c1e4m_results_R.tgz.tmp $O && mv -f c1e4m_results_R.tgz.tmp c1e4m_results_R.tgz
echo "chainR: Npro done, packed $(date -u +%FT%T)" >> /workspace/chainR.log
