#!/usr/bin/env bash
# rerun base (resumes: finished settings are skipped), then pilot
cd /workspace/jit-reflex
./scripts/gpu_c1e2.sh base 2>&1 | tee -a /workspace/console_base.txt
[ "${PIPESTATUS[0]}" = 0 ] || { echo "base failed: stop"; exit 1; }
./scripts/gpu_c1e2.sh pilot 2>&1 | tee -a /workspace/console_pilot.txt
