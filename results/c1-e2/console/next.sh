#!/usr/bin/env bash
# after a clean smoke: base, then pilot (the grid waits for the journaled coefficients from the Mac)
C=/workspace/console_smoke.txt
until grep -q "packed c1e2_results" $C 2>/dev/null; do sleep 20; done
grep -q "smoke: 26 of 26 records (6 + 16 + 4), 0 FAILED" $C && ! grep -q "smoke failed" $C || { echo "smoke not clean: stop"; exit 1; }
cd /workspace/jit-reflex
./scripts/gpu_c1e2.sh base 2>&1 | tee -a /workspace/console_base.txt
[ "${PIPESTATUS[0]}" = 0 ] || { echo "base failed: stop"; exit 1; }
./scripts/gpu_c1e2.sh pilot 2>&1 | tee -a /workspace/console_pilot.txt
