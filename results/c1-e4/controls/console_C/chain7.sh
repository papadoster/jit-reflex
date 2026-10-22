#!/bin/bash
# owner 2026-10-20: line 7 right after line 6. Waits for the CLINES='1 2 3 4 5 6' cgrid to end; if it ended cleanly
# (its 'On the Mac' line, no '!!!' at the end), cgrid with lines 1-7 (resumes: the runners skip done cases)
cd /workspace/jit-reflex
until tail -2 /workspace/cgrid_C.log | grep -q 'packed c1e4m_results_' && ! pgrep -f '^bash ./scripts/gpu_c1e4m.sh' >/dev/null; do sleep 60; done
sleep 10
if grep -q 'On the Mac: python scripts/c1e4c_summary.py' /workspace/cgrid_C.log && ! tail -30 /workspace/cgrid_C.log | grep -q '^!!!'; then
  POD=C CLINES='1 2 3 4 5 6 7' ./scripts/gpu_c1e4m.sh cgrid 2>&1 | tee /workspace/cgrid7_C.log
else
  echo "[$(date +%T)] chain7: the first cgrid did not end cleanly, line 7 not started" | tee /workspace/cgrid7_C.log
fi
