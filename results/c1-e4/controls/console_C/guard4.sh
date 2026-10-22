#!/bin/bash
# owner 2026-10-20: the VS branches are stopped. While the Mac is off the old cgrid (lines 1-6, with VS) goes on with
# lines 2 and 3 (PPCL and G-auto progress, VS records unread). This guard stops the stage when line 4 (pure VS, OFT+)
# would start; the chain for line 7 is killed so the old plan never runs line 7 with VS.
pkill -f chain7.sh
until grep -qE 'pass 1: .*(--brain oftplus|C4_oftplus)' /workspace/cgrid_C.log; do sleep 30; done
pkill -f '^bash ./scripts/gpu_c1e4m.sh'
for p in $(pgrep -f c1e4m_run.py); do kill -- -$(ps -o pgid= -p $p | tr -d ' ') 2>/dev/null; done
echo "[$(date -u +%T) UTC] guard4: line 4 was starting, the stage is stopped (the owner stopped VS)" | tee /workspace/guard4.txt
