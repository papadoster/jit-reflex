#!/bin/bash
# chainV3.sh (owner 2026-10-21 ~09:30 UTC: tops up $10, everything at full, no cuts). After cgrid's clean end: pull
# c1-e4, then vgrid in the owner's order: V0, smoke, V1, V2, V3 d 10 and d 20, the full noisy cell, V2eps (band > 15 mm).
W=/workspace; R=$W/jit-reflex; O=$R/results/c1-e4/controls/pod_C; L=$W/cgrid2_C.log
log() { echo "[$(date -u +%T)] chainV: $*" | tee -a $W/chainV.log; }
log "chainV3: waiting for cgrid's end (everything at full)"
while pgrep -f '^bash ./scripts/gpu_c1e4[m].sh cgrid' >/dev/null; do sleep 60; done
tail -3 $L | grep -q 'packed c1e4m_results_' || { log "cgrid ended without its pack line: not going on"; exit 1; }
[ -f $O/grid_all_C.jsonl ] || { log "no grid_all_C.jsonl: not going on"; exit 1; }
cd $R && git fetch -q origin c1-e4 && git pull -q --ff-only origin c1-e4 || { log "pull failed"; exit 1; }
grep -q NGLR_ARMS scripts/gpu_c1e4m.sh && grep -q V3d10 scripts/c1e4v_summary.py || { log "pulled code lacks V3d10/NGLR_ARMS"; exit 1; }
log "pulled $(git log --oneline -1 | cut -c1-80)"
env POD=C V3_OK=1 VLINES="V3d10 V3d20 Nglr V2eps" NGLR_ARMS="PPCL@glr,PPC@glr" ./scripts/gpu_c1e4m.sh vgrid 2>&1 \
  | tee -a $W/vgrid_C.log
log "vgrid exit ${PIPESTATUS[0]}"
