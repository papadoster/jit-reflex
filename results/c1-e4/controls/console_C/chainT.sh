#!/bin/bash
# chainT.sh (owner 2026-10-21: VS-T after V2eps, last). After chainV3 ends with "vgrid exit 0": pull c1-e4, run vtgrid.
# Nothing is pulled while vgrid can still run: a pull changes src and vgrid's CODE.v would refuse a resume.
W=/workspace; R=$W/jit-reflex
log() { echo "[$(date -u +%T)] chainT: $*" | tee -a $W/chainT.log; }
log "waiting for chainV3 and vgrid to end"
while pgrep -f 'chainV[3]\.sh' >/dev/null || pgrep -f '^bash \./scripts/gpu_c1e4[m]\.sh' >/dev/null; do sleep 60; done
grep 'chainV: vgrid exit' $W/chainV.log | tail -1 | grep -q 'exit 0$' \
  || { log "vgrid did not end cleanly (chainV.log): no pull, no vtgrid"; exit 1; }
cd $R && git fetch -q origin c1-e4 && git pull -q --ff-only origin c1-e4 || { log "pull failed"; exit 1; }
grep -q vtgrid scripts/gpu_c1e4m.sh || { log "the pulled code has no vtgrid"; exit 1; }
log "pulled $(git log --oneline -1 | cut -c1-80)"
env POD=C ./scripts/gpu_c1e4m.sh vtgrid 2>&1 | tee -a $W/vtgrid_C.log
log "vtgrid exit ${PIPESTATUS[0]}"
