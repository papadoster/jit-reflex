#!/bin/bash
# chainK.sh (owner 2026-10-22: C, the kappa law's ablation, runs before B). After chainT ends with "vtgrid exit 0":
# wait until origin c1-e4 has the kgrid stage (pushed with its journal), pull, run kgrid.
# Nothing is pulled while vtgrid can still run: a pull changes src and vtgrid's CODE.t would refuse a resume.
W=/workspace; R=$W/jit-reflex
log() { echo "[$(date -u +%T)] chainK: $*" | tee -a $W/chainK.log; }
log "waiting for chainT and vtgrid to end"
while pgrep -f 'chain[T]\.sh' >/dev/null || pgrep -f '^bash \./scripts/gpu_c1e4[m]\.sh' >/dev/null; do sleep 60; done
grep 'chainT: vtgrid exit' $W/chainT.log | tail -1 | grep -q 'exit 0$' \
  || { log "vtgrid did not end cleanly (chainT.log): no pull, no kgrid"; exit 1; }
cd $R || exit 1
log "vtgrid ended cleanly; waiting for kgrid on origin c1-e4"
until git fetch -q origin c1-e4 && git show origin/c1-e4:scripts/gpu_c1e4m.sh | grep -q '^  kgrid)'; do sleep 120; done
git pull -q --ff-only origin c1-e4 || { log "pull failed"; exit 1; }
log "pulled $(git log --oneline -1 | cut -c1-80)"
env POD=C ./scripts/gpu_c1e4m.sh kgrid 2>&1 | tee -a $W/kgrid_C.log
log "kgrid exit ${PIPESTATUS[0]}"
