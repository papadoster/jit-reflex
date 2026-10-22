#!/bin/bash
# chainM.sh (owner 2026-10-22: B, the VT-m branch, after C). After chainK ends with "kgrid exit 0":
# wait until origin c1-e4 has the mgrid stage with its last line Ed10 (owner 2026-10-22), pull, run mgrid.
# Nothing is pulled while kgrid can still run: a pull changes src and kgrid's CODE.k would refuse a resume.
W=/workspace; R=$W/jit-reflex
log() { echo "[$(date -u +%T)] chainM: $*" | tee -a $W/chainM.log; }
log "waiting for chainK and kgrid to end"
while pgrep -f 'chain[K]\.sh' >/dev/null || pgrep -f '^bash \./scripts/gpu_c1e4[m]\.sh' >/dev/null; do sleep 60; done
grep 'chainK: kgrid exit' $W/chainK.log | tail -1 | grep -q 'exit 0$' \
  || { log "kgrid did not end cleanly (chainK.log): no pull, no mgrid"; exit 1; }
cd $R || exit 1
log "kgrid ended cleanly; waiting for mgrid on origin c1-e4"
until git fetch -q origin c1-e4 && git show origin/c1-e4:scripts/gpu_c1e4m.sh | grep -q 'for L in Md0 Md10 Ed10'; do sleep 120; done
git pull -q --ff-only origin c1-e4 || { log "pull failed"; exit 1; }
log "pulled $(git log --oneline -1 | cut -c1-80)"
env POD=C ./scripts/gpu_c1e4m.sh mgrid 2>&1 | tee -a $W/mgrid_C.log
log "mgrid exit ${PIPESTATUS[0]}"
