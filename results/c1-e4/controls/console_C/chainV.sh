#!/bin/bash
# chainV.sh BAL0 T0 (owner 2026-10-21: no top-up, run through, no idling): after cgrid's clean end, pull c1-e4 and run
# vgrid with V3d10 Nglr V2eps; then V3d20 only if >= $0.5 is left after its forecast ($1.18 x 1.1). The balance is
# BAL0 at epoch T0 minus $0.371/h. /workspace/skip_d20 or /workspace/go_d20 (written from the Mac) override the rule.
BAL0=$1; T0=$2; R=/workspace/jit-reflex; O=$R/results/c1-e4/controls/pod_C; L=/workspace/cgrid2_C.log
log() { echo "[$(date -u +%T)] chainV: $*" | tee -a /workspace/chainV.log; }
log "waiting for cgrid's end (BAL0 \$$BAL0 at $T0)"
while pgrep -f '^bash ./scripts/gpu_c1e4[m].sh cgrid' >/dev/null; do sleep 60; done
tail -3 $L | grep -q 'packed c1e4m_results_' || { log "cgrid ended without its pack line: not going on"; exit 1; }
[ -f $O/grid_all_C.jsonl ] || { log "no grid_all_C.jsonl: not going on"; exit 1; }
cd $R
for i in $(seq 1 90); do  # the split code (V3d10) on origin, up to 3 h
  git fetch -q origin c1-e4 && git show origin/c1-e4:scripts/c1e4v_summary.py | grep -q V3d10 && break
  [ $i = 1 ] && log "waiting for the split code on origin"; sleep 120
done
git show origin/c1-e4:scripts/c1e4v_summary.py | grep -q V3d10 || { log "no split code after 3 h: not going on"; exit 1; }
git pull -q --ff-only origin c1-e4 || { log "pull failed"; exit 1; }
log "pulled $(git log --oneline -1 | cut -c1-80)"
POD=C V3_OK=1 VLINES="V3d10 Nglr V2eps" ./scripts/gpu_c1e4m.sh vgrid 2>&1 | tee /workspace/vgrid_C.log
rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { log "vgrid exit $rc: V3d20 not started"; exit 1; }
left=$(python3 -c "import time; print(round($BAL0 - 0.371 * (time.time() - $T0) / 3600, 2))")
if [ -f /workspace/skip_d20 ]; then log "skip_d20: V3d20 not run (left ~ \$$left)"; exit 0; fi
if [ -f /workspace/go_d20 ] || python3 -c "import sys; sys.exit(0 if $left - 1.30 >= 0.5 else 1)"; then
  log "V3d20 starts (left ~ \$$left)"
  POD=C V3_OK=1 VLINES="V3d20" ./scripts/gpu_c1e4m.sh vgrid 2>&1 | tee /workspace/vgrid2_C.log
  log "V3d20 exit ${PIPESTATUS[0]}"
else
  log "V3d20 not run: left ~ \$$left < its forecast \$1.30 + the \$0.5 reserve"
fi
