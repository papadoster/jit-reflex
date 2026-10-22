#!/bin/bash
# chainV.sh BAL0 T0 (owner 2026-10-21: no top-up, run through, no idling). After cgrid's clean end: pull c1-e4, then
# vgrid in steps, each later step only with >= $0.5 left after its forecast (the balance: BAL0 at epoch T0 minus
# $0.371/h). Files in /workspace written from the Mac override: nglr_arms (its NGLR_ARMS), go_/skip_v2eps, go_/skip_d20.
BAL0=$1; T0=$2; W=/workspace; R=$W/jit-reflex; O=$R/results/c1-e4/controls/pod_C; L=$W/cgrid2_C.log
log() { echo "[$(date -u +%T)] chainV: $*" | tee -a $W/chainV.log; }
left() { python3 -c "import time; print(round($BAL0 - 0.371 * (time.time() - $T0) / 3600, 2))"; }
fits() { python3 -c "import sys; sys.exit(0 if $(left) - $1 >= 0.5 else 1)"; }
vg() { local f=$1; shift; (cd $R && env POD=C V3_OK=1 "$@" ./scripts/gpu_c1e4m.sh vgrid) 2>&1 | tee -a $W/$f
  return ${PIPESTATUS[0]}; }
log "waiting for cgrid's end (BAL0 \$$BAL0 at $T0)"
while pgrep -f '^bash ./scripts/gpu_c1e4[m].sh cgrid' >/dev/null; do sleep 60; done
tail -3 $L | grep -q 'packed c1e4m_results_' || { log "cgrid ended without its pack line: not going on"; exit 1; }
[ -f $O/grid_all_C.jsonl ] || { log "no grid_all_C.jsonl: not going on"; exit 1; }
cd $R
for i in $(seq 1 23); do  # NGLR_ARMS on origin, up to 45 min; then whatever has V3d10
  git fetch -q origin c1-e4 && git show origin/c1-e4:scripts/gpu_c1e4m.sh | grep -q NGLR_ARMS && break
  [ $i = 1 ] && log "waiting for the NGLR_ARMS code on origin (up to 45 min)"; sleep 120
done
git show origin/c1-e4:scripts/c1e4v_summary.py | grep -q V3d10 || { log "no V3d10 code on origin: not going on"; exit 1; }
git pull -q --ff-only origin c1-e4 || { log "pull failed"; exit 1; }
log "pulled $(git log --oneline -1 | cut -c1-80); NGLR_ARMS in code: $(grep -c NGLR_ARMS scripts/gpu_c1e4m.sh)"
vg vgrid_C.log VLINES="V3d10" || { log "vgrid (V0..V2, V3d10) exit $?: not going on"; exit 1; }
arms=$(cat $W/nglr_arms 2>/dev/null || echo "PPCL@glr,PPC@glr")
log "Nglr starts, NGLR_ARMS=$arms (left ~ \$$(left))"
vg vgrid_C.log VLINES="Nglr" NGLR_ARMS="$arms" || { log "vgrid (Nglr) exit $?: not going on"; exit 1; }
if python3 -c "import json, sys; sys.exit(0 if json.load(open('$O/band.json'))['delta'] > 0.015 else 1)"; then
  if [ ! -f $W/skip_v2eps ] && { [ -f $W/go_v2eps ] || fits 1.08; }; then
    log "V2eps starts (left ~ \$$(left))"; vg vgrid_C.log VLINES="V2eps"; log "V2eps exit $?"
  else log "V2eps not run: left ~ \$$(left) < its forecast \$1.08 + the \$0.5 reserve (or skip_v2eps)"; fi
else log "V2eps not needed: the band is within 15 mm"; fi
if [ ! -f $W/skip_d20 ] && { [ -f $W/go_d20 ] || fits 1.30; }; then
  log "V3d20 starts (left ~ \$$(left))"; vg vgrid2_C.log VLINES="V3d20"; log "V3d20 exit $?"
else log "V3d20 not run: left ~ \$$(left) < its forecast \$1.30 + the \$0.5 reserve (or skip_d20)"; fi
log "done, left ~ \$$(left)"
