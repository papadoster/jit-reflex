#!/bin/bash
# pod P chain (owner 2026-11-08: "когда поймаешь, запускай прогон"): wait for setup's ok, then psmoke, then pgrid;
# any failure stops the chain (pgrid itself also refuses without psmoke's CODE.p). It starts while the clone still runs:
# setup counts as dead only after its process was seen once; until then up to START_MAX s for it to start.
set -o pipefail
OK=${OK:-/workspace/lmax-deps/c1e4m-setup-ok} REPO=${REPO:-/workspace/jit-reflex} LOGDIR=${LOGDIR:-/workspace}
START_MAX=${START_MAX:-2400} NAP=${NAP:-20} PAT=${PAT:-'^bash ./scripts/gpu_c1e4[m].sh setup'}
seen=
until [ -f "$OK" ]; do
  if pgrep -f "$PAT" >/dev/null; then seen=1
  elif [ -f "$OK" ]; then break
  elif [ -n "$seen" ]; then echo "chainP: setup died without its ok file $(date -u +%T)"; exit 1
  elif [ $SECONDS -gt "$START_MAX" ]; then echo "chainP: setup never started in $START_MAX s $(date -u +%T)"; exit 1
  fi
  sleep "$NAP"
done
cd "$REPO" || { echo "chainP: no $REPO"; exit 1; }
echo "chainP: setup ok $(date -u +%T)"
POD=P ./scripts/gpu_c1e4m.sh psmoke 2>&1 | tee "$LOGDIR/psmoke_P.log" || { echo "chainP: psmoke failed $(date -u +%T)"; exit 1; }
echo "chainP: psmoke ok $(date -u +%T)"
POD=P ./scripts/gpu_c1e4m.sh pgrid 2>&1 | tee "$LOGDIR/pgrid_P.log"
echo "chainP: pgrid exit $? $(date -u +%T)"
