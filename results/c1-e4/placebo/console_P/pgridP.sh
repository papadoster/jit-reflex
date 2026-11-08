#!/bin/bash
# pgrid by hand after chainP's pgrid was refused by its live-stage guard (the tmux server kept setup's command line)
set -o pipefail
cd /workspace/jit-reflex || exit 1
echo "chainP: pgrid started by hand $(date -u +%T)" >> /workspace/chainP.log
POD=P ./scripts/gpu_c1e4m.sh pgrid 2>&1 | tee /workspace/pgrid_P.log
echo "chainP: pgrid exit $? $(date -u +%T)" >> /workspace/chainP.log
