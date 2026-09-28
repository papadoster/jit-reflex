#!/usr/bin/env bash
# Mac rehearsal of scripts/gpu_a2c2_fix.sh at toy scale (spec §6): every stage on one level, one cell, 8 envs.
# Numbers mean nothing; the point is that each stage runs, writes what the next reads, and the zero head is naive.
set -euo pipefail
cd "$(dirname "$0")/.."
O=results/b2b5_fix/rehearsal
rm -rf $O && mkdir -p $O
L=trampoline
P=$(uv run --offline src/a2c2_fix.py experts | awk -v L=$L '$1 == L {print $2}')
[ -f "$P" ] || { echo "!!! $P missing: the rehearsal uses the expert already on the Mac"; exit 1; }
uv run --offline src/a2c2_fix.py latency --out-dir $O --repeats 5 --warmup 2
uv run --offline src/a2c2_fix.py zero-head-check --out-dir $O
uv run --offline src/a2c2.py relabel --level-path worlds/l/$L.json --experts "$P" --out-dir $O/a2c2_paper --paper \
  --num-envs 16 --num-chunks 8 --num-epochs 1
uv run --offline src/eval_flow.py --run-path checkpoints/bc --config.num-evals 8 --level-paths worlds/l/$L.json \
  --methods a2c2_paper --cells 3,5 --seeds 20 --heads-root $O --output-dir $O/eval
uv run --offline src/a2c2_fix.py summarize --new-dir $O --no-strict | tail -20
echo "rehearsal OK"
