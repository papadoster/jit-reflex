#!/usr/bin/env bash
# One-off helper for this pod (not in the repo): GCS throttles one connection to ~2 MB/s, so the expert data of spec
# §9 are fetched with 16 parallel byte ranges (<= 3 files ahead) while the heads train on the files already here. The
# training, SHA log and lock records are exactly gpu_b2b5.sh's A2C2 loop (same commit, same commands).
set -uo pipefail
cd /workspace/jit-reflex
O=results/b2b5
LEVELS="grasp_easy catapult cartpole_thrust hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker h17_unicycle
  chain_lander catcher_v3 trampoline car_launch"
mkdir -p $O/expert
fetch() {  # fetch <level>: 16 ranges, size-checked; 3 attempts, else <file>.failed
  local L=$1 u=https://storage.googleapis.com/rtc-assets/expert/data/worlds_l_$1.npz f=$O/expert/worlds_l_$1.npz n=16
  local size chunk a b i try
  size=$(curl -sI $u | awk 'tolower($1)=="content-length:"{print $2+0}')
  chunk=$(( (size + n - 1) / n ))
  for try in 1 2 3; do
    for i in $(seq 0 $((n - 1))); do
      a=$((i * chunk)); b=$((a + chunk - 1)); [ $b -ge $size ] && b=$((size - 1))
      curl -fsS --retry 5 -r $a-$b $u -o $f.p$i &
    done
    wait
    cat $(for i in $(seq 0 $((n - 1))); do echo $f.p$i; done) > $f.part 2>/dev/null; rm -f $f.p[0-9]*
    if [ "$(stat -c %s $f.part 2>/dev/null)" = "$size" ]; then mv $f.part $f; echo "[$(date +%T)] fetched $L"; return; fi
    echo "[$(date +%T)] fetch $L attempt $try: size mismatch"; rm -f $f.part
  done
  touch $f.failed
}
downloader() {
  for L in $LEVELS; do
    [ -f $O/a2c2/worlds_l_$L.pkl ] && continue
    while [ $(ls $O/expert/*.npz 2>/dev/null | wc -l) -ge 3 ]; do sleep 5; done
    fetch $L
  done
}
downloader &
for L in $LEVELS; do
  [ -f $O/a2c2/worlds_l_$L.pkl ] && continue
  f=$O/expert/worlds_l_$L.npz
  until [ -f $f ] || [ -f $f.failed ]; do sleep 5; done
  if [ ! -f $f ]; then echo "!!! A2C2 data for $L failed: the a2c2 configs will be skipped, the rest runs" | tee -a $O/a2c2_train.txt; continue; fi
  sha256sum $f | tee -a $O/expert_data.sha256
  uv run src/a2c2.py expert --data-dir $O/expert --level-paths worlds/l/$L.json 2>&1 | tee -a $O/a2c2_train.txt
  st=$?
  rm -f $f
  if [ $st = 0 ]; then JAX_PLATFORMS=cpu uv run src/b2b5.py lock --add $O/a2c2; else
    echo "!!! A2C2 for $L failed: the a2c2 configs will be skipped, the rest runs" | tee -a $O/a2c2_train.txt; fi
done
wait
echo "[$(date +%T)] A2C2 heads: $(ls $O/a2c2/*.pkl 2>/dev/null | wc -l)/12"
