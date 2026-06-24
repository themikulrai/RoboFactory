#!/bin/bash
# Launch S parallel LB shards, each running the full recovery/merged/clean mix
# over a DISJOINT seed block. Each shard is a self-contained mini-dataset;
# merge_shards.py later concatenates + interleaves them into one 200-ep dataset.
set -u
PY=/iris/u/mikulrai/data/miniforge3/envs/RoboFactory/bin/python
WT=/iris/u/mikulrai/RoboFactory-subtask-wt
ROOT=/iris/u/mikulrai/data/RoboFactory/subtask_combined_lb_v1/full200
TMP=/iris/u/mikulrai/.claude/jobs/a7194c42/tmp
S=5                 # shards
NUM=40              # per-shard budget (mix 0.3/0.3/0.4 -> 12/12/16) ; 5*40=200
SEEDS_PER=150       # disjoint candidate seeds per shard (>> needed at these yields)

rm -rf "$ROOT"; mkdir -p "$ROOT"
cd "$WT" || exit 1
for s in $(seq 0 $((S-1))); do
  # disjoint seed block: shard s -> [s*1000, s*1000+SEEDS_PER)
  lo=$((s*1000)); hi=$((lo+SEEDS_PER))
  seeds=$($PY -c "print(','.join(str(x) for x in range($lo,$hi)))")
  OUT="$ROOT/shard$s"
  LOG="$TMP/lb_shard$s.log"
  PYTHONPATH="$WT" setsid $PY scripts/dart/run_subtask_rollouts.py \
    --task LiftBarrier \
    --num $NUM --mix recovery=0.3,merged=0.3,clean=0.4 \
    --config-suffix _aug \
    --dart-sigma 0.05 --inject-floor 0.05 --grasp-settle 10 --shove-joints 0,1,2,3 \
    --jitter-frac 0.40 --jitter-sigma 0.05 \
    --per-attempt-timeout 300 --max-steps 600 \
    --seeds "$seeds" \
    --record-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
  echo "shard $s -> pid $! seeds[$lo,$hi) log=$LOG"
done
echo "launched $S shards"
