#!/usr/bin/env bash
# Run inside the MiniDayDreamer Isaac environment with the V12 run available.
set -euo pipefail
mkdir -p "$1"
run=$(realpath "$1")
prior=$(realpath "$2")
src=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$src:$src/scripts${PYTHONPATH:+:$PYTHONPATH}"
for name in flow.pt world.pt visual-starts.pt actor-{101,211,307,401,503}.pt; do
    if [[ ! -f "$run/$name" ]]; then cp "$prior/$name" "$run/$name"; fi
    cmp -s "$prior/$name" "$run/$name"
done
if [[ ! -d "$run/inputs" ]]; then ln -s "$prior/inputs" "$run/inputs"; fi
if [[ ! -f "$run/diagnosis.json" ]]; then
    python "$src/scripts/rgb_isaac.py" score "$run" --split test --label diagnosis --failure-study replay --recorded-resets "$prior/test-paired" --batches 8 --width 16 > "$run/replay.log" 2>&1
    python "$src/scripts/rgb_failure_events.py" "$run" "$prior/test-paired" > "$run/diagnosis.log"
fi
if [[ ! -f "$run/approach.pt" ]]; then python "$src/scripts/rgb_approach_probe.py" "$run" > "$run/approach.log"; fi
pids=()
for seed in 101 211 307 401 503; do
    if [[ ! -f "$run/target-$seed.pt" ]]; then
        python "$src/scripts/rgb_failure_actor.py" "$run" --seed "$seed" > "$run/train-$seed.log" 2>&1 &
        pids+=($!)
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
# Each physical scene is paired across all arms and both noise streams.
# Thirty-two scenes are assigned to each of the five matched pairs of training seeds.
pids=()
for first in 0 5; do
    python "$src/scripts/rgb_isaac.py" score "$run" --split test --label fresh --failure-study fresh --width 16 --batches 10 --first-batch "$first" --stop-batch "$((first+5))" > "$run/fresh-$first.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python "$src/scripts/summarize_rgb_target.py" "$run" > "$run/fresh-summary.log"
python "$src/scripts/audit_rgb_carry.py" "$run" > "$run/carry-audit.log"
