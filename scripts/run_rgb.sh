#!/usr/bin/env bash
# Execute inside the project's Isaac Lab environment.
set -euo pipefail
mkdir -p "$1"
run=$(realpath "$1")
src=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$src:$src/scripts${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -s "$run/inputs/valid-410015.npz" ]]; then
    python "$src/scripts/rgb_isaac.py" collect "$run" --width 8 >"$run/collect.log" 2>&1
    python "$src/scripts/fit_rgb.py" export "$run" --source "$run/episodes"
fi
if [[ ! -s "$run/bc.json" || ! -s "$run/flow.pt" ]]; then python "$src/scripts/fit_rgb.py" bc "$run" >"$run/bc.log" 2>&1; fi
python "$src/scripts/check_rgb_inputs.py" "$run" >"$run/image-check.log"
if [[ ! -s "$run/world.json" || ! -s "$run/world.pt" || ! -s "$run/visual-starts.pt" ]]; then
    python "$src/scripts/train_visual_dreams.py" world "$run" >"$run/world.log" 2>&1
fi
pids=()
for seed in 101 211 307 401 503; do
    if [[ ! -s "$run/actor-$seed.pt" ]]; then
        python "$src/scripts/train_visual_dreams.py" actor "$run" --seed "$seed" >"$run/actor-$seed.log" 2>&1 &
        pids+=($!)
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
python "$src/scripts/rgb_isaac.py" score "$run" --split dev --label paired --width 8 --batches 1 >"$run/dev.log" 2>&1
python "$src/scripts/summarize_rgb.py" "$run/dev-paired" >"$run/dev-summary.log"
pids=()
for start in 0 4; do
    python "$src/scripts/rgb_isaac.py" score "$run" --split test --label paired --width 16 --batches 8 --first-batch "$start" --stop-batch "$((start+4))" >"$run/test-$start.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python "$src/scripts/summarize_rgb.py" "$run/test-paired" >"$run/test-summary.log"
