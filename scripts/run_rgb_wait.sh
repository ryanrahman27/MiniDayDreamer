#!/usr/bin/env bash
set -euo pipefail
mkdir -p "$1"
run=$(realpath "$1")
prior=$(realpath "$2")
base=${3:-"$(dirname "$prior")/isaac-rgb-actions-v12"}
src=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$src:$src/scripts${PYTHONPATH:+:$PYTHONPATH}"
for f in inputs flow.pt world.pt visual-starts.pt release-geometry.pt release-{101,211,307,401,503}.pt; do
    path="$prior/$f"; if [[ ! -e "$path" ]]; then path="$base/$f"; fi
    [[ -e "$path" ]] || { echo "Missing $f" >&2; exit 1; }
    if [[ ! -e "$run/$f" ]]; then ln -s "$(realpath "$path")" "$run/$f"; fi
    if [[ -f "$path" ]]; then cmp -s "$path" "$run/$f"; fi
done
if [[ ! -f "$run/placement-view.pt" || ! -f "$run/placement-starts.pt" ]]; then python "$src/scripts/rgb_wait_geometry.py" "$run" > "$run/view.log"; fi
jobs=()
for seed in 101 211 307 401 503; do
    if [[ ! -f "$run/wait-$seed.pt" ]]; then
        python "$src/scripts/rgb_wait_actor.py" "$run" --seed "$seed" > "$run/learn-$seed.log" 2>&1 &
        jobs+=($!)
    fi
done
for pid in "${jobs[@]}"; do wait "$pid"; done
score() {
    local split=$1 total=$2 half=$(( $2 / 2 ))
    local pids=()
    for first in 0 "$half"; do
        python "$src/scripts/rgb_isaac.py" score "$run" --failure-study wait --label wait --split "$split" --width 16 --batches "$total" --first-batch "$first" --stop-batch "$(( first + half ))" > "$run/$split-$first.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    python "$src/scripts/summarize_rgb_wait.py" "$split" "$run" > "$run/$split-summary.log"
    python "$src/scripts/audit_rgb_wait_coverage.py" "$run" --split "$split" > "$run/$split-coverage.log"
}
if [[ ! -f "$run/selection.json" ]]; then score dev 4; fi
score test 10
