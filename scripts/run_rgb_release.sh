#!/usr/bin/env bash
# Requires the V13 actors and the earlier RGB/actions training corpus.
set -euo pipefail
mkdir -p "$1"
run=$(realpath "$1")
prior=$(realpath "$2")
base=${3:-"$(dirname "$prior")/isaac-rgb-actions-v12"}
src=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$src:$src/scripts${PYTHONPATH:+:$PYTHONPATH}"
for name in inputs flow.pt world.pt visual-starts.pt actor-{101,211,307,401,503}.pt target-{101,211,307,401,503}.pt; do
    artifact="$prior/$name"
    if [[ ! -e "$artifact" ]]; then artifact="$base/$name"; fi
    if [[ ! -e "$artifact" ]]; then echo "Missing training artifact: $name" >&2; exit 1; fi
    if [[ ! -e "$run/$name" ]]; then ln -s "$(realpath "$artifact")" "$run/$name"; fi
    if [[ -f "$artifact" ]]; then cmp -s "$artifact" "$run/$name"; fi
done
if [[ ! -f "$run/release-geometry.pt" ]]; then python "$src/scripts/rgb_release_geometry.py" "$run" > "$run/geometry.log"; fi
pids=()
for s in 101 211 307 401 503; do
    if [[ ! -f "$run/release-$s.pt" ]]; then
        python "$src/scripts/rgb_release_actor.py" "$run" --seed "$s" > "$run/train-$s.log" 2>&1 &
        pids+=($!)
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
score() {
    local split=$1 total=$2 half=$(( $2 / 2 ))
    local jobs=()
    for first in 0 "$half"; do
        python "$src/scripts/rgb_isaac.py" score "$run" --failure-study release --label release --split "$split" --width 16 --batches "$total" --first-batch "$first" --stop-batch "$(( first + half ))" > "$run/$split-$first.log" 2>&1 &
        jobs+=($!)
    done
    for pid in "${jobs[@]}"; do wait "$pid"; done
}
if [[ ! -f "$run/selection.json" ]]; then
    score dev 4
    python "$src/scripts/summarize_rgb_release.py" select "$run" > "$run/selection.log"
fi
score test 10
python "$src/scripts/summarize_rgb_release.py" compare "$run" > "$run/summary.log"
python "$src/scripts/audit_rgb_release_effect.py" "$run" > "$run/mechanism.log"
