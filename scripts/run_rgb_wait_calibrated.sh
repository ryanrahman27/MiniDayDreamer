#!/usr/bin/env bash
# Fit a development RGB height correction and independently qualify the retrained actor.
set -euo pipefail
mkdir -p "$1"
run=$(realpath "$1")
prior=$(realpath "$2")
src=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$src:$src/scripts${PYTHONPATH:+:$PYTHONPATH}"
base=${3:-"$(dirname "$prior")/isaac-rgb-actions-v12"}
old="$(dirname "$prior")/isaac-rgb-release-v14"
for f in flow.pt world.pt release-geometry.pt placement-view.pt placement-starts.pt release-{101,211,307,401,503}.pt; do
    path="$prior/$f"; if [[ ! -e "$path" ]]; then path="$old/$f"; fi
    if [[ ! -e "$path" ]]; then path="$base/$f"; fi
    [[ -f "$path" ]] || { echo "Missing $f" >&2; exit 1; }
    if [[ ! -e "$run/$f" ]]; then ln -s "$(realpath "$path")" "$run/$f"; fi
    cmp -s "$path" "$run/$f"
done
if [[ ! -f "$run/dev-height-calibration.json" ]]; then
    python "$src/scripts/audit_rgb_wait_dynamics.py" "$prior" "$run" --split dev > "$run/calibration.log"
    cp "$prior/dev-height-calibration.json" "$run/"
fi
pids=()
for seed in 101 211 307 401 503; do
    if [[ ! -f "$run/wait-$seed.pt" ]]; then
        python "$src/scripts/rgb_wait_actor.py" "$run" --seed "$seed" --calibration "$run/dev-height-calibration.json" > "$run/learn-$seed.log" 2>&1 &
        pids+=($!)
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
do_score() {
    python "$src/scripts/rgb_isaac.py" score "$run" --failure-study calibrated --label wait --width 16 --split "$1" --batches "$2" --first-batch "$3" --stop-batch "$4"
}
if [[ ! -f "$run/selection.json" ]]; then
    do_score dev 4 0 4 > "$run/dev.log" 2>&1
    python "$src/scripts/summarize_rgb_wait.py" dev "$run" > "$run/dev-summary.log"
fi
promote=$(python -c 'import sys,json; print(int(json.load(open(sys.argv[1]))["beat_original_fixed_while_lower"]))' "$run/selection.json")
if [[ "$promote" == "1" ]]; then
    pids=()
    for start in 0 5; do do_score test 10 "$start" "$((start+5))" > "$run/test-$start.log" 2>&1 & pids+=($!); done
    for pid in "${pids[@]}"; do wait "$pid"; done
    python "$src/scripts/summarize_rgb_wait.py" test "$run" > "$run/test-summary.log"
else
    echo "Development criterion missed. See $run/selection.json for the scores." >&2
fi
