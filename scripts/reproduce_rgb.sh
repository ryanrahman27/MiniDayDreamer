#!/usr/bin/env bash
# Run the RGB experiment from demonstrations through held-out evaluation.
set -euo pipefail
if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/reproduce_rgb.sh [run-directory]" >&2
    exit 2
fi
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "${1:-runs/rgb-reproduction}"
out=$(realpath "${1:-runs/rgb-reproduction}")
base="$out/isaac-rgb-actions-v12"
target="$out/isaac-rgb-target-v13"
release="$out/isaac-rgb-release-v14"
wait="$out/isaac-rgb-wait-v15"
bash "$root/scripts/run_rgb.sh" "$base"
bash "$root/scripts/run_rgb_target.sh" "$target" "$base"
bash "$root/scripts/run_rgb_release.sh" "$release" "$target" "$base"
bash "$root/scripts/run_rgb_wait.sh" "$wait" "$release" "$base"
bash "$root/scripts/run_rgb_wait_calibrated.sh" "$out/isaac-rgb-wait-v15-calibrated" "$wait" "$base"
