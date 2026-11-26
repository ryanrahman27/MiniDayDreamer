"""Measure release opportunities in the 23 diagnostic replays and the frozen flow's noise response."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from rgb_failure_events import dump_csv

from minidaydreamer.vision import VisualFlowHead


def read_csv(path):
    with path.open() as source:
        return list(csv.DictReader(source))


@torch.inference_mode()
def audit(prior, flow_path, output):
    cases = []
    for row in read_csv(prior / "original-127.csv"):
        if row["mode"] != "seated_no_open":
            continue
        scene = int(row["scene"])
        offset = int(row["noise_seed"]) - scene
        lane = int(row["lane"])
        with np.load(
            prior / "test-diagnosis" / f"{row['batch_seed']}-{offset}-rgb-base.npz"
        ) as trace:
            s = trace["obs"][: int(row["steps"]) + 1, lane]
        xy = 100 * np.linalg.norm(s[:, 3:5] - s[:, 6:8], axis=1)
        height = 100 * (s[:, 5] - s[:, 8]) - 46.8
        seat = (xy < 12) & (abs(height) < 8) & (s[:, 9] > 0.5)
        seat[: int(row["lift_tick"])] = False
        edges = np.diff(np.r_[False, seat, False].astype(int))
        spans = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        assert len(spans) > 0 and row["opening_tick"] == ""
        cases.append(
            {
                "scene": scene,
                "noise_seed": int(row["noise_seed"]),
                "first_seated_tick": int(np.flatnonzero(seat)[0]),
                "longest_consecutive_ticks": int(spans.max()),
                "total_seated_ticks": int(seat.sum()),
            }
        )
    assert len(cases) == 23
    dump_csv(output / "diagnostic-23.csv", cases)
    features = []
    for row in read_csv(prior / "fresh-outcomes.csv"):
        if row["arm"] == "rgb-base" and row["mode"] == "seated_no_open":
            offset = int(row["noise_seed"]) - int(row["scene"])
            with np.load(
                prior / "test-fresh" / f"{row['batch_seed']}-{offset}-rgb-base.npz"
            ) as trace:
                features.append(
                    trace["visual_features"][int(row["seated_tick"]), int(row["lane"])]
                )
    torch.set_num_threads(1)
    pi = VisualFlowHead().eval()
    pi.load_state_dict(
        torch.load(flow_path, weights_only=True, map_location="cpu")["flow"]
    )
    z = torch.from_numpy(np.stack(features)).float().repeat(15, 1)
    noise = torch.randn(len(z), 3, 4, generator=torch.Generator().manual_seed(491))
    base = pi.sample(z, noise)
    effects = {}
    for scale in (0, 0.5, 2, 6, 8, 10, 14, 18):
        shifted = noise.clone()
        shifted[:, :, 3] -= scale
        a = pi.sample(z, shifted)
        effects[str(scale)] = {
            "open_command_fraction": float((a[:, :, 3] <= 0).float().mean()),
            "mean_absolute_xyz_command_change": float(
                (a[:, :, :3] - base[:, :, :3]).abs().mean()
            ),
        }
    longest = [r["longest_consecutive_ticks"] for r in cases]
    stats = {
        "diagnostic_cases": len(cases),
        "scenes": len({r["scene"] for r in cases}),
        "held_eight_ticks": sum(n >= 8 for n in longest),
        "median_longest_dwell": float(np.median(longest)),
        "historical_fresh_features": len(features),
        "draws_per_feature": 15,
        "noise_response": effects,
    }
    (output / "diagnostic.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "flow", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    audit(a.prior, a.flow, a.output)
