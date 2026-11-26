"""Count failures reachable by a release-only intervention along archived unsteered paths."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rgb_wait_actor import WaitActor


@torch.inference_mode()
def audit(run, split):
    torch.set_num_threads(1)
    chosen = json.loads((run / "selection.json").read_text())["selected"]
    folder = run / f"{split}-wait"
    plan = json.loads((folder / "protocol.json").read_text())
    high, radial, minimum = plan["limits"][chosen]
    with (run / f"{split}-outcomes.csv").open() as f:
        modes = {
            (int(r["scene"]), int(r["noise_seed"])): r["mode"]
            for r in csv.DictReader(f)
            if r["arm"] == "rgb-base"
        }
    actors = {}
    for seed in plan["actor_seeds"]:
        actor = WaitActor().eval()
        actor.load_state_dict(
            torch.load(run / f"wait-{seed}.pt", weights_only=True)["actor"]
        )
        actors[seed] = actor
    totals = Counter()
    hits = {k: Counter() for k in ("original_fixed", "lower_gate_only", "learned_wait")}
    for p in sorted(folder.glob("*-rgb-base.json")):
        with np.load(p.with_suffix(".npz")) as tr:
            act = tr["actions"]
            features = tr["visual_features"]
            dec = tr["release_decisions"]
            view = tr["placement_decisions"]
        for r in json.loads(p.read_text()):
            i = r["lane"]
            n = r["steps"]
            mode = modes[r["scene"], r["noise_seed"]]
            totals[mode] += 1
            # An intervention needs at least eight ticks remaining to achieve the final stack score.
            ticks = np.arange(0, n - 7, 3)
            if not len(ticks):
                continue
            xyz = view[ticks // 3, i, :2]
            chunks = np.stack([act[t : t + 3, i] for t in ticks])
            advantage = actors[r["actor_seed"]](
                torch.from_numpy(features[ticks, i]).float(),
                torch.from_numpy(xyz),
                torch.from_numpy(chunks),
            ).numpy()
            for name, records in hits.items():
                streak = pulse = 0
                for j, t in enumerate(ticks):
                    h, l = xyz[j]
                    closed = t >= 12 and (act[t - 12 : t, i, 3] > 0).all()
                    allow = closed and dec[t // 3, i, 0] >= 0.28
                    if name != "original_fixed":
                        allow = (
                            allow
                            and view[t // 3, i, 2] > 0.5
                            and -2.15 <= h <= high
                            and l <= radial
                        )
                    streak = streak + 1 if allow else 0
                    trigger = streak == 2 if name == "original_fixed" else streak >= 2
                    if (
                        trigger
                        and pulse == 0
                        and (name != "learned_wait" or advantage[j] >= minimum)
                    ):
                        pulse = 16
                    if pulse and (chunks[j, :, 3] > 0).any():
                        records[mode] += 1
                        break
                    pulse = max(0, pulse - 1)
    result = {
        "base_modes": dict(totals),
        "eligibility_by_mode": {k: dict(v) for k, v in hits.items()},
        "base_failures": sum(totals.values()) - totals["stacked"],
        "eligible_base_failures": {
            k: sum(v.values()) - v["stacked"] for k, v in hits.items()
        },
    }
    (run / f"{split}-coverage.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=("dev", "test"), default="test")
    a = p.parse_args()
    audit(a.run, a.split)
