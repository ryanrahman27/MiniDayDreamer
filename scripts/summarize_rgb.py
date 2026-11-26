"""Summarize matched physical scenes, actor seeds and noise draws without counting duplicates as scenes."""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.folder / "protocol.json").read_text())
    collected = collections.defaultdict(dict)
    saved = {}
    for file in args.folder.glob("*.json"):
        if not file.name[0].isdigit():
            continue
        for row in json.loads(file.read_text()):
            scene = row["scene"]
            arm = row["arm"]
            offset = row["noise_seed"] - scene
            k = (scene, offset)
            if k in collected[arm]:
                raise ValueError("Duplicate scene and stream")
            collected[arm][k] = row
            if scene in saved:
                np.testing.assert_allclose(
                    saved[scene], row["initial"], atol=0.011, rtol=0
                )
            saved[scene] = row["initial"]
    scenes = sorted(saved)
    offsets = plan["noise_offsets"]
    assert len(scenes) == plan["width"] * plan["batches"]

    def arr(name, metric, streams):
        return np.array(
            [[collected[name][(s, o)][metric] for o in streams] for s in scenes]
        )

    actors = plan["actors"][1:]
    base = arr("rgb-base", "success", offsets)
    steered = np.stack([arr(a, "success", offsets) for a in actors], axis=1)
    differences = steered - base[:, None, :]
    batches = np.array(
        [collected["rgb-base"][(s, offsets[0])]["batch_seed"] for s in scenes]
    )
    group_means = np.stack(
        [differences[batches == b].mean(0) for b in np.unique(batches)]
    )
    rng = np.random.default_rng(125)
    interval = []
    group_interval = []
    for _ in range(10000):
        si = rng.integers(len(scenes), size=len(scenes))
        ai = rng.integers(len(actors), size=len(actors))
        interval.append(differences[si][:, ai, :].mean())
        gi = rng.integers(len(group_means), size=len(group_means))
        group_interval.append(group_means[gi][:, ai, :].mean())
    summaries = {}
    for arm in plan["actors"]:
        streams = offsets
        rows = list(collected[arm].values())
        measured = {
            name: arr(arm, name, streams).mean(axis=0).tolist()
            for name in ("success", "grasped", "lifted")
        }
        measured["reset_rgb_mae_max"] = max(r["reset_rgb_mae"] for r in rows)
        summaries[arm] = measured
    out = {
        "physical_scenes": len(scenes),
        "steering_seeds": len(actors),
        "noise_streams": len(offsets),
        "rgb_base": float(base.mean()),
        "rgb_steered": float(steered.mean()),
        "paired_gain": float(differences.mean()),
        "crossed_paired_95ci": np.quantile(interval, [0.025, 0.975]).tolist(),
        "lighting_groups": len(group_means),
        "group_and_actor_seed_95ci": np.quantile(
            group_interval, [0.025, 0.975]
        ).tolist(),
        "paired_gain_by_actor": dict(
            zip(actors, differences.mean(axis=(0, 2)).tolist())
        ),
        "paired_gain_by_stream": differences.mean(axis=(0, 1)).tolist(),
        "gained_pairs": int((differences == 1).sum()),
        "lost_pairs": int((differences == -1).sum()),
        "paired_comparisons": int(differences.size),
        "per_arm": summaries,
        "source_protocol": plan,
    }
    if plan.get("repeat_base_first_stream"):
        duplicate = arr("rgb-repeat", "success", [offsets[0]])[:, 0]
        out["base_replay"] = {
            "success": float(duplicate.mean()),
            "original_success": float(base[:, 0].mean()),
            "same_outcome_fraction": float((duplicate == base[:, 0]).mean()),
        }
    print(json.dumps(out, indent=2))
    (args.folder / "summary.json").write_text(json.dumps(out, indent=2))
    with (args.folder / "paired-outcomes.csv").open("w", newline="") as output:
        columns = (
            "scene",
            "batch_seed",
            "lane",
            "noise_seed",
            "arm",
            "success",
            "grasped",
            "lifted",
            "steps",
            "reset_rgb_mae",
        )
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for arm in sorted(collected):
            for key in sorted(collected[arm]):
                writer.writerow(collected[arm][key])


if __name__ == "__main__":
    main()
