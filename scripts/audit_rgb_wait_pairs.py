"""Audit release strategies on the same scene and Gaussian stream as each base failure."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_scene_rows(result):
    with (result / "test-outcomes.csv").open() as f:
        rows = defaultdict(dict)
        for row in csv.DictReader(f):
            key = (int(row["scene"]), int(row["noise_seed"]))
            if key in rows[row["arm"]]:
                raise ValueError(f"Duplicate paired trial {key}")
            rows[row["arm"]][key] = row
    base = rows["rgb-base"]
    keys = sorted(base)
    scenes = sorted({k[0] for k in keys})
    assert len(keys) == 2 * len(scenes) and len(scenes) == 160
    groups = [[k for k in keys if k[0] == scene] for scene in scenes]
    assert all(len(g) == 2 for g in groups)
    seeds = np.array([int(base[g[0]]["actor_seed"]) for g in groups])
    return rows, groups, seeds


def differences(actor, comparator, groups):
    return np.array(
        [
            [int(actor[k]["success"]) - int(comparator[k]["success"]) for k in group]
            for group in groups
        ]
    )


def estimate(diff, seeds, boot):
    scene_means = diff.mean(axis=1)
    return {
        "difference": float(diff.mean()),
        "scene_ci": np.quantile(scene_means[boot].mean(axis=1), [0.025, 0.975]).tolist(),
        "per_seed": {int(s): float(diff[seeds == s].mean()) for s in np.unique(seeds)},
        "rescued": int((diff == 1).sum()),
        "lost": int((diff == -1).sum()),
    }


def audit(result):
    summary = json.loads((result / "summary.json").read_text())
    rows, groups, seeds = load_scene_rows(result)
    base = rows["rgb-base"]
    keys = sorted(base)
    boot = np.random.default_rng(62153).integers(len(groups), size=(12000, len(groups)))
    wait_arm = summary["selected"]
    actors = (
        "fixed-permissive",
        "release-permissive",
        wait_arm.replace("wait-", "lower-", 1),
        wait_arm,
    )
    per_mode = defaultdict(Counter)
    comparisons = {}
    for actor in actors:
        assert rows[actor].keys() == base.keys()
        for k in keys:
            mode = base[k]["mode"]
            per_mode[mode][actor] += int(rows[actor][k]["success"])
        comparisons[actor] = estimate(differences(rows[actor], base, groups), seeds, boot)
    for row in base.values():
        per_mode[row["mode"]]["trials"] += 1
    for name, check in (
        ("fixed-permissive", summary["fixed_minus_base"]),
        (summary["selected"], summary["selected_minus"]["rgb-base"]),
    ):
        assert np.allclose(comparisons[name]["scene_ci"], check["scene_ci"])
        assert comparisons[name]["rescued"] == check["rescued"]
    discordant = defaultdict(Counter)
    actor = rows[summary["selected"]]
    fixed = rows["fixed-permissive"]
    for k in keys:
        a, b = actor[k], fixed[k]
        if a["success"] == b["success"]:
            continue
        winner = "wait_wins" if int(a["success"]) else "fixed_wins"
        changed = (bool(a["first_shift_tick"]), bool(b["first_shift_tick"]))
        intervention = {
            (True, True): "both_shift",
            (True, False): "only_wait_shifts",
            (False, True): "only_fixed_shifts",
            (False, False): "neither_shifts",
        }[changed]
        discordant[winner][intervention] += 1
    out = {
        "stacks_by_unsteered_mode": dict(per_mode),
        "controller_minus_base": comparisons,
        "interventions_in_discordant_pairs": dict(discordant),
    }
    (result / "matched-diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")


def pool_controls(first, second):
    banks = [load_scene_rows(first), load_scene_rows(second)]
    scene_ids = [{group[0][0] for group in groups} for _, groups, _ in banks]
    assert scene_ids[0].isdisjoint(scene_ids[1])
    rng = np.random.default_rng(62153)
    # Hold the bank weights fixed and keep both noise streams for each sampled scene.
    draws = []
    offset = 0
    for _, groups, _ in banks:
        draws.append(offset + rng.integers(len(groups), size=(12000, len(groups))))
        offset += len(groups)
    bootstrap = np.concatenate(draws, axis=1)
    seeds = np.concatenate([s for _, _, s in banks])
    out = {}
    for actor, reference in (
        ("fixed-permissive", "rgb-base"),
        ("release-permissive", "rgb-base"),
        ("release-permissive", "fixed-permissive"),
        ("lower-middle", "fixed-permissive"),
    ):
        d = np.concatenate([
            differences(rows[actor], rows[reference], groups)
            for rows, groups, _ in banks
        ])
        out[f"{actor}_minus_{reference}"] = estimate(d, seeds, bootstrap)
    (first / "pooled-controls.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    p.add_argument("--pool-with", type=Path, help="Second independent fresh bank")
    args = p.parse_args()
    audit(args.result)
    if args.pool_with:
        audit(args.pool_with)
        pool_controls(args.result, args.pool_with)
