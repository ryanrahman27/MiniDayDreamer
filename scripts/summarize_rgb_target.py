"""Paired fresh-scene comparison of frozen, incumbent and failure-targeted RGB policies."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rgb_failure_events import dump_csv, score_rows


def compare(run):
    diagnosis = json.loads((run / "diagnosis.json").read_text())
    mode = diagnosis["modal_failure"]
    root = run / "test-fresh"
    plan = json.loads((root / "protocol.json").read_text())
    rows = list(score_rows(root))
    lookup = {}
    for r in rows:
        key = (r["arm"], r["scene"], r["noise_seed"] - r["scene"])
        assert key not in lookup
        lookup[key] = r
        r.pop("initial")
    arms = ["rgb-base", "rgb-incumbent", "rgb-target"]
    offsets = plan["noise_offsets"]
    scenes = sorted({r["scene"] for r in rows})
    assert (
        len(scenes) == plan["width"] * plan["batches"] == 160 and len(rows) == 160 * 7
    )
    seeds = np.array([lookup[arms[0], s, 0]["actor_seed"] for s in scenes])
    group = np.array([lookup[arms[0], s, 0]["batch_seed"] for s in scenes])
    assert all(n == 32 for n in Counter(seeds).values())
    for s in scenes:
        for off in offsets:
            tr = [lookup[a, s, off] for a in arms]
            assert len({r["actor_seed"] for r in tr}) == 1

    def matrix(arm, field):
        values = [[lookup[arm, s, o][field] for o in offsets] for s in scenes]
        return np.asarray(values)

    result = {}
    for a in arms:
        score = matrix(a, "success")
        modes = matrix(a, "mode")
        result[a] = {
            "stacks": int(score.sum()),
            "trials": score.size,
            "success": float(score.mean()),
            "per_stream": score.mean(axis=0).tolist(),
            "modes": dict(Counter(modes.flatten())),
            "per_seed_stacks": {
                int(seed): int(score[seeds == seed].sum()) for seed in np.unique(seeds)
            },
        }

    def paired(delta):
        average = delta.mean(axis=1)
        rng = np.random.default_rng(19542)
        simple = average[rng.integers(len(scenes), size=(10000, len(scenes)))].mean(1)
        _, gi = np.unique(group, return_inverse=True)
        _, si = np.unique(seeds, return_inverse=True)
        gs = rng.multinomial(
            len(set(group)), np.full(len(set(group)), 1 / len(set(group))), 10000
        )
        ss = rng.multinomial(
            len(set(seeds)), np.full(len(set(seeds)), 1 / len(set(seeds))), 10000
        )
        weights = gs[:, gi] * ss[:, si]
        clustered = (weights @ average) / weights.sum(1)
        return {
            "delta": float(delta.mean()),
            "scene_ci": np.quantile(simple, [0.025, 0.975]).tolist(),
            "group_seed_ci": np.quantile(clustered, [0.025, 0.975]).tolist(),
            "per_seed": {
                int(s): float(delta[seeds == s].mean()) for s in sorted(set(seeds))
            },
            "plus": int((delta == 1).sum()),
            "minus": int((delta == -1).sum()),
        }

    pairs = {}
    for a in arms[:2]:
        pairs[a] = {
            "stacks": paired(matrix("rgb-target", "success") - matrix(a, "success")),
            "modal_failure_incidence": paired(
                (matrix("rgb-target", "mode") == mode).astype(int)
                - (matrix(a, "mode") == mode).astype(int)
            ),
        }
    repeat = np.array([lookup["rgb-repeat", s, 0]["success"] for s in scenes])
    base = matrix("rgb-base", "success")[:, 0]
    out = {
        "target": mode,
        "fresh_scenes": len(scenes),
        "steering_seeds": sorted(map(int, set(seeds))),
        "streams": offsets,
        "total_rollouts": len(rows),
        "arms": result,
        "target_minus": pairs,
        "replay": {
            "base": int(base.sum()),
            "repeat": int(repeat.sum()),
            "agree": int((base == repeat).sum()),
        },
        "max_native_initial_rgb_mae": max(r["reset_rgb_mae"] for r in rows),
    }
    dump_csv(run / "fresh-outcomes.csv", rows)
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    compare(p.parse_args().run)
