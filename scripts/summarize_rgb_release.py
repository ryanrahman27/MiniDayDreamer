"""Select a release gate on development scenes and score paired held-out outcomes."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rgb_failure_events import dump_csv, score_rows

CANDIDATES = (
    "release-careful",
    "release-balanced",
    "release-permissive",
    "release-prompt",
)
TARGET = "seated_no_open"


def read(run, split):
    folder = run / f"{split}-release"
    rows = list(score_rows(folder))
    plan = json.loads((folder / "protocol.json").read_text())
    for row in rows:
        row.pop("initial")
    arms = plan["actors"] + (["rgb-repeat"] if split == "test" else [])
    keys = {(r["scene"], r["noise_seed"] - r["scene"], r["arm"]) for r in rows}
    expected = {
        (s, o, a)
        for s in range(
            plan["first_scene"], plan["first_scene"] + plan["width"] * plan["batches"]
        )
        for a in arms
        for o in plan["noise_offsets"]
        if a != "rgb-repeat" or o == 0
    }
    assert keys == expected and len(rows) == len(keys)
    return rows, plan


def select(run):
    rows, plan = read(run, "dev")
    stats = {}
    for name in plan["actors"]:
        arm = [r for r in rows if r["arm"] == name]
        stats[name] = {
            "stacks": sum(r["success"] for r in arm),
            "trials": len(arm),
            "off_center_open": sum(r["mode"] == "opened_off_center" for r in arm),
            "held_in_position": sum(r["mode"] == TARGET for r in arm),
        }
    chosen = max(
        CANDIDATES, key=lambda n: (stats[n]["stacks"], -stats[n]["off_center_open"])
    )
    out = {
        "selected": chosen.removeprefix("release-"),
        "arms": stats,
        "rule": "Most dev stacks, then fewest off-center release failures, then candidate order.",
    }
    (run / "selection.json").write_text(json.dumps(out, indent=2))
    dump_csv(run / "dev-outcomes.csv", rows)
    print(json.dumps(out, indent=2))


def compare(run):
    rows, plan = read(run, "test")
    selection = json.loads((run / "selection.json").read_text())
    name = "release-" + selection["selected"]
    assert name in plan["actors"] and plan["selected"] == selection["selected"]
    scenes = sorted({r["scene"] for r in rows})
    noise = plan["noise_offsets"]
    data = {(r["scene"], r["noise_seed"] - r["scene"], r["arm"]): r for r in rows}

    def array(arm, field):
        return np.asarray([[data[s, o, arm][field] for o in noise] for s in scenes])

    seeds = array("rgb-base", "actor_seed")[:, 0]
    groups = array("rgb-base", "batch_seed")[:, 0]
    assert len(scenes) == 160 and all(n == 32 for n in Counter(seeds).values())

    def paired(diff):
        per_scene = diff.mean(1)
        rng = np.random.default_rng(41201)
        idx = rng.integers(len(scenes), size=(10000, len(scenes)))
        ci = np.quantile(per_scene[idx].mean(1), [0.025, 0.975])
        _, gi = np.unique(groups, return_inverse=True)
        _, si = np.unique(seeds, return_inverse=True)

        def weights(indices):
            n = len(np.unique(indices))
            return rng.multinomial(n, np.full(n, 1 / n), 10000)[:, indices]

        w = weights(gi) * weights(si)
        group_ci = np.quantile(w @ per_scene / w.sum(1), [0.025, 0.975])
        return {
            "delta": float(diff.mean()),
            "scene_ci": ci.tolist(),
            "group_seed_ci": group_ci.tolist(),
            "gained": int((diff == 1).sum()),
            "lost": int((diff == -1).sum()),
            "per_seed": {
                int(seed): float(diff[seeds == seed].mean())
                for seed in np.unique(seeds)
            },
        }

    arms = {}
    pairs = {}
    for arm in plan["actors"]:
        score = array(arm, "success")
        modes = array(arm, "mode")
        arms[arm] = {
            "stacks": int(score.sum()),
            "per_stream_stacks": score.sum(0).tolist(),
            "trials": int(score.size),
            "modes": dict(Counter(modes.flat)),
            "per_seed_stacks": {
                int(s): int(score[seeds == s].sum()) for s in np.unique(seeds)
            },
        }
        if arm != name:
            target = array(name, "mode")
            mask = modes == TARGET
            pairs[arm] = {
                "stacks": paired(array(name, "success") - score),
                "held_in_position": paired(
                    (target == TARGET).astype(int) - mask.astype(int)
                ),
                "on_comparator_held": dict(Counter(target[mask])),
            }
    # Audit physical geometry at the first visual trigger. Inputs to the controller remain RGB/commands.
    triggers = Counter()
    offsets = []
    for path in (run / "test-release").glob(f"*-{name}.json"):
        with np.load(path.with_suffix(".npz")) as f:
            decisions = f["release_decisions"]
            state = f["obs"]
            acts = f["actions"]
            for row in json.loads(path.read_text()):
                i = row["lane"]
                n = row["steps"]
                hits = np.flatnonzero(decisions[:, i, 1] > 0)
                hits = hits[hits * 3 < n]
                if not len(hits):
                    continue
                t = int(hits[0]) * 3
                xy = 100 * np.linalg.norm(state[t, i, 3:5] - state[t, i, 6:8])
                height = 100 * (state[t, i, 5] - state[t, i, 8]) - 46.8
                offsets.append((xy, height))
                triggers["triggered"] += 1
                triggers["held"] += int(state[t, i, 9] > 0.5)
                triggers["in_both_bands"] += int(xy < 12 and abs(height) < 8)
                triggers["opened_within_6_ticks"] += int(
                    (acts[t : min(t + 6, n), i, 3] <= 0).any()
                )
    base = array("rgb-base", "success")[:, 0]
    rep = np.array([data[s, 0, "rgb-repeat"]["success"] for s in scenes])
    out = {
        "selected": name,
        "fresh_scenes": len(scenes),
        "rollouts": len(rows),
        "arms": arms,
        "target_minus": pairs,
        "fixed_minus_base": paired(
            array("fixed-" + plan["selected"], "success") - array("rgb-base", "success")
        ),
        "repeat": {
            "base": int(base.sum()),
            "repeat": int(rep.sum()),
            "agree": int((base == rep).sum()),
        },
        "trigger": dict(triggers),
        "median_xy_and_signed_height_mm_at_trigger": np.median(offsets, axis=0).tolist()
        if offsets
        else None,
        "max_reset_rgb_mae": max(r["reset_rgb_mae"] for r in rows),
    }
    dump_csv(run / "fresh-outcomes.csv", rows)
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("select", "compare"))
    p.add_argument("run", type=Path)
    a = p.parse_args()
    (select if a.mode == "select" else compare)(a.run)
