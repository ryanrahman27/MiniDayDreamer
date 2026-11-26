"""Measure where the visual trigger intervenes and what happens to each frozen-base failure mode."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def audit(run):
    plan = json.loads((run / "test-release" / "protocol.json").read_text())
    name = "release-" + plan["selected"]
    threshold, reps, _ = plan["configs"][plan["selected"]]
    with (run / "fresh-outcomes.csv").open() as f:
        rows = list(csv.DictReader(f))
    data = {(r["arm"], int(r["scene"]), int(r["noise_seed"])): r for r in rows}
    changed = defaultdict(Counter)
    reach = Counter()
    for key, b in data.items():
        arm, s, n = key
        if arm == "rgb-base":
            changed[b["mode"]][data[name, s, n]["mode"]] += 1
    actual = Counter()
    geometry = []
    opening_geometry = []
    magnitudes = []
    for path in (run / "test-release").glob("[0-9]*.json"):
        metadata = json.loads(path.read_text())
        arm = metadata[0]["arm"]
        if arm not in (name, "rgb-base"):
            continue
        with np.load(path.with_suffix(".npz")) as trace:
            decisions = trace["release_decisions"]
            state = trace["obs"]
            commands = trace["actions"]
        for row in metadata:
            i = row["lane"]
            n = row["steps"]
            x = state[: n + 1, i]
            act = commands[:n, i]
            d = decisions[: (n + 2) // 3, i]
            xy = 100 * np.linalg.norm(x[:, 3:5] - x[:, 6:8], axis=1)
            h = 100 * (x[:, 5] - x[:, 8]) - 46.8
            seated = (xy < 12) & (abs(h) < 8) & (x[:, 9] > 0.5)
            if arm == "rgb-base":
                item = data[arm, row["scene"], row["noise_seed"]]
                if item["mode"] != "seated_no_open":
                    continue
                reach["cases"] += 1
                eligible = np.array(
                    [
                        t >= 12
                        and (act[t - 12 : t, 3] > 0).all()
                        and d[t // 3, 0] >= threshold
                        for t in range(0, n, 3)
                    ]
                )
                streak = 0
                starts = []
                for ix, good in enumerate(eligible):
                    streak = streak + 1 if good else 0
                    if streak == reps:
                        starts.append(3 * ix)
                reach["would_trigger_at_least_once"] += int(bool(starts))
                reach["would_first_trigger_in_both_bands_while_held"] += int(
                    bool(starts) and seated[starts[0]]
                )
                continue
            nz = np.flatnonzero(d[:, 2] != 0)
            if not len(nz):
                continue
            t = int(nz[0]) * 3
            geometry.append([xy[t], h[t]])
            magnitudes.extend(abs(d[nz, 2]).tolist())
            actual["episodes_with_shift"] += 1
            actual["held_at_first_shift"] += int(x[t, 9] > 0.5)
            actual["horizontally_aligned_at_first_shift"] += int(xy[t] < 12)
            actual["both_bands_at_first_shift"] += int(xy[t] < 12 and abs(h[t]) < 8)
            actual["opened_within_three_ticks"] += int((act[t : t + 3, 3] <= 0).any())
            opened = np.flatnonzero(act[t : t + 6, 3] <= 0)
            if len(opened):
                tick = t + int(opened[0])
                actual["opened_within_six_ticks"] += 1
                opening_geometry.append((xy[tick], h[tick]))
                actual["both_bands_at_open_within_six_ticks"] += int(
                    xy[tick] < 12 and abs(h[tick]) < 8
                )
            actual["stable_stacks"] += row["success"]
    out = {
        "selected": name,
        "base_to_steered": dict(changed),
        "would_trigger_on_unsteered_seated_failures": dict(reach),
        "first_applied_shift": dict(actual),
        "median_xy_and_signed_height_mm_at_open_within_six_ticks": np.median(
            opening_geometry, axis=0
        ).tolist()
        if opening_geometry
        else None,
        "median_xy_and_signed_height_mm_at_first_shift": np.median(
            geometry, axis=0
        ).tolist()
        if geometry
        else None,
        "median_absolute_applied_shift": float(np.median(magnitudes))
        if magnitudes
        else None,
    }
    (run / "mechanism.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    audit(p.parse_args().run)
