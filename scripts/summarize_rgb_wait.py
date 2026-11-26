"""Select waiting on independent development scenes and estimate paired physical effects."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rgb_failure_events import dump_csv, score_rows
from rgb_wait_control import LIMITS
from score_multiseed import summarize_trace


def geometry(o):
    return float(100 * np.linalg.norm(o[3:5] - o[6:8])), float(
        100 * (o[5] - o[8]) - 46.8
    )


def measure(run, split):
    folder = run / f"{split}-wait"
    plan = json.loads((folder / "protocol.json").read_text())
    rows = list(score_rows(folder))
    index = {}
    for r in rows:
        r.pop("initial")
        key = r["arm"], r["scene"], r["noise_seed"] - r["scene"]
        if key in index:
            raise ValueError("duplicate trial")
        index[key] = r
    groups = plan["batches"]
    w = plan["width"]
    start = plan["first_scene"]
    names = plan["actors"] + (["rgb-repeat"] if split == "test" else [])
    expected = {
        (a, s, off)
        for a in names
        for s in range(start, start + groups * w)
        for off in plan["noise_offsets"]
        if a != "rgb-repeat" or off == 0
    }
    assert set(index) == expected
    for p in sorted(folder.glob("[0-9]*.json")):
        with np.load(p.with_suffix(".npz")) as tr:
            d, s, a, v = [
                tr[k]
                for k in ("release_decisions", "obs", "actions", "placement_decisions")
            ]
            jaws, speeds = tr["jaws"], tr["speeds"]
        for row in json.loads(p.read_text()):
            r = index[row["arm"], row["scene"], row["noise_seed"] - row["scene"]]
            i = r["lane"]
            n = r["steps"]
            measured = summarize_trace(s[:n + 1, i], jaws[:n + 1, i], speeds[:n, i])
            if measured["success"] != r["success"]:
                raise ValueError(f"Released-stack label disagrees with trajectory in {p.name}, lane {i}")
            r.update(
                first_shift_tick=None,
                first_shift_height=None,
                first_shift_xy=None,
                first_shift_in_band=None,
                shift_triggered=False,
                open_within_six=False,
                first_steered_open_height=None,
                first_steered_open_xy=None,
                first_steered_open_in_band=None,
                deferred_decisions=0,
            )
            ds = d[: (n + 2) // 3, i]
            vs = v[: len(ds), i]
            applied = np.flatnonzero(ds[:, 2] != 0)
            if r["arm"].startswith("wait-"):
                threshold = LIMITS[r["arm"].split("-", 1)[1]][2]
                valid = vs[:, 4] > 0.5
                ready_twice = valid & np.r_[False, valid[:-1]]
                triggers = np.flatnonzero(ds[:, 1] > 0)
                end = triggers[0] if len(triggers) else len(ds)
                r["deferred_decisions"] = int(
                    (ready_twice[:end] & (vs[:end, 3] < threshold)).sum()
                )
            if not len(applied):
                continue
            t = int(applied[0]) * 3

            xy, h = geometry(s[t, i])
            r.update(
                first_shift_tick=t,
                first_shift_height=h,
                first_shift_xy=xy,
                first_shift_in_band=xy < 12 and abs(h) < 8,
                shift_triggered=True,
            )
            op = np.flatnonzero(a[t : min(t + 6, n), i, 3] <= 0)
            if len(op):
                xy, h = geometry(s[t + int(op[0]), i])
                r.update(
                    open_within_six=True,
                    first_steered_open_height=h,
                    first_steered_open_xy=xy,
                    first_steered_open_in_band=xy < 12 and abs(h) < 8,
                )

    def med(rr, k):
        v = [r[k] for r in rr if r[k] is not None]
        return float(np.median(v)) if v else None

    arms = {}
    for name in names:
        rr = [r for r in rows if r["arm"] == name]
        arms[name] = {
            "trials": len(rr),
            "stacks": sum(r["success"] for r in rr),
            "modes": dict(Counter(r["mode"] for r in rr)),
            "per_seed": {
                seed: sum(r["success"] for r in rr if r["actor_seed"] == seed)
                for seed in plan["actor_seeds"]
            },
            "per_stream": {
                off: sum(
                    r["success"] for r in rr if r["noise_seed"] - r["scene"] == off
                )
                for off in plan["noise_offsets"]
            },
            "shifted": sum(r["shift_triggered"] for r in rr),
            "open_within_six": sum(r["open_within_six"] for r in rr),
            "median_shift_height_mm": med(rr, "first_shift_height"),
            "median_shift_xy_mm": med(rr, "first_shift_xy"),
            "median_steered_open_height_mm": med(rr, "first_steered_open_height"),
            "median_steered_open_xy_mm": med(rr, "first_steered_open_xy"),
            "in_band_at_shift": sum(r["first_shift_in_band"] is True for r in rr),
            "in_band_at_steered_open": sum(
                r["first_steered_open_in_band"] is True for r in rr
            ),
            "episodes_actor_deferred": sum(r["deferred_decisions"] > 0 for r in rr),
        }
    dump_csv(run / f"{split}-outcomes.csv", rows)
    return plan, index, arms


def select(run):
    _, _, arms = measure(run, "dev")
    fixed = arms["fixed-permissive"]["median_shift_height_mm"]
    names = ["wait-" + k for k in LIMITS if "wait-" + k in arms]
    qualified = [
        n
        for n in names
        if arms[n]["shifted"]
        and arms[n]["median_shift_height_mm"] <= fixed - 4
        and arms[n]["median_shift_xy_mm"] < 12
    ]
    best = max(
        qualified or names,
        key=lambda n: (
            arms[n]["stacks"],
            -arms[n]["modes"].get("opened_off_center", 0),
        ),
    )
    out = {
        "selected": best.split("-", 1)[1],
        "qualified": qualified,
        "beat_original_fixed_while_lower": best in qualified
        and arms[best]["stacks"] > arms["fixed-permissive"]["stacks"],
        "arms": arms,
        "rule": "Among candidates with median intervention at least 4 mm lower than original fixed and median horizontal offset under 12 mm, maximize dev stacks, then minimize off-center opening failures; use all if none qualify.",
    }
    (run / "selection.json").write_text(json.dumps(out, indent=2))
    print(out["selected"], qualified, {a: v["stacks"] for a, v in arms.items()})


def test(run):
    plan, idx, arms = measure(run, "test")
    start = plan["first_scene"]
    scenes = range(start, start + plan["width"] * plan["batches"])
    sel = "wait-" + plan["selected"]
    fixed = "fixed-permissive"
    lower = "lower-" + plan["selected"]
    noise = plan["noise_offsets"]

    def scores(a):
        return np.array([[idx[a, s, o]["success"] for o in noise] for s in scenes])

    seeds = np.array([idx[sel, s, 0]["actor_seed"] for s in scenes])
    groups = np.array([idx[sel, s, 0]["batch_seed"] for s in scenes])
    assert len(scenes) == 160 and set(Counter(seeds).values()) == {32}
    rng = np.random.default_rng(62153)
    boot = rng.integers(len(scenes), size=(12000, len(scenes)))

    def weights(ids):
        u, rev = np.unique(ids, return_inverse=True)
        return rng.multinomial(len(u), np.full(len(u), 1 / len(u)), 12000)[:, rev]

    weights_cross = weights(groups) * weights(seeds)

    def pair(a, b):
        d = scores(a) - scores(b)
        av = d.mean(1)
        return {
            "difference": float(d.mean()),
            "scene_ci": np.quantile(av[boot].mean(1), [0.025, 0.975]).tolist(),
            "lighting_seed_ci": np.quantile(
                weights_cross @ av / weights_cross.sum(1), [0.025, 0.975]
            ).tolist(),
            "rescued": int((d == 1).sum()),
            "lost": int((d == -1).sum()),
            "per_seed": {int(s): float(d[seeds == s].mean()) for s in np.unique(seeds)},
        }

    pairs = {b: pair(sel, b) for b in plan["actors"] if b != sel}
    base = scores("rgb-base")[:, 0]
    rep = np.array([idx["rgb-repeat", s, 0]["success"] for s in scenes])
    matched = []
    for s in scenes:
        for o in noise:
            x, y = idx[sel, s, o], idx[fixed, s, o]
            if x["shift_triggered"] and y["shift_triggered"]:
                matched.append(
                    [
                        x["first_shift_tick"] - y["first_shift_tick"],
                        x["first_shift_height"] - y["first_shift_height"],
                        x["first_shift_xy"] - y["first_shift_xy"],
                    ]
                )
    out = {
        "selected": sel,
        "physical_scenes": len(scenes),
        "total_rollouts": len(idx),
        "arms": arms,
        "selected_minus": pairs,
        "fixed_minus_base": pair(fixed, "rgb-base"),
        "previous_actor_minus_fixed": pair("release-permissive", fixed),
        "matched_gate_fixed_minus_original_fixed": pair(lower, fixed),
        "base_repeat": {
            "original": int(base.sum()),
            "repeat": int(rep.sum()),
            "agree": int((base == rep).sum()),
        },
        "both_shifted_vs_original_fixed": len(matched),
        "median_paired_first_shift_difference_ticks_height_xy": np.median(
            matched, axis=0
        ).tolist()
        if matched
        else None,
    }
    (run / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("split", choices=("dev", "test"))
    p.add_argument("run", type=Path)
    a = p.parse_args()
    (select if a.split == "dev" else test)(a.run)
