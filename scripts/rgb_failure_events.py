"""Locate alignment, seating and opening events on measured RGB-policy evaluation traces."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def first(mask):
    candidates = np.flatnonzero(mask)
    return int(candidates[0]) if candidates.size else None


def events(obs, actions, success):
    # Positions are in 10-cm units. The evaluation checks the horizontal and vertical bands separately.
    xy = 100 * np.linalg.norm(obs[:, 3:5] - obs[:, 6:8], axis=1)
    height = 100 * (obs[:, 5] - obs[:, 8]) - 46.8
    held = obs[:, 9] > 0.5
    lift = first(held & (obs[:, 5] > 0.65))
    out = {
        "lift_tick": lift,
        "opening_tick": None,
        "detached_tick": None,
        "centered_tick": None,
        "seated_tick": None,
        "open_xy_mm": None,
        "open_height_mm": None,
        "min_held_xy_mm": None,
        "min_abs_centered_height_mm": None,
        "ever_open_after_lift": False,
        "regrasped": False,
        "ended_held": bool(held[-1]),
        "final_xy_mm": float(xy[-1]),
        "final_height_mm": float(height[-1]),
        "geometry_at_open": None,
    }
    if lift is None:
        return {**out, "mode": "stacked" if success else "no_lift"}
    time = np.arange(len(obs))
    opened = first((time[:-1] >= lift) & (actions[:, 3] <= 0))
    detached = first((time > lift) & ~held)
    # An opening is a command at tick t; geometry at t precedes the action.
    last = min(
        [len(obs) - 1]
        + ([opened] if opened is not None else [])
        + ([detached - 1] if detached is not None else [])
    )
    carrying = (time >= lift) & (time <= last) & held
    centered = carrying & (xy < 12)
    seated = centered & (np.abs(height) < 8)
    center = first(centered)
    seat = first(seated)
    out.update(
        opening_tick=opened,
        detached_tick=detached,
        centered_tick=center,
        seated_tick=seat,
        ever_open_after_lift=opened is not None,
        regrasped=bool(detached is not None and held[detached:].any()),
        min_held_xy_mm=float(xy[carrying].min()),
        min_abs_centered_height_mm=float(np.abs(height[centered]).min())
        if center is not None
        else None,
    )
    if opened is not None:
        out.update(
            open_xy_mm=float(xy[opened]),
            open_height_mm=float(height[opened]),
            geometry_at_open=("off_center" if xy[opened] >= 12 else "centered")
            + (
                "_high"
                if height[opened] >= 8
                else "_low"
                if height[opened] <= -8
                else "_height_ok"
            ),
        )
    if success:
        mode = "stacked"
    elif detached is not None and (opened is None or detached <= opened):
        mode = "lost_grasp_before_open"
    elif opened is None:
        if center is None:
            mode = "never_centered_no_open"
        elif seat is None:
            mode = "centered_not_seated_no_open"
        else:
            mode = "seated_no_open"
    elif xy[opened] >= 12:
        mode = "opened_off_center"
    elif height[opened] >= 8:
        mode = "opened_centered_high"
    elif height[opened] <= -8:
        mode = "opened_centered_low"
    else:
        mode = "opened_in_band_then_failed"
    return {**out, "mode": mode}


def score_rows(folder):
    for path in sorted(folder.glob("[0-9]*.json")):
        rows = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz")) as trace:
            obs, acts = trace["obs"], trace["actions"]
            for r in rows:
                i = r["lane"]
                n = r["steps"]
                yield {**r, **events(obs[: n + 1, i], acts[:n, i], bool(r["success"]))}


def dump_csv(path, rows):
    with path.open("w", newline="") as f:
        cols = list(rows[0])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def diagnose(run, prior):
    original = {}
    for path in sorted(prior.glob("*-rgb-base.json")):
        for r in json.loads(path.read_text()):
            original[r["scene"], r["noise_seed"]] = r
    assert len(original) == 256
    rows = []
    for r in score_rows(run / "test-diagnosis"):
        old = original[r["scene"], r["noise_seed"]]
        r.pop("initial", None)
        r.update(
            original_lifted=old["lifted"],
            original_success=old["success"],
            original_terminal_ungrasped=old["terminal_ungrasped"],
            original_ever=old["ever"],
        )
        rows.append(r)
    assert len(rows) == len({(r["scene"], r["noise_seed"]) for r in rows}) == 256
    selected = [r for r in rows if r["original_lifted"] and not r["original_success"]]
    assert len(selected) == 127
    persistent = [r for r in selected if r["lifted"] and not r["success"]]
    successful = [r for r in rows if r["success"]]

    def counts(group, field):
        return dict(Counter(r[field] for r in group))

    summary = {
        "original_runs": len(selected),
        "physical_scenes": len({r["scene"] for r in selected}),
        "original_grasp_detected_in_final_8": sum(
            not r["original_terminal_ungrasped"] for r in selected
        ),
        "original_stacked_for_eight_earlier": sum(r["original_ever"] for r in selected),
        "replay_stacked": sum(r["success"] for r in selected),
        "replay_no_lift": sum(r["mode"] == "no_lift" for r in selected),
        "replay_lifted_failed": len(persistent),
        "modes": counts(persistent, "mode"),
        "matched_success_reference": len(successful),
        "success_open_geometry": counts(successful, "geometry_at_open"),
        "replay_failed_open_geometry": counts(persistent, "geometry_at_open"),
        "replay_centered": sum(r["centered_tick"] is not None for r in persistent),
        "replay_seated": sum(r["seated_tick"] is not None for r in persistent),
        "replay_without_open": sum(not r["ever_open_after_lift"] for r in persistent),
    }
    summary["modal_failure"] = Counter(r["mode"] for r in persistent).most_common(1)[0][
        0
    ]
    dump_csv(run / "original-127.csv", selected)
    dump_csv(run / "all-replays.csv", rows)
    (run / "diagnosis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("prior", type=Path)
    a = p.parse_args()
    diagnose(a.run, a.prior)
