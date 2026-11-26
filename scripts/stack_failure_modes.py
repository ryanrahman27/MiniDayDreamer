"""Score the same grasp, descent and release events on archived and fresh Isaac traces."""

import argparse
import json
from collections import Counter
from pathlib import Path

from climb_timing import timing, write_csv
from placement_events import event_ticks
import numpy as np


def classify(success, event):
    if success:
        return "stable_stack"
    if event["first_lift"] is None:
        return "no_lift"
    if event["first_open"] is None:
        minimum = event["min_centered_held_z_error_mm"]
        if minimum is None:
            return "never_open_never_centered_while_lifted"
        if minimum >= 8:
            return "never_open_stayed_high_while_centered"
        return "never_open_other"
    if not event["grasped_at_open"]:
        return "first_open_after_grasp_loss"
    if event["open_xy_mm"] >= 12:
        return "first_open_off_center"
    if event["final_xy_mm"] >= 12 or abs(event["final_z_error_mm"]) >= 8:
        return "opened_centered_ended_outside_pose"
    return "opened_centered_ended_in_pose_without_stable_stack"


def measure(tr):
    e = event_ticks(tr)
    motion, _ = timing(tr)
    success = int(tr["success"])
    opened = e["first_open"] is not None
    return dict(
        success=success,
        mode=classify(success, e),
        lifted=e["first_lift"] is not None,
        reached_center=e["first_over"] is not None,
        never_open=e["first_lift"] is not None and not opened,
        first_open_tick=e["first_open"],
        first_open_grasped=e["grasped_at_open"],
        open_xy_mm=e["open_xy_mm"],
        open_z_error_mm=e["open_z_error_mm"],
        min_centered_height_error_mm=e["min_centered_held_z_error_mm"],
        open_off_center=bool(e["grasped_at_open"] and e["open_xy_mm"] >= 12),
        open_high=bool(e["grasped_at_open"] and e["open_z_error_mm"] >= 8),
        loss_off_center=e["grasp_loss_xy_mm"] is not None and e["grasp_loss_xy_mm"] >= 12,
        rising_at_alignment=motion.get("rising_at_12"),
        descent_by_12_after=motion.get("descent_by_12_after"),
        ended_in_pose=e["final_xy_mm"] < 12 and abs(e["final_z_error_mm"]) < 8,
    )


def counts(rows):
    return dict(
        episodes=len(rows),
        modes=dict(sorted(Counter(r["mode"] for r in rows).items())),
        **{
            k: sum(bool(r[k]) for r in rows)
            for k in (
                "success",
                "lifted",
                "reached_center",
                "never_open",
                "first_open_grasped",
                "open_off_center",
                "open_high",
                "loss_off_center",
                "rising_at_alignment",
                "descent_by_12_after",
                "ended_in_pose",
            )
        },
    )


def census(run, folders):
    rows = []
    result = {}
    configs = (
        ("midbase", ("mid_base", "mid_varied_learned", "mid_varied_direct")),
        ("target", ("base", "direct", "handoff")),
        ("approach", ("base", "handoff", "approach")),
    )
    for folder, (cohort, arms) in zip(folders, configs):
        rs = [json.loads(line) for f in folder.glob("eval-*.jsonl") for line in f.read_text().splitlines()]
        assert len({r["seed"] for r in rs}) == len(rs) == 100
        result[cohort] = {}
        for arm in arms:
            values = []
            for r in sorted(rs, key=lambda s: s["seed"]):
                with np.load(folder / "traces" / f"{arm}-{r['seed']}.npz") as tr:
                    assert int(tr["seed"]) == r["seed"] and int(tr["success"]) == r[arm]["success"]
                    values.append(dict(cohort=cohort, arm=arm, seed=r["seed"], **measure(tr)))
            result[cohort][arm] = {
                outcome: counts([v for v in values if v["success"] == s])
                for outcome, s in (("success", 1), ("failure", 0))
            }
            rows.extend(values)
    pooled = [r for r in rows if r["arm"] in ("mid_base", "base")]
    result["all_frozen_base"] = {
        v: counts([r for r in pooled if r["success"] == s]) for v, s in (("success", 1), ("failure", 0))
    }
    assert len({r["seed"] for r in pooled}) == len(pooled) == 300
    write_csv(run / "archived-failure-events.csv", rows)
    (run / "archived-failure-modes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["all_frozen_base"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("folders", nargs=3, type=Path)
    args = p.parse_args()
    census(args.run, args.folders)
