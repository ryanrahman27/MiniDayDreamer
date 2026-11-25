"""Match gripper proposals and measured grasp state to cube geometry on scored Isaac trials."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def release_events(obs, actions, proposed, aperture):
    # Actions[t] is issued from obs[t]. Grasp loss is observed in obs[t+1].
    states = obs[:-1]
    ticks = np.arange(len(actions))
    xy = 100 * np.linalg.norm(obs[:, 3:5] - obs[:, 6:8], axis=1)
    above = 100 * (obs[:, 5] - obs[:, 8]) - 46.8
    grasped = obs[:, 9] > .5
    lift = first(grasped & (obs[:, 5] - obs[0, 5] > .2))
    after_lift = ticks >= (lift if lift is not None else len(obs))
    opening_edges = after_lift & (states[:, 10] > .5) & (actions[:, 3] <= 0)
    opening = first(opening_edges)
    open_duration = None
    reclosed_within_chunk = lost_within_chunk = False
    if opening is not None:
        next_close = first(actions[opening:, 3] > 0)
        open_duration = next_close if next_close is not None else len(actions) - opening
        reclosed_within_chunk = next_close is not None and next_close <= 3
        lost_within_chunk = bool(grasped[opening] and (~grasped[opening+1:opening+4]).any())
    loss = first(after_lift & grasped[:-1] & ~grasped[1:])
    held = after_lift & grasped[:-1]
    in_zone = (xy < 12) & (np.abs(above) < 8)
    overrides = int(np.sum((proposed <= 0) & (actions[:, 3] > 0)))
    if lift is None:
        mode = "no_lift"
    elif opening is None:
        mode = "never_opened_after_lift"
    elif not grasped[opening]:
        mode = "first_open_without_grasp"
    elif xy[opening] >= 12:
        mode = "first_open_off_center"
    elif abs(above[opening]) >= 8:
        mode = "first_open_at_wrong_height"
    else:
        mode = "first_open_in_zone"
    return dict(lift_step=lift, first_open_step=opening, first_loss_step=loss,
                first_open_grasped=bool(grasped[opening]) if opening is not None else None,
                first_open_xy_mm=float(xy[opening]) if opening is not None else None,
                first_open_height_error_mm=float(above[opening]) if opening is not None else None,
                first_open_aperture_mm=float(aperture[opening] * 1000) if opening is not None and aperture is not None else None,
                first_open_duration_steps=open_duration, reclosed_within_three_steps=reclosed_within_chunk,
                lost_grasp_signal_within_three_steps=lost_within_chunk, postlift_openings=int(opening_edges.sum()),
                grasp_loss_before_open=bool(loss is not None and (opening is None or loss < opening)),
                first_loss_command_closed=bool(actions[loss, 3] > 0) if loss is not None else None,
                minimum_held_xy_mm=float(xy[:-1][held].min()) if held.any() else None,
                reached_zone_while_held=bool((held & in_zone[:-1]).any()),
                ended_grasped=bool(grasped[-1]), ended_closed=bool(obs[-1, 10] > .5),
                override_steps=overrides, opening_category=mode)


def describe(rows):
    categories = {}
    for r in rows:
        name = r["opening_category"]
        categories[name] = categories.get(name, 0) + 1
    opens = [r for r in rows if r["first_open_grasped"]]
    return dict(episodes=len(rows), categories=categories, commanded_open_while_grasped=len(opens),
                median_open_xy_mm=float(np.median([r["first_open_xy_mm"] for r in opens])) if opens else None,
                median_open_height_error_mm=float(np.median([r["first_open_height_error_mm"] for r in opens])) if opens else None,
                reached_zone_while_held=sum(r["reached_zone_while_held"] for r in rows),
                median_first_open_duration_steps=float(np.median([r["first_open_duration_steps"] for r in opens])) if opens else None,
                reclosed_within_three_steps=sum(r["reclosed_within_three_steps"] for r in opens),
                lost_grasp_signal_within_three_steps=sum(r["lost_grasp_signal_within_three_steps"] for r in opens),
                ended_grasped=sum(r["ended_grasped"] for r in rows),
                grasp_loss_before_open=sum(r["grasp_loss_before_open"] for r in rows),
                intervention_activated=sum(r["override_steps"] > 0 for r in rows))


def analyze(run):
    scores = [json.loads(line) for path in sorted(run.glob("eval-*.jsonl"))
              for line in path.read_text().splitlines() if line]
    rows = []
    for trial in sorted(scores, key=lambda r: r["seed"]):
        for arm, metrics in trial.items():
            if arm in ("seed", "start"):
                continue
            seed = trial["seed"]
            with np.load(run / "traces" / f"{arm}-{seed}.npz", allow_pickle=False) as tr:
                obs, actions = tr["obs"], tr["actions"]
                if (int(tr["seed"]) != seed or len(actions) != metrics["steps"] or len(obs) != len(actions) + 1
                        or int(tr["success"]) != metrics["success"] or not np.allclose(obs[0], trial["start"], atol=.01, rtol=0)):
                    raise ValueError(f"Trace does not match scored trial {seed}/{arm}")
                events = release_events(obs, actions, tr["proposed_grip"], tr["aperture_m"])
                if (events["lift_step"] is not None) != bool(metrics["lifted"]):
                    raise ValueError(f"Lift disagreement in {seed}/{arm}")
                rows.append(dict(seed=seed, policy=arm, success=metrics["success"], **events))
    with (run / "release-events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with np.load(run / "corpus.npz", allow_pickle=False) as d:
        expert = []
        for i in np.flatnonzero(d["split"] == "demo"):
            n = d["lengths"][i]
            actions = d["actions"][i, :n]
            expert.append(release_events(d["obs"][i, :n+1], actions, actions[:, 3], None))
    report = dict(expert=describe(expert), policies={})
    for arm in sorted({r["policy"] for r in rows}):
        subset = [r for r in rows if r["policy"] == arm]
        report["policies"][arm] = dict(all=describe(subset), failed=describe([r for r in subset if not r["success"]]))
    if "gated_base" in report["policies"]:
        base_success = {r["seed"]: r["success"] for r in rows if r["policy"] == "base"}
        report["gate_effect"] = {}
        for active in (True, False):
            subset = [r for r in rows if r["policy"] == "gated_base" and bool(r["override_steps"]) == active]
            delta = [r["success"] - base_success[r["seed"]] for r in subset]
            report["gate_effect"]["applied" if active else "unused"] = dict(
                scenes=len(subset), gained=delta.count(1), lost=delta.count(-1))
    (run / "release-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(analyze(parser.parse_args().run), indent=2))
