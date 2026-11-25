"""Locate the first carry stall, compare proposed and realized motion, and audit the XY intervention."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from minidaydreamer.stack import STEP
from minidaydreamer.transport import CarryStall, WINDOW


def gap_mm(obs):
    return 100 * np.linalg.norm(obs[..., 6:8] - obs[..., 3:5], axis=-1)


def window_motion(obs, actions, end):
    start = end - WINDOW
    states = obs[start:end]
    direction = states[:, 6:8] - states[:, 3:5]
    direction = direction / np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-7)
    xy = actions[start:end, :2] * (STEP * 1000)
    tool = 100 * np.diff(obs[start:end+1, :2], axis=0)
    cube = 100 * np.diff(obs[start:end+1, 3:5], axis=0)
    return dict(gap_mm=float(gap_mm(obs[end])), gap_reduction_mm=float(gap_mm(obs[start])-gap_mm(obs[end])),
                request_mm_tick=float((xy * direction).sum(1).mean()),
                sideways_request_mm_tick=float(np.abs(xy[:, 0]*direction[:, 1]-xy[:, 1]*direction[:, 0]).mean()),
                tool_mm_tick=float((tool * direction).sum(1).mean()), cube_mm_tick=float((cube * direction).sum(1).mean()),
                cube_tool_slip_mm_tick=float(np.linalg.norm(cube-tool, axis=1).mean()),
                z_request_mm_tick=float(actions[start:end, 2].mean() * STEP * 1000))


def diagnose(obs, actions):
    detector = CarryStall(obs[0])
    continuous = 0
    stalls = []
    moved_15 = False
    for t, state in enumerate(obs):
        if not detector.carrying(state) or gap_mm(state) <= 12:
            continuous = t + 1
        stalled = detector.observe(state)
        moved_15 |= detector.transported
        if stalled and not stalls:
            stalls.append((t, continuous))
    lifted = (obs[:, 9] > .5) & (obs[:, 5]-obs[0, 5] > .2)
    xy = gap_mm(obs)
    near = lifted & (xy < 12)
    result = dict(lifted=bool(lifted.any()), best_held_xy_mm=float(xy[lifted].min()) if lifted.any() else None,
                  reached_xy=bool(near.any()), reached_zone=bool((near & (np.abs(100*(obs[:, 5]-obs[:, 8])-46.8) < 8)).any()),
                  moved_15=bool(moved_15), stall_tick=None, approach_end=None)
    if not stalls:
        return result
    t, segment_start = stalls[0]
    result.update(stall_tick=t, **{"stall_"+k: v for k, v in window_motion(obs, actions, t).items()})
    # Most recent non-overlapping productive window in this same uninterrupted carry.
    for end in range(t-WINDOW, segment_start+WINDOW-1, -1):
        if gap_mm(obs[end-WINDOW]) - gap_mm(obs[end]) >= 6:
            result.update(approach_end=end, **{"approach_"+k: v for k, v in window_motion(obs, actions, end).items()})
            break
    return result


def group(rows):
    stalls = [r for r in rows if r["stall_tick"] is not None]
    pairs = [r for r in stalls if r["approach_end"] is not None]
    held = [r["best_held_xy_mm"] for r in rows if r["lifted"]]
    stats = dict(episodes=len(rows), lifts=sum(r["lifted"] for r in rows), transported_15mm=sum(r["moved_15"] for r in rows),
                 reached_xy=sum(r["reached_xy"] for r in rows), reached_zone=sum(r["reached_zone"] for r in rows),
                 median_best_held_xy_mm=float(np.median(held)) if held else None,
                 stalls=len(stalls), stalled_failures=sum(not r["success"] for r in stalls), paired_windows=len(pairs))
    if stalls:
        stats["median_stall_distance_mm"] = float(np.median([r["stall_gap_mm"] for r in stalls]))
    for phase in ("approach", "stall"):
        stats[phase] = {}
        if pairs:
            for k in pairs[0]:
                if k.startswith(phase+"_") and k != "approach_end" and k != "stall_tick":
                    stats[phase][k[len(phase)+1:]] = float(np.median([r[k] for r in pairs]))
    return stats


def pulse_outcome(run, seed, obs, trigger, applied):
    if trigger < 0 or applied == 0 or trigger+WINDOW >= len(obs):
        return None
    with np.load(run / "traces" / f"base-{seed}.npz") as base:
        other = base["obs"]
        if trigger+WINDOW >= len(other):
            return None
        prefix = (obs[:trigger+1, :9]-other[:trigger+1, :9]).reshape(-1, 3)
        prefix_mm = float(100 * np.linalg.norm(prefix, axis=1).max())
        matched = prefix_mm <= 2 and np.array_equal(obs[:trigger+1, 9:], other[:trigger+1, 9:])
        def progress(x):
            return float(gap_mm(x[trigger]) - gap_mm(x[trigger+WINDOW]))
        return dict(seed=seed, trigger=trigger, applied=applied, matched=bool(matched), prefix_mm=prefix_mm,
                    base_progress_mm=progress(other), pulse_progress_mm=progress(obs))


def analyze(run, output):
    scores = sorted((json.loads(line) for p in run.glob("eval-*.jsonl") for line in p.read_text().splitlines() if line),
                    key=lambda r: r["seed"])
    rows, pulses = [], []
    for record in scores:
        seed = record["seed"]
        for arm, metrics in record.items():
            if arm in ("start", "seed"):
                continue
            with np.load(run / "traces" / f"{arm}-{seed}.npz") as tr:
                obs, actions = tr["obs"], tr["actions"]
                if (len(obs) != len(actions)+1 or len(actions) != metrics["steps"]
                        or int(tr["success"]) != metrics["success"] or int(tr["seed"]) != seed):
                    raise ValueError(f"Trace mismatch for {arm}/{seed}")
                result = diagnose(obs, actions)
                rows.append(dict(seed=seed, policy=arm, success=metrics["success"], **result))
                if arm == "pulse_base":
                    trigger = int(tr["pulse_trigger"])
                    rows[-1]["pulse_trigger"] = trigger
                    rows[-1]["pulse_applied_ticks"] = int(tr["pulse_steps"])
                    if trigger != (result["stall_tick"] if result["stall_tick"] is not None and result["stall_tick"] < len(actions) else -1):
                        raise ValueError(f"Pulse trigger does not match recorded states on {seed}")
                    outcome = pulse_outcome(run, seed, obs, trigger, int(tr["pulse_steps"]))
                    if outcome is not None:
                        pulses.append(outcome)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (output / "transport-events.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with np.load(run / "corpus.npz") as corpus:
        experts = []
        for i in np.flatnonzero(corpus["split"] == "demo"):
            n = corpus["lengths"][i]
            experts.append(dict(success=1, **diagnose(corpus["obs"][i, :n+1], corpus["actions"][i, :n])))
    out = dict(expert=group(experts), policies={a: group([r for r in rows if r["policy"] == a])
                                               for a in sorted({r["policy"] for r in rows})}, pulses=pulses)
    pulse_trials = [r for r in rows if r["policy"] == "pulse_base"]
    if pulse_trials:
        out["pulse_counts"] = dict(enrolled=len(pulse_trials), triggered=sum(r["pulse_trigger"] >= 0 for r in pulse_trials),
                                   applied=sum(r["pulse_applied_ticks"] > 0 for r in pulse_trials),
                                   full_pulse=sum(r["pulse_applied_ticks"] == WINDOW for r in pulse_trials))
    (output / "transport-summary.json").write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run
    output.mkdir(exist_ok=True, parents=True)
    result = analyze(args.run, output)
    print(json.dumps(result["policies"]["base"], indent=2))
