"""Audit frozen DSRL training provenance, all fresh actions, paired stacks and failure rates."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from climb_timing import write_csv
from failure_dsrl import ARMS, SEEDS, mixed_starts
from minidaydreamer.data import Dataset
from minidaydreamer.experiment import load_policies, verify_weights
from placement_events import geometry
from probe_failure_steering import observed_windows
from reward_study import digest, load
from stack_failure_modes import counts, measure
from summarize_handoff import verify_commands
from summarize_stack import paired


def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return int(lengths.max(initial=0))


def release_dwell(trace, first_open):
    # The first scored position after the opening command is obs[first_open + 1].
    obs = trace["obs"][first_open + 1 :]
    aperture = trace["aperture_m"][first_open + 1 :]
    xy, height, _ = geometry(obs)
    clearance = np.linalg.norm(obs[:, :3] - obs[:, 3:6], axis=1) * 100
    pose = (xy < 12) & (abs(height) < 8) & (obs[:, 8] > 0.14) & (obs[:, 8] < 0.32)
    commanded = pose & (obs[:, 10] < 0.5)
    opened = commanded & (aperture > 0.035)
    return dict(
        pose_ticks=longest_run(pose),
        pose_cmd_open_ticks=longest_run(commanded),
        pose_jaws_open_ticks=longest_run(opened),
        pose_open_clear_ticks=longest_run(opened & (clearance > 50)),
        final_clearance_mm=float(clearance[-1]),
    )


def summarize(run, prior, archived):
    info = json.loads((run / "training.json").read_text())
    baseline = load(run / "selected-base.pt")
    old = load(run / "mid-varied-direct.pt")
    old_audit = json.loads((prior / "midbase-summary.json").read_text())
    assert (
        info["base_sha256"] == digest(run / "selected-base.pt") == old_audit["selection"]["checkpoint_sha256"]
    )
    assert (
        info["world_source_sha256"]
        == digest(run / "mid-varied-direct.pt")
        == old_audit["actors"]["direct"]["sha256"]
    )
    assert digest(prior / "corpus.npz") == info["corpus_sha256"]
    rng = load(archived / "varied.pt")["pre_actor_rng"]
    assert info["rng_sha256"] == old["actor_rng_sha256"] == hashlib.sha256(rng.numpy().tobytes()).hexdigest()
    policies = {}
    for arm in ARMS[1:]:
        f = run / ("mid-varied-direct.pt" if arm == "original_direct" else arm + ".pt")
        c, pi, wm, sac = load_policies(f, SimpleNamespace(backend="isaac"))
        verify_weights(wm, old["world"], arm + " fixed world")
        verify_weights(pi, baseline["policy"], arm + " fixed base")
        assert (
            c["base_checkpoint_sha256"] == info["base_sha256"] and c["actor_rng_sha256"] == info["rng_sha256"]
        )
        assert (c["imagination_chunks"], c["rl_steps"], c["imagined_ticks"]) == (2, 1500, 576000)
        if arm != "original_direct":
            assert digest(f) == info["actors"][arm]["sha256"]
            assert c["start_tensor_sha256"] == info["start_tensor_sha256"] and c["reward_mode"] == arm
        policies[arm] = (pi, wm, sac)
    policies["base"] = (pi, wm, None)
    data = Dataset(prior / "corpus.npz", 0)
    starts, source = mixed_starts(prior, wm, data, np.flatnonzero(data.split == "demo"))
    assert hashlib.sha256(starts.numpy().tobytes()).hexdigest() == info["start_tensor_sha256"]
    assert source["pool_counts"] == info["pool_counts"] and source["source_seeds"] == info["source_seeds"]
    assert {str(k): v for k, v in source["source_trace_sha256"].items()} == info["source_trace_sha256"]
    excluded = set(baseline["demo_seeds"]) | set(data.seeds.tolist())
    for folder in prior.parent.glob("isaac-*-v[1-7]"):
        for f in folder.glob("eval-*.jsonl"):
            excluded.update(json.loads(line)["seed"] for line in f.read_text().splitlines())
    for entry in json.loads((prior / "selection.json").read_text())["development"]:
        excluded.update(r["seed"] for r in entry["outcomes"])
    assert not excluded.intersection(SEEDS)
    scores = sorted(
        (json.loads(line) for f in run.glob("eval-*.jsonl") for line in f.read_text().splitlines()),
        key=lambda r: r["seed"],
    )
    assert [r["seed"] for r in scores] == list(SEEDS)
    records = []
    windows = {arm: {} for arm in ARMS}
    max_start = max_action = 0.0
    for scene in scores:
        seed = scene["seed"]
        assert set(scene) == {"seed", "start", *ARMS}
        starts_seen = []
        for arm in ARMS:
            with np.load(run / "traces" / f"{arm}-{seed}.npz") as tr:
                m = scene[arm]
                obs = tr["obs"]
                assert (
                    int(tr["seed"]) == seed
                    and int(tr["success"]) == m["success"]
                    and len(obs) - 1 == len(tr["actions"]) == m["steps"]
                )
                starts_seen.append(obs[0])
                np.testing.assert_allclose(obs[0], scene["start"], atol=0.01, rtol=0)
                max_action = max(max_action, verify_commands(tr, arm, *policies[arm], None))
                row = measure(tr)
                if row["first_open_tick"] is not None:
                    row.update(release_dwell(tr, row["first_open_tick"]))
                if row["success"]:
                    assert row["pose_open_clear_ticks"] >= 8
                for name, sample in observed_windows(obs).items():
                    values = windows[arm].setdefault(
                        name, dict(scenes=0, windows=0, opening_windows=0, total_height_change_mm=0.0)
                    )
                    values["scenes"] += int(sample["windows"] > 0)
                    for key, value in sample.items():
                        values[key] += value
                assert row["lifted"] == m["lifted"] and np.isclose(
                    np.linalg.norm(obs[-1, 3:5] - obs[-1, 6:8]) * 100, m["final_xy_error_mm"]
                )
                records.append(
                    dict(seed=seed, arm=arm, steps=m["steps"], episode_return=m["episode_return"], **row)
                )
        max_start = max(max_start, float(np.ptp(np.array(starts_seen)[:, :9], axis=0).max() * 100))
    assert len(records) == len(list((run / "traces").glob("*.npz"))) == 400 and max_start <= 1.0
    rows = {a: [r for r in records if r["arm"] == a] for a in ARMS}
    stats = {
        a: {
            outcome: counts([r for r in rs if test(r)])
            for outcome, test in (
                ("all", lambda r: True),
                ("failure", lambda r: r["success"] == 0),
                ("success", lambda r: r["success"] == 1),
            )
        }
        for a, rs in rows.items()
    }
    pairs = {}
    for ref, alt in (
        ("base", "targeted"),
        ("matched_direct", "targeted"),
        ("original_direct", "targeted"),
        ("original_direct", "matched_direct"),
        ("base", "matched_direct"),
        ("base", "original_direct"),
    ):
        a, b = (np.array([r["success"] for r in rows[k]]) for k in (ref, alt))
        pairs[f"{alt}_vs_{ref}"] = dict(
            **paired(a, b),
            gained_seeds=np.asarray(SEEDS)[b > a].tolist(),
            lost_seeds=np.asarray(SEEDS)[b < a].tolist(),
        )
    transfers = {}
    for ref in ARMS[:-1]:
        by_mode = {}
        for before, after in zip(rows[ref], rows["targeted"]):
            tab = by_mode.setdefault(before["mode"], {})
            tab[after["mode"]] = tab.get(after["mode"], 0) + 1
        transfers[ref] = by_mode
    dwell = {}
    for arm, arm_rows in rows.items():
        group = [r for r in arm_rows if r["mode"] == "opened_centered_ended_in_pose_without_stable_stack"]
        dwell[arm] = dict(
            episodes=len(group),
            eight_ticks={
                k: sum(r[k] >= 8 for r in group)
                for k in (
                    "pose_ticks",
                    "pose_cmd_open_ticks",
                    "pose_jaws_open_ticks",
                    "pose_open_clear_ticks",
                )
            },
            ever_open_and_clear_in_pose=sum(r["pose_open_clear_ticks"] > 0 for r in group),
            median_final_clearance_mm=float(np.median([r["final_clearance_mm"] for r in group]))
            if group
            else None,
        )
    result = dict(
        first_seed=SEEDS.start,
        last_seed=SEEDS.stop - 1,
        episodes=len(records),
        max_start_coordinate_difference_mm=max_start,
        max_replayed_action_error=max_action,
        policies=stats,
        paired=pairs,
        mode_transitions_to_targeted=transfers,
        observed_six_tick_windows=windows,
        dwell_after_centered_opening_on_in_pose_failures=dwell,
    )
    write_csv(run / "paired-outcomes.csv", records)
    (run / "failure-dsrl-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({a: v["all"] for a, v in stats.items()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--archived", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    summarize(args.run, args.prior, args.archived)
