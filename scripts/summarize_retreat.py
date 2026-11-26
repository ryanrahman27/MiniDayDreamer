"""Verify physics and policy traces, score both definitions, and compare paired outcomes."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from climb_timing import write_csv
from minidaydreamer.experiment import load_policies, verify_weights
from reward_study import digest, load
from score_retreat import ARMS, SEEDS, masks, results
from summarize_handoff import verify_commands
from summarize_stack import paired


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_physics(run, name, obs, jaws, logged):
    with np.load(run / "physics" / f"{name}.npz") as saved:
        speeds = saved["policy_speeds_mps"]
        assert speeds.shape == (len(obs) - 1, 2) and np.isfinite(speeds).all() and (speeds >= 0).all()
        outcome = results(obs[1:], jaws[1:], speeds)
        assert all(logged[k] == v for k, v in outcome.items())
        post, post_jaws, post_speed = (
            saved[k] for k in ("withdrawal_obs", "withdrawal_aperture_m", "withdrawal_speeds_mps")
        )
    tested = outcome["released_terminal"]
    assert len(post) == len(post_jaws) == len(post_speed) == 32 * tested
    survived = None
    checks = {}
    if tested:
        pose, _, clear = masks(post, post_jaws, post_speed)
        distance = np.linalg.norm(post[:, :3] - post[:, 3:6], axis=1) * 100
        survived = int(pose.all() and clear[-20:].all() and (post[-20:, 9] < 0.5).all())
        checks = dict(
            retained_pose_during_probe=int(pose.all()),
            gripper_clear_last_second=int((distance[-20:] > 50).all()),
            steady_pose_last_second=int((pose[-20:] & (post_speed[-20:] < 0.025).all(axis=1)).all()),
            min_clearance_last_second_mm=float(min(distance[-20:])),
        )
    assert (logged["withdrawal_tested"], logged["survived_withdrawal"]) == (tested, survived)
    assert max(outcome["strict"], tested) <= outcome["released_ever"]
    return dict(
        **outcome,
        terminal_ungrasped_stack=int(tested and (obs[-8:, 9] < 0.5).all()),
        withdrawal_tested=tested,
        survived_withdrawal=survived,
        **checks,
    )


def totals(rows):
    extra = [r for r in rows if not r["strict"] and r["released_terminal"]]
    return dict(
        episodes=len(rows),
        **{
            k: sum(int(r[k] or 0) for r in rows)
            for k in ("strict", "released_ever", "released_terminal", "survived_withdrawal")
        },
        extra_terminal=len(extra),
        terminal_ungrasped=sum(r["terminal_ungrasped_stack"] for r in rows),
        extra_survived=sum(r["survived_withdrawal"] for r in extra),
        extra_ever_only=sum(r["released_ever"] and not r["released_terminal"] for r in rows),
        extra_gripper_clear=sum(r["gripper_clear_last_second"] for r in extra),
        extra_pose_retained=sum(r["retained_pose_during_probe"] for r in extra),
        strict_survived=sum(int(r["survived_withdrawal"] or 0) for r in rows if r["strict"]),
    )


def archived_checks(run, previous):
    with (previous / "paired-outcomes.csv").open() as f:
        chosen = {
            int(r["seed"])
            for r in csv.DictReader(f)
            if r["arm"] == "targeted" and r["mode"] == "opened_centered_ended_in_pose_without_stable_stack"
        }
    archive = read_rows(run / "archived-replays.jsonl")
    assert len(archive) == len(chosen) == 19 and {r["seed"] for r in archive} == chosen
    records = []
    for r in archive:
        seed = r["seed"]
        name = f"archive-{seed}"
        with (
            np.load(previous / "traces" / f"targeted-{seed}.npz") as original,
            np.load(run / "traces" / f"{name}.npz") as replay,
        ):
            a, b = original["obs"], replay["obs"]
            np.testing.assert_array_equal(original["actions"], replay["actions"])
            assert not original["success"]
            error = float(np.linalg.norm((a[:, :9] - b[:, :9]).reshape(-1, 3, 3), axis=2).max() * 100)
            assert abs(error - r["max_xyz_error_mm"]) < 0.001
            checks = check_physics(run, name, b, replay["aperture_m"], r)
            # Speeds were not recorded in the original experiment. Compute its displacement proxy separately.
            proxy = np.linalg.norm(np.diff(a[:, 3:9], axis=0).reshape(-1, 2, 3), axis=2) * 0.1 / 0.05
            est = results(a[1:], original["aperture_m"][1:], proxy)
            records.append(
                dict(
                    **r,
                    **{k: v for k, v in checks.items() if k not in r},
                    original_proxy_ever=est["released_ever"],
                    original_proxy_terminal=est["released_terminal"],
                    within_one_mm=int(
                        error < 1 and r["max_jaw_error_mm"] < 1 and not r["max_grasp_flag_error"]
                    ),
                )
            )
    write_csv(run / "archived-outcomes.csv", records)
    return dict(
        **totals(records),
        original_proxy_ever=sum(x["original_proxy_ever"] for x in records),
        original_proxy_terminal=sum(x["original_proxy_terminal"] for x in records),
        matched_replays=totals([r for r in records if r["within_one_mm"]]),
        max_replay_xyz_error_mm=max(r["max_xyz_error_mm"] for r in records),
        median_replay_xyz_error_mm=float(np.median([r["max_xyz_error_mm"] for r in records])),
    )


def placement_stage(row):
    for stage in ("strict", "released_terminal", "released_ever"):
        if row[stage]:
            return stage
    return "no_released_stack"


def summarize(run, previous, midbase):
    src = json.loads((run / "training.json").read_text())
    old = json.loads((previous / "training.json").read_text())
    assert src["targeted_sha256"] == old["actors"]["targeted"]["sha256"]
    assert src["start_tensor_sha256"] == old["start_tensor_sha256"] and src["rng_sha256"] == old["rng_sha256"]
    assert digest(run / "selected-base.pt") == old["base_sha256"] == src["base_sha256"]
    models = {}
    target = load(run / "targeted.pt")
    base = load(run / "selected-base.pt")
    for arm in ARMS[1:]:
        assert digest(run / (arm + ".pt")) == src[arm + "_sha256"]
        ckpt, pi, wm, actor = load_policies(run / (arm + ".pt"), SimpleNamespace(backend="isaac"))
        verify_weights(pi, base["policy"], arm + " common base")
        verify_weights(wm, target["world"], arm + " common RSSM")
        assert ckpt["reward_mode"] == arm and ckpt["start_tensor_sha256"] == src["start_tensor_sha256"]
        models[arm] = (pi, wm, actor)
    models["base"] = (pi, wm, None)
    excluded = set(base["demo_seeds"])
    for file in ("corpus.npz", "extra-bc.npz"):
        with np.load(midbase / file) as data:
            excluded.update(data["seeds"].tolist())
    for selection in json.loads((midbase / "selection.json").read_text())["development"]:
        excluded.update(r["seed"] for r in selection["outcomes"])
    for folder in previous.parent.glob("isaac-*-v[1-8]"):
        for path in folder.glob("eval-*.jsonl"):
            excluded.update(r["seed"] for r in read_rows(path))
    assert not excluded.intersection(SEEDS)
    scenes = sorted((r for f in run.glob("eval-*.jsonl") for r in read_rows(f)), key=lambda r: r["seed"])
    assert [r["seed"] for r in scenes] == list(SEEDS)
    records = []
    max_action = max_start = 0.0
    for scene in scenes:
        seed, starts = scene["seed"], []
        for arm in ARMS:
            name = f"{arm}-{seed}"
            logged = scene[arm]
            with np.load(run / "traces" / (name + ".npz")) as tr:
                obs = tr["obs"]
                assert int(tr["seed"]) == seed and int(tr["success"]) == logged["strict"] == logged["success"]
                assert len(obs) - 1 == logged["steps"] == len(tr["actions"])
                np.testing.assert_allclose(obs[0], scene["start"], atol=0.01, rtol=0)
                starts.append(obs[0])
                max_action = max(max_action, verify_commands(tr, arm, *models[arm], None))
                checks = check_physics(run, name, obs, tr["aperture_m"], logged)
                assert checks["strict"] <= checks["released_terminal"]
                records.append(dict(seed=seed, arm=arm, steps=logged["steps"], **checks))
        max_start = max(max_start, float(np.ptp(starts, axis=0)[:9].max() * 100))
    assert max_start <= 1 and len(records) == 300
    assert len(list((run / "traces").glob("*.npz"))) == len(list((run / "physics").glob("*.npz"))) == 319
    by_arm = {a: [r for r in records if r["arm"] == a] for a in ARMS}
    pairs = {}
    for key in ("strict", "released_ever", "released_terminal", "survived_withdrawal"):
        comparisons = {}
        for ref, alt in (("base", "retreat"), ("targeted", "retreat"), ("base", "targeted")):
            a, b = (np.array([int(r[key] or 0) for r in by_arm[k]]) for k in (ref, alt))
            comparisons[f"{alt}_vs_{ref}"] = dict(
                **paired(a, b),
                gained_seeds=np.asarray(SEEDS)[b > a].tolist(),
                lost_seeds=np.asarray(SEEDS)[b < a].tolist(),
            )
        pairs[key] = comparisons
    stage_changes = {}
    for arm in ("base", "targeted"):
        changes = {}
        for before, after in zip(by_arm[arm], by_arm["retreat"]):
            source, dest = placement_stage(before), placement_stage(after)
            outcomes = changes.setdefault(source, {})
            outcomes[dest] = outcomes.get(dest, 0) + 1
        stage_changes[arm] = changes
    audit = dict(
        episodes=len(records),
        first_seed=SEEDS.start,
        last_seed=SEEDS.stop - 1,
        max_action_replay_error=max_action,
        max_start_error_mm=max_start,
        policies={k: totals(v) for k, v in by_arm.items()},
        paired=pairs,
        stage_changes_to_retreat=stage_changes,
        archive=archived_checks(run, previous),
    )
    write_csv(run / "paired-outcomes.csv", records)
    (run / "retreat-summary.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(audit["policies"], audit["archive"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "previous", "midbase"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(1)
    summarize(**vars(parser.parse_args()))
