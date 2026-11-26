"""Recheck clearance-limited probes with a withdrawal that ends at measured clearance."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from score_retreat import ARMS, masks, results


def candidates(run):
    for file in sorted([*run.glob("eval-*.jsonl"), *run.glob("archived-replays.jsonl")]):
        for line in file.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                continue
            scene = json.loads(line)
            seed = scene["seed"]
            trials = (
                [("archive", scene)] if file.name.startswith("archived") else [(a, scene[a]) for a in ARMS]
            )
            for arm, r in trials:
                if r["survived_withdrawal"] != 0:
                    continue
                name = f"{arm}-{seed}"
                with np.load(run / "physics" / (name + ".npz")) as f:
                    o = f["withdrawal_obs"]
                    pose, _, _ = masks(o, f["withdrawal_aperture_m"], f["withdrawal_speeds_mps"])
                    clear = np.linalg.norm(o[-20:, :3] - o[-20:, 3:6], axis=1) * 100 > 50
                if pose.all() and not clear.all():
                    yield name, seed


def recheck(run, dest, name, seed, env):
    with np.load(run / "traces" / (name + ".npz")) as f:
        original, actions = f["obs"], f["actions"]
    obs = [env.reset(seed)]
    jaws, speeds = [], []
    for action in actions:
        obs.append(env.step(action)[0])
        jaws.append(env.aperture)
        speeds.append(env.cube_speeds_mps)
    obs, jaws, speeds = np.asarray(obs), np.asarray(jaws), np.asarray(speeds)
    before = results(obs[1:], jaws, speeds)
    discrepancy = np.linalg.norm((obs[:, :9] - original[:, :9]).reshape(-1, 3, 3), axis=2) * 100
    state = obs[-1]
    post, opened, motion = [], [], []
    steps_out = 0
    acquired = False
    passed = None
    if before["released_terminal"]:

        def step(z):
            state = env.step(np.array([0, 0, z, -1], dtype=np.float32))[0]
            post.append(state)
            opened.append(env.aperture)
            motion.append(env.cube_speeds_mps)
            return state

        for _ in range(80):
            if np.linalg.norm(state[:3] - state[3:6]) * 100 > 70:
                break
            state = step(0.4)
            steps_out += 1
        acquired = bool(np.linalg.norm(state[:3] - state[3:6]) * 100 > 70)
        for _ in range(20):
            step(0.0)
        observed = np.asarray(post)
        pose, _, clear = masks(observed, np.asarray(opened), np.asarray(motion))
        passed = int(acquired and pose.all() and clear[-20:].all() and (observed[-20:, 9] < 0.5).all())
    with (dest / (name + ".npz")).open("xb") as f:
        np.savez_compressed(
            f,
            obs=obs,
            aperture_m=jaws,
            policy_speeds_mps=speeds,
            withdrawal_obs=np.asarray(post).reshape(-1, 11),
            withdrawal_aperture_m=opened,
            withdrawal_speeds_mps=np.asarray(motion).reshape(-1, 2),
        )
    return dict(
        name=name,
        seed=seed,
        **before,
        max_xyz_error_mm=float(discrepancy.max()),
        final_xyz_error_mm=float(discrepancy[-1].max()),
        steps_out=steps_out,
        acquired_70mm=int(acquired),
        survived=passed,
    )


def audit(run, dest):
    rows = [json.loads(line) for line in (dest / "adaptive.jsonl").read_text().splitlines()]
    assert {r["name"] for r in rows} == {name for name, _ in candidates(run)}
    assert len(rows) == len({r["name"] for r in rows})
    groups = {}
    for row in rows:
        name = row["name"]
        with np.load(dest / (name + ".npz")) as tr:
            before = results(tr["obs"][1:], tr["aperture_m"], tr["policy_speeds_mps"])
            assert all(before[k] == row[k] for k in before)
            post = tr["withdrawal_obs"]
            if before["released_terminal"]:
                assert len(post) == row["steps_out"] + 20
                pose, _, clear = masks(post, tr["withdrawal_aperture_m"], tr["withdrawal_speeds_mps"])
                last = post[row["steps_out"] - 1] if row["steps_out"] else tr["obs"][-1]
                acquired = int(np.linalg.norm(last[:3] - last[3:6]) * 100 > 70)
                assert acquired == row["acquired_70mm"]
                assert row["survived"] == int(
                    acquired and pose.all() and clear[-20:].all() and (post[-20:, 9] < 0.5).all()
                )
            else:
                assert len(post) == 0 and row["survived"] is None
        group = groups.setdefault(name.split("-")[0], dict(replayed=0, terminal_reproduced=0, survived=0))
        group["replayed"] += 1
        group["terminal_reproduced"] += row["released_terminal"]
        group["survived"] += int(row["survived"] or 0)
    report = dict(groups=groups, replays=rows)
    (dest / "adaptive-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument(
        "--follow",
        action="store_true",
        help="Follow live JSONL files until the scoring job writes score.exit",
    )
    args = p.parse_args()
    if args.audit_only:
        print(audit(args.run, args.output)["groups"])
        raise SystemExit
    if not args.follow:
        scored = sum(len(f.read_text().splitlines()) for f in args.run.glob("eval-*.jsonl"))
        if scored != 100 or len((args.run / "archived-replays.jsonl").read_text().splitlines()) != 19:
            p.error("Complete the 100-scene evaluation and 19 archived replays first, or use --follow")
    torch.set_num_threads(1)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False)
    try:
        from minidaydreamer.isaac import IsaacStack

        env = IsaacStack(cameras=False)
        try:
            args.output.mkdir(exist_ok=True)
            done = set()
            with (args.output / "adaptive.jsonl").open("x") as f:
                while True:
                    # Once the marker exists all primary files are flushed and immutable.
                    complete = not args.follow or (args.run / "score.exit").exists()
                    for name, seed in candidates(args.run):
                        if name in done:
                            continue
                        row = recheck(args.run, args.output, name, seed, env)
                        print(json.dumps(row), file=f, flush=True)
                        done.add(name)
                    if complete:
                        if args.follow:
                            assert (args.run / "score.exit").read_text().strip() == "0"
                        break
                    time.sleep(12)
            print(audit(args.run, args.output)["groups"])
        finally:
            env.close()
    finally:
        app.app.close()
