"""Score strict and released stacks on the same physical rollout and test terminal placements."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.experiment import load_policies, policy_episode, verify_weights
from reward_study import digest, load
from summarize_failure_dsrl import longest_run

ARMS = ("base", "targeted", "retreat")
SEEDS = range(708, 808)


def masks(obs, aperture, speeds):
    """All arrays are post-command samples. Speeds are instantaneous PhysX magnitudes."""
    tool, red, blue = (obs[:, i : i + 3] * 0.1 for i in (0, 3, 6))
    clearance = np.linalg.norm(tool - red, axis=1)
    pose = (
        (np.linalg.norm(red[:, :2] - blue[:, :2], axis=1) < 0.012)
        & (abs(red[:, 2] - blue[:, 2] - 0.0468) < 0.008)
        & (blue[:, 2] > 0.014)
        & (blue[:, 2] < 0.032)
    )
    released = pose & (obs[:, 10] < 0.5) & (aperture > 0.035) & (speeds < 0.025).all(axis=1)
    return pose, released, released & (clearance > 0.05)


def results(obs, aperture, speeds):
    _, released, clear = masks(obs, aperture, speeds)
    return dict(
        strict=int(longest_run(clear) >= 8),
        released_ever=int(longest_run(released) >= 8),
        released_terminal=int(len(released) >= 8 and released[-8:].all()),
        max_released_ticks=longest_run(released),
    )


def save_physics(env, run, name, obs, aperture, policy_success):
    path = run / "physics" / (name + ".npz")
    path.parent.mkdir(exist_ok=True)
    speeds = np.asarray(env.speeds, dtype=np.float64)
    assert len(obs) == len(aperture) == len(speeds) + 1
    result = results(obs[1:], aperture[1:], speeds)
    assert result["strict"] == policy_success
    withdrawal_obs, withdrawal_aperture = [], []
    survived = None
    if result["released_terminal"]:
        state = obs[-1]
        for tick in range(32):
            # Withdraw vertically for up to 12 ticks, then hold still and open for one second.
            z = 0.4 if tick < 12 and state[2] - state[5] < 1.0 else 0.0
            state, _, _ = env.step(np.array([0, 0, z, -1], dtype=np.float32))
            withdrawal_obs.append(state)
            withdrawal_aperture.append(env.aperture)
        post = np.asarray(withdrawal_obs)
        pose, _, clear = masks(post, np.asarray(withdrawal_aperture), np.asarray(env.speeds)[len(speeds) :])
        survived = int(pose.all() and clear[-20:].all() and (post[-20:, 9] < 0.5).all())
    with path.open("xb") as f:
        np.savez_compressed(
            f,
            policy_speeds_mps=speeds,
            withdrawal_obs=np.asarray(withdrawal_obs).reshape(-1, 11),
            withdrawal_aperture_m=withdrawal_aperture,
            withdrawal_speeds_mps=np.asarray(env.speeds)[len(speeds) :],
        )
    return dict(**result, withdrawal_tested=int(survived is not None), survived_withdrawal=survived)


def archived(run, previous, env, first_only=False):
    with (previous / "paired-outcomes.csv").open() as f:
        seeds = sorted(
            int(r["seed"])
            for r in csv.DictReader(f)
            if r["arm"] == "targeted" and r["mode"] == "opened_centered_ended_in_pose_without_stable_stack"
        )
    assert len(seeds) == 19
    if first_only:
        seeds = seeds[:1]
    with (run / "archived-replays.jsonl").open("x") as stream:
        for seed in seeds:
            with np.load(previous / "traces" / f"targeted-{seed}.npz") as tr:
                controls, original, src_jaws = tr["actions"], tr["obs"], tr["aperture_m"]
            states = [env.reset(seed)]
            jaws = [env.aperture]
            saw_strict = False
            for action in controls:
                state, success, _ = env.step(action)
                states.append(state)
                jaws.append(env.aperture)
                saw_strict |= bool(success)
            states, jaws = np.asarray(states), np.asarray(jaws)
            name = f"archive-{seed}"
            with (run / "traces" / (name + ".npz")).open("xb") as f:
                np.savez_compressed(
                    f, obs=states, aperture_m=jaws, actions=controls, seed=seed, success=saw_strict
                )
            result = save_physics(env, run, name, states, jaws, saw_strict)
            xyz_error = np.linalg.norm((states[:, :9] - original[:, :9]).reshape(-1, 3, 3), axis=2) * 100
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        **result,
                        max_xyz_error_mm=float(xyz_error.max()),
                        max_jaw_error_mm=float(1000 * abs(jaws - src_jaws).max()),
                        max_grasp_flag_error=float(abs(states[:, 9] - original[:, 9]).max()),
                    )
                ),
                file=stream,
                flush=True,
            )


def render_archive_example(run, previous, env):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    example = run / "illustration"
    (example / "traces").mkdir(parents=True)
    video = run / "videos" / "archived-withdrawal.mp4"
    if video.exists():
        raise FileExistsError(video)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    with imageio.get_writer(video, fps=round(1 / env.env.unwrapped.step_dt)) as writer:

        def capture():
            frame = Image.fromarray(env.rgb())
            draw = ImageDraw.Draw(frame)
            phase = "Recorded policy commands"
            if env.steps > 200:
                phase = "Withdrawal test" if env.steps <= 212 else "Open-gripper hold"
            draw.rectangle((0, 0, frame.width, 38), fill=(21, 28, 32))
            draw.text((12, 6), phase, font=font, fill="white")
            writer.append_data(np.asarray(frame))

        env.frame_callback = capture
        try:
            archived(example, previous, env, first_only=True)
        finally:
            env.frame_callback = None


def score(run, first, count, env, video):
    models = {}
    source = json.loads((run / "training.json").read_text())
    assert digest(run / "selected-base.pt") == source["base_sha256"]
    base = load(run / "selected-base.pt")
    for arm in ARMS[1:]:
        checkpoint = run / (arm + ".pt")
        assert digest(checkpoint) == source[arm + "_sha256"]
        ckpt, pi, wm, sac = load_policies(checkpoint, env)
        assert ckpt["reward_mode"] == arm
        verify_weights(pi, base["policy"], arm + " common base")
        models[arm] = (wm, sac)
    models["base"] = (None, None)
    with (run / f"eval-{first}.jsonl").open("x") as stream:
        for seed in range(first, first + count):
            assert seed in SEEDS
            row = dict(seed=seed)
            offset = seed % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                name = f"{arm}-{seed}"
                clip = run / "videos" / (name + ".mp4") if video and seed == first else None
                trace = run / "traces" / (name + ".npz")
                metrics, start = policy_episode(env, pi, *models[arm], seed, clip, trace)
                if "start" in row:
                    np.testing.assert_allclose(start, row["start"], rtol=0, atol=0.01)
                else:
                    row["start"] = start.tolist()
                with np.load(trace) as tr:
                    checks = save_physics(env, run, name, tr["obs"], tr["aperture_m"], metrics["success"])
                row[arm] = {**metrics, **checks}
            print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--archived", type=Path)
    parser.add_argument("--first", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=args.video)
    try:
        from minidaydreamer.isaac import IsaacStack

        class RecordingStack(IsaacStack):
            def reset(self, seed):
                self.speeds = []
                return super().reset(seed)

            def step(self, action):
                result = super().step(action)
                self.speeds.append(self.cube_speeds_mps)
                return result

        env = RecordingStack(cameras=args.video)
        try:
            (args.run / "traces").mkdir(exist_ok=True)
            if args.archived and args.video:
                render_archive_example(args.run, args.archived, env)
            elif args.archived:
                archived(args.run, args.archived, env)
            else:
                score(args.run, args.first, args.count, env, args.video)
        finally:
            env.close()
    finally:
        app.app.close()
