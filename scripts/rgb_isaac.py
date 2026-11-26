"""Collect time-aligned two-camera Isaac demonstrations and score paired RGB controllers."""

import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
from score_multiseed import extract, summarize_trace

from minidaydreamer.stack import STEP


def pixels(obs):
    views = [obs["policy"][name][..., :3] for name in ("table_cam", "wrist_cam")]
    if any(v.dtype != torch.uint8 for v in views):
        raise ValueError(f"Expected raw RGB bytes, got {[v.dtype for v in views]}")
    return torch.cat(views, dim=-1).permute(0, 3, 1, 2).contiguous()


def step(env, action):
    cmd = action.clamp(-1, 1).to(env.env.unwrapped.device)
    control = cmd.new_zeros(len(cmd), 7)
    control[:, :3] = cmd[:, :3] * (STEP / env.scale)
    control[:, 6] = torch.where(cmd[:, 3] > 0, -1.0, 1.0)
    obs, _, term, trunc, _ = env.env.step(control)
    if (term | trunc).any().item():
        raise RuntimeError("Isaac reset inside an episode")
    return obs


def speeds(env):
    scene = env.env.unwrapped.scene
    values = [scene[name].data.root_lin_vel_w for name in ("cube_1", "cube_2")]
    return (
        torch.stack([getattr(v, "torch", v).norm(dim=1) for v in values], 1)
        .cpu()
        .numpy()
    )


def targets(o, variants, rng):
    red, blue = o[:, 3:6] * 0.1, o[:, 6:9] * 0.1
    grasp, place = red.copy(), blue + [0, 0, 0.0488]
    grasp[variants == "miss", 0] += 0.027
    offset = variants == "offset"
    place[offset, :2] += rng.choice([-1, 1], (sum(offset), 2)) * rng.uniform(
        0.020, 0.043, (sum(offset), 2)
    )
    place[variants == "drop"] += [0, 0.045, 0.055]
    return np.stack(
        [
            grasp + [0, 0, 0.065],
            grasp,
            grasp,
            grasp + [0, 0, 0.095],
            place + [0, 0, 0.06],
            place,
            place,
            place + [0, 0, 0.08],
        ],
        1,
    )


def collect(env, args):
    dest = args.run / "episodes"
    dest.mkdir(exist_ok=True)
    n = args.width
    for split, required, offset in (
        ("demo", 72, 210000),
        ("extra", 72, 310000),
        ("valid", 16, 410000),
    ):
        kept = len(list(dest.glob(f"{split}-*.npz")))
        for batch in range(60):
            if kept >= required:
                break
            initial = offset + batch * n
            marker = dest / f"{split}-{initial}.done"
            if marker.exists():
                continue
            rng = np.random.default_rng(initial)
            variants = np.array(
                [
                    "demo"
                    if split == "demo"
                    else ("demo", "miss", "offset", "drop")[i % 4]
                    for i in range(n)
                ]
            )
            obs, _ = env.env.reset(seed=initial)
            closed = torch.zeros(n, device=env.env.unwrapped.device)
            o, jaw = extract(env, obs, closed)
            goals = targets(o.numpy(), variants, rng)
            phase, dwell = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
            states, jaws, frames = [o.numpy()], [jaw], [pixels(obs).cpu().numpy()]
            actions, vels = [], []
            for tick in range(200):
                delta = (goals[np.arange(n), phase] - o.numpy()[:, :3] * 0.1) / STEP
                action = np.column_stack(
                    [
                        np.clip(delta, -1, 1) * rng.uniform(0.70, 0.98, (n, 1)),
                        np.where((phase >= 2) & (phase <= 5), 1.0, -1.0),
                    ]
                ).astype(np.float32)
                cmd = torch.from_numpy(action)
                obs = step(env, cmd)
                o, jaw = extract(env, obs, cmd[:, 3].to(closed.device) > 0)
                dwell += (np.linalg.norm(delta, axis=1) < 0.22) & (phase < 7)
                advance = (dwell >= np.where((phase == 2) | (phase == 6), 7, 1)) & (
                    phase < 7
                )
                phase += advance
                dwell[advance] = 0
                states.append(o.numpy())
                jaws.append(jaw)
                actions.append(action)
                vels.append(speeds(env))
                frames.append(pixels(obs).cpu().numpy())
            states, jaws, actions, vels, frames = map(
                np.stack, (states, jaws, actions, vels, frames)
            )
            for i in range(n):
                success = summarize_trace(states[:, i], jaws[:, i], vels[:, i])[
                    "success"
                ]
                path = dest / f"{split}-{initial + i}.npz"
                if (
                    path.exists()
                    or kept >= required
                    or (split == "demo" and not success)
                ):
                    continue
                if min(frames[::10, i, :3].std(), frames[::10, i, 3:].std()) < 5:
                    raise ValueError("Empty camera view")
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.savez(
                        f,
                        rgb=frames[:, i],
                        obs=states[:, i],
                        actions=actions[:, i],
                        success=success,
                        speeds=vels[:, i],
                        aperture=jaws[:, i],
                        variant=variants[i],
                        batch_seed=initial,
                        lane=i,
                    )
                tmp.replace(path)
                kept += 1
            marker.touch()
            print(f"{split} {kept}/{required} after batch {batch}", flush=True)
        if kept != required:
            raise RuntimeError(f"Insufficient {split} episodes: {kept}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("collect", "score"))
    p.add_argument("run", type=Path)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--first-batch", type=int, default=0)
    p.add_argument("--stop-batch", type=int)
    p.add_argument("--split", choices=("dev", "test"), default="test")
    p.add_argument("--label", default="final")
    p.add_argument(
        "--failure-study", choices=("replay", "fresh", "release", "wait", "calibrated")
    )
    p.add_argument("--recorded-resets", type=Path,
                   help="Read bank-*.pt and reference-*.npz from a previous evaluation")
    args = p.parse_args()
    torch.set_num_threads(2)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True)
    exit_code = 1
    try:
        from minidaydreamer.isaac import IsaacStack

        env = IsaacStack(released_at_finish=True, num_envs=args.width, rgb_size=96)
        try:
            if args.mode == "collect":
                collect(env, args)
            else:
                from rgb_score import score

                score(env, args)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            env.close()
        exit_code = 0
    finally:
        app.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
