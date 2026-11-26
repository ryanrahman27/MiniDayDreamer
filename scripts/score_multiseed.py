"""Matched physical scenes and noise streams for multi-seed released-stack evaluation."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from minidaydreamer.experiment import load_policies, policy_episode, verify_weights
from minidaydreamer.models import ACTION_DIM, HORIZON
from minidaydreamer.released_steering import PriorNoiseSAC
from minidaydreamer.stack import STEP
from multiseed_study import SCENE_SETS, RECIPES, arms
from reward_study import digest, load
from score_retreat import masks


def controllers(run, names):
    meta = json.loads((run / "protocol.json").read_text())
    assert (
        digest(run / "previous.pt") == meta["previous_sha256"]
        and digest(run / "selected-base.pt") == meta["base_sha256"]
    )
    ck, pi, wm, prior = load_policies(run / "previous.pt", SimpleNamespace(backend="isaac"))
    verify_weights(pi, load(run / "selected-base.pt")["policy"], "base")
    models = {"base": None, "previous": prior}
    if "previous_scaled" in names:
        control = PriorNoiseSAC(wm.features, prior, 0.0)
        control.actor.load_state_dict(ck["actor"])
        control.evaluation_gain = json.loads((run / "gain-selection.json").read_text())["gain_percent"] / 100
        models["previous_scaled"] = control
    for name in names:
        if name in models:
            continue
        checkpoint, _, scale = name.partition("@")
        c = load(run / (checkpoint + ".pt"))
        assert (
            digest(run / (checkpoint + ".pt"))
            == json.loads((run / (checkpoint + ".json")).read_text())["sha256"]
        )
        assert c["previous_sha256"] == meta["previous_sha256"] and checkpoint == f"{c['recipe']}-{c['seed']}"
        assert c["prior_weight"] == RECIPES[c["recipe"]]
        actor = PriorNoiseSAC(wm.features, prior, c["prior_weight"])
        actor.actor.load_state_dict(c["actor"])
        if scale:
            actor.evaluation_gain = int(scale) / 100
        elif c["seed"] not in (11, 29):
            winner = json.loads((run / "selection.json").read_text())["winner"]
            if c["recipe"] == winner:
                actor.evaluation_gain = (
                    json.loads((run / "gain-selection.json").read_text())["gain_percent"] / 100
                )
        models[name] = actor
    return pi, wm, models


def scene_bank(env, seeds, path):
    records = {}
    for seed in seeds:
        obs = env.reset(seed)
        state = env.env.unwrapped.scene.get_state(is_relative=True)
        physical = {
            kind: {
                asset: {k: v.detach().cpu().clone() for k, v in fields.items()}
                for asset, fields in assets.items()
            }
            for kind, assets in state.items()
        }
        records[seed] = (torch.from_numpy(obs.copy()), physical)
    with path.open("xb") as f:
        torch.save(records, f)


def reset_lanes(env, bank, seeds):
    raw = env.env.unwrapped
    sample = bank[seeds[0]][1]
    state = {
        kind: {
            asset: {
                key: torch.cat([bank[s][1][kind][asset][key] for s in seeds]).to(raw.device) for key in fields
            }
            for asset, fields in assets.items()
        }
        for kind, assets in sample.items()
    }
    obs, _ = raw.reset_to(state, seed=seeds[0], is_relative=True)
    return obs


def extract(env, obs, closed):
    p = obs["policy"]
    n = len(closed)
    origins = env.env.unwrapped.scene.env_origins
    cubes = p["cube_positions"].reshape(n, 3, 3) - origins[:, None, :]
    xyz = torch.stack((p["eef_pos"].reshape(n, 3), cubes[:, 1], cubes[:, 0]), dim=1)
    xyz = (xyz - xyz.new_tensor([0.5, 0, 0])) / 0.1
    state = torch.cat(
        (xyz.flatten(1), obs["subtask_terms"]["grasp_1"].reshape(n, 1).float(), closed[:, None].float()),
        dim=1,
    )
    aperture = p["gripper_pos"].reshape(n, -1).abs().amin(dim=1)
    return state.cpu(), aperture.cpu().numpy()


def summarize_trace(obs, jaws, speeds):
    _, released, _ = masks(obs[1:], jaws[1:], speeds)
    flags = np.r_[False, released, False].astype(int)
    lengths = np.flatnonzero(np.diff(flags) == -1) - np.flatnonzero(np.diff(flags) == 1)
    return dict(
        success=int(len(released) >= 8 and released[-8:].all()),
        ever=int(lengths.max(initial=0) >= 8),
        steps=len(released),
        terminal_ungrasped=bool((obs[-8:, 9] < 0.5).all()),
    )


@torch.no_grad()
def batch_episode(env, bank, seeds, pi, wm, actor, folder, arm, noise_offset=0):
    device = env.env.unwrapped.device
    n = len(seeds)
    observation = reset_lanes(env, bank, seeds)
    o, jaw = extract(env, observation, torch.zeros(n, device=device))
    initial = torch.stack([bank[s][0] for s in seeds])
    torch.testing.assert_close(o, initial, atol=0.01, rtol=0)
    noise_generators = [torch.Generator().manual_seed(s + noise_offset) for s in seeds]
    latent = wm.initial(o) if actor is not None else None
    states, jaws, velocities, commands = [o.numpy().copy()], [jaw.copy()], [], []
    lens = np.full(n, env.max_steps, dtype=int)
    for t in range(env.max_steps):
        if t % HORIZON == 0:
            noise = torch.cat([torch.randn(1, HORIZON, ACTION_DIM, generator=g) for g in noise_generators])
            if actor is not None:
                noise += actor.draw(latent, deterministic=True)[0][:, None]
            chunk = pi.sample(o, noise)
        action = chunk[:, t % HORIZON]
        cmd = action.clamp(-1, 1).to(device)
        closed = cmd[:, 3] > 0
        control = torch.zeros(n, 7, device=device)
        control[:, :3] = cmd[:, :3] * (STEP / env.scale)
        control[:, 6] = torch.where(closed, -1.0, 1.0)
        observation, _, term, trunc, _ = env.env.step(control)
        if (term | trunc).any().item():
            raise RuntimeError("Unexpected Isaac auto-reset in batch")
        o, jaw = extract(env, observation, closed)
        if actor is not None:
            latent, _ = wm.filter(wm.advance(latent, action), o)
        scene = env.env.unwrapped.scene
        speed = torch.stack(
            [scene[k].data.root_lin_vel_w.torch.norm(dim=1) for k in ("cube_1", "cube_2")], dim=1
        )
        states.append(o.numpy().copy())
        jaws.append(jaw.copy())
        velocities.append(speed.cpu().numpy().copy())
        commands.append(action.numpy().copy())
        fallen = (o[:, [5, 8]].amin(dim=1) < -0.5).numpy()
        lens[fallen & (lens == env.max_steps)] = t + 1
    obs, apertures, speeds, actions = map(np.stack, (states, jaws, velocities, commands))
    out = []
    for i, (seed, steps) in enumerate(zip(seeds, lens)):
        s, j, v, a = obs[: steps + 1, i], apertures[: steps + 1, i], speeds[:steps, i], actions[:steps, i]
        score = summarize_trace(s, j, v)
        with (folder / "traces" / f"{arm}-{seed}.npz").open("xb") as f:
            np.savez_compressed(
                f,
                obs=s,
                actions=a,
                aperture_m=j,
                cube_speeds_mps=v,
                seed=seed,
                noise_seed=seed + noise_offset,
                success=score["success"],
            )
        out.append(score)
    return out


def batch_score(env, run, split, first, count, width):
    names = arms(run, split)
    pi, wm, models = controllers(run, names)
    folder = run / split
    (folder / "traces").mkdir(exist_ok=True)
    bank_folder = run / {"gain": "dev", "noise": "test", "prior_gain": "test"}.get(split, split)
    bank = torch.load(bank_folder / "scenes.pt", weights_only=True)
    assert count % width == 0
    env.env.reset(seed=first)
    with (folder / f"eval-{first}.jsonl").open("x") as f:
        for offset in range(first, first + count, width):
            seeds = list(range(offset, offset + width))
            rows = [dict(seed=s, start=bank[s][0].tolist()) for s in seeds]
            rotate = offset % len(names)
            for name in names[rotate:] + names[:rotate]:
                values = batch_episode(
                    env, bank, seeds, pi, wm, models[name], folder, name, 1000003 if split == "noise" else 0
                )
                for row, value in zip(rows, values):
                    row[name] = value
            for row in rows:
                print(json.dumps(row), file=f, flush=True)


def scalar_score(env, run, first, count):
    names = arms(run, "scalar")
    pi, wm, models = controllers(run, names)
    folder = run / "scalar"
    (folder / "traces").mkdir(exist_ok=True)
    with (folder / f"eval-{first}.jsonl").open("x") as f:
        for seed in range(first, first + count):
            row = dict(seed=seed)
            offset = seed % len(names)
            for name in names[offset:] + names[:offset]:
                path = folder / "traces" / f"{name}-{seed}.npz"
                actor = models[name]
                metrics, start = policy_episode(
                    env, pi, wm if actor is not None else None, actor, seed, trace=path
                )
                if "start" in row:
                    np.testing.assert_allclose(start, row["start"], rtol=0, atol=0.01)
                else:
                    row["start"] = start.tolist()
                with np.load(path) as tr:
                    score = summarize_trace(tr["obs"], tr["aperture_m"], np.asarray(env.speeds))
                assert score["success"] == metrics["success"] and score["steps"] == metrics["steps"]
                with path.with_suffix(".speeds.npy").open("xb") as raw:
                    np.save(raw, np.asarray(env.speeds))
                row[name] = score
            print(json.dumps(row), file=f, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("bank", "batch", "scalar"))
    p.add_argument("run", type=Path)
    p.add_argument("split", choices=SCENE_SETS)
    p.add_argument("--first", type=int)
    p.add_argument("--count", type=int)
    p.add_argument("--width", type=int, default=1)
    args = p.parse_args()
    seeds = SCENE_SETS[args.split]
    if args.split != "dev":
        assert (args.run / "selection.json").is_file()
    if args.split in ("test", "noise", "scalar"):
        assert (args.run / "gain-selection.json").is_file()
    if args.mode != "bank":
        assert set(range(args.first, args.first + args.count)).issubset(seeds)
    torch.set_num_threads(1)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False)
    try:
        from minidaydreamer.isaac import IsaacStack

        class Measured(IsaacStack):
            def reset(self, seed):
                self.speeds = []
                return super().reset(seed)

            def step(self, action):
                output = super().step(action)
                self.speeds.append(self.cube_speeds_mps)
                return output

        env = Measured(cameras=False, released_at_finish=True, num_envs=args.width)
        try:
            if args.mode == "bank":
                scene_bank(env, seeds, args.run / args.split / "scenes.pt")
            elif args.mode == "batch":
                batch_score(env, args.run, args.split, args.first, args.count, args.width)
            else:
                scalar_score(env, args.run, args.first, args.count)
        finally:
            env.close()
    finally:
        app.app.close()
