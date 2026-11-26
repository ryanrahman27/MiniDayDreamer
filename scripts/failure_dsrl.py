"""Train two matched noise-space SAC actors in the frozen RSSM and score four policies in Isaac."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from minidaydreamer.experiment import (
    fit_steering,
    load_policies,
    policy_episode,
    posterior_states,
    verify_weights,
)
from minidaydreamer.models import HORIZON, make_policy, make_world
from minidaydreamer.placement_reward import PlacementReward, placement_terms, relative_pose
from minidaydreamer.state_reward import StateReward
from reward_study import digest, load


ARMS = ("base", "original_direct", "matched_direct", "targeted")
SEEDS = range(608, 708)


@torch.no_grad()
def mixed_starts(prior, wm, data, demos):
    demo = posterior_states(wm, data, demos)
    pools = {"approach": [], "placement": []}
    source_seeds = []
    trace_hashes = {}
    for path in sorted((prior / "traces").glob("mid_base-*.npz")):
        with np.load(path) as tr:
            obs, actions = torch.from_numpy(tr["obs"]), torch.from_numpy(tr["actions"])
            seed = int(tr["seed"])
            source_seeds.append(seed)
            trace_hashes[seed] = digest(path)
            st = wm.initial(obs[:1])
            for t in range(len(actions)):
                if t % HORIZON == 0 and obs[t, 9] > 0.5 and obs[t, 10] > 0.5 and obs[t, 5] - obs[0, 5] > 0.2:
                    xy, _ = relative_pose(obs[t])
                    if xy < 80:
                        pools["placement" if xy < 20 else "approach"].append(st)
                st, _ = wm.filter(wm.advance(st, actions[t : t + 1]), obs[t + 1 : t + 2])
    assert sorted(source_seeds) == list(range(308, 408))
    gen = torch.Generator().manual_seed(17)
    parts = [demo]
    sample_counts = {}
    for name, states in pools.items():
        n = len(demo) // 2 if name == "approach" else len(demo) - len(demo) // 2
        indices = torch.randint(len(states), (n,), generator=gen)
        parts.append(torch.cat(states)[indices])
        sample_counts[name] = dict(available=len(states), sampled=n)
    return torch.cat(parts), dict(
        demo_states=len(demo),
        pool_counts=sample_counts,
        source_seeds=sorted(source_seeds),
        source_trace_sha256=trace_hashes,
    )


def test_terms():
    # A held cube 20 mm from the target and too high; all translations below are in mm.
    s = torch.tensor([[0.0, 0.0, 2.0, 0.2, 0.0, 1.6, 0.0, 0.0, 0.234, 1.0, 1.0]])

    def change(axis, mm):
        nxt = s.clone()
        nxt[:, axis] += mm / 100
        return placement_terms(s, nxt)

    assert change(5, -3)[0] > 0 and change(5, 3)[0] < 0
    assert change(3, -3)[1] > 0 and change(3, 3)[1] < 0
    open_state = s.clone()
    open_state[:, 9:11] = 0
    assert placement_terms(s, open_state)[3] < 0
    centered = s.clone()
    centered[:, 3] = 0
    centered[:, 5] = centered[:, 8] + 0.568
    opened = centered.clone()
    opened[:, 9:11] = 0
    terms = placement_terms(centered, opened)
    assert terms[2] > 0 and terms[3] == 0 and terms[0] == 0
    assert all(x == 0 for x in placement_terms(centered, centered))


def train(run, prior, archived):
    test_terms()
    orig = load(run / "mid-varied-direct.pt")
    base = load(run / "selected-base.pt")
    old_audit = json.loads((prior / "midbase-summary.json").read_text())
    assert digest(run / "mid-varied-direct.pt") == old_audit["actors"]["direct"]["sha256"]
    assert digest(run / "selected-base.pt") == old_audit["selection"]["checkpoint_sha256"]
    pi = make_policy(base["policy_kind"]).eval().requires_grad_(False)
    pi.load_state_dict(base["policy"])
    verify_weights(pi, orig["policy"], "frozen base")
    wm = make_world("residual").eval().requires_grad_(False)
    wm.load_state_dict(orig["world"])
    ref = load(archived / "varied.pt")
    verify_weights(wm, ref["world"], "frozen varied RSSM")
    data = Dataset(prior / "corpus.npz", 0)
    demos = np.flatnonzero(data.split == "demo")
    states, info = mixed_starts(prior, wm, data, demos)
    rng = ref["pre_actor_rng"]
    info.update(
        start_tensor_sha256=hashlib.sha256(states.numpy().tobytes()).hexdigest(),
        rng_sha256=hashlib.sha256(rng.numpy().tobytes()).hexdigest(),
        corpus_sha256=digest(prior / "corpus.npz"),
        base_sha256=digest(run / "selected-base.pt"),
        world_source_sha256=digest(run / "mid-varied-direct.pt"),
        actors={},
    )
    assert info["rng_sha256"] == orig["actor_rng_sha256"]
    for mode, reward_type in (("matched_direct", StateReward), ("targeted", PlacementReward)):
        reward = reward_type(data, demos)
        # Additional starts all have the gripper closed and a measured grasp, hence zero stable dwell.
        reward.starts = torch.cat(
            (reward.starts, torch.zeros(len(states) - len(reward.starts), dtype=torch.long))
        )
        assert len(reward.starts) == len(states)
        torch.set_rng_state(rng)
        actor = fit_steering(wm, pi, states, 1500, 2, reward)
        assert reward.ticks == 576000
        verify_weights(wm, orig["world"], "unchanged world")
        verify_weights(pi, base["policy"], "unchanged base")
        ckpt = dict(
            orig,
            actor=actor.actor.state_dict(),
            reward_mode=mode,
            direct_bonuses=reward.bonuses,
            start_tensor_sha256=info["start_tensor_sha256"],
            imagination_start_episodes=len(demos) + 100,
        )
        dest = run / f"{mode}.pt"
        torch.save(ckpt, dest)
        info["actors"][mode] = dict(sha256=digest(dest), ticks=reward.ticks, stable_bonuses=reward.bonuses)
        print(mode, info["actors"][mode], flush=True)
    (run / "training.json").write_text(json.dumps(info, indent=2) + "\n")


def score(run, first, count, video, env):
    pi = None
    controllers = {"base": (None, None)}
    base = load(run / "selected-base.pt")
    for arm in ARMS[1:]:
        filename = "mid-varied-direct.pt" if arm == "original_direct" else arm + ".pt"
        _, pi, wm, actor = load_policies(run / filename, env)
        verify_weights(pi, base["policy"], "common frozen base")
        controllers[arm] = (wm, actor)
    with (run / f"eval-{first}.jsonl").open("x") as stream:
        for seed in range(first, first + count):
            if seed not in SEEDS:
                raise ValueError("Seed outside the fresh test")
            row = {"seed": seed}
            offset = seed % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                wm, actor = controllers[arm]
                clip = run / "videos" / f"{arm}-{seed}.mp4" if video and seed == first else None
                metrics, start = policy_episode(
                    env, pi, wm, actor, seed, clip, run / "traces" / f"{arm}-{seed}.npz"
                )
                if "start" in row:
                    np.testing.assert_allclose(start, row["start"], atol=0.01, rtol=0)
                else:
                    row["start"] = start.tolist()
                row[arm] = metrics
            print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("train", "score"))
    p.add_argument("run", type=Path)
    p.add_argument("--prior", type=Path)
    p.add_argument("--archived", type=Path)
    p.add_argument("--first", type=int)
    p.add_argument("--count", type=int)
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.command == "train":
        train(a.run, a.prior, a.archived)
    else:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=a.video)
        try:
            from minidaydreamer.isaac import IsaacStack

            env = IsaacStack(cameras=a.video)
            try:
                score(a.run, a.first, a.count, a.video, env)
            finally:
                env.close()
        finally:
            app.app.close()
