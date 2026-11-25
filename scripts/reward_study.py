"""Train and score a paired frozen-base reward ablation in Isaac."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from minidaydreamer.experiment import fit_steering, load_policies, policy_episode, posterior_states, verify_weights
from minidaydreamer.models import HORIZON, make_policy, make_world
from minidaydreamer.state_reward import StateReward


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def augment(obs, actions):
    """Reflect and permute x/y together for the two cubes, tool and Cartesian commands."""
    n = len(obs)
    signs = torch.randint(2, (n, 1, 2), device=obs.device)*2-1
    swap = torch.rand(n, 1, 1, device=obs.device) < .5
    pos = obs[:, :9].reshape(n, 3, 3)
    def transform(xy):
        return signs * torch.where(swap, xy.flip(-1), xy)
    pos = torch.cat((transform(pos[:, :, :2]), pos[:, :, 2:]), dim=2).reshape(n, 9)
    act = torch.cat((transform(actions[:, :, :2]), actions[:, :, 2:]), dim=2)
    return torch.cat((pos, obs[:, 9:]), dim=1), act


def fit_bc(run):
    # The world-model dataset and the additional cloning-only data have separate files.
    obs, act, lens, seeds = [], [], [], []
    for name, expected in (("corpus.npz", 24), ("extra-bc.npz", 48)):
        with np.load(run / name) as d:
            ids = np.flatnonzero(d["split"] == "demo")
            assert len(ids) == expected and all(d["success"][i, d["lengths"][i]-1] for i in ids)
            obs.extend(d["obs"][ids]); act.extend(d["actions"][ids]); lens.extend(d["lengths"][ids]); seeds.extend(d["seeds"][ids])
    assert len(set(seeds)) == 72
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs = torch.tensor(np.stack(obs), device=device)
    act = torch.tensor(np.stack(act), device=device)
    lengths = torch.tensor(lens, device=device)
    torch.manual_seed(0)
    policy = make_policy("relative").to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=.001)
    batch = 128
    for step in range(1, 30001):
        ep = torch.randint(len(lengths), (batch,), device=device)
        tick = (torch.rand(batch, device=device) * (lengths[ep]-HORIZON+1)).long()
        o, a = augment(obs[ep, tick], act[ep[:, None], tick[:, None]+torch.arange(HORIZON, device=device)])
        loss = policy.loss(o, a)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5)
        opt.step()
        if step in (12000, 30000):
            torch.save(dict(policy={k: v.cpu() for k, v in policy.state_dict().items()}, policy_kind="relative",
                            bc_steps=step, demo_seeds=list(map(int, seeds)), source_sha256=digest(run/"extra-bc.npz")),
                       run/f"candidate-{step}.pt")
            print(f"Saved relative flow base after {step} updates", flush=True)


def fit_actors(args):
    run = args.run
    data = Dataset(run/"corpus.npz", 0)
    ids = np.flatnonzero(data.split == "demo")
    base = load(run/("base.pt" if args.group == "weak" else "selected-base.pt"))
    assert load(run/"base.pt")["corpus_sha256"] == digest(run/"corpus.npz")
    kind = base.get("policy_kind", "original")
    pi = make_policy(kind).eval().requires_grad_(False)
    pi.load_state_dict(base["policy"])
    for coverage in (("demos", "varied") if args.group == "weak" else ("varied",)):
        short = load(args.short/f"{coverage}.pt")
        ref = load(args.rng/f"{coverage}.pt")
        wm = make_world("residual").eval().requires_grad_(False)
        wm.load_state_dict(short["world"])
        verify_weights(wm, ref["world"], "shared RSSM")
        if args.group == "weak":
            verify_weights(pi, short["policy"], "original base")
        starts = posterior_states(wm, data, ids)
        for mode in ("learned", "direct"):
            direct = StateReward(data, ids) if mode == "direct" else None
            torch.set_rng_state(ref["pre_actor_rng"])
            actor = fit_steering(wm, pi, starts, 1500, 2, state_reward=direct)
            if args.group == "weak" and mode == "learned":
                verify_weights(actor.actor, short["actor"], "existing learned-reward actor")
            checkpoint = dict(short, policy=pi.state_dict(), policy_kind=kind, actor=actor.actor.state_dict(),
                              reward_mode=mode, imagination_chunks=2, rl_steps=1500, base_group=args.group,
                              imagined_ticks=576000, direct_bonuses=direct.bonuses if direct else None,
                              actor_rng_sha256=hashlib.sha256(ref["pre_actor_rng"].numpy().tobytes()).hexdigest())
            if direct and direct.ticks != checkpoint["imagined_ticks"]:
                raise ValueError("Imagined transition budget differs")
            torch.save(checkpoint, run/f"{args.group}-{coverage}-{mode}.pt")
            print(args.group, coverage, mode, checkpoint["direct_bonuses"], flush=True)


def calibrate(args, env):
    history = []
    for step in (12000, 30000):
        c = load(args.run/f"candidate-{step}.pt")
        pi = make_policy(c["policy_kind"]).eval()
        pi.load_state_dict(c["policy"])
        outcomes = [{"seed": seed, **policy_episode(env, pi, None, None, seed)[0]} for seed in range(5200, 5212)]
        count = sum(x["success"] for x in outcomes)
        history.append(dict(step=step, stacks=count, outcomes=outcomes))
        print(f"Candidate {step}: {count}/12 dev stacks", flush=True)
        if count >= 4:
            break
    selected = max(history, key=lambda row: row["stacks"])
    c = load(args.run/f"candidate-{selected['step']}.pt")
    c["calibration"] = history
    torch.save(c, args.run/"selected-base.pt")
    (args.run/"calibration.json").write_text(json.dumps(history, indent=2)+"\n")


def score(args, env):
    run, group = args.run, args.group
    base = load(run/("base.pt" if group == "weak" else "selected-base.pt"))
    pi = make_policy(base.get("policy_kind", "original")).eval()
    pi.load_state_dict(base["policy"])
    arms = [(group+"_base", None, None)]
    for coverage in (("demos", "varied") if group == "weak" else ("varied",)):
        for mode in ("learned","direct"):
            c, cp, wm, sac = load_policies(run/f"{group}-{coverage}-{mode}.pt", env)
            verify_weights(cp, pi.state_dict(), group+" frozen base")
            if c["reward_mode"] != mode or c["base_group"] != group or c["coverage"] != coverage:
                raise ValueError("Mislabelled score arm")
            arms.append((f"{group}_{coverage}_{mode}",wm,sac))
    with (run/f"eval-{group}-{args.first}.jsonl").open("x") as f:
        for seed in range(args.first, args.first+args.count):
            row = dict(seed=seed)
            off = seed % len(arms)
            for name, wm, sac in arms[off:]+arms[:off]:
                video = run/"videos"/f"{name}-{seed}.mp4" if args.video and seed == args.first else None
                metrics, start = policy_episode(env, pi, wm, sac, seed, video, run/"traces"/f"{name}-{seed}.npz")
                if "start" in row and not np.allclose(row["start"], start, atol=.01, rtol=0):
                    raise ValueError(f"Unequal starting scene {seed}")
                row["start"] = start.tolist()
                row[name] = metrics
            print(json.dumps(row), file=f, flush=True)
            print(f"Scored {group} scene {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bc","calibrate","actors","score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--group", choices=("weak","strong"), default="weak")
    parser.add_argument("--short", type=Path)
    parser.add_argument("--rng", type=Path)
    parser.add_argument("--first", type=int, default=258)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "bc":
        fit_bc(args.run)
    elif args.command == "actors":
        fit_actors(args)
    else:
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True, enable_cameras=args.video)
        try:
            from minidaydreamer.isaac import IsaacStack
            env = IsaacStack(cameras=args.video)
            try:
                (calibrate if args.command == "calibrate" else score)(args, env)
            finally:
                env.close()
        finally:
            launcher.app.close()


if __name__ == "__main__":
    main()
