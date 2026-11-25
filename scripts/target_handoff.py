"""Score the frozen base, always-on direct steering and an over-target handoff in Isaac."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.experiment import load_policies, policy_episode, verify_weights
from reward_study import digest, load


ARMS = ("base", "direct", "handoff")
SEEDS = range(408, 508)


def score(run, prior, first, count, cameras, env):
    audit = json.loads((prior / "midbase-summary.json").read_text())
    actor_path = run / "mid-varied-direct.pt"
    base_path = run / "selected-base.pt"
    if digest(actor_path) != audit["actors"]["direct"]["sha256"] or digest(base_path) != audit["selection"]["checkpoint_sha256"]:
        raise ValueError("Evaluation requires the frozen checkpoints from the matched 100-scene study")
    c, pi, wm, actor = load_policies(actor_path, env)
    verify_weights(pi, load(base_path)["policy"], "development-selected baseline")
    if (c["base_checkpoint_sha256"], c["coverage"], c["reward_mode"]) != (digest(base_path), "varied", "direct"):
        raise ValueError("Unexpected steering checkpoint")
    pi.eval()
    wm.eval()
    with (run / f"eval-{first}.jsonl").open("x") as stream:
        for seed in range(first, first + count):
            if seed not in SEEDS:
                raise ValueError("Seed outside the specified fresh test set")
            row = {"seed": seed}
            offset = seed % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                video = run / "videos" / f"{arm}-{seed}.mp4" if cameras and seed == first else None
                metrics, start = policy_episode(env, pi, wm if arm != "base" else None,
                                                actor if arm != "base" else None, seed, video,
                                                run / "traces" / f"{arm}-{seed}.npz",
                                                handoff_at_target=arm == "handoff")
                if "start" in row:
                    np.testing.assert_allclose(row["start"], start, atol=.01, rtol=0)
                else:
                    row["start"] = start.tolist()
                row[arm] = metrics
            print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=args.video)
    try:
        from minidaydreamer.isaac import IsaacStack
        task = IsaacStack(cameras=args.video)
        try:
            score(args.run, args.prior, args.first, args.count, args.video, task)
        finally:
            task.close()
    finally:
        app.app.close()
