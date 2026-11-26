"""Evaluate the incumbent, matched residuals and frozen base on identical Isaac resets."""

import argparse
import json
from pathlib import Path

import torch

from margin_study import SCENES, STREAMS, controllers_for
from score_multiseed import batch_episode, scene_bank


def score(env, run, old, split, first, count, width):
    pi, wm, models = controllers_for(run, old)
    names = list(models)
    folder = run / split
    bank = torch.load(run / "scenes.pt", weights_only=True)
    assert set(bank) == set(SCENES) and set(range(first, first + count)).issubset(bank)
    assert count % width == 0
    env.env.reset(seed=first)
    with (folder / f"eval-{first}.jsonl").open("x") as f:
        for offset in range(first, first + count, width):
            seeds = list(range(offset, offset + width))
            rows = [dict(seed=s, start=bank[s][0].tolist()) for s in seeds]
            rotate = ((offset - SCENES.start) // width + int(split == "noise")) % len(names)
            for name in names[rotate:] + names[:rotate]:
                values = batch_episode(env, bank, seeds, pi, wm, models[name], folder, name, STREAMS[split])
                for row, value in zip(rows, values):
                    row[name] = value
            for row in rows:
                print(json.dumps(row), file=f, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("bank", "score"))
    p.add_argument("run", type=Path)
    p.add_argument("old", type=Path)
    p.add_argument("--split", choices=STREAMS, default="test")
    p.add_argument("--first", type=int, default=SCENES.start)
    p.add_argument("--count", type=int, default=len(SCENES))
    p.add_argument("--width", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(1)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=False)
    try:
        from minidaydreamer.isaac import IsaacStack

        env = IsaacStack(
            cameras=False, released_at_finish=True, num_envs=1 if args.mode == "bank" else args.width
        )
        try:
            if args.mode == "bank":
                scene_bank(env, SCENES, args.run / "scenes.pt")
            else:
                score(env, args.run, args.old, args.split, args.first, args.count, args.width)
        finally:
            env.close()
    finally:
        app.app.close()
