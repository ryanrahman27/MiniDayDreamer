"""Offline experiment CLI. Isaac users launch collect/eval through isaaclab.sh -p."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Expected a positive count")
    return n


def main():
    parser = argparse.ArgumentParser(prog="minidaydreamer")
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--demos", type=positive, default=24)
    c.add_argument("--extra", type=int, default=72)
    c.add_argument("--holdout", type=positive, default=12)
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--coverage", choices=["demos", "varied"], default="varied")
    for option, default in (("bc", 3000), ("model", 1800), ("rl", 1500)):
        t.add_argument(f"--{option}-steps", type=positive, default=default)
    e = sub.add_parser("eval")
    e.add_argument("--model", type=Path, required=True)
    e.add_argument("--episodes", type=positive, default=50)
    e.add_argument("--video-dir", type=Path, help="Record the first matched Isaac scene for each policy")
    v = sub.add_parser("preview")
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--seed", type=int, default=0)
    for p in (c, e):
        p.add_argument("--backend", choices=["isaac", "mujoco"], default="isaac")
    args = parser.parse_args()
    if getattr(args, "video_dir", None) and args.backend != "isaac":
        parser.error("--video-dir requires the Isaac camera")
    if args.command == "collect" and args.extra < 0:
        parser.error("--extra must be nonnegative")
    if hasattr(args, "out"):
        if args.out.exists():
            parser.error(f"Output already exists: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        env = None
        if args.command == "preview" or getattr(args, "backend", None) == "isaac":
            from isaaclab.app import AppLauncher
            launcher = AppLauncher(headless=True, enable_cameras=True)
            resources.callback(launcher.app.close)
            from .isaac import IsaacStack
            env = IsaacStack()
            resources.callback(env.close)
        elif getattr(args, "backend", None) == "mujoco":
            from .stack import CubeStack
            env = CubeStack()
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(getattr(args, "seed", 0))
        if args.command == "collect":
            from .data import collect
            collect(args.out, args.demos, args.extra, args.holdout, args.seed, env)
        elif args.command == "train":
            from .experiment import train
            train(args)
        elif args.command == "preview":
            import imageio.v2 as imageio
            from .stack import scripted_episode
            with imageio.get_writer(args.out, fps=round(1 / env.env.unwrapped.step_dt)) as video:
                env.frame_callback = lambda: video.append_data(env.rgb())
                *_, success = scripted_episode(env, args.seed)
            print(json.dumps({"success": bool(success[-1]), "frames": len(success), "seed": args.seed}))
        else:
            from .experiment import evaluate
            print(json.dumps(evaluate(args, env), indent=2))


if __name__ == "__main__":
    main()
