"""Paired vertical-only intervention on previously stalled scenes. Run in an Isaac Python environment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(args):
    import torch
    from isaaclab.app import AppLauncher
    from minidaydreamer.experiment import policy_episode
    from minidaydreamer.models import FlowPolicy
    from minidaydreamer.transport import WINDOW
    from diagnose_transport import gap_mm, window_motion

    with args.events.open() as f:
        seeds = sorted(int(r["seed"]) for r in csv.DictReader(f) if r["policy"] == "base" and r["stall_tick"]
                       and int(r["stall_tick"]) <= 200-3*WINDOW)[:12]
    args.out.mkdir(parents=True, exist_ok=False)
    launcher = AppLauncher(headless=True, enable_cameras=False)
    try:
        from minidaydreamer.isaac import IsaacStack
        env = IsaacStack(cameras=False)
        try:
            torch.set_num_threads(2)
            policy = FlowPolicy().eval()
            policy.load_state_dict(torch.load(args.base, weights_only=True)["policy"])
            records = []
            for i, seed in enumerate(seeds):
                row = {"seed": seed}
                modes = ("base", "height") if i % 2 == 0 else ("height", "base")
                starts = []
                for mode in modes:
                    path = args.out / f"{mode}-{seed}.npz"
                    metrics, first = policy_episode(env, policy, None, None, seed, trace=path,
                                                    transport_pulse="height" if mode == "height" else None)
                    row[mode] = metrics
                    starts.append(first)
                if not np.allclose(*starts, rtol=0, atol=.01):
                    raise ValueError(f"Height intervention scene {seed} did not match")
                with np.load(args.out / f"height-{seed}.npz") as h, np.load(args.out / f"base-{seed}.npz") as b:
                    t, applied = int(h["pulse_trigger"]), int(h["pulse_steps"])
                    row.update(trigger=t, applied=applied)
                    if not np.array_equal(h["actions"][:, [0, 1, 3]], h["proposals"][:, [0, 1, 3]]):
                        raise ValueError("Vertical intervention changed another action component")
                    if t >= 0 and applied and t+2*WINDOW < min(len(h["obs"]), len(b["obs"])):
                        pos = (h["obs"][:t+1, :9]-b["obs"][:t+1, :9]).reshape(-1, 3)
                        prefix_mm = float(100 * np.linalg.norm(pos, axis=1).max())
                        row["matched"] = prefix_mm <= 2 and np.array_equal(h["obs"][:t+1, 9:], b["obs"][:t+1, 9:])
                        row["prefix_mm"] = prefix_mm
                        for mode, tr in (("base", b), ("height", h)):
                            obs = tr["obs"]
                            row[mode].update(
                                red_rise_mm=float(100*(obs[t+WINDOW, 5]-obs[t, 5])),
                                progress_24_mm=float(gap_mm(obs[t])-gap_mm(obs[t+2*WINDOW])),
                                postpulse_request_mm_tick=window_motion(obs, tr["actions"], t+2*WINDOW)["request_mm_tick"])
                records.append(row)
                print(f"Completed height pair {i+1}/{len(seeds)} on {seed}", flush=True)
                (args.out / "height-check.json").write_text(json.dumps(records, indent=2, allow_nan=False)+"\n")
        finally:
            env.close()
    finally:
        launcher.app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
