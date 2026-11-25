"""Resample the frozen flow policy at its recorded first off-target opening commands."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_stack import release_events
from minidaydreamer.models import FlowPolicy, HORIZON
from minidaydreamer.stack import ACTION_DIM


@torch.no_grad()
def analyze(run, draws=512):
    policy = FlowPolicy().eval()
    policy.load_state_dict(torch.load(run / "base.pt", weights_only=True)["policy"])
    generator = torch.Generator().manual_seed(731)
    rows = []
    for path in sorted((run / "traces").glob("base-*.npz")):
        with np.load(path) as d:
            events = release_events(d["obs"], d["actions"], d["proposed_grip"], d["aperture_m"])
            if events["opening_category"] not in ("first_open_off_center", "first_open_at_wrong_height"):
                continue
            seed = int(d["seed"])
            t = events["first_open_step"]
            chunk, offset = divmod(t, HORIZON)
            obs = torch.from_numpy(d["obs"][chunk * HORIZON])[None]
            # The policy reads the observation only at chunk boundaries.
            replay_rng = torch.Generator().manual_seed(seed)
            for _ in range(chunk + 1):
                original_noise = torch.randn(1, HORIZON, ACTION_DIM, generator=replay_rng)
            actual = float(d["actions"][t, 3])
            replay = policy.sample(obs, original_noise)[0, offset, 3].item()
            if abs(actual - replay) > 1e-4:
                raise ValueError(f"Original gripper proposal did not replay on scene {seed}")
            noise = torch.randn(draws, HORIZON, ACTION_DIM, generator=generator)
            grip = policy.sample(obs.expand(draws, -1), noise)[:, offset, 3]
            rows.append(dict(seed=seed, opening_step=t, chunk_index=offset, executed_grip=actual,
                             fraction_open=(grip <= 0).float().mean().item()))
    probs = [row["fraction_open"] for row in rows]
    result = dict(draws_per_state=draws, eligible_scenes=len(rows), rows=rows,
                  median_fraction_open=float(np.median(probs)) if probs else None,
                  majority_open=sum(p > .5 for p in probs))
    (run / "gripper-noise.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    torch.set_num_threads(2)
    print(json.dumps(analyze(parser.parse_args().run), indent=2))
