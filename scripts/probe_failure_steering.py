"""Compare the trained shifts at identical held-out carry states and stochastic RSSM futures."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from minidaydreamer.experiment import load_policies
from minidaydreamer.models import ACTION_DIM, HORIZON
from minidaydreamer.placement_reward import placement_terms, relative_pose
from minidaydreamer.stack import STEP
from minidaydreamer.state_reward import transition


def carry_bands(obs):
    xy, height = relative_pose(obs)
    held = (obs[:, 9] > 0.5) & (obs[:, 10] > 0.5) & (obs[:, 5] - obs[0, 5] > 0.2)
    return {
        "elevated_approach": held & (xy < 40) & (height > 30),
        "ready_to_release": held & (xy < 12) & (height > -8) & (height < 25),
    }


def observed_windows(obs):
    """Measure the same two phases along an actor's actual Isaac trajectory."""
    obs = torch.from_numpy(obs)
    _, height = relative_pose(obs)
    results = {}
    for name, mask in carry_bands(obs).items():
        # Require all six subsequent control ticks to be recorded.
        starts = torch.arange(0, max(0, len(obs) - 2 * HORIZON), HORIZON)
        starts = starts[mask[starts]]
        future = starts[:, None] + torch.arange(1, 2 * HORIZON + 1)
        opened = (obs[future, 10] < 0.5).any(dim=1)
        results[name] = dict(
            windows=len(starts),
            opening_windows=int(opened.sum()),
            total_height_change_mm=float((height[starts + 2 * HORIZON] - height[starts]).sum()),
        )
    return results


@torch.no_grad()
def probe(run, confirmation):
    actors = {"base": None}
    for arm, name in (
        ("original_direct", "mid-varied-direct"),
        ("matched_direct", "matched_direct"),
        ("targeted", "targeted"),
    ):
        _, pi, wm, actors[arm] = load_policies(run / (name + ".pt"), SimpleNamespace(backend="isaac"))
    banks = {"elevated_approach": [], "ready_to_release": []}
    for path in sorted((confirmation / "traces").glob("base-*.npz")):
        with np.load(path) as tr:
            o = torch.from_numpy(tr["obs"])
            a = torch.from_numpy(tr["actions"])
            bands = carry_bands(o)
            state = wm.initial(o[:1])
            for t in range(len(a)):
                if t % HORIZON == 0:
                    for label, eligible in bands.items():
                        if eligible[t]:
                            banks[label].append(state)
                state, _ = wm.filter(wm.advance(state, a[t : t + 1]), o[t + 1 : t + 2])
    output = {}
    for label, states in banks.items():
        initial = torch.cat(states)
        output[label] = {"histories": len(initial), "policies": {}}
        for arm, actor in actors.items():
            # Restore the same random draws per state and tick in every arm.
            torch.manual_seed(73)
            st = initial
            original = wm.decode(st)
            dwell = torch.zeros(len(st), dtype=torch.long)
            total = torch.zeros(len(st))
            extra = torch.zeros_like(total)
            discount = torch.ones_like(total)
            opened = torch.zeros(len(st), dtype=torch.bool)
            strength = torch.zeros_like(total)
            first_z = None
            for chunk in range(2):
                shift = (
                    torch.zeros(len(st), ACTION_DIM)
                    if actor is None
                    else actor.draw(st, deterministic=True)[0]
                )
                strength += shift.abs().mean(1) / 2
                noise = torch.randn(len(st), HORIZON, ACTION_DIM) + shift[:, None]
                commands = pi.sample(wm.decode(st), noise)
                if chunk == 0:
                    first_z = float((commands[:, 0, 2] * STEP * 1000).mean())
                for action in commands.unbind(1):
                    prev = wm.decode(st)
                    st = wm.imagine(st, action, sample=True)
                    nxt = wm.decode(st)
                    direct, dwell, _ = transition(prev, nxt, dwell)
                    total += discount * direct
                    extra += discount * sum(placement_terms(prev, nxt))
                    opened |= (prev[:, 10] > 0.5) & (nxt[:, 10] < 0.5)
                    discount *= 0.99 * (1 - wm.success(st)[:, 0].sigmoid())
            xy1, z1 = relative_pose(wm.decode(st))
            xy0, z0 = relative_pose(original)
            output[label]["policies"][arm] = dict(
                mean_first_vertical_command_mm=first_z,
                mean_six_tick_height_change_mm=float((z1 - z0).mean()),
                mean_six_tick_xy_error_change_mm=float((xy1 - xy0).mean()),
                open_fraction=float(opened.float().mean()),
                mean_abs_shift=float(strength.mean()),
                mean_direct_return=float(total.mean()),
                mean_targeted_return=float((total + extra).mean()),
            )
    (run / "model-probe.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--confirmation", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    probe(a.run, a.confirmation)
