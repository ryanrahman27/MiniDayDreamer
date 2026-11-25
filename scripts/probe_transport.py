"""Probe the frozen policy at recorded stalls with common noise and small changes to the state input."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.models import FlowPolicy, HORIZON, NoiseSAC, make_world
from minidaydreamer.stack import ACTION_DIM, STEP


def expert_carries(corpus):
    carries = []
    for i in np.flatnonzero(corpus["split"] == "demo"):
        n = corpus["lengths"][i]
        s = corpus["obs"][i, :n]
        a = corpus["actions"][i, :n]
        delta = s[:, 6:8] - s[:, 3:5]
        gap = np.linalg.norm(delta, axis=1)
        toward = (a[:, :2] * delta).sum(1) / gap.clip(1e-7)
        idx = (s[:, 9] > .5) & (s[:, 10] > .5) & (s[:, 5]-s[0, 5] > .2) & (gap > .12) & (toward > .4) & (a[:, 3] > 0)
        if idx.any():
            carries.append(s[idx])
    return carries


@torch.no_grad()
def run_probes(run, analysis, draws=512, short=None, long=None):
    with np.load(run / "corpus.npz") as corpus:
        carries = expert_carries(corpus)
    pi = FlowPolicy().eval()
    pi.load_state_dict(torch.load(run / "base.pt", weights_only=True)["policy"])
    controllers = {}
    if (short is None) != (long is None):
        raise ValueError("Specify both horizon checkpoint folders")
    if short is not None:
        for coverage in ("demos", "varied"):
            for depth, folder in ((6, short), (24, long)):
                c = torch.load(folder / f"{coverage}.pt", weights_only=True)
                if not all(torch.equal(v, c["policy"][k]) for k, v in pi.state_dict().items()):
                    raise ValueError("Noise probe needs identical bases")
                wm = make_world(c["world_model"]).eval()
                wm.load_state_dict(c["world"])
                sac = NoiseSAC(wm.features)
                sac.actor.load_state_dict(c["actor"])
                controllers[f"{coverage}_{depth}"] = (wm, sac)
    conditions = ("recorded", "demo_height", "demo_tool_offset", "demo_height_and_offset",
                  "farther_goal", "nearer_goal", "higher_10mm", "lower_10mm")
    rows = []
    with (analysis / "transport-events.csv").open() as f:
        stalls = [r for r in csv.DictReader(f) if r["policy"] == "base" and r["stall_tick"]]
    for r in stalls:
        seed, tick = int(r["seed"]), int(r["stall_tick"])
        with np.load(run / "traces" / f"base-{seed}.npz") as tr:
            if tick >= len(tr["actions"]):
                continue
            start = tick - tick % HORIZON
            history = torch.from_numpy(tr["obs"][:start+1])
            commands = torch.from_numpy(tr["actions"][:start])
            obs = history[-1].numpy().copy()
        direction = obs[6:8] - obs[3:5]
        direction /= np.linalg.norm(direction)
        # Match on horizontal cube and target locations only, one nearest state from each demo.
        nearest = []
        xy = [3, 4, 6, 7]
        for states in carries:
            d = np.linalg.norm(states[:, xy] - obs[xy], axis=1)
            j = d.argmin()
            nearest.append((float(d[j]), states[j]))
        best = sorted(nearest, key=lambda item: item[0])[:5]
        ref = np.stack([s for _, s in best])
        dz = np.median(ref[:, 5]) - obs[5]
        offset = np.median(ref[:, :3]-ref[:, 3:6], axis=0)
        variants = np.tile(obs, (len(conditions), 1))
        for j in (1, 3):
            variants[j, [2, 5]] += dz
        for j in (2, 3):
            variants[j, :3] = variants[j, 3:6] + offset
        variants[4, 6:8] += .1 * direction
        variants[5, 6:8] -= .1 * direction
        variants[6, [2, 5]] += .1
        variants[7, [2, 5]] -= .1
        generator = torch.Generator().manual_seed(87000 + seed)
        noise = torch.randn(draws, HORIZON, ACTION_DIM, generator=generator)
        scores = {}
        for name, state in zip(conditions, variants, strict=True):
            actions = pi.sample(torch.from_numpy(state)[None].expand(draws, -1), noise)[:, tick-start, :2]
            scores[name] = float((actions @ torch.from_numpy(direction)).mean() * STEP * 1000)
        # Condition every noise actor on the same recorded base-policy history.
        for name, (wm, sac) in controllers.items():
            state = wm.initial(history[:1])
            for j in range(start):
                state, _ = wm.filter(wm.advance(state, commands[j:j+1]), history[j+1:j+2])
            shift, _ = sac.draw(state, deterministic=True)
            actions = pi.sample(history[-1:].expand(draws, -1), noise + shift[:, None])[:, tick-start, :2]
            scores[name] = float((actions @ torch.from_numpy(direction)).mean() * STEP * 1000)
        rows.append(dict(seed=seed, tick=tick, height_adjustment_mm=float(dz*100),
                         demo_xy_match_mm=float(np.median([d for d, _ in best])*100), **scores))
    summary = dict(draws=draws, stalls_with_action=len(rows), rows=rows)
    if rows:
        summary["median_requested_mm_tick"] = {k: float(np.median([r[k] for r in rows])) for k in conditions}
        summary["median_paired_change_mm_tick"] = {
            k: float(np.median([r[k]-r["recorded"] for r in rows])) for k in conditions[1:]}
        summary["median_height_adjustment_mm"] = float(np.median([r["height_adjustment_mm"] for r in rows]))
        summary["median_demo_xy_match_mm"] = float(np.median([r["demo_xy_match_mm"] for r in rows]))
        if controllers:
            summary["median_steering_change_mm_tick"] = {
                k: float(np.median([r[k]-r["recorded"] for r in rows])) for k in controllers}
            summary["median_horizon_change_mm_tick"] = {
                a: float(np.median([r[a+"_24"]-r[a+"_6"] for r in rows])) for a in ("demos", "varied")}
    (analysis / "transport-probes.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--short", type=Path)
    parser.add_argument("--long", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run_probes(args.run, args.analysis or args.run, short=args.short, long=args.long)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
