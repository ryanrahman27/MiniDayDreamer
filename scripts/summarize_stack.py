"""Audit the collected corpus and summarize a paired MiniDayDreamer run."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from minidaydreamer.models import HORIZON, RSSM


ARMS = ("base", "demos", "varied")


def paired(reference, treatment):
    wins = int(((reference == 0) & (treatment == 1)).sum())
    losses = int(((reference == 1) & (treatment == 0)).sum())
    discordant = wins + losses
    # Exact, two-sided McNemar test conditional on the discordant scenes.
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    return dict(gained=wins, lost=losses, both_succeeded=int((reference & treatment).sum()),
                difference_pp=100 * (wins - losses) / len(reference), exact_p=p)


@torch.no_grad()
def cube_forecast(model_weights, data):
    """Open-loop six-action forecast from recorded histories, including windows where the cube moves."""
    model = RSSM().eval()
    model.load_state_dict(model_weights)
    horizon = 2 * HORIZON
    errors, movements = [], []
    for i in np.flatnonzero(data.split == "holdout"):
        end = int(data.lengths[i])
        obs, actions = data.obs[i, :end+1], data.actions[i, :end]
        state = model.initial(obs[:1])
        starts, times = [], []
        for t in range(end):
            if t % horizon == 0 and t + horizon <= end:
                starts.append(state)
                times.append(t)
            state, _ = model.filter(model.advance(state, actions[t:t+1]), obs[t+1:t+2])
        if not times:
            continue
        imagined = torch.cat(starts)
        ticks = torch.tensor(times)
        for k in range(horizon):
            imagined = model.imagine(imagined, actions[ticks + k])
        actual = obs[ticks + horizon, 3:6]
        errors.append((model.decode(imagined)[:, 3:6] - actual).norm(dim=1) * 100)
        movements.append((obs[ticks, 3:6] - actual).norm(dim=1) * 100)
    err, motion = torch.cat(errors), torch.cat(movements)
    moving = motion >= 5
    return dict(horizon=horizon, windows=len(err), mean_error_mm=err.mean().item(),
                static_error_mm=motion.mean().item(), moving_windows=int(moving.sum()),
                moving_error_mm=err[moving].mean().item() if moving.any() else None,
                moving_static_error_mm=motion[moving].mean().item() if moving.any() else None)


def summarize(folder, expected_scenes):
    with np.load(folder / "corpus.npz", allow_pickle=False) as d:
        split, variants, seeds, lens = (d[key] for key in ("split", "variant", "seeds", "lengths"))
        success = d["success"][np.arange(len(lens)), lens-1].astype(int)
        counts = {}
        for group in np.unique(split):
            counts[group] = {}
            for kind in np.unique(variants[split == group]):
                mask = (split == group) & (variants == kind)
                counts[group][kind] = dict(episodes=int(mask.sum()), stacks=int(success[mask].sum()),
                                           transitions=int(lens[mask].sum()))
        partitions = [set(seeds[split == group]) for group in ("demo", "extra", "holdout")]
        if any(a & b for i, a in enumerate(partitions) for b in partitions[i+1:]):
            raise ValueError("Corpus splits overlap")
        training_seeds = set(seeds)
        first_eval_seed = int(seeds.max()) + 1
    base = torch.load(folder / "base.pt", map_location="cpu", weights_only=True)
    dataset_hash = hashlib.sha256((folder / "corpus.npz").read_bytes()).hexdigest()
    if dataset_hash != base["corpus_sha256"]:
        raise ValueError("Base policy checkpoint does not match the corpus")
    rows = sorted((json.loads(line) for p in folder.glob("eval-*.jsonl")
                   for line in p.read_text().splitlines() if line), key=lambda r: r["seed"])
    if [r["seed"] for r in rows] != list(range(first_eval_seed, first_eval_seed + expected_scenes)):
        raise ValueError("Incomplete or overlapping final evaluation seeds")
    eval_seeds = {r["seed"] for r in rows}
    calibration_seeds = {r["seed"] for c in base["calibration"] for r in c["outcomes"]}
    if (training_seeds & eval_seeds) or (calibration_seeds & (training_seeds | eval_seeds)):
        raise ValueError("Training, calibration and final evaluation scenes must be disjoint")
    wm = {}
    data = Dataset(folder / "corpus.npz", seed=0)
    for arm in ("demos", "varied"):
        c = torch.load(folder / f"{arm}.pt", map_location="cpu", weights_only=True)
        if c["policy"].keys() != base["policy"].keys() or any(
            not torch.equal(x, c["policy"][k]) for k, x in base["policy"].items()
        ):
            raise ValueError(f"Base weights differ in {arm}")
        wm[arm] = dict(world_episodes=c["world_episodes"], imagination_start_episodes=c["imagination_start_episodes"],
                       six_step_cube_forecast=cube_forecast(c["world"], data), **c["diagnostics"])
    stats = {}
    for arm in ARMS:
        values = [r[arm] for r in rows]
        stats[arm] = dict(stacks=sum(v["success"] for v in values), lifts=sum(v["lifted"] for v in values),
                          scenes=len(rows), mean_return=float(np.mean([v["episode_return"] for v in values])),
                          median_final_xy_mm=float(np.median([v["final_xy_error_mm"] for v in values])))
    scores = {arm: np.array([r[arm]["success"] for r in rows]) for arm in ARMS}
    comparison = {f"{b}_vs_{a}": paired(scores[a], scores[b])
                  for a, b in (("base", "demos"), ("base", "varied"), ("demos", "varied"))}
    with (folder / "paired-outcomes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "policy", *rows[0]["base"]])
        writer.writeheader()
        for r in rows:
            for arm in ARMS:
                writer.writerow(dict(seed=r["seed"], policy=arm, **r[arm]))
    summary = dict(corpus_sha256=dataset_hash, corpus=counts, bc_updates=base["bc_steps"],
                   calibration=base["calibration"], test_seeds=[min(eval_seeds), max(eval_seeds)],
                   world_models=wm, policies=stats, paired=comparison)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = summarize(args.run, args.episodes)
    print(json.dumps({k: result[k] for k in ("policies", "paired", "world_models")}, indent=2))
