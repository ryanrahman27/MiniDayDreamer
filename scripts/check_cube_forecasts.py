"""Evaluate frozen original and residual RSSMs on identical six-action forecast windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from minidaydreamer.models import HORIZON, make_world
from summarize_stack import cube_forecast


@torch.no_grad()
def bounded_history(weights, data, length):
    """Change only the amount of filtering history supplied to a frozen original model."""
    model = make_world("absolute").eval()
    model.load_state_dict(weights)
    horizon = 2 * HORIZON
    filtered, forecast = [], []
    for i in np.flatnonzero(data.split == "holdout"):
        ticks = torch.arange(0, int(data.lengths[i]) - horizon + 1, horizon)
        history = ticks.clamp(max=length)
        for n in history.unique().tolist():
            t = ticks[history == n]
            state = model.initial(data.obs[i, t-n])
            for j in range(n):
                state, _ = model.filter(model.advance(state, data.actions[i, t-n+j]), data.obs[i, t-n+j+1])
            filtered.append(100 * (model.decode(state)[:, 3:6] - data.obs[i, t, 3:6]).norm(dim=1))
            for j in range(horizon):
                state = model.imagine(state, data.actions[i, t+j])
            forecast.append(100 * (model.decode(state)[:, 3:6] - data.obs[i, t+horizon, 3:6]).norm(dim=1))
    return dict(max_history_steps=length, filtered_start_error_mm=torch.cat(filtered).mean().item(),
                mean_error_mm=torch.cat(forecast).mean().item())


def audit(run, original, first_test, count):
    fitting = Dataset(run / "corpus.npz", 0)
    confirm = Dataset(run / "forecast-check.npz", 0)
    base = torch.load(run / "base.pt", weights_only=True)
    calibration = {r["seed"] for c in base["calibration"] for r in c["outcomes"]}
    test = set(range(first_test, first_test + count))
    confirmation = set(confirm.seeds[confirm.split == "holdout"])
    if confirmation & (set(fitting.seeds) | calibration | test) or test & (calibration | set(fitting.seeds)):
        raise ValueError("Confirmation or policy-test scenes overlap an earlier split")
    result = {}
    for label, data in (("development", fitting), ("confirmation", confirm)):
        ids = np.flatnonzero(data.split == "holdout")
        row = {"episodes": len(ids), "seeds": sorted(map(int, set(data.seeds[ids]))), "models": {}}
        for version, folder in (("original", original), ("residual", run)):
            for arm in ("demos", "varied"):
                c = torch.load(folder / f"{arm}.pt", weights_only=True)
                metrics = cube_forecast(c["world"], data, c.get("world_model", "absolute"))
                if version == "original":
                    metrics["bounded_history"] = [bounded_history(c["world"], data, n) for n in (0, 8)]
                row["models"][f"{version}_{arm}"] = metrics
        result[label] = row
    (run / "forecast-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--first-test", type=int, required=True)
    parser.add_argument("--scenes", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.run, args.original, args.first_test, args.scenes), indent=2))
