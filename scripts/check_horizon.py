"""Score six- and 24-tick open-loop forecasts on exactly the same independent recorded histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from minidaydreamer.models import make_world


@torch.no_grad()
def check(path, data):
    c = torch.load(path, weights_only=True)
    wm = make_world(c["world_model"]).eval()
    wm.load_state_dict(c["world"])
    errors = {k: [] for k in (6, 24)}
    for i in np.flatnonzero(data.split == "holdout"):
        n = int(data.lengths[i])
        obs = data.obs[i, :n+1]
        ticks = torch.arange(0, n-24+1, 6)
        if not len(ticks):
            continue
        state = wm.initial(obs[:1])
        starts = []
        for t in range(int(ticks[-1])+1):
            if t % 6 == 0:
                starts.append(state)
            state, _ = wm.filter(wm.advance(state, data.actions[i, t:t+1]), obs[t+1:t+2])
        state = torch.cat(starts)
        origin = obs[ticks]
        carrying = (origin[:, 9] > .5) & (origin[:, 5]-obs[0, 5] > .2)
        predicted_return = torch.zeros(len(ticks))
        observed_return = torch.zeros(len(ticks))
        for t in range(24):
            state = wm.imagine(state, data.actions[i, ticks+t])
            predicted_return += .99**t * wm.reward(state)[:, 0]
            observed_return += .99**t * data.rewards[i, ticks+t]
            if t+1 not in errors:
                continue
            truth = obs[ticks+t+1]
            p = wm.decode(state)
            cube_error = 100 * (p[:, 3:6]-truth[:, 3:6]).norm(dim=1)
            stationary = 100 * (origin[:, 3:6]-truth[:, 3:6]).norm(dim=1)
            errors[t+1].append(torch.stack((cube_error, stationary, carrying.float(), predicted_return, observed_return), 1))
    result = {}
    for h, parts in errors.items():
        e, s, held, pred_r, real_r = torch.cat(parts).T
        mask = held.bool()
        result[h] = dict(windows=len(e), error_mm=e.mean().item(), stationary_mm=s.mean().item(),
                         carrying_windows=int(mask.sum()), carrying_mm=e[mask].mean().item(),
                         carrying_stationary_mm=s[mask].mean().item(), return_mae=(pred_r-real_r).abs().mean().item(),
                         predicted_mean_return=pred_r.mean().item(), measured_mean_return=real_r.mean().item(),
                         zero_return_mae=real_r.abs().mean().item(),
                         return_correlation=float(np.corrcoef(pred_r, real_r)[0, 1]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--diagnostic-data", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = Dataset(args.diagnostic_data, 0)
    scores = {a: check(args.run / f"{a}.pt", data) for a in ("demos", "varied")}
    scores["diagnostic_seeds"] = sorted(map(int, set(data.seeds[data.split == "holdout"])))
    text = json.dumps(scores, indent=2, allow_nan=False)
    (args.run / "horizon-forecast.json").write_text(text+"\n")
    print(text)
