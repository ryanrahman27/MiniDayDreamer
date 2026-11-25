"""Compare state-derived and learned rewards on recorded Isaac trajectories and frozen RSSM forecasts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from minidaydreamer.models import make_world
from minidaydreamer.state_reward import StateReward, dwell_history, potential, transition


def observed_report(episodes):
    table = dict(tp=0, fp=0, fn=0, tn=0, exact_terminal_tick=0)
    timings, errors = [], []
    for obs, reward, success in episodes:
        counts = dwell_history(obs)
        hits = torch.where((counts[1:] == 8) & (counts[:-1] < 8))[0]
        pred = int(hits[0]) if len(hits) else None
        actual = torch.where(success > .5)[0]
        truth = int(actual[0]) if len(actual) else None
        key = ("t" if (pred is None) == (truth is None) else "f") + ("p" if pred is not None else "n")
        # A missed success is a false negative; a spurious stack is a false positive.
        table[key] += 1
        if pred is not None and truth is not None:
            table["exact_terminal_tick"] += int(pred == truth)
            timings.append(pred-truth)
        if reward is not None:
            dense = torch.diff(potential(obs))
            errors.append(float((dense - (reward - 2 * success)).abs().max()))
    table.update(episodes=sum(table[k] for k in ("tp","fp","fn","tn")),
                 event_tick_errors=timings, max_dense_error=max(errors) if errors else None)
    return table


def from_dataset(data, ids):
    for i in ids:
        n = int(data.lengths[i])
        yield data.obs[i,:n+1], data.rewards[i,:n], data.success[i,:n]


def from_traces(folder):
    for path in sorted(folder.glob("*.npz")):
        with np.load(path) as d:
            obs = torch.from_numpy(d["obs"])
            success = torch.zeros(len(obs)-1)
            success[-1] = int(d["success"])
            yield obs, None, success


def recorded_windows(data, ids):
    for i in ids:
        n = int(data.lengths[i])
        yield data.obs[i,:n+1], data.actions[i,:n], data.rewards[i,:n], data.success[i,:n]


def policy_recordings(run, arm):
    for path in sorted((run / "traces").glob(f"{arm}-*.npz")):
        with np.load(path) as d:
            obs = torch.from_numpy(d["obs"])
            actions = torch.from_numpy(d["actions"])
            success = torch.zeros(len(actions))
            success[-1] = int(d["success"])
            yield obs, actions, torch.diff(potential(obs)) + 2*success, success


@torch.no_grad()
def forecast(episodes, model, horizon):
    rows = []
    for obs, actions, rewards, success in episodes:
        n = len(actions)
        ticks = sorted(set(range(0, n-horizon+1, 6)) | {n-horizon})
        if ticks[0] < 0:
            continue
        wanted = set(ticks)
        st = model.initial(obs[:1])
        starts = []
        for t in range(ticks[-1]+1):
            if t in wanted:
                starts.append(st)
            st, _ = model.filter(model.advance(st, actions[t:t+1]), obs[t+1:t+2])
        ts = torch.tensor(ticks)
        state = torch.cat(starts)
        count = dwell_history(obs)[ts]
        learned = torch.zeros(len(ticks))
        direct = torch.zeros_like(learned)
        truth = torch.zeros_like(learned)
        real_hit = torch.zeros(len(ticks), dtype=torch.bool)
        pred_hit = real_hit.clone()
        for t in range(horizon):
            before = model.decode(state)
            state = model.imagine(state, actions[ts+t])
            r, count, hit = transition(before, model.decode(state), count)
            learned += .99**t * model.reward(state)[:,0]
            direct += .99**t * r
            truth += .99**t * rewards[ts+t]
            pred_hit |= hit
            real_hit |= success[ts+t] > .5
        rows.append(torch.stack((truth,learned,direct,real_hit,pred_hit),dim=1))
    truth, learned, direct, real_hit, pred_hit = torch.cat(rows).T.numpy()
    terminal = real_hit.astype(bool)
    out = dict(windows=len(truth), true_terminal_windows=int(terminal.sum()), predicted_terminal_windows=int(pred_hit.sum()),
               matched_terminal_windows=int((terminal & pred_hit.astype(bool)).sum()), mean_recorded=float(truth.mean()))
    for mode, p in (("learned",learned),("direct",direct)):
        err = np.abs(p-truth)
        out[mode] = dict(mean=float(p.mean()), mae=float(err.mean()), correlation=float(np.corrcoef(p, truth)[0,1]),
                         terminal_mae=float(err[terminal].mean()) if terminal.any() else None,
                         nonterminal_mae=float(err[~terminal].mean()) if (~terminal).any() else None)
    return out


def main(args):
    torch.set_num_threads(1)
    raw = Dataset(args.run/"corpus.npz",0)
    starts = StateReward(raw,np.flatnonzero(raw.split == "demo")).starts
    dwell_counts = torch.bincount(starts,minlength=9).tolist()
    diagnostic = Dataset(args.short/"forecast-check.npz",0)
    ids = np.flatnonzero(diagnostic.split == "holdout")
    result = dict(diagnostic_seeds=sorted(map(int,set(diagnostic.seeds[ids]))),
                  demonstration_starts=dict(total=len(starts), dwell_counts=dwell_counts,
                                            eligible_for_eight_in_six=sum(dwell_counts[2:8])),
                  measured=dict(corpus=observed_report(from_dataset(raw, np.arange(len(raw.lengths)))),
                                confirmation=observed_report(from_dataset(diagnostic,ids)),
                                archived_policies=observed_report(from_traces(args.archived/"traces"))), forecasts={})
    for a in ("demos","varied"):
        model = make_world("residual").eval()
        model.load_state_dict(torch.load(args.short/f"{a}.pt", weights_only=True)["world"])
        result["forecasts"][a] = {h: forecast(recorded_windows(diagnostic,ids),model,h) for h in (6,24)}
    (args.run/"reward-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["forecasts"],indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--short",type=Path,required=True)
    parser.add_argument("--archived",type=Path,required=True)
    main(parser.parse_args())
