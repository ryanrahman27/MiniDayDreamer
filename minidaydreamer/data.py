"""Fixed episode-level splits. The simulator is used only during collection and evaluation."""

import numpy as np
import torch

from .stack import ACTION_DIM, STATE_DIM, scripted_episode


def collect(path, demos, extra, holdout, seed, env):
    variants = ("demo", "miss", "offset", "drop")
    rows, episodes = [], []
    cursor = seed
    while len(rows) < demos and cursor - seed < 5 * demos:
        ep = scripted_episode(env, cursor)
        if ep[3][-1]:
            rows.append(("demo", "demo", cursor))
            episodes.append(ep)
        cursor += 1
    if len(rows) < demos:
        raise RuntimeError(f"Scripted expert stacked {len(rows)} times in {cursor-seed} attempts. Inspect preview.")
    print(f"Accepted {demos} successful demonstrations from {cursor-seed} attempts.", flush=True)
    rows += [("extra", variants[i % 4], cursor + i) for i in range(extra)]
    # Each held-out initial condition has both a good trajectory and a counterfactual failure.
    rows += [("holdout", v, cursor + extra + i)
             for i in range(holdout) for v in ("demo", variants[1 + i % 3])]
    n = len(rows)
    obs = np.zeros((n, env.max_steps + 1, STATE_DIM), np.float32)
    actions = np.zeros((n, env.max_steps, ACTION_DIM), np.float32)
    rewards = np.zeros((n, env.max_steps), np.float32)
    success = np.zeros_like(rewards)
    lengths = np.empty(n, np.int32)
    for i, (_, variant, reset_seed) in enumerate(rows):
        o, a, r, s = episodes[i] if i < demos else scripted_episode(env, reset_seed, variant)
        t = len(a)
        obs[i, :t+1], actions[i, :t], rewards[i, :t], success[i, :t], lengths[i] = o, a, r, s, t
    splits, kinds, seeds = zip(*rows)
    # Exclusive creation protects a fixed corpus from accidental replacement.
    with path.open("xb") as f:
        np.savez_compressed(f, obs=obs, actions=actions, rewards=rewards, success=success,
                            lengths=lengths, split=splits, variant=kinds, seeds=seeds, backend=env.backend)


class Dataset:
    def __init__(self, path, seed):
        with np.load(path, allow_pickle=False) as d:
            self.backend = str(d["backend"])
            self.obs, self.actions, self.rewards, self.success = (
                torch.from_numpy(d[key]) for key in ("obs", "actions", "rewards", "success"))
            self.lengths, self.split, self.seeds = (d[k] for k in ("lengths", "split", "seeds"))
        self.rng = np.random.default_rng(seed)

    def sample(self, ids, batch, width):
        candidates = ids[self.lengths[ids] >= width]
        if not len(candidates):
            raise ValueError(f"No training episode has {width} transitions")
        episodes = self.rng.choice(candidates, size=batch)
        start = self.rng.integers(self.lengths[episodes] - width + 1)
        ticks = start[:, None] + np.arange(width + 1)
        ix = episodes[:, None], ticks[:, :-1]
        return self.obs[episodes[:, None], ticks], self.actions[ix], self.rewards[ix], self.success[ix]
