"""Train the retreat-reward actor with the targeted actor's frozen model and start states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from failure_dsrl import mixed_starts
from minidaydreamer.data import Dataset
from minidaydreamer.experiment import fit_steering, verify_weights
from minidaydreamer.models import make_policy, make_world
from minidaydreamer.placement_reward import RetreatReward, retreat_progress
from minidaydreamer.state_reward import transition
from reward_study import digest, load


def check_reward():
    a = torch.tensor([[0.0, 0.0, 0.8, 0.0, 0.0, 0.7, 0.0, 0.0, 0.232, 0.0, 0.0]])
    b = a.clone()
    b[:, 2] += 0.6
    torch.testing.assert_close(retreat_progress(a, b), torch.tensor([0.8]))
    torch.testing.assert_close(retreat_progress(b, a), torch.tensor([-0.8]))
    assert retreat_progress(b, b) == 0
    for index, value in ((10, 1.0), (9, 1.0), (3, 0.13), (5, 0.8), (6, 0.02)):
        bad = b.clone()
        bad[:, index] = value
        assert retreat_progress(a, bad) == 0
    d = torch.zeros(1, dtype=torch.long)
    for _ in range(7):
        _, d, bonus = transition(b, b, d)
        assert not bonus
    assert transition(b, a, d)[1] == 0
    _, d, bonus = transition(b, b, d)
    assert bonus and d == 8
    assert not transition(b, b, d)[2]


def train(run, previous, midbase, horizon):
    check_reward()
    src = json.loads((previous / "training.json").read_text())
    for name, expect in (
        ("targeted.pt", src["actors"]["targeted"]["sha256"]),
        ("selected-base.pt", src["base_sha256"]),
    ):
        assert digest(run / name) == expect
    target = load(run / "targeted.pt")
    base = load(run / "selected-base.pt")
    wm = make_world("residual").eval().requires_grad_(False)
    wm.load_state_dict(target["world"])
    pi = make_policy(base["policy_kind"]).eval().requires_grad_(False)
    pi.load_state_dict(base["policy"])
    verify_weights(pi, target["policy"], "common base")
    rng = load(horizon / "varied.pt")["pre_actor_rng"]
    assert hashlib.sha256(rng.numpy().tobytes()).hexdigest() == src["rng_sha256"]
    assert digest(midbase / "corpus.npz") == src["corpus_sha256"]
    data = Dataset(midbase / "corpus.npz", 0)
    demos = np.flatnonzero(data.split == "demo")
    states, _ = mixed_starts(midbase, wm, data, demos)
    assert hashlib.sha256(states.numpy().tobytes()).hexdigest() == src["start_tensor_sha256"]
    reward = RetreatReward(data, demos)
    reward.starts = torch.cat(
        (reward.starts, torch.zeros(len(states) - len(reward.starts), dtype=torch.long))
    )
    torch.set_rng_state(rng)
    actor = fit_steering(wm, pi, states, target["rl_steps"], target["imagination_chunks"], reward)
    assert reward.ticks == target["imagined_ticks"] == 576000 and reward.moving_out and reward.dwell_advances
    verify_weights(pi, target["policy"], "base after training")
    verify_weights(wm, target["world"], "world after training")
    assert not (run / "retreat.pt").exists()
    torch.save(
        dict(target, actor=actor.actor.state_dict(), reward_mode="retreat", direct_bonuses=reward.bonuses),
        run / "retreat.pt",
    )
    report = dict(
        base_sha256=src["base_sha256"],
        targeted_sha256=src["actors"]["targeted"]["sha256"],
        retreat_sha256=digest(run / "retreat.pt"),
        start_tensor_sha256=src["start_tensor_sha256"],
        rng_sha256=src["rng_sha256"],
        imagined_ticks=reward.ticks,
        moving_out=reward.moving_out,
        moving_in=reward.moving_in,
        dwell_advances=reward.dwell_advances,
        completions=reward.bonuses,
    )
    (run / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "previous", "midbase", "horizon"):
        parser.add_argument(name, type=Path)
    torch.set_num_threads(2)
    train(**vars(parser.parse_args()))
