"""Train matched noise actors for the RGB carry-alignment failure inside the frozen RSSM."""

import argparse
import json
from pathlib import Path

import torch
from rgb_approach_probe import ApproachProbe
from train_visual_dreams import digest

from minidaydreamer.experiment import fit_steering
from minidaydreamer.models import NoiseSAC
from minidaydreamer.vision import VisualFlowHead
from minidaydreamer.visual_rssm import ProgressReward, VisualRSSM


class CarryAlignmentReward(ProgressReward):
    uses_action = True

    def __init__(self, starts, probe):
        super().__init__(starts)
        self.probe = probe

    def step_action(self, previous, current, action, dwell):
        progress, dwell = self.step(previous, current, dwell)
        before = self.probe(previous[:, :16])
        after = self.probe(current[:, :16])
        xy = before[:, :2]
        length = xy.norm(dim=1, keepdim=True)
        direction = xy / torch.sqrt(length.square() + 0.25**2)
        toward = (direction * action[:, :2]).sum(1, keepdim=True).clamp(-1, 1)
        closer = torch.exp(-after[:, :2].norm(dim=1, keepdim=True) / 0.5) - torch.exp(
            -length / 0.5
        )
        carry = before[:, 3:4].sigmoid()
        return progress + carry * (0.45 * toward + 2 * closer), dwell


def load_target(run, world, seed, base_sha):
    ck = torch.load(run / f"target-{seed}.pt", weights_only=True, map_location="cpu")
    if (
        ck["base_sha256"] != base_sha
        or ck["world_sha256"] != digest(run / "world.pt")
        or ck["seed"] != seed
    ):
        raise ValueError("Target actor checkpoint provenance differs")
    learner = NoiseSAC(world.features)
    learner.actor.load_state_dict(ck["actor"])
    return learner


def train(run, seed):
    observed = json.loads((run / "diagnosis.json").read_text())
    if observed["modal_failure"] != "never_centered_no_open":
        raise ValueError(
            "This reward was designed for a held cube that never reached alignment"
        )
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    base = torch.load(run / "flow.pt", weights_only=True, map_location="cpu")
    trained = torch.load(run / "world.pt", weights_only=True, map_location="cpu")
    base_sha = digest(run / "flow.pt")
    assert trained["base_sha256"] == base_sha
    pi = VisualFlowHead().eval().requires_grad_(False)
    pi.load_state_dict(base["flow"])
    wm = VisualRSSM().eval().requires_grad_(False)
    wm.load_state_dict(trained["rssm"])
    starts = torch.load(run / "visual-starts.pt", weights_only=True)
    # Preserve the incumbent's seed-specific initialization and rollout RNG.
    with torch.random.fork_rng(devices=[]):
        probe = ApproachProbe().eval().requires_grad_(False)
        fitted = torch.load(run / "approach.pt", weights_only=True, map_location="cpu")
        assert fitted["base_sha256"] == base_sha
        probe.load_state_dict(fitted["probe"])
    reward = CarryAlignmentReward(len(starts), probe)
    learner = NoiseSAC(wm.features)
    fit_steering(wm, pi, starts, 1500, 4, reward, sac=learner)
    assert reward.ticks == 576000
    state = {
        "actor": learner.actor.state_dict(),
        "seed": seed,
        "base_sha256": base_sha,
        "world_sha256": digest(run / "world.pt"),
        "proxy_sha256": digest(run / "approach.pt"),
        "starts_sha256": digest(run / "visual-starts.pt"),
        "imagined_ticks": reward.ticks,
        "target": observed["modal_failure"],
    }
    tmp = run / f"target-{seed}.tmp"
    torch.save(state, tmp)
    tmp.replace(run / f"target-{seed}.pt")
    print(seed, reward.ticks, observed["modal_failure"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    train(args.run, args.seed)
