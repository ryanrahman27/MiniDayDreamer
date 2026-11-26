"""Decide when to use the reliable fixed gripper shift based on RGB and imagined returns."""

import json

import torch
from rgb_release_actor import load_actor
from rgb_release_control import SEEDS
from rgb_wait_actor import WaitActor, release_chunk
from rgb_wait_geometry import PlacementView
from train_visual_dreams import digest

# Bounds are in whitened two-view image coordinates, as measured on expert releases.
# The third value is the predicted advantage required for stopping now.
LIMITS = {
    "middle": (1.85, 2.05, -0.045),
    "tight": (1.20, 1.55, -0.020),
    "wide": (2.45, 2.35, -0.065),
}


def configure_wait(run, split, wm, calibrated=False):
    saved = torch.load(run / "placement-view.pt", weights_only=True)
    assert saved["base_sha256"] == digest(run / "flow.pt") and saved[
        "world_sha256"
    ] == digest(run / "world.pt")
    view = PlacementView().cuda().eval()
    view.load_state_dict(saved["view"])
    previous = {seed: load_actor(run, wm, seed) for seed in SEEDS}
    actors = {}
    for seed in SEEDS:
        ck = torch.load(run / f"wait-{seed}.pt", weights_only=True)
        assert ck["seed"] == seed and ck["view_sha256"] == digest(
            run / "placement-view.pt"
        )
        assert (
            ck["base_sha256"] == saved["base_sha256"]
            and ck["world_sha256"] == saved["world_sha256"]
        )
        actor = WaitActor().eval()
        actor.load_state_dict(ck["actor"])
        actors[seed] = actor
    chosen = None
    names = ["rgb-base", "fixed-permissive"]
    if split == "test":
        chosen = json.loads((run / "selection.json").read_text())["selected"]
        assert chosen in LIMITS
        names += ["release-permissive", "lower-" + chosen, "wait-" + chosen]
    else:
        names += (
            ["wait-" + key for key in LIMITS] if not calibrated else ["wait-middle"]
        )
    return view, previous, actors, names, chosen


class WaitControl:
    def __init__(self, arm, width):
        prefix, key = arm.split("-", 1)
        self.learned = prefix == "wait"
        self.high, self.radial, self.threshold = LIMITS[key]
        self.closed = torch.zeros(width, dtype=torch.long)
        self.streak = torch.zeros_like(self.closed)
        self.pulse = torch.zeros_like(self.closed)

    def observe(self, act):
        self.closed = torch.where(act[:, 3] > 0, self.closed + 1, 0)

    def sample(self, pi, condition, noise, ready, coords, visible, actors, seeds):
        plan = pi.sample(condition, noise)
        h, r = coords.T
        eligible = (
            (self.closed >= 12)
            & visible
            & (ready >= 0.28)
            & (h <= self.high)
            & (h >= -2.15)
            & (r <= self.radial)
        )
        self.streak = torch.where(eligible, self.streak + 1, 0)
        adv = torch.zeros(len(h))
        if self.learned:
            for seed, actor in actors.items():
                ids = seeds == seed
                if ids.any():
                    adv[ids] = actor(condition[ids], coords[ids], plan[ids])
        voted = adv >= self.threshold if self.learned else torch.ones_like(eligible)
        # Keep reconsidering a valid placement after waiting. Only the actor can defer past the guard.
        start = (self.streak >= 2) & (self.pulse == 0) & voted
        self.pulse = torch.where(start, 16, self.pulse)
        ids = torch.where((self.pulse > 0) & (plan[:, :, 3] > 0).any(1))[0]
        self.pulse = (self.pulse - 1).clamp_min(0)
        used = torch.zeros(len(h))
        if len(ids):
            chunk, shift = release_chunk(
                pi,
                condition[ids],
                noise[ids],
                torch.full((len(ids),), -14.0),
                plan[ids],
            )
            plan[ids] = chunk
            used[ids] = shift
        return plan, start, used, adv, eligible
