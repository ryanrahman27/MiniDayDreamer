"""Apply gripper-noise steering only when both RGB views support release."""

import json

import torch
from rgb_release_actor import load_actor
from rgb_release_geometry import Geometry

SEEDS = (101, 211, 307, 401, 503)
# Minimum similarity, consecutive three-tick decisions, duration in decisions.
CONFIGS = {
    "careful": (0.73, 2, 8),
    "balanced": (0.50, 2, 12),
    "permissive": (0.28, 2, 16),
    "prompt": (0.56, 1, 24),
}


def configure(run, split, wm):
    geom = Geometry().cuda().eval()
    geom.load_state_dict(
        torch.load(run / "release-geometry.pt", weights_only=True)["geometry"]
    )
    actors = {s: load_actor(run, wm, s) for s in SEEDS}
    if split == "dev":
        names = ["rgb-base", *["release-" + n for n in CONFIGS]]
        selected = None
    else:
        chosen = json.loads((run / "selection.json").read_text())
        selected = chosen["selected"]
        assert selected in CONFIGS
        names = [
            "rgb-base",
            "rgb-target",
            "rgb-incumbent",
            "release-" + selected,
            "fixed-" + selected,
        ]
    return geom, actors, names, selected


class ReleaseControl:
    def __init__(self, name, width):
        self.fixed = name.startswith("fixed-")
        self.threshold, self.required, self.duration = CONFIGS[name.split("-", 1)[1]]
        self.closed = torch.zeros(width, dtype=torch.long)
        self.streak = torch.zeros_like(self.closed)
        self.remaining = torch.zeros_like(self.closed)

    def observe(self, action):
        self.closed = torch.where(action[:, 3] > 0, self.closed + 1, 0)

    def sample(self, pi, condition, noise, readiness, latent, actors, lane_seeds):
        eligible = (readiness >= self.threshold) & (self.closed >= 12)
        self.streak = torch.where(eligible, self.streak + 1, 0)
        # A fresh pulse starts on the required consecutive observation.
        start = (self.streak == self.required) & (self.remaining == 0)
        self.remaining = torch.where(start, self.duration, self.remaining)
        nominal = pi.sample(condition, noise)
        active = (self.remaining > 0) & (nominal[:, :, 3] > 0).any(1)
        self.remaining = (self.remaining - 1).clamp_min(0)
        shift = torch.zeros(len(noise))
        if not active.any():
            return nominal, start, shift
        if self.fixed:
            shift[active] = -14.0
        else:
            for seed, actor in actors.items():
                ids = active & (lane_seeds == seed)
                if ids.any():
                    shift[ids] = actor.draw(latent[ids], deterministic=True)[0][:, 3]
        # Choose the smallest of three fractions of the proposed shift yielding an
        # open chunk, limiting incidental movement of the other flow outputs.
        out = nominal.clone()
        pending = active.clone()
        used = torch.zeros_like(shift)
        for factor in (0.65, 0.82, 1.0):
            ids = torch.where(pending)[0]
            if not len(ids):
                break
            adjusted = noise[ids].clone()
            adjusted[:, :, 3] += factor * shift[ids, None]
            candidate = pi.sample(condition[ids], adjusted)
            take = (
                (candidate[:, :, 3] <= 0).all(1)
                if factor < 1
                else torch.ones(len(ids), dtype=torch.bool)
            )
            out[ids[take]] = candidate[take]
            used[ids[take]] = factor * shift[ids[take]]
            pending[ids[take]] = False
        return out, start, used
