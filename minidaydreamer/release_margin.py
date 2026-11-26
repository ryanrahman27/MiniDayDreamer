"""An imagined release-margin reward and a local correction to the frozen noise controller."""

import torch

from .models import NoiseSAC, ResidualRSSM
from .placement_reward import relative_pose
from .released_steering import ReleasedReward


def approach_weight(obs):
    xy, height = relative_pose(obs)
    held = (obs[:, 9] > 0.5) & (obs[:, 10] > 0.5)
    return (
        held
        * ((50 - xy) / 20).clamp(0, 1)
        * ((55 - height) / 25).clamp(0, 1)
        * ((height + 20) / 10).clamp(0, 1)
    )


def margin_terms(previous, current):
    xy, _ = relative_pose(previous)
    next_xy, next_height = relative_pose(current)
    weight = approach_weight(previous)
    closed = current[:, 10] > 0.5
    improvement = ((xy - 6).clamp_min(0) - (next_xy - 6).clamp_min(0)) / 4
    progress = weight * closed * (current[:, 9] > 0.5) * improvement.clamp(-1, 1)
    # Reward a well-seated opening, penalize one outside the margin. Remaining closed earns neither.
    alignment = ((9 - torch.maximum(xy, next_xy)) / 4).clamp(-1, 1)
    height_credit = ((12 - next_height.abs()) / 6).clamp(0, 1)
    opening = weight * (~closed) * (alignment.clamp_max(0) + alignment.clamp_min(0) * height_credit)
    return progress, opening


class MarginReward(ReleasedReward):
    def step(self, previous, current, dwell):
        reward, count = super().step(previous, current, dwell)
        progress, opening = margin_terms(previous, current)
        return reward + (0.7 * progress + 1.5 * opening)[:, None], count


class LocalNoiseSAC(NoiseSAC):
    def __init__(self, features, incumbent):
        super().__init__(features)
        self.incumbent = incumbent

    def draw(self, state, deterministic=False):
        obs = ResidualRSSM.decode(state)
        # Expose signed local displacement to the residual actor, preserving the original state for critics.
        local = state.clone()
        local_obs = ResidualRSSM.decode(local)
        local_obs[:, 3:6] = 5 * (obs[:, 3:6] - obs[:, 6:9] - obs.new_tensor([0, 0, 0.468]))
        correction, logp = super().draw(local, deterministic)
        with torch.no_grad():
            anchor = self.incumbent.draw(state, deterministic=True)[0]
        return anchor + 0.75 * approach_weight(obs)[:, None] * correction, logp
