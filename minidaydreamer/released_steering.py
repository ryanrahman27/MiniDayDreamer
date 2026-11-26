"""Noise-space SAC for a released stack at a fixed endpoint."""

import torch

from .models import NoiseSAC
from .placement_reward import placement_terms, relative_pose
from .state_reward import next_dwell, potential


def released(previous, current):
    xy, height = relative_pose(current)
    slow = ((current[..., 3:9] - previous[..., 3:9]).reshape(*xy.shape, 2, 3).norm(dim=-1) < 0.0125).all(-1)
    return (
        slow
        & (xy < 12)
        & (height.abs() < 8)
        & (current[..., 8] > 0.14)
        & (current[..., 8] < 0.32)
        & (current[..., 9] < 0.5)
        & (current[..., 10] < 0.5)
    )


class ReleasedReward:
    # The historical RSSM success head was trained on the old clearance rule.
    fixed_horizon = True

    def __init__(self, data, demos, total_states):
        starts = []
        for ep in demos:
            obs = data.obs[ep, : int(data.lengths[ep]) + 1]
            count = torch.tensor(0, dtype=torch.long)
            for qualifies in released(obs[:-1], obs[1:]):
                starts.append(count)
                count = next_dwell(count, qualifies)
        # The additional recorded histories end with a closed gripper holding the cube.
        self.starts = torch.cat(
            (torch.stack(starts), torch.zeros(total_states - len(starts), dtype=torch.long))
        )
        self.ticks = self.qualified = self.bonuses = 0

    def step(self, previous, current, dwell):
        ok = released(previous, current)
        count = next_dwell(dwell, ok)
        bonus = (count == 8) & (dwell < 8)
        self.ticks += len(count)
        self.qualified += int(ok.sum())
        self.bonuses += int(bonus.sum())
        reward = (
            potential(current)
            - potential(previous)
            + sum(placement_terms(previous, current))
            + 0.25 * ok
            + 0.1 * (count - dwell)
            + 2 * bonus
        )
        return reward[:, None], count


class PriorNoiseSAC(NoiseSAC):
    """Train critics on the applied shift. Entropy is in the new actor's own coordinates."""

    def __init__(self, features, prior, prior_weight):
        super().__init__(features)
        self.prior = prior
        self.prior_weight = prior_weight
        self.evaluation_gain = 1.0

    def draw(self, state, deterministic=False):
        shift, logp = super().draw(state, deterministic)
        if self.prior_weight:
            with torch.no_grad():
                anchor = self.prior.draw(state, deterministic=True)[0]
            shift = torch.lerp(shift, anchor, self.prior_weight)
        return shift * self.evaluation_gain if deterministic else shift, logp
