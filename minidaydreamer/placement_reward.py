"""Extra imagined reward for descending on approach and releasing a centered cube."""

from .state_reward import StateReward


def relative_pose(obs):
    delta = (obs[..., 3:6] - obs[..., 6:9]) * 100  # mm
    return delta[..., :2].norm(dim=-1), delta[..., 2] - 46.8


def released_pose(obs, xy, height):
    centered = ((18 - xy) / 10).clamp(0, 1)
    above_support = ((height + 16) / 8).clamp(0, 1)
    near_support = ((40 - height) / 15).clamp(0, 1)
    return (1 - obs[..., 10]) * centered * above_support * near_support


def placement_terms(previous, current):
    xy, z = relative_pose(previous)
    next_xy, next_z = relative_pose(current)
    held = (previous[..., 9] > 0.5) & (previous[..., 10] > 0.5)
    carrying = held & (current[..., 9] > 0.5) & (current[..., 10] > 0.5)
    proximity = ((80 - xy) / 40).clamp(0, 1)
    # Compare both heights against the same clearance at the beginning of the tick.
    clearance = 8 + 60 * ((xy - 12) / 48).clamp(0, 1)
    down = ((z - clearance).clamp_min(0) - (next_z - clearance).clamp_min(0)) / 5
    center = ((xy - 8).clamp_min(0) - (next_xy - 8).clamp_min(0)) / 5
    descent_reward = 0.5 * carrying * proximity * down.clamp(-1, 1)
    center_reward = 0.3 * carrying * proximity * center.clamp(-1, 1)
    release_reward = 0.45 * (released_pose(current, next_xy, next_z) - released_pose(previous, xy, z))
    opened = held & (current[..., 10] < 0.5)
    off_center = -0.5 * opened * ((xy - 12) / 10).clamp(0, 1)
    return descent_reward, center_reward, release_reward, off_center


class PlacementReward(StateReward):
    def step(self, previous, current, dwell):
        reward, count = super().step(previous, current, dwell)
        return reward + sum(placement_terms(previous, current))[:, None], count


def retreat_progress(previous, current):
    """Credit motion away from a released, seated cube up to 50 mm of clearance."""

    def seated(obs):
        xy, height = relative_pose(obs)
        return (
            (xy < 12)
            & (height.abs() < 8)
            & (obs[..., 8] > 0.14)
            & (obs[..., 8] < 0.32)
            & (obs[..., 9] < 0.5)
            & (obs[..., 10] < 0.5)
        )

    def clearance(obs):
        return (2 * (obs[..., :3] - obs[..., 3:6]).norm(dim=-1)).clamp(max=1)

    red_slow = (current[..., 3:6] - previous[..., 3:6]).norm(dim=-1) < 0.0125
    blue_slow = (current[..., 6:9] - previous[..., 6:9]).norm(dim=-1) < 0.0125
    return (
        seated(previous) * seated(current) * red_slow * blue_slow * (clearance(current) - clearance(previous))
    )


class RetreatReward(PlacementReward):
    def __init__(self, data, episodes):
        super().__init__(data, episodes)
        self.moving_out = self.moving_in = self.dwell_advances = 0

    def step(self, previous, current, dwell):
        reward, count = super().step(previous, current, dwell)
        motion = retreat_progress(previous, current)
        self.moving_out += int((motion > 0).sum())
        self.moving_in += int((motion < 0).sum())
        self.dwell_advances += int((count > dwell).sum())
        # Losing an incomplete streak gives back its credit. Completed dwell is absorbing.
        return reward + (motion + 0.8 * (count - dwell) / 8)[:, None], count
