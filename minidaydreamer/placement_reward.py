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
