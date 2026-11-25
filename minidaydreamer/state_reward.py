"""Isaac shaping and a once-only stable-stack bonus from observable and predicted states."""

import torch


def potential(obs):
    tool, red, blue = obs[..., :3], obs[..., 3:6], obs[..., 6:9]
    lift = ((red[..., 2]-.2)/.6).clamp(0, 1)
    goal = blue + obs.new_tensor([0., 0., .4])
    return (.2 * (-(tool-red).norm(dim=-1)).exp() + .25*obs[..., 9] + .2*lift
            + .5 * (-2*(red-goal).norm(dim=-1)).exp())


def stable(previous, current):
    # Pose units are 100 mm. Finite differences cover one 50 ms control tick.
    red, blue = current[..., 3:6], current[..., 6:9]
    slow_red = (red-previous[..., 3:6]).norm(dim=-1) < .0125
    slow_blue = (blue-previous[..., 6:9]).norm(dim=-1) < .0125
    return ((current[..., 10] < .5) & (current[..., 9] < .5) & ((current[..., :3]-red).norm(dim=-1) > .5)
            & ((red[..., :2]-blue[..., :2]).norm(dim=-1) < .12)
            & ((red[..., 2]-blue[..., 2]-.468).abs() < .08)
            & (blue[..., 2] > .14) & (blue[..., 2] < .32) & slow_red & slow_blue)


def next_dwell(count, qualifies):
    # Eight is an absorbing state so a completed imagined stack earns exactly one bonus.
    return torch.where(count == 8, count, (count + 1) * qualifies)


def dwell_history(obs):
    flags = stable(obs[:-1], obs[1:])
    counts = [torch.zeros((), dtype=torch.long, device=obs.device)]
    for flag in flags:
        counts.append(next_dwell(counts[-1], flag))
    return torch.stack(counts)


def transition(previous, current, dwell):
    count = next_dwell(dwell, stable(previous, current))
    bonus = (count == 8) & (dwell < 8)
    return potential(current) - potential(previous) + 2*bonus, count, bonus


class StateReward:
    def __init__(self, data, episodes):
        if data.backend != "isaac":
            raise ValueError("The release geometry is specific to the Isaac stacking task")
        self.starts = torch.cat([dwell_history(data.obs[i, :int(data.lengths[i])+1])[:-1] for i in episodes])
        self.ticks = self.bonuses = 0

    def step(self, previous, current, dwell):
        reward, count, bonus = transition(previous, current, dwell)
        self.ticks += reward.numel()
        self.bonuses += int(bonus.sum())
        return reward[:, None], count
