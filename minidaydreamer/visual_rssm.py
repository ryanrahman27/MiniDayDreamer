"""Recurrent dynamics and a demonstration-progress reward for camera features."""

import torch
from torch import nn
from torch.nn import functional as F

from .models import ACTION_DIM, FEATURES, HIDDEN, RSSM, STOCH, mlp

CODE = 17  # 16 behavior-cloning features and one visual progress estimate.


class VisualRSSM(nn.Module):
    features = FEATURES + CODE

    def __init__(self):
        super().__init__()
        self.memory = nn.GRUCell(STOCH + CODE + ACTION_DIM, HIDDEN)
        self.posterior = mlp(HIDDEN + CODE, 2 * STOCH)
        self.prior = mlp(HIDDEN, 2 * STOCH)
        self.delta = mlp(self.features + ACTION_DIM, CODE)

    def decode(self, state):
        return state[:, FEATURES:]

    def filter(self, context, code, sample=False):
        h, _, _ = context
        q = RSSM.distribution(self.posterior(torch.cat((h, code), 1)))
        return torch.cat((h, q.rsample() if sample else q.mean, code), 1), q

    def initial(self, code):
        h = code.new_zeros(len(code), HIDDEN)
        return self.filter((h, None, None), code)[0]

    def advance(self, state, action):
        h = self.memory(torch.cat((state[:, HIDDEN:], action), 1), state[:, :HIDDEN])
        return h, self.decode(state), action

    def predict(self, context, sample):
        h, code, act = context
        p = RSSM.distribution(self.prior(h))
        z = p.rsample() if sample else p.mean
        change = self.delta(torch.cat((h, z, code, act), 1)) * 0.20
        nxt = code + change
        nxt = torch.cat((nxt[:, :16].clamp(-5, 5), nxt[:, 16:].clamp(0, 1)), 1)
        return torch.cat((h, z, nxt), 1), p

    def imagine(self, state, action, sample=False):
        return self.predict(self.advance(state, action), sample)[0]

    def prediction_loss(self, state, target):
        diff = self.decode(state) - target
        features = F.smooth_l1_loss(
            diff[:, :16], torch.zeros_like(diff[:, :16]), beta=0.12
        )
        return features + 2 * diff[:, 16].abs().mean()

    def loss(self, code, actions):
        state = self.initial(code[:, 0])
        loss = code.new_zeros(())
        for i, action in enumerate(actions.unbind(1)):
            context = self.advance(state, action)
            pred, p = self.predict(context, True)
            nxt = code[:, i + 1]
            state, q = self.filter(context, nxt, True)

            loss = (
                loss
                + self.prediction_loss(pred, nxt)
                + 0.015 * torch.distributions.kl_divergence(q, p).mean()
            )
            if i == 2:
                imagined = state
            if i >= 3:
                imagined = self.imagine(imagined, action, True)
                loss = loss + self.prediction_loss(imagined, nxt)
        return loss / (2 * actions.shape[1] - 3)


class ProgressReward:
    fixed_horizon = True

    def __init__(self, count):
        self.starts = torch.zeros(count, 1)
        self.ticks = 0

    def step(self, previous, current, dwell):
        self.ticks += len(current)
        progress = current[:, 16:17] - previous[:, 16:17]
        return 4 * progress + 0.04 * current[:, 16:17], dwell
