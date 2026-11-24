"""Small state-input flow policy, recurrent state-space model, and noise-space SAC."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .stack import ACTION_DIM, STATE_DIM


HORIZON = 3
HIDDEN = 64
STOCH = 12
FEATURES = HIDDEN + STOCH


def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN),
                         nn.SiLU(), nn.Linear(HIDDEN, outputs))


class FlowPolicy(nn.Module):
    """Conditional flow matching from iid Gaussian noise to an action chunk."""

    def __init__(self):
        super().__init__()
        self.width = HORIZON * ACTION_DIM
        self.velocity = mlp(STATE_DIM + self.width + 1, self.width)

    def loss(self, obs, action_chunk):
        actions = action_chunk.flatten(1)
        noise = torch.randn_like(actions)
        t = torch.rand_like(actions[:, :1])
        x = noise.lerp(actions, t)
        return F.mse_loss(self.velocity(torch.cat((obs, x, t), -1)), actions - noise)

    @torch.no_grad()
    def sample(self, obs, noise):
        x = noise.flatten(1)
        for i in range(8):
            t = torch.full_like(x[:, :1], i / 8)
            x = x + self.velocity(torch.cat((obs, x, t), -1)) / 8
        return x.reshape(-1, HORIZON, ACTION_DIM).clamp(-1, 1)


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = nn.GRUCell(STOCH + ACTION_DIM, HIDDEN)
        self.prior = mlp(HIDDEN, 2 * STOCH)
        self.posterior = mlp(HIDDEN + STATE_DIM, 2 * STOCH)
        self.decode = mlp(FEATURES, STATE_DIM)
        self.reward = mlp(FEATURES, 1)
        self.success = mlp(FEATURES, 1)

    @staticmethod
    def distribution(raw):
        mean, scale = raw.chunk(2, -1)
        return torch.distributions.Normal(mean, F.softplus(scale) + .05)

    def filter(self, h, obs, sample=False):
        q = self.distribution(self.posterior(torch.cat((h, obs), -1)))
        return torch.cat((h, q.rsample() if sample else q.mean), -1), q

    def advance(self, state, action):
        return self.memory(torch.cat((state[:, HIDDEN:], action), -1), state[:, :HIDDEN])

    def initial(self, obs):
        return self.filter(obs.new_zeros((len(obs), HIDDEN)), obs)[0]

    def imagine(self, state, action, sample=False):
        h = self.advance(state, action)
        p = self.distribution(self.prior(h))
        return torch.cat((h, p.rsample() if sample else p.mean), -1)

    def loss(self, obs, actions, rewards, successes):
        state = self.initial(obs[:, 0])
        loss = F.mse_loss(self.decode(state), obs[:, 0])
        for j in range(actions.shape[1]):
            h = self.advance(state, actions[:, j])
            p = self.distribution(self.prior(h))
            state, q = self.filter(h, obs[:, j + 1], sample=True)
            kl = torch.distributions.kl_divergence(q, p).mean()
            # Supervise prior predictions as well: actor training only sees prior successors.
            prior_state = torch.cat((h, p.rsample()), -1)
            target = obs[:, j + 1]
            for features in (state, prior_state):
                reconstruction = F.mse_loss(self.decode(features), target)
                r_loss = F.mse_loss(self.reward(features)[:, 0], rewards[:, j])
                done_loss = F.binary_cross_entropy_with_logits(self.success(features)[:, 0], successes[:, j])
                loss = loss + .5 * (reconstruction + r_loss + done_loss)
            loss = loss + .05 * kl
        return loss / actions.shape[1]


class NoiseSAC:
    """Black-box steering. Replay actions are bounded noise shifts, one per chunk."""

    def __init__(self):
        self.actor = mlp(FEATURES, 2 * ACTION_DIM)
        self.critics = nn.ModuleList([mlp(FEATURES + ACTION_DIM, 1) for _ in range(2)])
        self.targets = copy.deepcopy(self.critics).requires_grad_(False)
        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q_opt = torch.optim.Adam(self.critics.parameters(), lr=3e-4)

    def draw(self, state, deterministic=False):
        mean, log_std = self.actor(state).chunk(2, -1)
        dist = torch.distributions.Normal(mean, log_std.clamp(-5, 1).exp())
        raw = mean if deterministic else dist.rsample()
        unit = raw.tanh()
        # Log density on the normalized shift coordinates.
        log_jacobian = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return .5 * unit, (dist.log_prob(raw) - log_jacobian).sum(-1, keepdim=True)

    def update(self, state, shift, reward, discount, successor):
        with torch.no_grad():
            a, logp = self.draw(successor)
            x = torch.cat((successor, a), -1)
            value = torch.minimum(self.targets[0](x), self.targets[1](x)) - .03 * logp
            target = reward + discount * value
        x = torch.cat((state, shift), -1)
        loss = sum(F.mse_loss(q(x), target) for q in self.critics)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        self.critics.requires_grad_(False)
        a, logp = self.draw(state)
        x = torch.cat((state, a), -1)
        pi_loss = (.03 * logp - torch.minimum(self.critics[0](x), self.critics[1](x))).mean()
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        self.critics.requires_grad_(True)
        with torch.no_grad():
            for target_q, q in zip(self.targets.parameters(), self.critics.parameters()):
                target_q.lerp_(q, .01)
