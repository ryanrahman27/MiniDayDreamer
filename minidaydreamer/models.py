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

    def condition(self, obs):
        return obs

    def loss(self, obs, action_chunk):
        obs = self.condition(obs)
        actions = action_chunk.flatten(1)
        noise = torch.randn_like(actions)
        t = torch.rand_like(actions[:, :1])
        x = noise.lerp(actions, t)
        return F.mse_loss(self.velocity(torch.cat((obs, x, t), -1)), actions - noise)

    @torch.no_grad()
    def sample(self, obs, noise):
        obs = self.condition(obs)
        x = noise.flatten(1)
        for i in range(8):
            t = torch.full_like(x[:, :1], i / 8)
            x = x + self.velocity(torch.cat((obs, x, t), -1)) / 8
        return x.reshape(-1, HORIZON, ACTION_DIM).clamp(-1, 1)


class RelativeFlowPolicy(FlowPolicy):
    """Flow cloning with relative tool/cube geometry and no absolute horizontal origin."""

    def __init__(self):
        nn.Module.__init__(self)
        self.width = HORIZON * ACTION_DIM
        self.velocity = nn.Sequential(nn.Linear(16 + self.width + 1, 160), nn.SiLU(),
                                      nn.Linear(160, 160), nn.SiLU(), nn.Linear(160, 160), nn.SiLU(),
                                      nn.Linear(160, self.width))

    def condition(self, obs):
        tool, red, blue = obs[:, :3], obs[:, 3:6], obs[:, 6:9]
        return torch.cat((tool-red, red-blue, tool-blue, obs[:, [2, 5, 8]],
                          (red[:, :2]-blue[:, :2]).norm(dim=1, keepdim=True),
                          (tool-red).norm(dim=1, keepdim=True), obs[:, 9:]), dim=1)


def make_policy(kind):
    return {"original": FlowPolicy, "relative": RelativeFlowPolicy}[kind]()


class RSSM(nn.Module):
    features = FEATURES

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


class ResidualRSSM(nn.Module):
    """Carry measured poses through filtering and learn action-conditioned pose increments."""

    features = FEATURES + STATE_DIM + 9

    def __init__(self):
        super().__init__()
        self.memory = nn.GRUCell(STOCH + STATE_DIM + 9 + ACTION_DIM, HIDDEN)
        self.prior = mlp(HIDDEN, 2 * STOCH)
        self.posterior = mlp(HIDDEN + STATE_DIM, 2 * STOCH)
        self.motion = mlp(self.features + ACTION_DIM, 10)
        self.reward = mlp(self.features, 1)
        self.success = mlp(self.features, 1)

    @staticmethod
    def decode(state):
        return state[:, FEATURES:FEATURES + STATE_DIM]

    def observed(self, h, obs, velocity, sample):
        q = RSSM.distribution(self.posterior(torch.cat((h, obs), -1)))
        z = q.rsample() if sample else q.mean
        return torch.cat((h, z, obs, velocity), -1), q

    def initial(self, obs):
        h = obs.new_zeros((len(obs), HIDDEN))
        return self.observed(h, obs, torch.zeros_like(obs[:, :9]), False)[0]

    def advance(self, state, action):
        # Pose units are 100 mm. Scale the last observed displacement for the networks.
        velocity = state[:, -9:]
        inputs = torch.cat((state[:, HIDDEN:FEATURES], self.decode(state), 10 * velocity, action), -1)
        h = self.memory(inputs, state[:, :HIDDEN])
        return h, self.decode(state), velocity, action

    def filter(self, context, obs, sample=False):
        h, previous, _, _ = context
        return self.observed(h, obs, obs[:, :9] - previous[:, :9], sample)

    def predict(self, context, sample):
        h, obs, velocity, action = context
        p = RSSM.distribution(self.prior(h))
        z = p.rsample() if sample else p.mean
        latent = torch.cat((h, z), -1)
        motion = self.motion(torch.cat((latent, obs, 10 * velocity, action), -1))
        delta = .1 * motion[:, :9]
        grasp = motion[:, 9:].sigmoid()
        closed = (action[:, 3:] > 0).to(obs.dtype)
        predicted = torch.cat((obs[:, :9] + delta, grasp, closed), -1)
        return torch.cat((latent, predicted, delta), -1), p

    def imagine(self, state, action, sample=False):
        return self.predict(self.advance(state, action), sample)[0]

    def prediction_loss(self, state, target, reward, success):
        predicted = self.decode(state)
        # A 5 mm position error is one training unit.
        position = F.smooth_l1_loss(20 * predicted[:, :9], 20 * target[:, :9])
        grasp = F.binary_cross_entropy(predicted[:, 9], target[:, 9])
        return (position + .2 * grasp + F.mse_loss(self.reward(state)[:, 0], reward)
                + F.binary_cross_entropy_with_logits(self.success(state)[:, 0], success))

    def loss(self, obs, actions, rewards, successes):
        state = self.initial(obs[:, 0])
        warmup = actions.shape[1] - 2 * HORIZON
        if warmup < 0:
            raise ValueError("Residual RSSM needs a full imagination horizon")
        loss = obs.new_zeros(())
        for t, action in enumerate(actions.unbind(1)):
            if t == warmup:
                imagined = state
            context = self.advance(state, action)
            prior, p = self.predict(context, sample=True)
            target = obs[:, t + 1]
            state, q = self.filter(context, target, sample=True)
            loss += self.prediction_loss(prior, target, rewards[:, t], successes[:, t])
            loss += .05 * torch.distributions.kl_divergence(q, p).mean()
            if t >= warmup:
                imagined = self.imagine(imagined, action, sample=True)
                loss += self.prediction_loss(imagined, target, rewards[:, t], successes[:, t])
        return loss / (actions.shape[1] + 2 * HORIZON)


def make_world(kind):
    return {"absolute": RSSM, "residual": ResidualRSSM}[kind]()


class NoiseSAC:
    """Black-box steering. Replay actions are bounded noise shifts, one per chunk."""

    def __init__(self, features=FEATURES):
        self.actor = mlp(features, 2 * ACTION_DIM)
        self.critics = nn.ModuleList([mlp(features + ACTION_DIM, 1) for _ in range(2)])
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
