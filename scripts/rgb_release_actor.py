"""Learn a release-specific gripper noise shift in the frozen visual RSSM."""

import argparse
from pathlib import Path

import torch
from rgb_release_geometry import ImaginedGeometry
from train_visual_dreams import digest

from minidaydreamer.experiment import fit_steering
from minidaydreamer.models import NoiseSAC
from minidaydreamer.vision import VisualFlowHead
from minidaydreamer.visual_rssm import ProgressReward, VisualRSSM


class ReleaseSAC(NoiseSAC):
    """Use the existing SAC critics and a one-dimensional bounded flow-noise action."""

    def draw(self, state, deterministic=False):
        mean, log_std = self.actor(state).chunk(2, -1)
        dist = torch.distributions.Normal(
            mean[:, 3:], log_std[:, 3:].clamp(-5, 1).exp()
        )
        raw = dist.mean if deterministic else dist.rsample()
        unit = raw.tanh()
        # The tanh coordinate has unit range. The physical shift spans -18 to 0.
        shift = torch.cat((torch.zeros_like(mean[:, :3]), -9 * (unit + 1)), 1)
        log_jac = 2 * (0.69314718056 - raw - torch.nn.functional.softplus(-2 * raw))
        return shift, dist.log_prob(raw) - log_jac


class ReleaseReward(ProgressReward):
    uses_action = True

    def __init__(self, count, readout):
        super().__init__(count)
        self.readout = readout

    def step_action(self, previous, current, action, dwell):
        progress, dwell = self.step(previous, current, dwell)
        ready = self.readout(previous[:, :16])
        retained = self.readout(current[:, :16])
        opened = torch.sigmoid(-6 * action[:, 3:])
        # Reward open commands when the two-view geometry is near an expert release.
        # Penalize opening away from that geometry and moving the cube during release.
        reward = opened * (2.8 * ready + 1.2 * retained - 1.4 * (1 - ready))
        reward -= 0.35 * ready * opened * action[:, :3].square().sum(1, keepdim=True)
        return progress + reward, dwell


def load_actor(run, wm, seed):
    ck = torch.load(run / f"release-{seed}.pt", map_location="cpu", weights_only=True)
    assert ck["seed"] == seed and ck["base_sha256"] == digest(run / "flow.pt")
    assert ck["world_sha256"] == digest(run / "world.pt") and ck[
        "geometry_sha256"
    ] == digest(run / "release-geometry.pt")
    sac = ReleaseSAC(wm.features)
    sac.actor.load_state_dict(ck["actor"])
    return sac


def train(run, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    flow, wm = (
        VisualFlowHead().eval().requires_grad_(False),
        VisualRSSM().eval().requires_grad_(False),
    )
    f = torch.load(run / "flow.pt", map_location="cpu", weights_only=True)
    w = torch.load(run / "world.pt", map_location="cpu", weights_only=True)
    geom = torch.load(
        run / "release-geometry.pt", map_location="cpu", weights_only=True
    )
    assert geom["base_sha256"] == w["base_sha256"] == digest(run / "flow.pt")
    flow.load_state_dict(f["flow"])
    wm.load_state_dict(w["rssm"])
    readout = ImaginedGeometry().eval().requires_grad_(False)
    readout.load_state_dict(geom["imagined"])
    starts = torch.load(run / "visual-starts.pt", weights_only=True)
    code = wm.decode(starts)
    with torch.no_grad():
        nominal = flow.sample(code, torch.zeros(len(code), 3, 4))[:, :, 3].mean(1)
        carrying = (code[:, 16] > 0.35) & (code[:, 16] < 0.86) & (nominal > 0)
        ids = torch.where(carrying)[0]
        readiness = readout(code[ids, :16]).flatten()
        late = ids[readiness >= torch.quantile(readiness, 0.76)]
        # Half the posterior start pool emphasizes closed-command carries near release.
        pool = torch.cat((starts, starts[late].repeat((len(starts) // len(late), 1))))
    reward = ReleaseReward(len(pool), readout)
    sac = ReleaseSAC(wm.features)
    fit_steering(wm, flow, pool, 1500, 4, reward, sac=sac)
    assert reward.ticks == 576000
    tmp = run / f"release-{seed}.tmp"
    torch.save(
        {
            "seed": seed,
            "actor": sac.actor.state_dict(),
            "imagined_ticks": reward.ticks,
            "late_start_states": len(late),
            "base_sha256": digest(run / "flow.pt"),
            "world_sha256": digest(run / "world.pt"),
            "starts_sha256": digest(run / "visual-starts.pt"),
            "geometry_sha256": digest(run / "release-geometry.pt"),
        },
        tmp,
    )
    tmp.replace(run / f"release-{seed}.pt")
    print(seed, reward.ticks, len(late), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    train(a.run, a.seed)
