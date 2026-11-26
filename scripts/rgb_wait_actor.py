"""Fit a stop-or-wait noise policy to counterfactual returns in a frozen visual RSSM."""

import argparse
import json
from pathlib import Path

import torch
from rgb_wait_geometry import PlacementHead, PlacementView
from torch import nn
from torch.nn import functional as F
from train_visual_dreams import digest

from minidaydreamer.vision import VisualFlowHead
from minidaydreamer.visual_rssm import VisualRSSM


def release_chunk(flow, code, noise, proposal, nominal):
    out = nominal.clone()
    pending = (nominal[:, :, 3] > 0).any(1)
    used = torch.zeros(len(code), device=code.device)
    for fraction in (0.65, 0.82, 1.0):
        ids = torch.where(pending)[0]
        if not len(ids):
            break
        adjusted = noise[ids].clone()
        adjusted[:, :, 3] += fraction * proposal[ids, None]
        candidate = flow.sample(code[ids], adjusted)
        take = (
            (candidate[:, :, 3] <= 0).all(1)
            if fraction < 1
            else torch.ones(len(ids), dtype=torch.bool, device=code.device)
        )
        selected = ids[take]
        out[selected] = candidate[take]
        used[selected] = fraction * proposal[selected]
        pending[selected] = False
    return out, used


class WaitActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16 + 2 + 3 + 1, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, code, geometry, nominal):
        # The action-dependent inputs are predictions of the frozen flow for this Gaussian draw.
        x = torch.cat(
            (
                code[:, :16],
                (geometry / 4).clamp(-5, 5),
                nominal[:, :, :3].mean(1),
                nominal[:, :, 3].mean(1, keepdim=True),
            ),
            1,
        )
        return self.net(x).flatten()


def utility(geometry):
    h, r = geometry.unbind(1)
    # The expert release is centered at zero. High and sideward openings are costly.
    cost = ((h - 0.10).clamp_min(0) / 1.65).square() + (
        (-h - 1.15).clamp_min(0) / 2.0
    ).square()
    cost += ((r - 0.75).clamp_min(0) / 1.8).square()
    return 2.0 * torch.exp(-0.5 * cost) - 0.95


@torch.no_grad()
def batch(wm, flow, head, view, starts, errors, height_scale):
    s = starts
    anchor = errors - head(wm.decode(s)[:, :16])
    initial_height = (errors * view.up).sum(1)

    def geometry(residual):
        coord = view.components(residual)
        if height_scale != 1.0:
            origin = initial_height.repeat(len(coord) // len(initial_height))
            coord[:, 0] = origin + height_scale * (coord[:, 0] - origin)
        return coord

    states = []
    coords = []
    plans = []
    waits = []
    stops = []
    opens = []
    for _ in range(8):
        code = wm.decode(s)
        geo = geometry(head(code[:, :16]) + anchor)
        noise = torch.randn(len(s), 3, 4, device=s.device)
        nominal = flow.sample(code, noise)
        shifted, _ = release_chunk(
            flow, code, noise, noise.new_full((len(noise),), -14.0), nominal
        )
        branches = torch.cat((s, s))
        corrections = torch.cat((anchor, anchor))
        planned = torch.cat((nominal, shifted))
        opened = torch.zeros(len(branches), dtype=torch.bool, device=s.device)
        value = branches.new_zeros(len(branches))
        for tick in range(3):
            act = planned[:, tick]
            before = geometry(head(wm.decode(branches)[:, :16]) + corrections)
            successor = wm.imagine(branches, act, sample=False)
            after = geometry(head(wm.decode(successor)[:, :16]) + corrections)
            first = (~opened) & (act[:, 3] <= 0)
            value = torch.where(
                first, 0.8 * utility(before) + 0.2 * utility(after), value
            )
            opened |= first
            branches = successor
        wait, stop = value.chunk(2)
        wo, so = opened.chunk(2)
        drift = (shifted[:, :, :3] - nominal[:, :, :3]).abs().mean((1, 2))
        # A release attempt with no opening receives a failed-option return.
        stops.append(
            torch.where(so, stop, torch.full_like(stop, -0.8)) - 0.28 * drift - 0.025
        )
        waits.append(wait)
        opens.append(wo)
        states.append(code[:, :16])
        coords.append(geo)
        plans.append(nominal)
        s = branches[: len(s)]
    # Waiting preserves the frozen flow and can open naturally. A closed wait leads
    # to the next decision. Rollout horizon is 24 ticks; target fits use first half.
    future = s.new_full((len(s),), -0.60)
    advantages = []
    for i in reversed(range(8)):
        qwait = torch.where(opens[i], waits[i], -0.036 + 0.985 * future)
        advantages.append(stops[i] - qwait)
        future = torch.maximum(stops[i], qwait)
    return tuple(
        torch.cat(v[:4]) for v in (states, coords, plans, list(reversed(advantages)))
    )


def fit(run, seed, height_scale=1.0):
    if not 0 <= height_scale <= 1:
        raise ValueError("Height scale must be in [0, 1]")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = "cpu"
    frozen = torch.load(run / "flow.pt", map_location=device, weights_only=True)
    world = torch.load(run / "world.pt", map_location=device, weights_only=True)
    geometry = torch.load(
        run / "placement-view.pt", map_location=device, weights_only=True
    )
    assert world["base_sha256"] == geometry["base_sha256"] == digest(run / "flow.pt")
    assert geometry["world_sha256"] == digest(run / "world.pt")
    wm = VisualRSSM().eval().requires_grad_(False)
    wm.load_state_dict(world["rssm"])
    flow = VisualFlowHead().eval().requires_grad_(False)
    flow.load_state_dict(frozen["flow"])
    head = PlacementHead().eval().requires_grad_(False)
    head.load_state_dict(geometry["head"])
    view = PlacementView().eval()
    view.load_state_dict(geometry["view"])
    pool = torch.load(run / "placement-starts.pt", weights_only=True)
    states, errs = pool["latent"], pool["errors"]
    learner = WaitActor()
    opt = torch.optim.AdamW(learner.parameters(), lr=0.0008)
    ticks = 0
    for _ in range(280):
        ids = torch.randint(len(states), (96,))
        z, g, n, y = batch(wm, flow, head, view, states[ids], errs[ids], height_scale)
        ticks += 96 * 2 * 24
        for idx in torch.randperm(len(y)).split(96):
            pred = learner(z[idx], g[idx], n[idx])
            loss = F.smooth_l1_loss(pred, y[idx], beta=0.15)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        ids = torch.randint(len(states), (384,))
        z, g, n, y = batch(wm, flow, head, view, states[ids], errs[ids], height_scale)
        v = learner(z, g, n)
        clear = y.abs() > 0.08
        metrics = {
            "seed": seed,
            "imagined_training_ticks": ticks,
            "imagined_height_scale": height_scale,
            "advantage_mae": float((v - y).abs().mean()),
            "clear_choice_agreement": float(
                ((v[clear] > 0) == (y[clear] > 0)).float().mean()
            ),
            "clear_fraction": float(clear.float().mean()),
            "stop_fraction": float((v > 0).float().mean()),
        }
    torch.save(
        {
            "actor": learner.state_dict(),
            "seed": seed,
            "base_sha256": digest(run / "flow.pt"),
            "world_sha256": digest(run / "world.pt"),
            "view_sha256": digest(run / "placement-view.pt"),
            "starts_sha256": digest(run / "placement-starts.pt"),
            **metrics,
        },
        run / f"wait-{seed}.pt",
    )
    (run / f"wait-{seed}.json").write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--seed", type=int, required=True)
    calibration = p.add_mutually_exclusive_group()
    calibration.add_argument("--height-scale", type=float, default=1.0)
    calibration.add_argument("--calibration", type=Path, help="Development height-calibration JSON")
    a = p.parse_args()
    scale = json.loads(a.calibration.read_text())["scale"] if a.calibration else a.height_scale
    fit(a.run, a.seed, scale)
