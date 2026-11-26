"""Fit camera-feature dynamics, then train noise steering exclusively in imagination."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from fit_rgb import active_lengths, read
from torch import nn
from torch.nn import functional as F

from minidaydreamer.experiment import fit_steering
from minidaydreamer.models import NoiseSAC
from minidaydreamer.vision import CameraEncoder, VisualFlowHead
from minidaydreamer.visual_rssm import ProgressReward, VisualRSSM


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProgressHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(16, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )


def progress_targets(actions):
    """Place training-only phase anchors at closing, reopening and parking in expert action logs."""
    t = torch.arange(actions.shape[1] + 1, device=actions.device)[None, :]
    closed = actions[:, :, 3] > 0
    first = closed.float().argmax(1)[:, None]
    opens = (~closed) & (t[:, :-1] > first)
    assert closed.any(1).all() and opens.any(1).all()
    release = opens.float().argmax(1)[:, None]
    end = active_lengths(actions)[:, None].clamp_min(release + 1)
    return (
        0.30 * (t / first.clamp_min(1)).clamp(0, 1)
        + 0.48 * ((t - first) / (release - first).clamp_min(1)).clamp(0, 1)
        + 0.22 * ((t - release) / (end - release).clamp_min(1)).clamp(0, 1)
    )


@torch.no_grad()
def encode(paths, encoder):
    x, a = read(paths, "cpu")
    width = x.shape[1]
    features = torch.cat(
        [encoder(part.cuda()).cpu() for part in x.flatten(0, 1).split(160)]
    )
    return features.reshape(len(paths), width, -1).cuda(), a.cuda()


@torch.no_grad()
def diagnostics(wm, codes, actions):
    # Use identical recorded actions for open-loop prediction and compare to the frozen starting feature.
    horizons = (3, 6, 12)
    stats = {h: [] for h in horizons}
    for start in range(10, 175, 8):
        state = wm.initial(codes[:, start - 6])
        for k in range(start - 6, start):
            state, _ = wm.filter(wm.advance(state, actions[:, k]), codes[:, k + 1])
        for j in range(1, 13):
            state = wm.imagine(state, actions[:, start + j - 1], False)
            if j in stats:
                target = codes[:, start + j]
                pred = wm.decode(state)
                still = codes[:, start]
                stats[j].append(
                    torch.stack(
                        (
                            (pred[:, :16] - target[:, :16]).square().mean(),
                            (still[:, :16] - target[:, :16]).square().mean(),
                            (pred[:, 16] - target[:, 16]).abs().mean(),
                            (still[:, 16] - target[:, 16]).abs().mean(),
                        )
                    ).cpu()
                )
    return {
        h: dict(
            zip(
                (
                    "rssm_feature_mse",
                    "static_feature_mse",
                    "rssm_progress_mae",
                    "static_progress_mae",
                ),
                torch.stack(rows).mean(0).tolist(),
            )
        )
        for h, rows in stats.items()
    }


def world(run):
    torch.manual_seed(725)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    frozen = run / "flow.pt"
    enc = CameraEncoder().cuda().eval()
    enc.load_state_dict(torch.load(frozen, weights_only=True)["encoder"])
    inp = run / "inputs"
    demos = sorted(inp.glob("demo-*.npz"))
    extra = sorted(inp.glob("extra-*.npz"))
    valid = sorted(inp.glob("valid-*.npz"))
    assert (len(demos), len(extra), len(valid)) == (72, 72, 16)
    dz, da = encode(demos, enc)
    ez, ea = encode(extra, enc)
    vz, va = encode(valid, enc)
    head = ProgressHead().cuda()
    target = progress_targets(da)
    opt = torch.optim.AdamW(head.parameters(), lr=0.001)
    for _ in range(1600):
        e = torch.randint(len(demos), (256,), device="cuda")
        t = torch.randint(3, 201, (256,), device="cuda")
        loss = F.smooth_l1_loss(head(dz[e, t])[:, 0], target[e, t], beta=0.06)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval().requires_grad_(False)

    def code(z):
        return torch.cat((z, head(z)), dim=-1)

    codes = code(torch.cat((dz[:24], ez)))
    acts = torch.cat((da[:24], ea))
    vcode = code(vz)
    valid_target = progress_targets(va[::4])
    valid_mae = float((vcode[::4, :, 16] - valid_target).abs().mean())
    wm = VisualRSSM().cuda()
    opt = torch.optim.AdamW(wm.parameters(), lr=0.0008)
    span = torch.arange(16, device="cuda")
    best = float("inf")
    history = []
    for it in range(1, 2401):
        episodes = torch.randint(len(codes), (48,), device="cuda")
        ticks = torch.randint(3, 185, (48, 1), device="cuda") + span
        loss = wm.loss(
            codes[episodes[:, None], ticks], acts[episodes[:, None], ticks[:, :-1]]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 5)
        opt.step()
        if it % 400 == 0:
            m = diagnostics(wm, vcode, va)
            history.append({"update": it, "diagnostics": m})
            print(it, m, flush=True)
            metric = m[12]["rssm_feature_mse"] + m[12]["rssm_progress_mae"]
            if metric < best:
                best = metric
                tmp = run / "world.tmp"
                torch.save(
                    {
                        "rssm": wm.state_dict(),
                        "progress": head.state_dict(),
                        "base_sha256": digest(frozen),
                        "update": it,
                    },
                    tmp,
                )
                tmp.replace(run / "world.pt")
    wm.load_state_dict(torch.load(run / "world.pt", weights_only=True)["rssm"])
    with torch.no_grad():
        starts = []
        seq = code(dz[:24])
        state = wm.initial(seq[:, 0])
        for t in range(180):
            if t >= 3 and t % 3 == 0:
                starts.append(state.cpu())
            state, _ = wm.filter(wm.advance(state, da[:24, t]), seq[:, t + 1])
        torch.save(torch.cat(starts), run / "visual-starts.pt")
    (run / "world.json").write_text(
        json.dumps(
            {
                "base_sha256": digest(frozen),
                "world_sha256": digest(run / "world.pt"),
                "observation": "16 frozen BC camera features and the camera-predicted demonstration progress",
                "world_train_demos": 24,
                "world_train_varied": 72,
                "goal_train_demos": 72,
                "validation_episodes": 16,
                "expert_progress_validation_mae": valid_mae,
                "candidates": history,
            },
            indent=2,
        )
    )


def actor(run, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    base = torch.load(run / "flow.pt", weights_only=True, map_location="cpu")
    trained = torch.load(run / "world.pt", weights_only=True, map_location="cpu")
    assert trained["base_sha256"] == digest(run / "flow.pt")
    pi = VisualFlowHead().eval().requires_grad_(False)
    pi.load_state_dict(base["flow"])
    wm = VisualRSSM().eval().requires_grad_(False)
    wm.load_state_dict(trained["rssm"])
    starts = torch.load(run / "visual-starts.pt", weights_only=True)
    reward = ProgressReward(len(starts))
    sac = NoiseSAC(wm.features)
    fit_steering(wm, pi, starts, 1500, 4, reward, sac=sac)
    assert reward.ticks == 576000
    tmp = run / f"actor-{seed}.tmp"
    torch.save(
        {
            "actor": sac.actor.state_dict(),
            "seed": seed,
            "base_sha256": digest(run / "flow.pt"),
            "world_sha256": digest(run / "world.pt"),
            "imagined_ticks": reward.ticks,
        },
        tmp,
    )
    tmp.replace(run / f"actor-{seed}.pt")
    print(seed, reward.ticks, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("world", "actor"))
    p.add_argument("run", type=Path)
    p.add_argument("--seed", type=int, default=101)
    args = p.parse_args()
    if args.mode == "world":
        world(args.run)
    else:
        actor(args.run, args.seed)
