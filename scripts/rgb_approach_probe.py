"""Learn the remaining expert carry from RGB features and demonstrated commands alone."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from train_visual_dreams import digest, encode

from minidaydreamer.vision import CameraEncoder


class ApproachProbe(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(16, 96), nn.SiLU(), nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, 4)
        )


def targets(actions):
    closed = actions[:, :, 3] > 0
    t = torch.arange(actions.shape[1], device=actions.device)[None, :]
    first = closed.float().argmax(1)
    released = ((~closed) & (t > first[:, None])).float().argmax(1)
    assert torch.all(released > first + 5)
    carry = (t >= first[:, None]) & (t <= released[:, None])
    # Relative command integrals are in action units; they do not measure cube distances.
    path = F.pad(actions[:, :, :3].cumsum(1), (0, 0, 1, 0))
    remaining = (
        path[torch.arange(len(actions), device=t.device), released][:, None]
        - path[:, :-1]
    )
    return remaining, carry


def fit(run):
    torch.manual_seed(913)
    encoder = CameraEncoder().cuda().eval()
    encoder.load_state_dict(torch.load(run / "flow.pt", weights_only=True)["encoder"])
    demos = sorted((run / "inputs").glob("demo-*.npz"))
    vals = sorted((run / "inputs").glob("valid-*.npz"))[::4]
    assert len(demos) == 72 and len(vals) == 4
    z, a = encode(demos, encoder)
    vz, va = encode(vals, encoder)
    y, mask = targets(a)
    vy, vmask = targets(va)
    scale = torch.quantile(y[mask].abs(), 0.85, dim=0).clamp_min(1.0)
    y = (y / scale).clamp(-3, 3)
    vy = (vy / scale).clamp(-3, 3)
    head = ApproachProbe().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.0008)
    ids = torch.nonzero(mask)
    for _ in range(1600):
        ix = ids[torch.randint(len(ids), (192,), device="cuda")]
        pred = head(z[ix[:, 0], ix[:, 1]])
        reg = F.smooth_l1_loss(pred[:, :3], y[ix[:, 0], ix[:, 1]], beta=0.13)
        e = torch.randint(len(z), (256,), device="cuda")
        t = torch.randint(3, a.shape[1], (256,), device="cuda")
        flag = F.binary_cross_entropy_with_logits(
            head(z[e, t])[:, 3], mask[e, t].float()
        )
        opt.zero_grad()
        (reg + 0.35 * flag).backward()
        opt.step()
    with torch.no_grad():
        pred = head(vz[:, :-1])
        errors = (pred[vmask][:, :3] - vy[vmask]).abs().mean(0)
        const = (vy[vmask] - y[mask].median(0).values).abs().mean(0)
        vec = vy[vmask][:, :2]
        commanded = vec.norm(dim=1) > 0.25
        agreement = F.cosine_similarity(
            pred[vmask][commanded, :2], vec[commanded], dim=1
        ).mean()
        metrics = {
            "remaining_xyz_mae": errors.tolist(),
            "constant_xyz_mae": const.tolist(),
            "horizontal_direction_cosine": float(agreement),
            "carry_phase_accuracy": float(
                ((pred[:, :, 3] > 0) == vmask).float().mean()
            ),
            "xyz_command_scales": scale.tolist(),
            "training_demos": len(demos),
            "validation_demos": len(vals),
        }
    tmp = run / "approach.tmp"
    torch.save(
        {
            "probe": head.state_dict(),
            "scale": scale.cpu(),
            "base_sha256": digest(run / "flow.pt"),
        },
        tmp,
    )
    tmp.replace(run / "approach.pt")
    (run / "approach.json").write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    fit(p.parse_args().run)
