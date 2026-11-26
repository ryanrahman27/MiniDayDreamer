"""Learn a two-view release template from the RGB and actions of successful demonstrations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_visual_dreams import digest

from minidaydreamer.vision import CameraEncoder


def cube_offsets(rgb):
    frames = rgb.reshape(-1, 2, 3, 96, 96).float()
    red, green, blue = frames.unbind(2)
    masks = torch.stack(
        (
            (red > green * 6 + 5) & (red > blue * 6 + 5),
            (blue > red * 5 + 4) & (blue > green * 5 + 4),
        ),
        2,
    ).float()
    axis = torch.arange(96, device=rgb.device, dtype=torch.float32)
    area = masks.sum((-2, -1))
    x = (masks.sum(-2) * axis).sum(-1) / area.clamp_min(1)
    y = (masks.sum(-1) * axis).sum(-1) / area.clamp_min(1)
    offset = torch.stack((x[:, :, 0] - x[:, :, 1], y[:, :, 0] - y[:, :, 1]), -1)
    return offset.flatten(1), (area.amin(dim=(1, 2)) >= 5)


class Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", torch.zeros(4))
        self.register_buffer("inverse", torch.eye(4))

    def similarity(self, offset, visible):
        delta = offset - self.center
        d2 = ((delta @ self.inverse) * delta).sum(-1)
        return (-0.125 * d2.clamp_min(0)).exp() * visible

    def forward(self, rgb):
        return self.similarity(*cube_offsets(rgb))


class ImaginedGeometry(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(16, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )


@torch.no_grad()
def features(paths, encoder):
    codes, offsets, visible, anchors = [], [], [], []
    for path in paths:
        with np.load(path) as ep:
            assert set(ep.files) == {"rgb", "actions"}
            rgb = torch.from_numpy(ep["rgb"]).cuda()
            actions = torch.from_numpy(ep["actions"])
        codes.append(torch.cat([encoder(x) for x in rgb.split(96)]))
        parts = [cube_offsets(x) for x in rgb.split(96)]
        offsets.append(torch.cat([p[0] for p in parts]))
        visible.append(torch.cat([p[1] for p in parts]))
        first = int(torch.nonzero(actions[:, 3] > 0)[0])
        releases = torch.nonzero(actions[first:, 3] <= 0)
        anchors.append(first + int(releases[0]) if releases.numel() else -1)
    return (
        torch.stack(codes),
        torch.stack(offsets),
        torch.stack(visible),
        torch.tensor(anchors, device="cuda"),
    )


def fit(run):
    torch.manual_seed(645)
    enc = CameraEncoder().cuda().eval().requires_grad_(False)
    enc.load_state_dict(torch.load(run / "flow.pt", weights_only=True)["encoder"])
    inp = run / "inputs"
    demos = sorted(inp.glob("demo-*.npz"))
    extras = sorted(inp.glob("extra-*.npz"))
    valid = sorted(inp.glob("valid-*.npz"))
    assert (len(demos), len(extras), len(valid)) == (72, 72, 16)
    z, x, vis, anchor = features(demos, enc)
    t = torch.arange(x.shape[1], device="cuda")[None, :]
    near = ((t - anchor[:, None]).abs() <= 2) & vis
    pts = x[near]
    center = pts.median(dim=0).values
    spread = (pts - center).abs().median(0).values.clamp_min(1)
    keep = ((pts - center).abs() / spread).amax(-1) < 5
    geometry = Geometry().cuda()
    geometry.center.copy_(center)
    cov = torch.cov((pts[keep] - center).T)
    # A few pixels cover subpixel centroid changes and partial finger occlusion.
    floor = cov.new_tensor([1.1, 1.25, 2.2, 2.0])
    geometry.inverse.copy_(torch.linalg.inv(cov + floor.square().diag()))
    ez, ex, ev, _ = features(extras, enc)
    vz, vx, vv, va = features(valid, enc)
    labels = geometry.similarity(
        torch.cat((x, ex)).flatten(0, 1), torch.cat((vis, ev)).flatten()
    )
    features_flat = torch.cat((z, ez)).flatten(0, 1)
    near_ids = torch.where(labels > 0.50)[0]
    head = ImaginedGeometry().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=0.001)
    for _ in range(2200):
        selected = torch.cat(
            (
                torch.randint(len(labels), (256,), device="cuda"),
                near_ids[torch.randint(len(near_ids), (128,), device="cuda")],
            )
        )
        loss = F.binary_cross_entropy(
            head(features_flat[selected]).flatten(), labels[selected]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        scores = geometry.similarity(vx.flatten(0, 1), vv.flatten()).reshape(
            vz.shape[:2]
        )
        vm = head(vz).squeeze(-1)
        exp_scores = scores[::4]
        expert_rel = va[::4]
        r = torch.arange(len(expert_rel), device="cuda")
        stats = {
            "demo_anchors": len(demos),
            "visible_anchor_frames": len(pts),
            "fit_anchor_frames": int(keep.sum()),
            "center_px": center.tolist(),
            "validation_episodes": len(valid),
            "expert_geometry_at_release": exp_scores[r, expert_rel].tolist(),
            "expert_geometry_15_ticks_before": exp_scores[r, expert_rel - 15].tolist(),
            "distilled_validation_mae": float((scores - vm).abs().mean()),
            "distilled_validation_corr": float(
                torch.corrcoef(torch.stack((scores.flatten(), vm.flatten())))[0, 1]
            ),
        }
    torch.save(
        {
            "geometry": geometry.cpu().state_dict(),
            "imagined": head.cpu().state_dict(),
            "base_sha256": digest(run / "flow.pt"),
        },
        run / "release-geometry.pt",
    )
    (run / "geometry.json").write_text(json.dumps(stats, indent=2))
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    fit(p.parse_args().run)
