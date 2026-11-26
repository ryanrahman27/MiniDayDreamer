"""Estimate signed height and lateral image errors from RGB/action demonstrations."""

import argparse
import json
from pathlib import Path

import torch
from rgb_release_geometry import Geometry, cube_offsets, features
from torch import nn
from torch.nn import functional as F
from train_visual_dreams import ProgressHead, digest

from minidaydreamer.vision import CameraEncoder
from minidaydreamer.visual_rssm import VisualRSSM


class PlacementView(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = Geometry()
        self.register_buffer("whitener", torch.eye(4))
        self.register_buffer("up", torch.zeros(4))

    def components(self, errors):
        height = (errors * self.up).sum(-1, keepdim=True)
        lateral = (errors - height * self.up).norm(dim=-1, keepdim=True)
        return torch.cat((height, lateral), -1)

    def errors(self, offset):
        return (offset - self.reference.center) @ self.whitener

    def forward(self, rgb):
        offset, visible = cube_offsets(rgb)
        errors = self.errors(offset)
        return (
            self.reference.similarity(offset, visible),
            self.components(errors),
            visible,
        )


class PlacementHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(16, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 4)
        )


def fit(run):
    torch.manual_seed(766)
    dev = "cuda"
    base = torch.load(run / "flow.pt", weights_only=True, map_location="cpu")
    world = torch.load(run / "world.pt", weights_only=True, map_location="cpu")
    encoder = CameraEncoder().to(dev).eval().requires_grad_(False)
    encoder.load_state_dict(base["encoder"])
    view = PlacementView().to(dev)
    prior = torch.load(
        run / "release-geometry.pt", weights_only=True, map_location="cpu"
    )
    assert prior["base_sha256"] == world["base_sha256"] == digest(run / "flow.pt")
    view.reference.load_state_dict(prior["geometry"])
    eig, rot = torch.linalg.eigh(view.reference.inverse)
    view.whitener.copy_((rot * eig.clamp_min(0).sqrt()) @ rot.T)
    inp = run / "inputs"
    demo, extra, valid = [
        sorted(inp.glob(k + "-*.npz")) for k in ("demo", "extra", "valid")
    ]
    assert list(map(len, (demo, extra, valid))) == [72, 72, 16]
    z, x, vis, anchor = features(demo, encoder)
    ids = torch.arange(len(demo), device=dev)
    delta = view.errors(x[ids, anchor - 14]) - view.errors(x[ids, anchor])
    ok = vis[ids, anchor - 14] & vis[ids, anchor]
    axis = delta[ok].median(0).values
    view.up.copy_(F.normalize(axis, dim=0))
    ez, ex, ev, _ = features(extra, encoder)
    vz, vx, vv, va = features(valid, encoder)
    inp_z = torch.cat((z, ez)).flatten(0, 1)
    truth = view.errors(torch.cat((x, ex)).flatten(0, 1))
    visible = torch.cat((vis, ev)).flatten()
    near = visible & (truth.norm(dim=-1) < 10)
    all_ids = torch.where(visible)[0]
    near_ids = torch.where(near)[0]
    head = PlacementHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=0.001)
    for _ in range(2500):
        ix = torch.cat(
            (
                all_ids[torch.randint(len(all_ids), (96,), device=dev)],
                near_ids[torch.randint(len(near_ids), (288,), device=dev)],
            )
        )
        loss = F.smooth_l1_loss(head(inp_z[ix]), truth[ix].clamp(-14, 14), beta=0.6)
        opt.zero_grad()
        loss.backward()
        opt.step()
    wm = VisualRSSM().to(dev).eval().requires_grad_(False)
    wm.load_state_dict(world["rssm"])
    progress = ProgressHead().to(dev).eval()
    progress.load_state_dict(world["progress"])
    import numpy as np

    def actions(paths):
        parts = []
        for p in paths:
            with np.load(p) as f:
                assert set(f.files) == {"rgb", "actions"}
                parts.append(torch.from_numpy(f["actions"]))
        return torch.stack(parts).to(dev)

    with torch.no_grad():
        # Restrict posterior starts to sustained closed commands in the approach region.
        allz = torch.cat((z, ez))
        allx = torch.cat((x, ex))
        allv = torch.cat((vis, ev))
        a = actions(demo + extra)
        code = torch.cat((allz, progress(allz)), 2)
        s = wm.initial(code[:, 0])
        starts = []
        anchors = []
        for t in range(190):
            if t >= 15 and t % 2 == 0:
                e = view.errors(allx[:, t])
                mask = (
                    (a[:, t - 12 : t, 3] > 0).all(1)
                    & allv[:, t]
                    & (e.norm(dim=-1) < 12)
                )
                if mask.any():
                    starts.append(s[mask].cpu())
                    anchors.append(e[mask].cpu())
            s, _ = wm.filter(wm.advance(s, a[:, t]), code[:, t + 1])
        ve = view.errors(vx)
        vnear = vv & (ve.norm(dim=-1) < 8)
        pred = view.components(head(vz))
        observed = view.components(ve)
        residual = (pred[vnear] - observed[vnear]).abs().mean(0)
        vals = observed[::4]
        anc = va[::4]
        n = torch.arange(len(anc), device=dev)
        metrics = {
            "demo_pairs": int(ok.sum()),
            "start_states": sum(map(len, starts)),
            "validation_near_frames": int(vnear.sum()),
            "real_feature_mae_height_lateral": residual.tolist(),
            "up_direction": view.up.tolist(),
            "validation_expert_at_release": vals[n, anc].tolist(),
            "validation_expert_14_ticks_early": vals[n, anc - 14].tolist(),
        }
    torch.save(
        {
            "view": view.cpu().state_dict(),
            "head": head.cpu().state_dict(),
            "base_sha256": digest(run / "flow.pt"),
            "world_sha256": digest(run / "world.pt"),
        },
        run / "placement-view.pt",
    )
    torch.save(
        {"latent": torch.cat(starts), "errors": torch.cat(anchors)},
        run / "placement-starts.pt",
    )
    (run / "placement-view.json").write_text(json.dumps(metrics, indent=2))
    print(metrics, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    fit(p.parse_args().run)
