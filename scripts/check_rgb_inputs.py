"""Check image dependence against the expert actions at the same task tick in another scene."""

import argparse
import json
from pathlib import Path

import torch
from fit_rgb import active_lengths, read

from minidaydreamer.models import HORIZON
from minidaydreamer.vision import CameraEncoder, VisualFlowHead


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    ck = torch.load(args.run / "flow.pt", weights_only=True, map_location="cpu")
    enc, pi = CameraEncoder().cuda().eval(), VisualFlowHead().cuda().eval()
    enc.load_state_dict(ck["encoder"])
    pi.load_state_dict(ck["flow"])
    x, a = read(sorted((args.run / "inputs").glob("valid-*.npz"))[::4], "cuda")
    z = torch.cat([enc(v) for v in x.flatten(0, 1).split(128)]).reshape(
        len(x), x.shape[1], -1
    )
    tt = torch.arange(a.shape[1], device="cuda")[None, :]
    e, t = torch.where((tt >= 3) & (tt < active_lengths(a)[:, None]))
    target = a[e[:, None], t[:, None] + torch.arange(HORIZON, device="cuda")]
    eps = torch.randn(
        len(t),
        HORIZON,
        4,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(399),
    )

    def errors(features):
        diff = (pi.sample(features, eps) - target).abs()
        return {
            "all_actions": float(diff.mean()),
            "cartesian": float(diff[:, :, :3].mean()),
        }

    matches = errors(z[e, t])
    controls = [errors(z[(e + shift) % len(x), t]) for shift in range(1, len(x))]
    out = {
        "matched": matches,
        "other_scene_same_tick": {
            k: sum(d[k] for d in controls) / len(controls) for k in matches
        },
    }
    (args.run / "image-check.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
