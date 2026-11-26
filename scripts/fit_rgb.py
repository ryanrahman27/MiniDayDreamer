"""Train an RGB-conditioned flow policy using only byte images and demonstration actions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.models import HORIZON
from minidaydreamer.vision import CameraEncoder, VisualFlowHead


def export(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for kind, count in (("demo", 72), ("extra", 72), ("valid", 16)):
        files = sorted(source.glob(f"{kind}-*.npz"))
        if len(files) != count:
            raise ValueError(f"Expected {count} {kind} recordings")
        for path in files:
            out = dest / path.name
            if out.exists():
                with np.load(out) as existing:
                    assert set(existing.files) == {"rgb", "actions"}
                continue
            with np.load(path) as raw, out.with_suffix(".tmp").open("wb") as f:
                np.savez(f, rgb=raw["rgb"], actions=raw["actions"])
            out.with_suffix(".tmp").replace(out)


def read(paths, device):
    x, a = [], []
    for path in paths:
        with np.load(path) as record:
            if set(record.files) != {"rgb", "actions"}:
                raise ValueError(
                    "Training recordings must contain exactly RGB frames and actions"
                )
            x.append(torch.from_numpy(record["rgb"]))
            a.append(torch.from_numpy(record["actions"]))
    return torch.stack(x).to(device), torch.stack(a).to(device)


def active_lengths(actions):
    # The expert parks after withdrawing. Trim its long idle tail using only the command trace.
    moving = actions[:, :, :3].abs().amax(-1) > 0.12
    time = torch.arange(actions.shape[1], device=actions.device)
    return (moving * time).amax(1).add(10).clamp(12, actions.shape[1] - HORIZON)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("export", "bc"))
    p.add_argument("run", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--steps", type=int, default=8000)
    args = p.parse_args()
    if args.mode == "export":
        export(args.source, args.run / "inputs")
        return
    torch.manual_seed(812)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    files = sorted((args.run / "inputs").glob("demo-*.npz"))
    assert len(files) == 72
    x, a = read(files, "cuda")
    val = sorted((args.run / "inputs").glob("valid-*.npz"))[::4]
    vx, va = read(val, "cuda")
    enc, pi = CameraEncoder().cuda(), VisualFlowHead().cuda()
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(pi.parameters()), lr=5e-4, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=5e-5)
    lengths = active_lengths(a)
    changes = torch.nonzero((a[:, 1:, 3] * a[:, :-1, 3]) < 0)
    ve, vt = torch.where(
        (
            torch.arange(va.shape[1], device="cuda")[None, :]
            < active_lengths(va)[:, None]
        )
        & (torch.arange(va.shape[1], device="cuda")[None, :] >= 3)
    )
    span = torch.arange(HORIZON, device="cuda")
    gt = va[ve[:, None], vt[:, None] + span]
    noise = torch.randn(
        len(gt),
        HORIZON,
        4,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(399),
    )
    history = []
    best = float("inf")
    for update in range(1, args.steps + 1):
        enc.train()
        pi.train()
        ep = torch.randint(len(files), (96,), device="cuda")
        tick = (torch.rand(96, device="cuda") * (lengths[ep] - 3)).long() + 3
        switching = changes[torch.randint(len(changes), (16,), device="cuda")]
        ep[:16] = switching[:, 0]
        tick[:16] = (
            switching[:, 1] + torch.randint(-5, 6, (16,), device="cuda")
        ).clamp(3, a.shape[1] - HORIZON)
        obs = enc(x[ep, tick])
        target = a[ep[:, None], tick[:, None] + span]
        loss = pi.loss(obs, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(pi.parameters()), 5
        )
        opt.step()
        sched.step()
        if update % 500 == 0 or update == args.steps:
            with torch.no_grad():
                enc.eval()
                pi.eval()
                z = torch.cat([enc(b) for b in vx[ve, vt].split(128)])
                actions = pi.sample(z, noise)
                value = (actions - gt).abs().mean().item()
                log = {
                    "update": update,
                    "expert_action_mae": value,
                    "gripper_agreement": float(
                        ((actions[:, :, 3] > 0) == (gt[:, :, 3] > 0)).float().mean()
                    ),
                }
                if value < best:
                    best = value
                    ck = {
                        "encoder": enc.state_dict(),
                        "flow": pi.state_dict(),
                        "update": update,
                    }
                    path = args.run / "flow.tmp"
                    torch.save(ck, path)
                    path.replace(args.run / "flow.pt")
                print(json.dumps(log), flush=True)
                history.append(log)
    (args.run / "bc.json").write_text(
        json.dumps(
            {
                "input_keys": ["rgb", "actions"],
                "training_episodes": len(files),
                "validation_demonstration_sequences": len(val),
                "candidates": history,
                "sha256": hashlib.sha256(
                    (args.run / "flow.pt").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
