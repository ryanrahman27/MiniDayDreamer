"""Check imagined descent against recorded unsteered RGB on real wait opportunities."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rgb_wait_actor import WaitActor
from rgb_wait_geometry import PlacementHead, PlacementView
from train_visual_dreams import ProgressHead

from minidaydreamer.visual_rssm import VisualRSSM


@torch.inference_mode()
def audit(run, base, split):
    torch.set_num_threads(1)
    chosen = json.loads((run / "selection.json").read_text())["selected"]
    folder = run / f"{split}-wait"
    plan = json.loads((folder / "protocol.json").read_text())
    high, rad, threshold = plan["limits"][chosen]
    ck = torch.load(run / "placement-view.pt", weights_only=True)
    head = PlacementHead().eval()
    head.load_state_dict(ck["head"])
    view = PlacementView().eval()
    view.load_state_dict(ck["view"])
    w = torch.load(base / "world.pt", weights_only=True, map_location="cpu")
    wm = VisualRSSM().eval()
    wm.load_state_dict(w["rssm"])
    progress = ProgressHead().eval()
    progress.load_state_dict(w["progress"])
    actors = {}
    scales = set()
    for seed in plan["actor_seeds"]:
        actor = WaitActor().eval()
        checkpoint = torch.load(run / f"wait-{seed}.pt", weights_only=True)
        actor.load_state_dict(checkpoint["actor"])
        scales.add(float(checkpoint.get("imagined_height_scale", 1.0)))
        actors[seed] = actor
    if len(scales) != 1:
        raise ValueError("The cohort must share a forecast calibration")
    height_scale = scales.pop()
    records = defaultdict(list)
    for path in sorted(folder.glob("*-rgb-base.json")):
        rows = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz")) as d:
            z = torch.from_numpy(d["visual_features"]).float()
            a = torch.from_numpy(d["actions"])
            ready = d["release_decisions"][:, :, 0]
            vis = d["placement_decisions"]
        code = torch.cat((z, progress(z)), 2)
        state = wm.initial(code[0])
        post = [state]
        for t in range(len(a)):
            state, _ = wm.filter(wm.advance(state, a[t]), code[t + 1])
            post.append(state)
        for row in rows:
            i = row["lane"]
            n = row["steps"]
            k0 = None
            streak = 0
            for k in range((n - 2) // 3):
                t = 3 * k
                h, r, visible = vis[k, i, :3]
                good = (
                    t >= 12
                    and (a[t - 12 : t, i, 3] > 0).all()
                    and ready[k, i] >= 0.28
                    and visible
                    and -2.15 <= h <= high
                    and r <= rad
                )
                streak = streak + 1 if good else 0
                if streak >= 2 and (a[t : t + 3, i, 3] > 0).any():
                    k0 = k
                    break
            if k0 is None:
                continue
            t = 3 * k0
            pred = post[t][i : i + 1]
            h0 = float(vis[k0, i, 0])
            pred0 = float((head(wm.decode(pred)[:, :16]) * view.up).sum())
            adv = float(
                actors[row["actor_seed"]](
                    z[t, i : i + 1],
                    torch.from_numpy(vis[k0, i : i + 1, :2]),
                    a[t : t + 3, i].unsqueeze(0),
                ).item()
            )
            for dt in range(1, min(24, n - t) + 1):
                if a[t + dt - 1, i, 3] <= 0:
                    break
                pred = wm.imagine(pred, a[t + dt - 1, i : i + 1], sample=False)
                if dt not in (3, 6, 12, 24) or k0 + dt // 3 >= len(vis):
                    continue
                future = vis[k0 + dt // 3, i]
                if not future[2]:
                    continue
                seen = float(future[0]) - h0
                imagined = (
                    float((head(wm.decode(pred)[:, :16]) * view.up).sum()) - pred0
                )
                records[dt].append((seen, imagined, adv < threshold))
    out = {}
    if split == "dev" and records[6]:
        cand = np.asarray(records[6])
        observed, predicted = cand[:, 0], cand[:, 1]
        scale = float(
            np.clip(
                (observed @ predicted) / max(float(predicted @ predicted), 1e-9), 0, 1
            )
        )
        (run / "dev-height-calibration.json").write_text(
            json.dumps(
                {
                    "scale": scale,
                    "sample_count": len(cand),
                    "horizon": 6,
                    "source": "unsteered development RGB and executed commands",
                },
                indent=2,
            )
        )
    for dt, examples in records.items():
        values = np.asarray(examples)

        def metrics(v):
            if not len(v):
                return {"count": 0}
            seen, pred = v[:, 0], v[:, 1]
            return {
                "count": len(v),
                "median_observed_change": float(np.median(seen)),
                "median_predicted_change": float(np.median(pred)),
                "rssm_mae": float(np.abs(pred - seen).mean()),
                "training_height_scale": height_scale,
                "scaled_median_predicted_change": float(np.median(height_scale * pred)),
                "scaled_mae": float(np.abs(height_scale * pred - seen).mean()),
                "static_mae": float(np.abs(seen).mean()),
                "mean_prediction_minus_observed": float(np.mean(pred - seen)),
            }

        out[dt] = {
            "all": metrics(values),
            "actor_would_wait": metrics(values[values[:, 2] > 0]),
        }
    (run / f"{split}-dynamics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("base", type=Path)
    p.add_argument("--split", choices=("dev", "test"), default="test")
    a = p.parse_args()
    audit(a.run, a.base, a.split)
