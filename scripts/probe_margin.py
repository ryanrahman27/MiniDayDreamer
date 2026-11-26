"""Check whether the frozen model resolves a 9 mm release margin along archived action sequences."""

import argparse
from pathlib import Path

import numpy as np
import torch

from audit_margin import events, write_csv
from margin_study import controllers_for, dump
from minidaydreamer.placement_reward import relative_pose
from multiseed_study import TEST


@torch.no_grad()
def probe(run, old):
    _, wm, _ = controllers_for(run, old, [101], False)
    rows = []
    geometry = []
    for scene in TEST:
        with np.load(old / "test" / "traces" / f"prior80-101-{scene}.npz") as tr:
            e = events(tr)
            obs = torch.from_numpy(tr["obs"])
            actions = torch.from_numpy(tr["actions"])
        opened = e["first_open_tick"]
        if opened is None or not e["was_grasped_at_open"]:
            continue
        geometry.append(e)
        history = [wm.initial(obs[:1])]
        for t in range(opened):
            history.append(wm.filter(wm.advance(history[-1], actions[t : t + 1]), obs[t + 1 : t + 2])[0])
        actual_xy, _ = relative_pose(obs[opened : opened + 2])
        true_xy = float(actual_xy.max())
        for ticks in (3, 12):
            first = opened + 1 - ticks
            if first < 0:
                continue
            state = history[first]
            for t in range(first, opened + 1):
                before = wm.decode(state)
                state = wm.imagine(state, actions[t : t + 1])
            predicted_xy = float(relative_pose(torch.cat([before, wm.decode(state)]))[0].max())
            rows.append(
                dict(
                    scene=scene,
                    horizon_ticks=ticks,
                    actual_open_span_xy_mm=true_xy,
                    predicted_open_span_xy_mm=predicted_xy,
                    stationary_xy_mm=float(relative_pose(obs[first : first + 1])[0][0]),
                )
            )
    stats = {}
    for ticks in (3, 12):
        group = [r for r in rows if r["horizon_ticks"] == ticks]
        actual, pred, still = (
            np.array([r[k] for r in group])
            for k in ("actual_open_span_xy_mm", "predicted_open_span_xy_mm", "stationary_xy_mm")
        )

        def group_stats(mask):
            return dict(
                openings=int(mask.sum()),
                actual_inside=int((actual[mask] < 9).sum()),
                model_inside=int((pred[mask] < 9).sum()),
                model_correct_side=int(((pred[mask] < 9) == (actual[mask] < 9)).sum()),
                stationary_correct_side=int(((still[mask] < 9) == (actual[mask] < 9)).sum()),
                model_mae_mm=float(np.abs(pred[mask] - actual[mask]).mean()),
                stationary_mae_mm=float(np.abs(still[mask] - actual[mask]).mean()),
            )

        stats[ticks] = dict(
            all=group_stats(np.ones(len(actual), dtype=bool)), near_margin=group_stats(abs(actual - 9) <= 4)
        )
    write_csv(run / "archived-release-forecasts.csv", rows)
    groups = {}
    for label, success in (("stacked", 1), ("failed", 0)):
        subset = [e for e in geometry if e["success"] == success]
        xy = np.array([e["open_xy_mm"] for e in subset])
        height = np.array([e["open_height_error_mm"] for e in subset])
        groups[label] = dict(
            openings=len(subset),
            off_center=int((xy >= 12).sum()),
            above_8mm=int((height >= 8).sum()),
            both=int(((xy >= 12) & (height >= 8)).sum()),
            median_xy_mm=float(np.median(xy)),
            median_height_mm=float(np.median(height)),
        )
    dump(run / "archived-opening-geometry.json", groups)
    dump(run / "release-forecasts.json", stats)
    print(stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("old", type=Path)
    a = p.parse_args()
    torch.set_num_threads(1)
    probe(a.run, a.old)
