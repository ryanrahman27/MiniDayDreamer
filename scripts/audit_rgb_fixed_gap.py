"""Compare the two historical controllers on paired trials and identical saved camera states."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rgb_failure_events import dump_csv
from rgb_release_actor import ReleaseSAC
from rgb_release_geometry import ImaginedGeometry
from rgb_wait_actor import release_chunk
from train_visual_dreams import ProgressHead

from minidaydreamer.vision import VisualFlowHead
from minidaydreamer.visual_rssm import VisualRSSM


def median(vals):
    return float(np.median(vals)) if len(vals) else None


@torch.inference_mode()
def audit(prior, base, output):
    torch.set_num_threads(1)
    with (prior / "fresh-outcomes.csv").open() as f:
        records = {
            (r["arm"], int(r["scene"]), int(r["noise_seed"])): r
            for r in csv.DictReader(f)
        }
    flow = VisualFlowHead().eval()
    flow.load_state_dict(torch.load(base / "flow.pt", weights_only=True)["flow"])
    w = torch.load(base / "world.pt", weights_only=True)
    wm = VisualRSSM().eval()
    wm.load_state_dict(w["rssm"])
    head = ProgressHead().eval()
    head.load_state_dict(w["progress"])
    readiness = ImaginedGeometry().eval()
    readiness.load_state_dict(
        torch.load(prior / "release-geometry.pt", weights_only=True)["imagined"]
    )
    actors = {}
    for seed in (101, 211, 307, 401, 503):
        sac = ReleaseSAC(wm.features)
        sac.actor.load_state_dict(
            torch.load(prior / f"release-{seed}.pt", weights_only=True)["actor"]
        )
        actors[seed] = sac
    old = "release-permissive"
    fixed = "fixed-permissive"
    detail = {}
    cf = []
    for path in sorted((prior / "test-release").glob("[0-9]*.json")):
        rows = json.loads(path.read_text())
        arm = rows[0]["arm"]
        if arm not in (old, fixed):
            continue
        with np.load(path.with_suffix(".npz")) as d:
            dec, states, actions, features = (
                d[k] for k in ("release_decisions", "obs", "actions", "visual_features")
            )
        z = torch.from_numpy(features).float()
        act = torch.from_numpy(actions)
        initial = torch.cat((z[0], head(z[0])), 1)
        latent = wm.initial(initial)
        rng = [torch.Generator().manual_seed(r["noise_seed"]) for r in rows]
        for r in rows:
            key = arm, r["scene"], r["noise_seed"]
            rec = records[key]
            detail[key] = {
                "success": r["success"],
                "mode": rec["mode"],
                "shift": None,
                "shift_tick": None,
                "height": None,
                "xy": None,
                "open_delay": None,
            }
        for t in range(len(act)):
            if t % 3 == 0:
                noise = torch.cat([torch.randn(1, 3, 4, generator=g) for g in rng])
                for i, r in enumerate(rows):
                    key = arm, r["scene"], r["noise_seed"]
                    v = detail[key]
                    recorded = dec[t // 3, i, 2]
                    if v["shift"] is not None or not recorded or t >= r["steps"]:
                        continue
                    st = states[t, i]
                    v["height"] = 100 * float(st[5] - st[8]) - 46.8
                    v["xy"] = 100 * float(np.linalg.norm(st[3:5] - st[6:8]))
                    v["shift"] = float(recorded)
                    v["shift_tick"] = t
                    opening = np.flatnonzero(actions[t : r["steps"], i, 3] <= 0)
                    if len(opening):
                        v["open_delay"] = int(opening[0])
                    code = z[t, i : i + 1]
                    n = noise[i : i + 1]
                    nom = flow.sample(code, n)
                    prop = actors[r["actor_seed"]].draw(latent[i : i + 1], True)[0][
                        :, 3
                    ]
                    ai, ash = release_chunk(flow, code, n, prop, nom)
                    fi, fsh = release_chunk(
                        flow, code, n, torch.full_like(prop, -14.0), nom
                    )
                    actual = ai if arm == old else fi
                    observed = act[t : t + 3, i]
                    cf.append(
                        {
                            "arm": arm,
                            "scene": r["scene"],
                            "noise_seed": r["noise_seed"],
                            "actor_proposal": float(prop.item()),
                            "image_readiness": float(dec[t // 3, i, 0]),
                            "reward_readout": float(readiness(code).item()),
                            "actor_shift": float(ash.item()),
                            "fixed_shift": float(fsh.item()),
                            "actor_all_open": bool((ai[:, :, 3] <= 0).all()),
                            "fixed_all_open": bool((fi[:, :, 3] <= 0).all()),
                            "actor_drift": float(
                                (ai[:, :, :3] - nom[:, :, :3]).abs().mean()
                            ),
                            "fixed_drift": float(
                                (fi[:, :, :3] - nom[:, :, :3]).abs().mean()
                            ),
                            "command_reconstruction_mae": float(
                                (actual[0, : len(observed)] - observed).abs().mean()
                            ),
                        }
                    )
            code = torch.cat((z[t + 1], head(z[t + 1])), 1)
            latent, _ = wm.filter(wm.advance(latent, act[t]), code)
    pairs = []
    reasons = {}
    for k, fix in detail.items():
        if k[0] != fixed:
            continue
        _, s, n = k
        rl = detail[old, s, n]
        group = (
            "fixed_only"
            if fix["success"] > rl["success"]
            else "dsrl_only"
            if rl["success"] > fix["success"]
            else "agree"
        )
        pairs.append(
            {
                "scene": s,
                "noise_seed": n,
                "group": group,
                **{"fixed_" + k: v for k, v in fix.items()},
                **{"dsrl_" + k: v for k, v in rl.items()},
            }
        )
    for group in ("fixed_only", "dsrl_only", "agree"):
        g = [r for r in pairs if r["group"] == group]
        reasons[group] = {
            "trials": len(g),
            "neither_shifted": sum(
                r["fixed_shift"] is None and r["dsrl_shift"] is None for r in g
            ),
            "only_one_shifted": sum(
                (r["fixed_shift"] is None) != (r["dsrl_shift"] is None) for r in g
            ),
            "both_shifted": sum(
                r["fixed_shift"] is not None and r["dsrl_shift"] is not None for r in g
            ),
        }
        if group != "agree":
            loser = "dsrl_" if group == "fixed_only" else "fixed_"
            reasons[group]["failed_modes"] = dict(Counter(r[loser + "mode"] for r in g))
    summary = {"pairs": reasons}
    for arm in (fixed, old):
        ar = [v for k, v in detail.items() if k[0] == arm and v["shift"] is not None]
        summary[arm] = {
            "episodes_with_shift": len(ar),
            "first_shift_median": median([v["shift"] for v in ar]),
            "median_first_height": median([v["height"] for v in ar]),
            "median_first_xy": median([v["xy"] for v in ar]),
            "open_within_6": sum(
                v["open_delay"] is not None and v["open_delay"] < 6 for v in ar
            ),
        }
    summary["same_observation"] = {
        "first_interventions": len(cf),
        "actor_proposal_median": median([r["actor_proposal"] for r in cf]),
        "actor_applied_median": median([r["actor_shift"] for r in cf]),
        "fixed_applied_median": median([r["fixed_shift"] for r in cf]),
        "actor_opens_whole_chunk": sum(r["actor_all_open"] for r in cf),
        "fixed_opens_whole_chunk": sum(r["fixed_all_open"] for r in cf),
        "fixed_only_full_open": sum(
            r["fixed_all_open"] and not r["actor_all_open"] for r in cf
        ),
        "actor_only_full_open": sum(
            r["actor_all_open"] and not r["fixed_all_open"] for r in cf
        ),
        "actor_mean_xyz_drift": float(np.mean([r["actor_drift"] for r in cf])),
        "fixed_mean_xyz_drift": float(np.mean([r["fixed_drift"] for r in cf])),
        "reconstruction_mae": float(
            np.mean([r["command_reconstruction_mae"] for r in cf])
        ),
    }
    for opened in (False, True):
        subset = [r for r in cf if r["actor_all_open"] == opened]
        summary["same_observation"]["full_open" if opened else "incomplete_open"] = {
            k: median([r[k] for r in subset])
            for k in ("actor_proposal", "image_readiness", "reward_readout")
        }
    dump_csv(output / "historical-pairs.csv", pairs)
    dump_csv(output / "same-observation.csv", cf)
    (output / "historical-gap.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "base", "output"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    audit(a.prior, a.base, a.output)
