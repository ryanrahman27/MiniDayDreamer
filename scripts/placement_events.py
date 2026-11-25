"""Compare archived direct-steering losses with the successful base on the same seeds."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from diagnose_stack import first, release_events
from minidaydreamer.stack import STEP


def geometry(obs):
    xy = np.linalg.norm(obs[:, 3:5] - obs[:, 6:8], axis=1) * 100
    height = (obs[:, 5] - obs[:, 8]) * 100 - 46.8
    lifted = (obs[:, 9] > .5) & (obs[:, 5] - obs[0, 5] > .2)
    return xy, height, lifted


def event_ticks(trace):
    obs, actions, jaws = trace["obs"], trace["actions"], trace["aperture_m"]
    original = release_events(obs, actions, trace["proposed_grip"], jaws)
    xy, height, lifted = geometry(obs)
    target = first(lifted & (xy < 12))
    opened = original["first_open_step"]
    loss = original["first_loss_step"]
    loss_obs = loss + 1 if loss is not None else None
    cleared = None
    if opened is not None:
        # The first measured-open sample associated with the first continuous open command.
        duration = original["first_open_duration_steps"]
        reached = first(jaws[opened+1:opened+1+duration] > .035)
        if reached is not None:
            cleared = opened + 1 + reached
    held_over = lifted & (xy < 12)
    clearance = np.linalg.norm(obs[:, :3] - obs[:, 3:6], axis=1) * 100
    open_clear_aligned = ((xy < 12) & (abs(height) < 8) & (clearance > 50) &
                          (jaws > .035) & (obs[:, 10] < .5))
    row = dict(first_lift=original["lift_step"], first_over=target, first_open=opened,
               first_grasp_loss_obs=loss_obs, first_measured_open=cleared,
               opening_category=original["opening_category"], grasped_at_open=original["first_open_grasped"],
               first_open_steps=original["first_open_duration_steps"],
               ticks_over_to_open=opened-target if opened is not None and target is not None else None,
               ended_grasped=original["ended_grasped"], ended_closed=original["ended_closed"],
               final_xy_mm=float(xy[-1]), final_z_error_mm=float(height[-1]),
               final_jaw_mm=float(jaws[-1]*1000), final_clearance_mm=float(clearance[-1]), steps=len(actions),
               first_open_clear_aligned=first(open_clear_aligned), open_clear_aligned_obs=int(open_clear_aligned.sum()),
               min_centered_held_z_error_mm=float(height[held_over].min()) if held_over.any() else None,
               centered_held_in_z_band=bool((held_over & (abs(height) < 8)).any()))
    for name, t in (("over",target), ("open",opened), ("grasp_loss",loss_obs), ("measured_open",cleared)):
        row[name+"_xy_mm"] = float(xy[t]) if t is not None else None
        row[name+"_z_error_mm"] = float(height[t]) if t is not None else None
        row[name+"_jaw_mm"] = float(jaws[t]*1000) if t is not None else None
    full_window = target is not None and target+12 < len(obs)
    row["cube_dz_next_12_mm"] = float((obs[target+12,5]-obs[target,5])*100) if full_window else None
    row["mean_z_command_next_12_mm"] = float(actions[target:target+12,2].mean()*STEP*1000) if full_window else None
    return row


def median(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    return float(np.median(values)) if values else None


def counts(rows):
    opened = [r for r in rows if r["first_open"] is not None]
    held = [r for r in rows if r["first_open"] is None]
    return dict(episodes=len(rows), opened=len(opened), never_opened=len(held),
                ended_holding_among_never_open=sum(r["ended_grasped"] for r in held),
                opened_before_first_over=sum(r["ticks_over_to_open"] < 0 for r in opened if r["ticks_over_to_open"] is not None),
                opens_off_center=sum(r["open_xy_mm"] >= 12 for r in opened),
                opens_high=sum(r["open_z_error_mm"] >= 8 for r in opened),
                opens_low=sum(r["open_z_error_mm"] <= -8 for r in opened),
                opens_in_xyz=sum(r["open_xy_mm"] < 12 and abs(r["open_z_error_mm"]) < 8 for r in opened),
                losses_high_at_grasp_loss=sum(r["grasp_loss_z_error_mm"] is not None and r["grasp_loss_z_error_mm"] >= 8 for r in opened),
                losses_off_center_at_grasp_loss=sum(r["grasp_loss_xy_mm"] is not None and r["grasp_loss_xy_mm"] >= 12 for r in opened),
                median_open_z_error_mm=median(opened,"open_z_error_mm"), median_open_xy_mm=median(opened,"open_xy_mm"),
                median_ticks_over_to_open=median(opened,"ticks_over_to_open"),
                grasp_losses=sum(r["first_grasp_loss_obs"] is not None for r in rows),
                jaws_reached_open=sum(r["first_measured_open"] is not None for r in opened),
                median_z_at_grasp_loss_mm=median(opened,"grasp_loss_z_error_mm"),
                median_z_at_measured_open_mm=median(opened,"measured_open_z_error_mm"),
                median_final_z_never_open_mm=median(held,"final_z_error_mm"),
                median_min_centered_z_never_open_mm=median(held,"min_centered_held_z_error_mm"),
                never_open_always_high_while_centered=sum(r["min_centered_held_z_error_mm"] is not None
                    and r["min_centered_held_z_error_mm"] >= 8 for r in held),
                never_open_reached_xyz=sum(r["centered_held_in_z_band"] for r in held),
                reached_xyz_while_held=sum(r["centered_held_in_z_band"] for r in rows))


def diagnose(prior, out):
    archived = json.loads((prior / "midbase-summary.json").read_text())
    seeds = archived["paired"]["mid_varied_direct_vs_mid_base"]["lost_seeds"]
    assert len(set(seeds)) == len(seeds) == 24
    groups = {"mid_base": [], "mid_varied_direct": []}
    for seed in sorted(seeds):
        for arm, rows in groups.items():
            with np.load(prior / "traces" / f"{arm}-{seed}.npz") as tr:
                assert int(tr["seed"]) == seed and bool(tr["success"]) == (arm == "mid_base")
                rows.append(dict(seed=seed, arm=arm, **event_ticks(tr)))
    base, direct = groups.values()
    pairs = [(b,d) for b,d in zip(base,direct) if d["first_open"] is not None and b["first_open"] is not None]
    changes = {}
    for key in ("first_open", "ticks_over_to_open", "open_z_error_mm", "open_xy_mm"):
        delta = [d[key]-b[key] for b,d in pairs if d[key] is not None and b[key] is not None]
        changes[key] = dict(pairs=len(delta), median_difference=float(np.median(delta)) if delta else None,
                            lower=sum(v < 0 for v in delta), equal=sum(v == 0 for v in delta), higher=sum(v > 0 for v in delta))
    never_seeds = {r["seed"] for r in direct if r["first_open"] is None}
    descent = {}
    for name, rows in groups.items():
        matched = [r for r in rows if r["seed"] in never_seeds and r["cube_dz_next_12_mm"] is not None]
        descent[name] = dict(complete_windows=len(matched), rising=sum(r["cube_dz_next_12_mm"] > 0 for r in matched),
                             median_cube_dz_mm=median(matched,"cube_dz_next_12_mm"),
                             median_mean_z_command_mm=median(matched,"mean_z_command_next_12_mm"),
                             median_min_centered_z_error_mm=median(matched,"min_centered_held_z_error_mm"))
    result = dict(seeds=sorted(seeds), base=counts(base), direct=counts(direct), never_open_matched_descent=descent,
                  base_on_direct_open_subset=counts([b for b,_ in pairs]), paired_direct_minus_base=changes)
    with (out / "archived-placement-events.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(base[0]))
        writer.writeheader()
        writer.writerows(sorted(base + direct, key=lambda r: (r["seed"], r["arm"])))
    (out / "archived-placement-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prior", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    print(json.dumps(diagnose(args.prior, args.out)))
