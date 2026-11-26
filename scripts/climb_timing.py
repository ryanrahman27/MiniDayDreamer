"""Locate sustained cube ascent and descent in archived matched stacking losses."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from minidaydreamer.stack import STEP
from placement_events import first, geometry


RADII = (100, 80, 60, 50, 40, 30, 20, 12)
WIDTH = 5


def median(values):
    kept = [v for v in values if v is not None]
    return float(np.median(kept)) if kept else None


def timing(trace):
    obs, act = trace["obs"], trace["actions"]
    xy, height, lifted = geometry(obs)
    held = obs[:,9] > .5
    z = obs[:,5] * 100
    cmd = act[:,2] * (STEP * 1000)
    crossing = {r:first(lifted[:-1] & (xy[:-1] < r)) for r in RADII}
    t = crossing[12]
    row = dict(aligned=t is not None, align_tick=t, approach_tick=crossing[80])
    if t is None:
        return row, []
    # Use only the continuous grasp that led to alignment. End before commands to open that grasp.
    earlier = np.flatnonzero(~held[:t])
    start = int(earlier[-1] + 1) if len(earlier) else 0
    opened = first(act[start:,3] <= 0)
    lost = first(~held[start:])
    end = min(len(act), start + opened if opened is not None else len(act),
              start + lost - 1 if lost is not None else len(act))
    dz = np.diff(z)
    def events(signal, minimum_mean):
        windows = np.lib.stride_tricks.sliding_window_view(signal, WIDTH)
        valid = (np.arange(len(windows)) >= start) & (np.arange(len(windows))+WIDTH <= end)
        return [valid & (sign*windows.mean(axis=1) >= minimum_mean)
                & ((sign*windows > 0).sum(axis=1) >= WIDTH-1) for sign in (1,-1)]
    up, down = events(dz, .5)
    cmd_up, cmd_down = events(cmd, 1.)
    # Only anchors on this continuous carry can define approach onset.
    approach = first(lifted[start:t+1] & (xy[start:t+1] < 80)) + start
    row.update(carry_start=start, carry_end=end, approach_tick=approach, approach_lead=t-approach)
    for name, flags, after in (("climb",up,start), ("command_climb",cmd_up,start),
                               ("approach_climb",up,approach), ("descent",down,approach),
                               ("command_descent",cmd_down,approach)):
        hits = np.flatnonzero(flags & (np.arange(len(flags)) >= after))
        k = int(hits[0]) if len(hits) else None
        row.update({name+"_tick":k, name+"_relative_tick":k-t if k is not None else None,
                    name+"_xy_mm":float(xy[k]) if k is not None else None})
    row.update(rising_at_80=bool(up[approach]) if approach < len(up) else False,
               rising_at_12=bool(up[t]) if t < len(up) else False,
               descent_by_alignment=row["descent_tick"] is not None and row["descent_tick"] <= t,
               descent_by_12_after=row["descent_tick"] is not None and row["descent_tick"] <= t+12)
    radial = []
    for radius in RADII:
        k = first(lifted[start:t+1] & (xy[start:t+1] < radius)) + start
        if k + WIDTH > end:
            continue
        radial.append(dict(radius=radius, tick=k, relative_tick=k-t, height_error_mm=float(height[k]),
                           next_five_dz_mm=float(z[k+WIDTH]-z[k]), mean_z_command_mm=float(cmd[k:k+WIDTH].mean())))
    return row, radial


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def diagnose(run, original, previous):
    records, bands = [], []
    profiles = []
    for source, path, base, direct, summary, key in (
        ("original",original,"mid_base","mid_varied_direct","midbase-summary.json","mid_varied_direct_vs_mid_base"),
        ("replication",previous,"base","direct","handoff-summary.json","direct_vs_base")):
        seeds = json.loads((path/summary).read_text())["paired"][key]["lost_seeds"]
        for seed in seeds:
            with np.load(path/"traces"/f"{direct}-{seed}.npz") as tr:
                obs=tr["obs"]
                lift=first(geometry(obs)[2])
                never=lift is not None and not (tr["actions"][lift:,3] <= 0).any()
            for arm,label in ((base,"base"),(direct,"direct")):
                with np.load(path/"traces"/f"{arm}-{seed}.npz") as tr:
                    assert int(tr["seed"]) == seed and bool(tr["success"]) == (label=="base")
                    row,radial=timing(tr)
                    tags=dict(source=source,seed=seed,arm=label,direct_never_open=bool(never))
                    records.append(dict(**tags,**row))
                    bands.extend(dict(**tags,**entry) for entry in radial)
                    if never and row["aligned"]:
                        obs=tr["obs"]
                        xy,height,_=geometry(obs)
                        for rel in range(-25,17):
                            t=row["align_tick"]+rel
                            if row["carry_start"] <= t and t+WIDTH <= row["carry_end"]:
                                dz=np.diff(obs[t:t+WIDTH+1,5])*100
                                profiles.append(dict(source=source,seed=seed,arm=label,relative_tick=rel,
                                    after_approach=t>=row["approach_tick"],xy_mm=float(xy[t]),height_mm=float(height[t]),
                                    command_mm=float(tr["actions"][t:t+WIDTH,2].mean()*STEP*1000),
                                    dz_mm=float(dz.sum()),rising=bool(dz.mean()>=.5 and (dz>0).sum()>=WIDTH-1)))
    out={}
    for source in ("original","replication"):
        out[source]={}
        for never in (True,False):
            name="never_open" if never else "other_losses"
            subset=[r for r in records if r["source"]==source and r["direct_never_open"]==never]
            eligible=set(r["seed"] for r in subset if r["arm"]=="base" and r["aligned"]) & set(
                         r["seed"] for r in subset if r["arm"]=="direct" and r["aligned"])
            group=dict(seeds=len(subset)//2,aligned_pairs=len(eligible),arms={},radii={})
            for arm in ("base","direct"):
                rows=[r for r in subset if r["arm"]==arm and r["seed"] in eligible]
                group["arms"][arm]={k:sum(bool(r[k]) for r in rows) for k in
                                    ("rising_at_80","rising_at_12","descent_by_alignment","descent_by_12_after")}
                for event in ("climb","command_climb","approach_climb","descent","command_descent"):
                    times=[r[event+"_relative_tick"] for r in rows]
                    group["arms"][arm][event]=dict(observed=sum(t is not None for t in times),
                        median_relative_tick=median(times),median_xy_mm=median(r[event+"_xy_mm"] for r in rows))
            for radius in RADII:
                samples={a:{r["seed"]:r for r in bands if r["source"]==source and r["seed"] in eligible
                            and r["arm"]==a and r["radius"]==radius} for a in ("base","direct")}
                both=sorted(samples["base"].keys() & samples["direct"].keys())
                stats={"pairs":len(both)}
                for arm in samples:
                    stats[arm]={k:median(samples[arm][s][k] for s in both) for k in
                                ("relative_tick","height_error_mm","next_five_dz_mm","mean_z_command_mm")}
                stats["paired_dz_difference_mm"]=median(samples["direct"][s]["next_five_dz_mm"]-
                                                        samples["base"][s]["next_five_dz_mm"] for s in both)
                group["radii"][radius]=stats
            out[source][name]=group
    separation=[]
    for source in ("original","replication"):
        samples={(r["seed"],r["arm"],r["relative_tick"]):r for r in profiles if r["source"]==source}
        seeds=sorted({r["seed"] for r in records if r["source"]==source and r["direct_never_open"]})
        for seed in seeds:
            # Compare windows at the same time relative to each run's own alignment.
            candidates=[]
            for rel in range(-25,13):
                b=samples.get((seed,"base",rel))
                d=samples.get((seed,"direct",rel))
                if b and d and b["after_approach"] and d["after_approach"] and d["rising"] and b["dz_mm"]<=0:
                    candidates.append(d)
            hit=candidates[0] if candidates else None
            separation.append(dict(source=source,seed=seed,relative_tick=hit["relative_tick"] if hit else None,
                                   direct_xy_mm=hit["xy_mm"] if hit else None))
        hits=[r for r in separation if r["source"]==source and r["relative_tick"] is not None]
        out[source]["never_open"]["opposite_vertical_motion"]=dict(observed=len(hits),
            median_relative_tick=median(r["relative_tick"] for r in hits),median_xy_mm=median(r["direct_xy_mm"] for r in hits),
            before_alignment=sum(r["relative_tick"]<0 for r in hits))
    successes=[]
    for source,path,base,direct in (("original",original,"mid_base","mid_varied_direct"),
                                    ("replication",previous,"base","direct")):
        rows=[json.loads(line) for f in path.glob("eval-*.jsonl") for line in f.read_text().splitlines()]
        seeds=sorted(r["seed"] for r in rows if r[base]["success"] and r[direct]["success"])
        out[source]["both_stacked"]={}
        for arm,label in ((base,"base"),(direct,"direct")):
            arm_rows=[]
            for seed in seeds:
                with np.load(path/"traces"/f"{arm}-{seed}.npz") as tr:
                    assert int(tr["success"])==1
                    row,_=timing(tr)
                    arm_rows.append(dict(source=source,seed=seed,arm=label,**row))
            successes.extend(arm_rows)
            aligned=[r for r in arm_rows if r["aligned"]]
            out[source]["both_stacked"][label]=dict(scenes=len(seeds),aligned=len(aligned),
                rising_at_12=sum(r["rising_at_12"] for r in aligned),
                descent_by_12_after=sum(r["descent_by_12_after"] for r in aligned))
    write_csv(run/"successful-pair-timing.csv",successes)
    write_csv(run/"climb-separation.csv",separation)
    write_csv(run/"climb-events.csv",records)
    write_csv(run/"approach-distances.csv",bands)
    write_csv(run/"aligned-profiles.csv",profiles)
    (run/"climb-summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    args=p.parse_args()
    diagnose(args.run,args.original,args.previous)
