"""Audit the fresh approach handoff trial and its comparison with the over-target handoff."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from climb_timing import median, timing, write_csv
from minidaydreamer.experiment import load_policies, verify_weights
from placement_events import first, geometry
from reward_study import digest, load
from summarize_handoff import verify_commands
from summarize_stack import paired
from target_handoff import APPROACH_ARMS as ARMS, APPROACH_SEEDS as SEEDS, APPROACH_RADIUS_MM


def compare(reference, alternative, seeds):
    ref=np.array([r["success"] for r in reference])
    alt=np.array([r["success"] for r in alternative])
    delta=alt-ref
    return dict(**paired(ref,alt), gained_seeds=seeds[delta==1].tolist(),lost_seeds=seeds[delta==-1].tolist())


def summarize(run, original, previous):
    prior=json.loads((original/"midbase-summary.json").read_text())
    base_hash,actor_hash=digest(run/"selected-base.pt"),digest(run/"mid-varied-direct.pt")
    assert base_hash==prior["selection"]["checkpoint_sha256"] and actor_hash==prior["actors"]["direct"]["sha256"]
    c,pi,wm,actor=load_policies(run/"mid-varied-direct.pt",SimpleNamespace(backend="isaac"))
    verify_weights(pi,load(run/"selected-base.pt")["policy"],"frozen baseline")
    assert (c["base_checkpoint_sha256"],c["coverage"],c["reward_mode"])==(base_hash,"varied","direct")
    excluded=set(load(run/"selected-base.pt")["demo_seeds"])
    for folder in previous.parent.glob("isaac-*-v[1-6]"):
        for path in folder.glob("eval-*.jsonl"):
            excluded.update(json.loads(line)["seed"] for line in path.read_text().splitlines())
    for key in ("corpus.npz","extra-bc.npz"):
        with np.load(original/key) as data:
            excluded.update(data["seeds"].tolist())
    for entry in json.loads((original/"selection.json").read_text())["development"]:
        excluded.update(r["seed"] for r in entry["outcomes"])
    assert not excluded.intersection(SEEDS)
    scenes=sorted((json.loads(line) for path in run.glob("eval-*.jsonl") for line in path.read_text().splitlines()),
                  key=lambda r:r["seed"])
    assert [r["seed"] for r in scenes]==list(SEEDS)
    records,prefixes=[],[]
    max_start=max_action=0.
    for scene in scenes:
        seed=scene["seed"]
        assert set(scene)=={"seed","start",*ARMS}
        starts,pre=[],{}
        for arm in ARMS:
            with np.load(run/"traces"/f"{arm}-{seed}.npz") as tr:
                obs,actions=tr["obs"],tr["actions"]
                metrics=scene[arm]
                assert int(tr["seed"])==seed and int(tr["success"])==metrics["success"]
                assert len(obs)-1==len(actions)==metrics["steps"]
                np.testing.assert_allclose(obs[0],scene["start"],atol=.01,rtol=0)
                starts.append(obs[0])
                xy,_,lifted=geometry(obs)
                lift=first(lifted)
                t80=first(lifted[:-1] & (xy[:-1]<APPROACH_RADIUS_MM))
                t12=first(lifted[:-1] & (xy[:-1]<12))
                if arm=="base":
                    trigger=None
                else:
                    trigger=t80 if arm=="approach" else t12
                    assert int(tr["steering_off_step"]) == (-1 if trigger is None else trigger)
                    pre[arm]=(t80,obs[:t80+1] if t80 is not None else obs)
                max_action=max(max_action,verify_commands(tr,arm,pi,wm,actor,trigger))
                assert bool(lifted.any())==metrics["lifted"] and np.isclose(xy[-1],metrics["final_xy_error_mm"])
                events,_=timing(tr)
                records.append(dict(seed=seed,arm=arm,**metrics,first_lift=lift,first_80=t80,first_12=t12,
                    switch_tick=trigger,lead_to_align=t12-t80 if t12 is not None else None,
                    never_open=lift is not None and not (actions[lift:,3]<=0).any(),
                    rising_at_alignment=events.get("rising_at_12"),descent_by_12_after=events.get("descent_by_12_after")))
        max_start=max(max_start,float(np.ptp(np.array(starts)[:,:9],axis=0).max()*100))
        a,early=pre["approach"]
        b,late=pre["handoff"]
        n=min(len(early),len(late))
        gap=float(np.linalg.norm((early[:n,:9]-late[:n,:9]).reshape(n,3,3),axis=2).max()*100)
        close=a==b and len(early)==len(late) and gap<=2 and np.array_equal(early[:n,9:],late[:n,9:])
        prefixes.append(dict(seed=seed,early_first80=a,late_first80=b,max_gap_mm=gap,close=bool(close)))
    assert len(records)==len(list((run/"traces").glob("*.npz")))==300 and max_start<=1
    by_arm={a:[r for r in records if r["arm"]==a] for a in ARMS}
    pairs={f"{b}_vs_{a}":compare(by_arm[a],by_arm[b],np.asarray(SEEDS))
           for a,b in (("handoff","approach"),("base","approach"),("base","handoff"))}
    policies={}
    for arm,rows in by_arm.items():
        triggered=[r for r in rows if r["switch_tick"] is not None]
        policies[arm]=dict(stacks=sum(r["success"] for r in rows),lifted=sum(r["lifted"] for r in rows),
            reached_80=sum(r["first_80"] is not None for r in rows),reached_12=sum(r["first_12"] is not None for r in rows),
            switches=len(triggered),switches_on_first_lift=sum(r["switch_tick"]==r["first_lift"] for r in triggered),
            median_switch_tick=median(r["switch_tick"] for r in triggered),
            median_80_to_12_ticks=median(r["lead_to_align"] for r in rows),
            never_open=sum(r["never_open"] for r in rows),
            rising_at_alignment=sum(r["rising_at_alignment"] is True for r in rows),
            descent_by_12_after=sum(r["descent_by_12_after"] is True for r in rows))
    close=np.array([r["close"] for r in prefixes])
    close_pair=compare([r for r,ok in zip(by_arm["handoff"],close) if ok],
                       [r for r,ok in zip(by_arm["approach"],close) if ok],np.asarray(SEEDS)[close]) if close.any() else None
    late_no_open=[e for h,e in zip(by_arm["handoff"],by_arm["approach"]) if h["never_open"]]
    result=dict(first_seed=SEEDS.start,last_seed=SEEDS.stop-1,episodes=len(records),policies=policies,paired=pairs,
                late_never_open=dict(scenes=len(late_no_open),opened_early=sum(not r["never_open"] for r in late_no_open),
                                     stacked_early=sum(r["success"] for r in late_no_open)),
                close_prefix_pairs=int(close.sum()),approach_vs_handoff_close=close_pair,
                max_start_coordinate_difference_mm=max_start,max_replayed_action_error=max_action,
                base_sha256=base_hash,direct_actor_sha256=actor_hash,approach_radius_mm=APPROACH_RADIUS_MM)
    write_csv(run/"paired-outcomes.csv",records)
    write_csv(run/"paired-prefixes.csv",prefixes)
    (run/"approach-summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(policies),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--previous",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    summarize(args.run,args.original,args.previous)
