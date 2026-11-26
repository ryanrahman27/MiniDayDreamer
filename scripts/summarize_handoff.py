"""Audit each per-tick handoff and compare all three frozen policies on matched Isaac scenes."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from minidaydreamer.experiment import load_policies, verify_weights
from minidaydreamer.models import ACTION_DIM, HORIZON
from placement_events import counts, diagnose, event_ticks, first, geometry
from reward_study import digest, load
from summarize_stack import paired
from target_handoff import ARMS, SEEDS


@torch.no_grad()
def verify_commands(trace, arm, pi, wm, actor, trigger):
    """Reconstruct both candidate chunks and test every executed command on its own observed history."""
    obs, actions = (torch.from_numpy(trace[k]) for k in ("obs","actions"))
    generator = torch.Generator().manual_seed(int(trace["seed"]))
    noise = torch.cat([torch.randn(1, HORIZON, ACTION_DIM, generator=generator)
                       for _ in range(0, len(actions), HORIZON)])
    starts = obs[:-1:HORIZON]
    plain = pi.sample(starts, noise).flatten(0, 1)[:len(actions)]
    if arm == "base":
        expected = plain
    else:
        state = wm.initial(obs[:1])
        history = []
        for t in range(len(actions)):
            if t % HORIZON == 0:
                history.append(state)
            state, _ = wm.filter(wm.advance(state, actions[t:t+1]), obs[t+1:t+2])
        shift = actor.draw(torch.cat(history), deterministic=True)[0]
        expected = pi.sample(starts, noise + shift[:, None]).flatten(0, 1)[:len(actions)]
        if arm in ("handoff", "approach") and trigger is not None:
            expected[trigger:] = plain[trigger:]
    torch.testing.assert_close(actions, expected, rtol=0., atol=2e-4)
    return float((actions - expected).abs().max())


def summarize(run, prior):
    source = json.loads((prior / "midbase-summary.json").read_text())
    base_hash, actor_hash = digest(run / "selected-base.pt"), digest(run / "mid-varied-direct.pt")
    assert base_hash == source["selection"]["checkpoint_sha256"] and actor_hash == source["actors"]["direct"]["sha256"]
    c, pi, wm, actor = load_policies(run / "mid-varied-direct.pt", SimpleNamespace(backend="isaac"))
    verify_weights(pi, load(run / "selected-base.pt")["policy"], "frozen base")
    assert c["base_checkpoint_sha256"] == base_hash and c["coverage"] == "varied" and c["reward_mode"] == "direct"
    excluded = set(load(run / "selected-base.pt")["demo_seeds"])
    for folder in prior.parent.glob("isaac-*-v[1-5]"):
        for path in folder.glob("eval-*.jsonl"):
            excluded.update(json.loads(line)["seed"] for line in path.read_text().splitlines())
    for name in ("corpus.npz", "extra-bc.npz"):
        with np.load(prior / name) as data:
            excluded.update(data["seeds"].tolist())
    for entry in json.loads((prior / "selection.json").read_text())["development"]:
        excluded.update(row["seed"] for row in entry["outcomes"])
    assert not excluded.intersection(SEEDS)
    scenes = sorted((json.loads(line) for f in run.glob("eval-*.jsonl") for line in f.read_text().splitlines()),
                    key=lambda scene: scene["seed"])
    assert [s["seed"] for s in scenes] == list(SEEDS)
    records = []
    prefixes = []
    max_start_error = max_action_error = 0.
    for scene in scenes:
        seed = scene["seed"]
        assert set(scene) == {"seed","start",*ARMS}
        starts = []
        before_switch = {}
        for arm in ARMS:
            m = scene[arm]
            with np.load(run / "traces" / f"{arm}-{seed}.npz") as tr:
                obs = tr["obs"]
                starts.append(obs[0])
                np.testing.assert_allclose(obs[0], scene["start"], atol=.01, rtol=0)
                assert int(tr["success"]) == m["success"] and int(tr["seed"]) == seed
                assert len(obs)-1 == len(tr["actions"]) == m["steps"]
                xy, _, lifted = geometry(obs)
                trigger = first(lifted[:-1] & (xy[:-1] < 12))
                if arm in ("direct", "handoff"):
                    before_switch[arm] = (trigger, obs[:trigger+1] if trigger is not None else obs)
                assert bool(lifted.any()) == m["lifted"] and np.isclose(xy[-1], m["final_xy_error_mm"])
                if arm == "handoff":
                    assert int(tr["steering_off_step"]) == (trigger if trigger is not None else -1)
                max_action_error = max(max_action_error, verify_commands(tr, arm, pi, wm, actor, trigger))
                e = event_ticks(tr)
                records.append(dict(seed=seed, arm=arm, **m, first_eligible=trigger,
                                    steering_off_step=trigger if arm=="handoff" else None,
                                    **{k:v for k,v in e.items() if k!="steps"}))
        max_start_error = max(max_start_error,float(np.ptp(np.array(starts)[:,:9],axis=0).max()*100))
        dt, d = before_switch["direct"]
        ht, h = before_switch["handoff"]
        n = min(len(d),len(h))
        gaps = (d[:n,:9] - h[:n,:9]).reshape(n,3,3)
        gap_mm = float(np.linalg.norm(gaps,axis=2).max()*100)
        matched = dt == ht and len(d) == len(h) and gap_mm <= 2 and np.array_equal(d[:n,9:],h[:n,9:])
        prefixes.append(dict(seed=seed, direct_trigger=dt, handoff_trigger=ht,
                             max_gap_mm=gap_mm, close_through_switch=bool(matched)))
    assert max_start_error <= 1 and len(records)==len(list((run/"traces").glob("*.npz")))==300
    rows = {arm:[r for r in records if r["arm"] == arm] for arm in ARMS}
    scores = {arm: np.array([r["success"] for r in values]) for arm,values in rows.items()}
    pairs = {}
    for ref,alt in (("direct","handoff"),("base","handoff"),("base","direct")):
        delta = scores[alt] - scores[ref]
        pairs[f"{alt}_vs_{ref}"] = dict(**paired(scores[ref],scores[alt]),
            gained_seeds=np.asarray(SEEDS)[delta==1].tolist(), lost_seeds=np.asarray(SEEDS)[delta==-1].tolist())
    stats = {}
    for arm,values in rows.items():
        stats[arm] = dict(stacks=int(scores[arm].sum()), lifts=sum(r["lifted"] for r in values),
                          first_eligible=sum(r["first_eligible"] is not None for r in values),
                          never_open_after_lift=sum(r["lifted"] and r["first_open"] is None for r in values))
    switched = [r["steering_off_step"] for r in rows["handoff"] if r["steering_off_step"] is not None]
    no_open = [i for i,r in enumerate(rows["direct"]) if r["lifted"] and r["first_open"] is None]
    direct_losses = (scores["base"] == 1) & (scores["direct"] == 0)
    result = dict(first_seed=SEEDS.start,last_seed=SEEDS.stop-1,episodes=len(records),policies=stats,paired=pairs,
                  trigger=dict(count=len(switched),midchunk=sum(t%HORIZON!=0 for t in switched),
                               median_tick=float(np.median(switched)) if switched else None),
                  failed_to_open_on_direct=dict(scenes=len(no_open),
                       opened_with_handoff=sum(rows["handoff"][i]["first_open"] is not None for i in no_open),
                       stacked_with_handoff=int(scores["handoff"][no_open].sum())),
                  fresh_base_success_direct_failure=dict(scenes=int(direct_losses.sum()),
                       stacked_with_handoff=int(scores["handoff"][direct_losses].sum()),
                       placement={arm:counts([r for r,keep in zip(rows[arm],direct_losses) if keep]) for arm in ARMS}),
                  max_start_coordinate_difference_mm=max_start_error,max_action_replay_error=max_action_error,
                  base_sha256=base_hash,direct_checkpoint_sha256=actor_hash)
    consistent = np.array([r["close_through_switch"] for r in prefixes])
    result["paired_prefix_check"] = dict(same_trigger_tick=sum(r["direct_trigger"]==r["handoff_trigger"] for r in prefixes),
                        close_pairs=int(consistent.sum()), median_max_gap_mm=float(np.median([r["max_gap_mm"] for r in prefixes])),
                        handoff_vs_direct=paired(scores["direct"][consistent],scores["handoff"][consistent]) if consistent.any() else None)
    with (run/"paired-prefixes.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(prefixes[0]))
        writer.writeheader()
        writer.writerows(prefixes)
    with (run/"paired-outcomes.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (run/"handoff-summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(stats), flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--prior",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    diagnose(args.prior,args.run)
    summarize(args.run,args.prior)
