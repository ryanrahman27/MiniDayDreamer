"""Audit paired Isaac scores, frozen checkpoints, and data splits for the reward comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from check_state_reward import forecast, from_traces, observed_report, policy_recordings
from minidaydreamer.models import ACTION_DIM, HORIZON, NoiseSAC, make_policy, make_world
from minidaydreamer.stack import STEP
from reward_study import digest, load
from summarize_stack import paired


def equal_weights(actual, expected, name):
    if actual.keys() != expected.keys() or any(not torch.equal(actual[k], v) for k, v in expected.items()):
        raise ValueError(f"Different weights: {name}")


def audit_training(run, short, archived, seeds):
    original = load(run / "base.pt")
    selected = load(run / "selected-base.pt")
    assert digest(run / "base.pt") == digest(short / "base.pt")
    assert digest(run / "corpus.npz") == original["corpus_sha256"]
    assert selected["source_sha256"] == digest(run / "extra-bc.npz")
    history = json.loads((run / "calibration.json").read_text())
    assert selected["calibration"] == history
    best = max(history, key=lambda c: c["stacks"])
    assert best["step"] == selected["bc_steps"]
    assert len(history) == (1 if history[0]["stacks"] >= 4 else 2)
    equal_weights(selected["policy"], load(run / f"candidate-{best['step']}.pt")["policy"], "calibrated base")
    dev = set(range(5200, 5212))
    for candidate in history:
        assert [row["seed"] for row in candidate["outcomes"]] == sorted(dev)
        assert candidate["stacks"] == sum(row["success"] for row in candidate["outcomes"])
    old_dev = {o["seed"] for c in original["calibration"] for o in c["outcomes"]}
    with np.load(run / "corpus.npz") as data, np.load(run / "extra-bc.npz") as extra:
        training = set(data["seeds"])
        extra_demos = set(extra["seeds"][extra["split"] == "demo"])
        demo = set(data["seeds"][data["split"] == "demo"])
        assert len(extra_demos) == 48 and len(demo) == 24 and not training & extra_demos
        assert set(selected["demo_seeds"]) == demo | extra_demos
    previous_eval = {json.loads(line)["seed"] for f in archived.glob("eval-*.jsonl") for line in f.read_text().splitlines()}
    diagnostics = set(json.loads((run / "reward-audit.json").read_text())["diagnostic_seeds"])
    assert not (set(seeds) & (training | extra_demos | old_dev | dev | previous_eval | diagnostics))
    assert not (dev & (training | extra_demos | old_dev | previous_eval | diagnostics))
    record = {}
    for group, base, coverage in (("weak",original,"demos"), ("weak",original,"varied"), ("strong",selected,"varied")):
        prior = load(short / f"{coverage}.pt")
        rng = load(archived / f"{coverage}.pt")["pre_actor_rng"]
        for mode in ("learned","direct"):
            name = f"{group}_{coverage}_{mode}"
            c = load(run / (name.replace("_","-") + ".pt"))
            equal_weights(c["policy"], base["policy"], name + " policy")
            equal_weights(c["world"], prior["world"], name + " RSSM")
            if mode == "learned" and group == "weak":
                equal_weights(c["actor"], prior["actor"], name + " actor")
            assert c["actor_rng_sha256"] == hashlib.sha256(rng.numpy().tobytes()).hexdigest()
            assert (c["reward_mode"], c["base_group"], c["coverage"]) == (mode, group, coverage)
            assert (c["imagination_chunks"], c["rl_steps"], c["imagined_ticks"]) == (2,1500,576000)
            assert c["imagination_start_episodes"] == 24 and c["world_episodes"] == (24 if coverage == "demos" else 96)
            record[name] = dict(imagined_ticks=c["imagined_ticks"], direct_bonuses=c["direct_bonuses"],
                                actor_rng_sha256=c["actor_rng_sha256"])
    return dict(original_bc_updates=original["bc_steps"], selected_bc_updates=selected["bc_steps"],
                calibration=[dict(step=c["step"], stacks=c["stacks"]) for c in history],
                extra_demo_seeds=sorted(map(int,extra_demos)), actors=record)


@torch.no_grad()
def steering_on_base_traces(run, group, world):
    """Replay the same state and noise at each base chunk boundary, changing only the noise shift."""
    base = load(run / ("base.pt" if group == "weak" else "selected-base.pt"))
    pi = make_policy(base.get("policy_kind","original")).eval()
    pi.load_state_dict(base["policy"])
    histories, observations, noises, recorded, heights = [], [], [], [], []
    for path in sorted((run / "traces").glob(f"{group}_base-*.npz")):
        with np.load(path) as trace:
            o, a = (torch.from_numpy(trace[key]) for key in ("obs","actions"))
            gen = torch.Generator().manual_seed(int(trace["seed"]))
            st = world.initial(o[:1])
            for t in range(len(a)):
                if t % HORIZON == 0:
                    histories.append(st)
                    observations.append(o[t])
                    noises.append(torch.randn(1,HORIZON,ACTION_DIM,generator=gen))
                    recorded.append(a[t])
                    heights.append(o[t,5]-o[0,5])
                st,_ = world.filter(world.advance(st,a[t:t+1]),o[t+1:t+2])
    o, st, noise = torch.stack(observations), torch.cat(histories), torch.cat(noises)
    original = pi.sample(o,noise)[:,0]
    torch.testing.assert_close(original,torch.stack(recorded),atol=2e-4,rtol=0)
    direction = o[:,6:8]-o[:,3:5]
    distance = direction.norm(dim=1)
    carry = (torch.stack(heights)>.2) & (o[:,9]>.5) & (o[:,10]>.5)
    far = carry & (distance>.12)
    axis = direction / distance[:,None].clamp_min(1e-8)
    result = dict(chunk_starts=len(o), carrying=int(carry.sum()), carrying_outside_xy=int(far.sum()), actors={})
    for mode in ("learned","direct"):
        actor = NoiseSAC(world.features)
        actor.actor.load_state_dict(load(run/f"{group}-varied-{mode}.pt")["actor"])
        shift = actor.draw(st,deterministic=True)[0]
        steered = pi.sample(o,noise+shift[:,None])[:,0]
        toward = ((steered[:,:2]-original[:,:2])*axis).sum(dim=1)*STEP*1000
        flips = (steered[:,3]>0) != (original[:,3]>0)
        result["actors"][mode] = dict(
            mean_targetward_change_mm=float(toward[far].mean()) if far.any() else None,
            median_absolute_targetward_change_mm=float(toward[far].abs().median()) if far.any() else None,
            grip_flips_while_carrying=int((flips & carry).sum()),
            mean_absolute_noise_shift=float(shift.abs().mean()))
    return result


def summarize(run, short, archived):
    seeds = range(258, 308)
    groups = dict(weak=["weak_base","weak_demos_learned","weak_demos_direct","weak_varied_learned","weak_varied_direct"],
                  strong=["strong_base","strong_varied_learned","strong_varied_direct"])
    starts, records = {}, []
    for group, arms in groups.items():
        rows = [json.loads(line) for f in run.glob(f"eval-{group}-*.jsonl") for line in f.read_text().splitlines()]
        assert sorted(r["seed"] for r in rows) == list(seeds)
        for row in sorted(rows, key=lambda r: r["seed"]):
            seed = row["seed"]
            assert set(row) == {"seed", "start", *arms}
            first = starts.setdefault(seed, np.array(row["start"]))
            for arm in arms:
                m = row[arm].copy()
                with np.load(run / "traces" / f"{arm}-{seed}.npz") as d:
                    o = d["obs"]
                    np.testing.assert_allclose(o[0], first, atol=.01, rtol=0)
                    assert int(d["seed"]) == seed and bool(d["success"]) == bool(m["success"])
                    assert len(o) - 1 == m["steps"] == len(d["actions"])
                    lifted = (o[:,9] > .5) & (o[:,5] - o[0,5] > .2)
                    xy = np.linalg.norm(o[:,3:5]-o[:,6:8],axis=1)*100
                    z_error = np.abs((o[:,5]-o[:,8])*100 - 46.8)
                    assert bool(lifted.any()) == m["lifted"] and np.isclose(xy[-1], m["final_xy_error_mm"])
                    m["reached_xy_lifted"] = bool((lifted & (xy < 12)).any())
                    m["reached_xyz_lifted"] = bool((lifted & (xy < 12) & (z_error < 8)).any())
                records.append(dict(seed=seed, arm=arm, **m))
    assert len(list((run / "traces").glob("*.npz"))) == len(records) == 400
    training = audit_training(run,short,archived,seeds)
    stats, scores = {}, {}
    for arm in groups["weak"] + groups["strong"]:
        values = [r for r in records if r["arm"] == arm]
        scores[arm] = np.array([r["success"] for r in values], dtype=int)
        stats[arm] = {k: sum(int(r[k]) for r in values) for k in ("success","lifted","reached_xy_lifted","reached_xyz_lifted")}
        stats[arm].update(scenes=len(values), successful_seeds=[r["seed"] for r in values if r["success"]],
                          mean_return=float(np.mean([r["episode_return"] for r in values])))
    comparisons = [(group+"_base", arm) for group, arms in groups.items() for arm in arms[1:]]
    comparisons.extend((a, a.replace("learned","direct")) for a in stats if a.endswith("learned"))
    comparisons.append(("weak_base","strong_base"))
    pairs = {f"{b}_vs_{a}": paired(scores[a],scores[b]) for a,b in comparisons}
    # Match all four outcomes by seed. Positive values favor a larger incremental effect with the 72-demo base.
    interaction = {}
    for reference in ("base","varied_learned"):
        delta = ((scores["strong_varied_direct"] - scores["strong_"+reference])
                 - (scores["weak_varied_direct"] - scores["weak_"+reference]))
        interaction["direct_vs_"+reference] = dict(
            difference_in_differences_pp=100 * float(delta.mean()),
            per_scene_counts={str(i): int((delta == i).sum()) for i in range(-2,3)})
    base_forecasts = {}
    for coverage in ("demos", "varied"):
        model = make_world("residual").eval()
        model.load_state_dict(load(short / f"{coverage}.pt")["world"])
        base_forecasts[coverage] = {
            a: forecast(policy_recordings(run,a),model,6) for a in ("weak_base","strong_base")
        }
    varied_model = make_world("residual").eval()
    varied_model.load_state_dict(load(short / "varied.pt")["world"])
    probe = {g: steering_on_base_traces(run,g,varied_model) for g in groups}
    result = dict(first_seed=seeds.start, last_seed=seeds.stop-1, episodes=len(records), policies=stats,
                  paired=pairs, training=training, measured_fresh=observed_report(from_traces(run / "traces")),
                  base_trajectory_forecasts=base_forecasts, steering_on_base_traces=probe,
                  baseline_interaction=interaction)
    (run / "reward-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    with (run / "paired-outcomes.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(sorted(records,key=lambda r: (r["seed"],r["arm"])))
    for arm, v in stats.items():
        print(arm, v["success"], v["lifted"], v["reached_xyz_lifted"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--short",type=Path,required=True)
    p.add_argument("--archived",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    summarize(args.run,args.short,args.archived)
