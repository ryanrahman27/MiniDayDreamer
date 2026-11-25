"""Verify the development lock, paired Isaac episodes, and frozen reward comparison."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from check_state_reward import forecast, from_traces, observed_report, policy_recordings
from minidaydreamer.models import make_world
from reward_study import digest, load
from select_midbase import ACCEPT, DEV, next_candidate
from summarize_reward import equal_weights, steering_on_base_traces
from summarize_stack import paired


ARMS = ("mid_base", "mid_varied_learned", "mid_varied_direct")
TEST = range(308, 408)


def audit_selection(run, prior):
    lock = json.loads((run / "selection.json").read_text())
    scores = {}
    for entry in lock["development"]:
        step = entry["step"]
        assert next_candidate(scores) == step
        assert [r["seed"] for r in entry["outcomes"]] == list(DEV)
        scores[step] = sum(r["success"] for r in entry["outcomes"])
        assert scores[step] == entry["stacks"]
    step = lock["selected_step"]
    assert list(ACCEPT) == lock["accepted_counts"] and step == list(scores)[-1]
    assert scores[step] in ACCEPT and all(v not in ACCEPT for v in list(scores.values())[:-1])
    assert digest(run / "selected-base.pt") == digest(run / f"candidate-{step}.pt") == lock["selected_sha256"]
    base = load(run / "selected-base.pt")
    assert base["bc_steps"] == step and base["policy_kind"] == "relative"
    equal_weights(load(run / "candidate-12000.pt")["policy"], load(prior / "candidate-12000.pt")["policy"],
                  "unchanged 12,000-update endpoint")
    assert digest(run / "extra-bc.npz") == base["source_sha256"] == digest(prior / "extra-bc.npz")
    assert digest(run / "corpus.npz") == digest(prior / "corpus.npz") == load(run / "base.pt")["corpus_sha256"]
    excluded = set()
    demos = []
    for f in ("corpus.npz", "extra-bc.npz"):
        with np.load(run / f) as d:
            excluded.update(map(int, d["seeds"]))
            demos.extend(map(int, d["seeds"][d["split"] == "demo"]))
    assert base["demo_seeds"] == demos and len(set(demos)) == 72
    for folder in run.parent.glob("isaac-*-v[1-4]"):
        for f in folder.glob("eval-*.jsonl"):
            excluded.update(json.loads(line)["seed"] for line in f.read_text().splitlines())
    excluded.update(json.loads((prior / "reward-audit.json").read_text())["diagnostic_seeds"])
    for records in (load(run / "base.pt")["calibration"], json.loads((prior / "calibration.json").read_text())):
        excluded.update(r["seed"] for c in records for r in c["outcomes"])
    assert not set(DEV) & (excluded | set(TEST)) and not set(TEST) & excluded
    return {"bc_updates": step, "stacks": scores[step], "scenes": len(DEV),
            "checkpoint_sha256": lock["selected_sha256"], "candidate_counts": scores}, base


def summarize(run, prior, short, archived):
    selection, base = audit_selection(run, prior)
    source = load(short / "varied.pt")
    reference = load(archived / "varied.pt")
    equal_weights(reference["world"], source["world"], "archived varied RSSM")
    rng_hash = hashlib.sha256(reference["pre_actor_rng"].numpy().tobytes()).hexdigest()
    actors = {}
    for mode in ("learned", "direct"):
        c = load(run / f"mid-varied-{mode}.pt")
        equal_weights(c["policy"], base["policy"], mode + " frozen baseline")
        assert c["base_checkpoint_sha256"] == selection["checkpoint_sha256"] and c["base_bc_steps"] == base["bc_steps"]
        equal_weights(c["world"], source["world"], mode + " frozen RSSM")
        assert (c["coverage"], c["base_group"], c["reward_mode"]) == ("varied", "mid", mode)
        assert (c["world_episodes"], c["imagination_start_episodes"]) == (96, 24)
        assert (c["imagination_chunks"], c["rl_steps"], c["imagined_ticks"]) == (2, 1500, 576000)
        assert c["actor_rng_sha256"] == rng_hash
        actors[mode] = {"sha256": digest(run / f"mid-varied-{mode}.pt"), "imagined_ticks": c["imagined_ticks"],
                        "direct_bonuses": c["direct_bonuses"]}
    scenes = sorted((json.loads(line) for f in run.glob("eval-mid-*.jsonl")
                     for line in f.read_text().splitlines()), key=lambda r: r["seed"])
    assert [r["seed"] for r in scenes] == list(TEST)
    records = []
    max_start_error = 0.
    for scene in scenes:
        seed = scene["seed"]
        assert set(scene) == {"seed", "start", *ARMS}
        starts = []
        for arm in ARMS:
            metrics = scene[arm]
            with np.load(run / "traces" / f"{arm}-{seed}.npz") as trace:
                obs = trace["obs"]
                starts.append(obs[0])
                assert int(trace["seed"]) == seed and int(trace["success"]) == metrics["success"]
                assert len(obs) - 1 == metrics["steps"] == len(trace["actions"])
                np.testing.assert_allclose(obs[0], scene["start"], atol=.01, rtol=0)
                lifted = (obs[:, 9] > .5) & (obs[:, 5] - obs[0, 5] > .2)
                assert bool(lifted.any()) == metrics["lifted"]
                xy = np.linalg.norm(obs[:, 3:5] - obs[:, 6:8], axis=1) * 100
                assert np.isclose(xy[-1], metrics["final_xy_error_mm"])
                records.append(dict(seed=seed, arm=arm, **metrics, reached_xy_lifted=bool((lifted & (xy < 12)).any())))
        max_start_error = max(max_start_error, float(np.ptp(np.array(starts)[:, :9], axis=0).max() * 100))
    assert max_start_error <= 1.0 and len(records) == len(list((run / "traces").glob("*.npz"))) == 300
    scores, stats = {}, {}
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        scores[arm] = np.array([r["success"] for r in rows])
        stats[arm] = {key: sum(int(r[key]) for r in rows) for key in ("success", "lifted", "reached_xy_lifted")}
        stats[arm]["mean_return"] = float(np.mean([r["episode_return"] for r in rows]))
    pairs = {}
    for ref, alt in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
        delta = scores[alt] - scores[ref]
        result = paired(scores[ref], scores[alt])
        result["gained_seeds"] = np.asarray(TEST)[delta == 1].tolist()
        result["lost_seeds"] = np.asarray(TEST)[delta == -1].tolist()
        pairs[f"{alt}_vs_{ref}"] = result
    wm = make_world("residual").eval()
    wm.load_state_dict(source["world"])
    result = dict(selection=selection, first_seed=TEST.start, last_seed=TEST.stop-1, episodes=len(records),
                  max_paired_start_coordinate_error_mm=max_start_error, actors=actors, policies=stats, paired=pairs,
                  actor_rng_sha256=rng_hash, action_probe=steering_on_base_traces(run, "mid", wm),
                  six_tick_base_forecast=forecast(policy_recordings(run, ARMS[0]), wm, 6),
                  measured_reward_check=observed_report(from_traces(run / "traces")))
    with (run / "paired-outcomes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (run / "midbase-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    for name in ("prior", "short", "archived"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    summarize(args.run, args.prior, args.short, args.archived)
