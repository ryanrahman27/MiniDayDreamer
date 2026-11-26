"""Development selection and independently seeded DSRL training for released stacking."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from failure_dsrl import mixed_starts
from minidaydreamer.data import Dataset
from minidaydreamer.experiment import fit_steering, load_policies, verify_weights
from minidaydreamer.released_steering import PriorNoiseSAC, ReleasedReward
from reward_study import digest, load

RECIPES = {"solo": 0.0, "prior50": 0.5, "prior80": 0.8}
DEV_TRAIN = (11, 29)
TEST_TRAIN = (101, 211, 307, 401, 503)
DEV = range(808, 840)
TEST = range(840, 940)
SCALAR = range(940, 972)
SCENE_SETS = {"dev": DEV, "gain": DEV, "test": TEST, "noise": TEST, "prior_gain": TEST, "scalar": SCALAR}
CHUNKS, UPDATES = 4, 1500
GAINS = (100, 75, 125, 150)


def candidate(recipe, seed):
    return f"{recipe}-{seed}"


def arms(run, split):
    if split == "prior_gain":
        return ["previous_scaled"]
    if split == "dev":
        return ["base", "previous", *(candidate(r, s) for r in RECIPES for s in DEV_TRAIN)]
    recipe = json.loads((run / "selection.json").read_text())["winner"]
    if split == "gain":
        return [f"{candidate(recipe, s)}@{gain}" for gain in GAINS if gain != 100 for s in DEV_TRAIN]
    controls = (
        [] if recipe == "solo" or split in ("scalar", "noise") else [candidate("solo", s) for s in TEST_TRAIN]
    )
    return ["base", "previous", *(candidate(recipe, s) for s in TEST_TRAIN), *controls]


def prepare(run, old, midbase):
    src = json.loads((old / "training.json").read_text())
    assert (
        digest(run / "previous.pt") == src["retreat_sha256"]
        and digest(run / "selected-base.pt") == src["base_sha256"]
    )
    ck, pi, wm, prior = load_policies(run / "previous.pt", SimpleNamespace(backend="isaac"))
    assert ck["reward_mode"] == "retreat"
    verify_weights(pi, load(run / "selected-base.pt")["policy"], "base")
    data = Dataset(midbase / "corpus.npz", 0)
    demos = np.flatnonzero(data.split == "demo")
    states, _ = mixed_starts(midbase, wm, data, demos)
    assert hashlib.sha256(states.numpy().tobytes()).hexdigest() == src["start_tensor_sha256"]
    torch.save(states, run / "starts.pt")
    manifest = dict(
        recipes=RECIPES,
        dev_actor_seeds=DEV_TRAIN,
        test_actor_seeds=TEST_TRAIN,
        dev_scenes=list(DEV),
        test_scenes=list(TEST),
        chunks=CHUNKS,
        updates=UPDATES,
        base_sha256=src["base_sha256"],
        previous_sha256=src["retreat_sha256"],
        starts_sha256=src["start_tensor_sha256"],
        corpus_sha256=digest(midbase / "corpus.npz"),
        selection="Highest summed released-terminal successes across both development actor seeds, then higher worst seed, then recipe order.",
        score="Final eight post-command ticks at 200 ticks or physical drop; open jaws, aligned and slow cubes, no tool-distance requirement.",
        scratch_controls="If an anchored recipe wins, test a from-scratch actor for each validation seed.",
        scalar_confirmation=list(SCALAR),
        development_gain_percent=list(GAINS),
        second_test_noise_seed_offset=1000003,
    )
    (run / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")


def train(run, midbase, recipe, seed):
    meta = json.loads((run / "protocol.json").read_text())
    if seed in TEST_TRAIN and candidate(recipe, seed) not in arms(run, "test"):
        raise ValueError("Actor is outside locked selection")
    if seed not in (*DEV_TRAIN, *TEST_TRAIN):
        raise ValueError("Unknown actor seed")
    name = candidate(recipe, seed)
    ck, pi, wm, prior = load_policies(run / "previous.pt", SimpleNamespace(backend="isaac"))
    assert digest(run / "previous.pt") == meta["previous_sha256"]
    pi.eval().requires_grad_(False)
    wm.eval().requires_grad_(False)
    prior.actor.eval().requires_grad_(False)
    states = torch.load(run / "starts.pt", weights_only=True)
    assert hashlib.sha256(states.numpy().tobytes()).hexdigest() == meta["starts_sha256"]
    data = Dataset(midbase / "corpus.npz", 0)
    reward = ReleasedReward(data, np.flatnonzero(data.split == "demo"), len(states))
    torch.manual_seed(seed)
    learner = PriorNoiseSAC(wm.features, prior, RECIPES[recipe])
    learner = fit_steering(wm, pi, states, UPDATES, CHUNKS, reward, sac=learner)
    verify_weights(pi, ck["policy"], "frozen imitation")
    verify_weights(wm, ck["world"], "frozen RSSM")
    verify_weights(prior.actor, ck["actor"], "frozen prior")
    assert reward.ticks == 576000
    dest = run / (name + ".pt")
    with dest.open("xb") as f:
        torch.save(
            dict(
                actor=learner.actor.state_dict(),
                seed=seed,
                recipe=recipe,
                prior_weight=RECIPES[recipe],
                previous_sha256=meta["previous_sha256"],
                imagined_ticks=reward.ticks,
            ),
            f,
        )
    stats = dict(
        name=name,
        sha256=digest(dest),
        seed=seed,
        recipe=recipe,
        ticks=reward.ticks,
        qualified=reward.qualified,
        bonuses=reward.bonuses,
    )
    (run / (name + ".json")).write_text(json.dumps(stats, indent=2) + "\n")
    print(stats, flush=True)


def scores(run, split):
    scenes = sorted(
        (json.loads(line) for f in (run / split).glob("eval-*.jsonl") for line in f.read_text().splitlines()),
        key=lambda s: s["seed"],
    )
    assert [r["seed"] for r in scenes] == list(SCENE_SETS[split])
    assert all(set(r) == {"seed", "start", *arms(run, split)} for r in scenes)
    return scenes


def select(run):
    rows = scores(run, "dev")
    counts = {a: sum(r[a]["success"] for r in rows) for a in arms(run, "dev")}
    ranked = {recipe: [counts[candidate(recipe, s)] for s in DEV_TRAIN] for recipe in RECIPES}
    winner = max(RECIPES, key=lambda r: (sum(ranked[r]), min(ranked[r])))
    decision = dict(
        winner=winner,
        dev_stacks=counts,
        per_recipe=ranked,
        protocol_sha256=digest(run / "protocol.json"),
        dev_file_sha256={f.name: digest(f) for f in sorted((run / "dev").glob("eval-*.jsonl"))},
    )
    with (run / "selection.json").open("x") as f:
        json.dump(decision, f, indent=2)
    print(decision, flush=True)


def select_gain(run):
    rows = scores(run, "gain")
    old = json.loads((run / "selection.json").read_text())
    recipe = old["winner"]
    counts = {100: old["per_recipe"][recipe]}
    for gain in GAINS[1:]:
        counts[gain] = [
            sum(row[f"{candidate(recipe, s)}@{gain}"]["success"] for row in rows) for s in DEV_TRAIN
        ]
    best = max(GAINS, key=lambda g: (sum(counts[g]), min(counts[g]), -abs(g - 100)))
    decision = dict(
        gain_percent=best,
        counts=counts,
        recipe_selection_sha256=digest(run / "selection.json"),
        gain_file_sha256={f.name: digest(f) for f in sorted((run / "gain").glob("eval-*.jsonl"))},
    )
    with (run / "gain-selection.json").open("x") as f:
        json.dump(decision, f, indent=2)
    print(decision, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "train", "select", "gain"))
    p.add_argument("run", type=Path)
    p.add_argument("--old", type=Path)
    p.add_argument("--midbase", type=Path)
    p.add_argument("--recipe", choices=RECIPES)
    p.add_argument("--seed", type=int)
    a = p.parse_args()
    torch.set_num_threads(1)
    if a.mode == "prepare":
        prepare(a.run, a.old, a.midbase)
    elif a.mode == "gain":
        select_gain(a.run)
    elif a.mode == "select":
        select(a.run)
    else:
        train(a.run, a.midbase, a.recipe, a.seed)
