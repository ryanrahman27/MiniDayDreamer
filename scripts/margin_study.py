"""Train matched release-margin and unchanged-reward residuals in the frozen varied RSSM."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from minidaydreamer.experiment import fit_steering, verify_weights
from minidaydreamer.models import HORIZON
from minidaydreamer.release_margin import LocalNoiseSAC, MarginReward, approach_weight, margin_terms
from minidaydreamer.released_steering import ReleasedReward
from multiseed_study import SCENE_SETS, TEST_TRAIN
from reward_study import digest, load
from score_multiseed import controllers

SEED_PAIRS = dict(zip(TEST_TRAIN, (1201, 1213, 1237, 1259, 1277), strict=True))
SCENES = range(972, 1100)
STREAMS = {"test": 0, "noise": 1000003}
RECIPES = {"control": ReleasedReward, "margin": MarginReward}


def dump(path, data):
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def prior_name(seed):
    return f"prior80-{seed}"


def controllers_for(run, old, seeds=TEST_TRAIN, trained=True):
    old_names = [prior_name(seed) for seed in seeds]
    pi, wm, models = controllers(old, ["base", *old_names])
    pi.eval().requires_grad_(False)
    wm.eval().requires_grad_(False)
    for seed in seeds:
        incumbent = models[prior_name(seed)]
        incumbent.actor.eval().requires_grad_(False)
        incumbent.prior.actor.eval().requires_grad_(False)
        if trained:
            for recipe in RECIPES:
                name = f"{recipe}-{seed}"
                path = run / (name + ".pt")
                manifest = json.loads((run / (name + ".json")).read_text())
                assert manifest["sha256"] == digest(path) and manifest["protocol_sha256"] == digest(
                    run / "protocol.json"
                )
                learner = LocalNoiseSAC(wm.features, incumbent)
                learner.actor.load_state_dict(load(path))
                models[name] = learner
    return pi, wm, models


def tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def prepare(run, old, midbase):
    # All earlier simulator evaluations, including gain selection and single-env confirmations.
    used = set().union(*SCENE_SETS.values())
    used.update(load(old / "selected-base.pt")["demo_seeds"])
    for name in ("corpus.npz", "extra-bc.npz"):
        with np.load(midbase / name) as data:
            used.update(data["seeds"].tolist())
    for group in json.loads((midbase / "selection.json").read_text())["development"]:
        used.update(r["seed"] for r in group["outcomes"])
    for path in old.parent.glob("isaac-*-v[1-9]/eval-*.jsonl"):
        used.update(json.loads(line)["seed"] for line in path.read_text().splitlines())
    assert not used.intersection(SCENES)
    _, wm, _ = controllers_for(run, old, [101], False)
    states = load(old / "starts.pt")
    prior_protocol = json.loads((old / "protocol.json").read_text())
    assert tensor_digest(states) == prior_protocol["starts_sha256"]
    sources, focused = {}, []
    for seed in SCENE_SETS["test"]:
        path = old / "test" / "traces" / f"prior80-101-{seed}.npz"
        sources[seed] = digest(path)
        with np.load(path) as tr:
            obs, cmd = torch.from_numpy(tr["obs"]), torch.from_numpy(tr["actions"])
        selected = approach_weight(obs) > 0.5
        latent = wm.initial(obs[:1])
        for t, action in enumerate(cmd):
            if t % HORIZON == 0 and selected[t]:
                focused.append(latent)
            latent, _ = wm.filter(wm.advance(latent, action[None]), obs[t + 1 : t + 2])
    pool = torch.cat(focused)
    idx = torch.randint(len(pool), (len(states),), generator=torch.Generator().manual_seed(6601))
    starts = torch.cat([states, pool[idx]])
    torch.save(starts, run / "starts.pt")
    protocol = dict(
        score="Final 8 released, aligned, slow ticks; no hand clearance threshold. 200 ticks or physical drop.",
        physical_scenes=list(SCENES),
        noise_offsets=STREAMS,
        training_seed_pairs=SEED_PAIRS,
        archived_focus_trace_sha256=sources,
        unique_focus_histories=len(pool),
        original_starts=len(states),
        focused_starts=len(idx),
        starts_sha256=tensor_digest(starts),
        checkpoint_sha256={
            n: digest(old / n)
            for n in (
                "previous.pt",
                "selected-base.pt",
                "gain-selection.json",
                *(prior_name(s) + ".pt" for s in TEST_TRAIN),
            )
        },
        corpus_sha256=digest(midbase / "corpus.npz"),
        updates=1500,
        chunks=4,
        imagined_ticks_per_actor=576000,
        objective="ReleasedReward plus 0.7 clipped progress toward 6 mm and 1.5 opening-margin credit/penalty crossing zero at 9 mm. Same local steering window and focused starts in the reward ablation.",
        primary="Two-stream mean within each scene, crossed paired bootstrap over scenes and five actor seed pairs; margin vs incumbent and base. Control isolates reward change. Report both streams, each pair and failure rates.",
    )
    dump(run / "protocol.json", protocol)
    print("Physical scenes, histories, sampled starts:", len(SCENES), len(pool), len(starts), flush=True)


def train(run, old, midbase, seed, recipe):
    meta = json.loads((run / "protocol.json").read_text())
    for n, expected in meta["checkpoint_sha256"].items():
        assert digest(old / n) == expected
    pi, wm, models = controllers_for(run, old, [seed], False)
    incumbent = models[prior_name(seed)]
    starts = load(run / "starts.pt")
    assert (
        tensor_digest(starts) == meta["starts_sha256"]
        and digest(midbase / "corpus.npz") == meta["corpus_sha256"]
    )
    data = Dataset(midbase / "corpus.npz", 0)
    reward = RECIPES[recipe](data, np.flatnonzero(data.split == "demo"), len(starts))
    torch.manual_seed(SEED_PAIRS[seed])
    learner = LocalNoiseSAC(wm.features, incumbent)
    fit_steering(wm, pi, starts, meta["updates"], meta["chunks"], reward, sac=learner)
    ck = load(old / "previous.pt")
    verify_weights(pi, ck["policy"], "imitation policy")
    verify_weights(wm, ck["world"], "world")
    verify_weights(incumbent.actor, load(old / (prior_name(seed) + ".pt"))["actor"], "incumbent")
    verify_weights(incumbent.prior.actor, ck["actor"], "shared steering prior")
    assert reward.ticks == meta["imagined_ticks_per_actor"]
    name = f"{recipe}-{seed}"
    path = run / (name + ".pt")
    with path.open("xb") as f:
        torch.save(learner.actor.state_dict(), f)
    dump(
        run / (name + ".json"),
        dict(
            sha256=digest(path),
            protocol_sha256=digest(run / "protocol.json"),
            incumbent_seed=seed,
            residual_seed=SEED_PAIRS[seed],
            ticks=reward.ticks,
            bonuses=reward.bonuses,
        ),
    )


def check_reward():
    s = torch.tensor([[0.0, 0.0, 2.0, 0.14, 0.0, 0.698, 0.0, 0.0, 0.23, 1.0, 1.0]])
    centered = s.clone()
    centered[:, 3] = 0.05
    assert margin_terms(s, centered)[0] > 0 and margin_terms(centered, s)[0] < 0
    for obs, sign in ((s, -1), (centered, 1)):
        opened = obs.clone()
        opened[:, 9:11] = 0
        assert sign * margin_terms(obs, opened)[1] > 0
        assert margin_terms(obs, obs)[1] == 0 and approach_weight(opened) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("run", type=Path)
    parser.add_argument("old", type=Path)
    parser.add_argument("midbase", type=Path)
    parser.add_argument("--seed", type=int, choices=SEED_PAIRS)
    parser.add_argument("--recipe", choices=RECIPES)
    a = parser.parse_args()
    torch.set_num_threads(1)
    check_reward()
    if a.mode == "prepare":
        prepare(a.run, a.old, a.midbase)
    else:
        train(a.run, a.old, a.midbase, a.seed, a.recipe)
