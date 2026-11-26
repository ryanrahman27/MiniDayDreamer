"""Audit matched trajectories and estimate uncertainty over scenes and independently trained actors."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from climb_timing import write_csv
from multiseed_study import DEV, TEST, SCALAR, DEV_TRAIN, TEST_TRAIN, RECIPES, GAINS, arms, candidate, scores
from reward_study import digest, load
from score_multiseed import controllers, summarize_trace
from summarize_handoff import verify_commands
from summarize_stack import paired


def comparison(matrix, reference):
    """Cross actor and scene draws. A shared baseline gets the identical sampled scene indices."""
    delta = matrix - reference[None, :]
    n_actors, n_scenes = delta.shape
    rng = np.random.default_rng(81473)
    estimates = []
    for _ in range(40):
        actor = rng.integers(n_actors, size=(500, n_actors, 1))
        scene = rng.integers(n_scenes, size=(500, 1, n_scenes))
        estimates.extend(delta[actor, scene].mean(axis=(1, 2)) * 100)
    return dict(
        mean_stacks=float(matrix.sum(axis=1).mean()),
        mean_success_pct=float(100 * matrix.mean()),
        reference_success_pct=float(100 * reference.mean()),
        mean_difference_pp=float(100 * delta.mean()),
        crossed_bootstrap_95_pp=np.percentile(estimates, [2.5, 97.5]).tolist(),
        wins_by_actor=[int((r > reference).sum()) for r in matrix],
        losses_by_actor=[int((r < reference).sum()) for r in matrix],
        actor_success_totals=matrix.sum(axis=1).tolist(),
        actor_differences_pp=(100 * delta.mean(axis=1)).tolist(),
        positive_actors=int((delta.mean(axis=1) > 0).sum()),
    )


@torch.no_grad()
def shift_check(folder, wm, models, recipe):
    histories = []
    for seed in sorted(TEST)[:20]:
        with np.load(folder / "traces" / f"base-{seed}.npz") as tr:
            obs = torch.from_numpy(tr["obs"])
            actions = torch.from_numpy(tr["actions"])
            latent = wm.initial(obs[:1])
            for t, action in enumerate(actions):
                if t % 3 == 0:
                    histories.append(latent)
                latent, _ = wm.filter(wm.advance(latent, action[None]), obs[t + 1 : t + 2])
    states = torch.cat(histories)
    shifts = torch.stack(
        [models[candidate(recipe, s)].draw(states, deterministic=True)[0] for s in TEST_TRAIN]
    )
    previous = models["previous"].draw(states, deterministic=True)[0]
    return dict(
        histories=len(states),
        source="First 20 unsteered primary test scenes, every chunk boundary.",
        mean_effective_shift_by_seed=shifts.mean(dim=1).tolist(),
        mean_abs_difference_from_previous=float((shifts - previous).abs().mean()),
        mean_between_seed_std=float(shifts.std(dim=0, unbiased=False).mean()),
        maximum_between_seed_range=float((shifts.amax(0) - shifts.amin(0)).max()),
    )


def audit_split(run, split, recipe, all_rows):
    rows = scores(run, split)
    names = arms(run, split)
    pi, wm, models = controllers(run, names)
    starts_error = action_error = 0.0
    drop_count = 0
    folder = run / split
    bank_folder = run / {"gain": "dev", "noise": "test", "prior_gain": "test"}.get(split, split)
    bank = None if split == "scalar" else torch.load(bank_folder / "scenes.pt", weights_only=True)
    if bank is not None:
        assert set(bank) == {s["seed"] for s in rows}
    for scene in rows:
        seed = scene["seed"]
        starts = []
        if bank is not None:
            np.testing.assert_array_equal(scene["start"], bank[seed][0].numpy())
        for name in names:
            path = folder / "traces" / f"{name}-{seed}.npz"
            with np.load(path) as tr:
                obs, cmd = tr["obs"], tr["actions"]
                starts.append(obs[0])
                np.testing.assert_allclose(obs[0], scene["start"], rtol=0, atol=0.01)
                assert len(cmd) == len(obs) - 1 and int(tr["seed"]) == seed
                assert np.isfinite(obs).all() and np.isfinite(cmd).all()
                speed = (
                    tr["cube_speeds_mps"] if bank is not None else np.load(path.with_suffix(".speeds.npy"))
                )
                assert speed.shape == (len(cmd), 2) and np.isfinite(speed).all() and (speed >= 0).all()
                value = summarize_trace(obs, tr["aperture_m"], speed)
                assert value == scene[name] and value["success"] == tr["success"]
                assert len(cmd) == 200 or min(obs[-1, 5], obs[-1, 8]) < -0.5
                drop_count += int(len(cmd) < 200)
                noise_seed = seed + 1000003 if split == "noise" else seed
                assert int(tr.get("noise_seed", seed)) == noise_seed
                inputs = dict(obs=obs, actions=cmd, seed=noise_seed)
                action_error = max(action_error, verify_commands(inputs, name, pi, wm, models[name], None))
                all_rows.append(dict(split=split, scene=seed, arm=name, **value))
        starts_error = max(starts_error, float(np.ptp(starts, axis=0)[:9].max() * 100))
    assert len(list((folder / "traces").glob("*.npz"))) == len(names) * len(rows) and starts_error <= 1
    values = {n: np.array([r[n]["success"] for r in rows]) for n in names}
    report = dict(
        scenes=len(rows),
        totals={k: int(v.sum()) for k, v in values.items()},
        early_drops=drop_count,
        max_action_error=action_error,
        max_paired_start_error_mm=starts_error,
    )
    if split in ("dev", "gain", "prior_gain"):
        return report
    winner = np.stack([values[candidate(recipe, s)] for s in TEST_TRAIN])
    report.update(
        scenes_with_seed_disagreement=int((np.ptp(winner, axis=0) > 0).sum()),
        winner_vs_base=comparison(winner, values["base"]),
        winner_vs_previous=comparison(winner, values["previous"]),
        previous_vs_base=comparison(values["previous"][None, :], values["base"]),
        paired_by_actor={candidate(recipe, s): paired(values["base"], v) for s, v in zip(TEST_TRAIN, winner)},
        scene_blocks=[
            dict(
                first=rows[i[0]]["seed"],
                last=rows[i[-1]]["seed"],
                base=int(values["base"][i].sum()),
                previous=int(values["previous"][i].sum()),
                winning_actors=winner[:, i].sum(axis=1).tolist(),
            )
            for i in np.array_split(np.arange(len(rows)), 4)
        ],
    )
    if split == "test" and recipe != "solo":
        cold = np.stack([values[candidate("solo", s)] for s in TEST_TRAIN])
        report["from_scratch_vs_base"] = comparison(cold, values["base"])
    if split == "test":
        report["common_state_shifts"] = shift_check(folder, wm, models, recipe)
    return report


def complete_test_cohort(run, split):
    return sum(len(f.read_text().splitlines()) for f in (run / split).glob("eval-*.jsonl")) == len(TEST)


def summarize(run, midbase):
    protocol = json.loads((run / "protocol.json").read_text())
    lock = json.loads((run / "selection.json").read_text())
    assert digest(run / "protocol.json") == lock["protocol_sha256"]
    for name, expected in lock["dev_file_sha256"].items():
        assert digest(run / "dev" / name) == expected
    assert (
        digest(run / "previous.pt") == protocol["previous_sha256"]
        and digest(run / "selected-base.pt") == protocol["base_sha256"]
    )
    assert not set(DEV_TRAIN).intersection(TEST_TRAIN)
    excluded = set(load(run / "selected-base.pt")["demo_seeds"])
    for path in midbase.parent.glob("isaac-*-v[1-9]/eval-*.jsonl"):
        excluded.update(json.loads(line)["seed"] for line in path.read_text().splitlines())
    for name in ("corpus.npz", "extra-bc.npz"):
        with np.load(midbase / name) as data:
            excluded.update(data["seeds"].tolist())
    for group in json.loads((midbase / "selection.json").read_text())["development"]:
        excluded.update(r["seed"] for r in group["outcomes"])
    assert len(set(DEV) | set(TEST) | set(SCALAR)) == len(DEV) + len(TEST) + len(SCALAR)
    assert not excluded.intersection(set(DEV) | set(TEST) | set(SCALAR))
    recipe = lock["winner"]
    records = []
    gain_lock = json.loads((run / "gain-selection.json").read_text())
    assert gain_lock["recipe_selection_sha256"] == digest(run / "selection.json")
    for name, expected in gain_lock["gain_file_sha256"].items():
        assert digest(run / "gain" / name) == expected
    report = {s: audit_split(run, s, recipe, records) for s in ("dev", "gain", "test", "scalar")}
    if complete_test_cohort(run, "prior_gain"):
        report["prior_gain"] = audit_split(run, "prior_gain", recipe, records)
        control = np.array([r["previous_scaled"]["success"] for r in scores(run, "prior_gain")])
        test_rows = scores(run, "test")
        selected = np.array([[r[candidate(recipe, s)]["success"] for r in test_rows] for s in TEST_TRAIN])
        report["prior_gain"]["winner_vs_scaled_previous"] = comparison(selected, control)
    if complete_test_cohort(run, "noise"):
        report["noise"] = audit_split(run, "noise", recipe, records)
        # First average the two outcomes of each policy on each physical scene.
        # The bootstrap continues to use 100 independent scene units.
        streams = [scores(run, s) for s in ("test", "noise")]

        def average(arm):
            return np.mean([[r[arm]["success"] for r in rows] for rows in streams], axis=0)

        matrix = np.stack([average(candidate(recipe, s)) for s in TEST_TRAIN])
        report["both_noise_streams"] = {}
        for arm in ("base", "previous"):
            stat = comparison(matrix, average(arm))
            report["both_noise_streams"][f"winner_vs_{arm}"] = {
                k: stat[k]
                for k in (
                    "mean_success_pct",
                    "reference_success_pct",
                    "mean_difference_pp",
                    "crossed_bootstrap_95_pp",
                    "positive_actors",
                )
            }
    by_gain = {100: lock["per_recipe"][recipe]}
    for gain in GAINS[1:]:
        by_gain[gain] = [report["gain"]["totals"][f"{candidate(recipe, s)}@{gain}"] for s in DEV_TRAIN]
    assert gain_lock["counts"] == {str(k): v for k, v in by_gain.items()}
    assert gain_lock["gain_percent"] == max(
        GAINS, key=lambda g: (sum(by_gain[g]), min(by_gain[g]), -abs(g - 100))
    )
    report["gain_percent"] = gain_lock["gain_percent"]
    assert report["dev"]["totals"] == lock["dev_stacks"]
    assert max(RECIPES, key=lambda r: (sum(lock["per_recipe"][r]), min(lock["per_recipe"][r]))) == recipe
    report["winning_recipe"] = recipe
    report["episodes"] = len(records)
    report["training"] = {
        n: json.loads((run / (n + ".json")).read_text())
        for n in sorted(set(arms(run, "dev") + arms(run, "test")) - {"base", "previous"})
    }
    write_csv(run / "paired-outcomes.csv", records)
    (run / "multiseed-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for s in ("dev", "test", "scalar"):
        print(s, report[s]["totals"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("midbase", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    summarize(args.run, args.midbase)
