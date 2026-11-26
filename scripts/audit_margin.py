"""Audit archived failures, paired Isaac outcomes and the noise commands that produced them."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from margin_study import SCENES, SEED_PAIRS, STREAMS, controllers_for, dump, prior_name
from minidaydreamer.models import ACTION_DIM, HORIZON
from multiseed_study import TEST
from placement_events import event_ticks
from score_multiseed import summarize_trace
from stack_failure_modes import classify


def write_csv(path, rows):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def events(tr):
    obs, actions, jaws = (tr[k] for k in ("obs", "actions", "aperture_m"))
    e = event_ticks(dict(obs=obs, actions=actions, aperture_m=jaws, proposed_grip=actions[:, 3]))
    success = int(tr["success"])
    return dict(
        success=success,
        mode=classify(success, e),
        first_open_tick=e["first_open"],
        was_grasped_at_open=bool(e["grasped_at_open"]),
        open_xy_mm=e["open_xy_mm"],
        open_height_error_mm=e["open_z_error_mm"],
        final_xy_mm=e["final_xy_mm"],
        final_height_error_mm=e["final_z_error_mm"],
        never_open=e["first_lift"] is not None and e["first_open"] is None,
        opened_off_center=bool(e["grasped_at_open"] and e["open_xy_mm"] >= 12),
    )


def archive(run, old):
    rows, by_actor, failed_keys = [], {}, None
    folder = old / "test" / "traces"
    for s in SEED_PAIRS:
        failures = []
        for scene in TEST:
            with np.load(folder / f"{prior_name(s)}-{scene}.npz") as tr:
                if int(tr["success"]):
                    continue
                actor = events(tr)
            with np.load(folder / f"base-{scene}.npz") as tr:
                base = events(tr)
            failures.append((scene, base, actor))
            if s == 101:
                for name, v in (("base", base), ("incumbent-101", actor)):
                    rows.append(dict(scene=scene, base_won=base["success"], policy=name, **v))
        keys = [scene for scene, _, _ in failures]
        if failed_keys is None:
            failed_keys = keys
        assert keys == failed_keys
        by_actor[s] = {
            group: dict(Counter(a["mode"] for _, b, a in failures if group == "all" or b["success"]))
            for group in ("all", "base_won")
        }
    write_csv(run / "archived-failure-pairs.csv", rows)
    dump(run / "archived-failures.json", by_actor)


def paired(a, b):
    a, b = np.broadcast_arrays(a, b)
    delta = a - b
    rng = np.random.default_rng(9917)
    n, k = delta.shape
    blocks = []
    for _ in range(40):
        actors = rng.integers(n, size=(500, n, 1))
        scenes = rng.integers(k, size=(500, 1, k))
        blocks.append(delta[actors, scenes].mean(axis=(1, 2)) * 100)
    return dict(
        success_pct=100 * a.mean(),
        reference_pct=100 * b.mean(),
        difference_pp=100 * delta.mean(),
        interval_pp=np.percentile(np.concatenate(blocks), [2.5, 97.5]).tolist(),
        by_pair_pp=(100 * delta.mean(axis=1)).tolist(),
    )


@torch.no_grad()
def verify_batch(traces, actor, pi, wm):
    lengths = np.array([len(t["actions"]) for t in traces])
    duration = max(lengths)
    obs = torch.stack(
        [
            torch.from_numpy(np.pad(t["obs"], ((0, duration - n), (0, 0)), mode="edge"))
            for n, t in zip(lengths, traces)
        ]
    )
    actions = torch.stack(
        [torch.from_numpy(np.pad(t["actions"], ((0, duration - n), (0, 0)))) for n, t in zip(lengths, traces)]
    )
    generators = [torch.Generator().manual_seed(int(t["noise_seed"])) for t in traces]
    latent = wm.initial(obs[:, 0]) if actor is not None else None
    errors = []
    for t in range(duration):
        if t % HORIZON == 0:
            noise = torch.cat([torch.randn(1, HORIZON, ACTION_DIM, generator=g) for g in generators])
            if actor is not None:
                noise += actor.draw(latent, deterministic=True)[0][:, None]
            chunk = pi.sample(obs[:, t], noise)
        active = lengths > t
        errors.append((actions[active, t] - chunk[active, t % HORIZON]).abs().max())
        if actor is not None:
            latent, _ = wm.filter(wm.advance(latent, actions[:, t]), obs[:, t + 1])
    error = float(torch.stack(errors).max())
    assert error < 2e-4, error
    return error


def audit(run, old):
    pi, wm, models = controllers_for(run, old)
    bank = torch.load(run / "scenes.pt", weights_only=True)
    assert sorted(bank) == list(SCENES)
    tables, records, checks = {}, [], {}
    for split, noise_offset in STREAMS.items():
        folder = run / split
        data = sorted(
            (json.loads(line) for p in folder.glob("eval-*.jsonl") for line in p.read_text().splitlines()),
            key=lambda r: r["seed"],
        )
        assert [r["seed"] for r in data] == list(SCENES)
        assert all(set(r) == {"seed", "start", *models} for r in data)
        assert len(list((folder / "traces").glob("*.npz"))) == len(models) * len(SCENES)
        outcome, max_error, start_error = {}, 0.0, 0.0
        starts = {}
        for name, actor in models.items():
            entries = []
            for i in range(0, len(data), 32):
                group = []
                for row in data[i : i + 32]:
                    seed = row["seed"]
                    path = folder / "traces" / f"{name}-{seed}.npz"
                    with np.load(path) as f:
                        tr = dict(f)
                    obs, cmd, speeds = tr["obs"], tr["actions"], tr["cube_speeds_mps"]
                    assert np.isfinite(obs).all() and np.isfinite(cmd).all() and np.isfinite(speeds).all()
                    assert np.isfinite(tr["aperture_m"]).all() and (speeds >= 0).all()
                    assert len(cmd) == len(obs) - 1 and speeds.shape == (len(cmd), 2)
                    assert len(cmd) == 200 or min(obs[-1, 5], obs[-1, 8]) < -0.5
                    assert int(tr["seed"]) == seed and int(tr["noise_seed"]) == seed + noise_offset
                    np.testing.assert_array_equal(row["start"], bank[seed][0].numpy())
                    np.testing.assert_allclose(obs[0], row["start"], atol=0.01, rtol=0)
                    if seed not in starts:
                        starts[seed] = obs[0]
                    start_error = max(start_error, float(np.max(np.abs(obs[0, :9] - starts[seed][:9])) * 100))
                    value = summarize_trace(obs, tr["aperture_m"], speeds)
                    assert value == row[name] and int(tr["success"]) == value["success"]
                    e = events(tr)
                    entries.append(e)
                    records.append(dict(stream=split, scene=seed, arm=name, steps=len(cmd), **e))
                    group.append(tr)
                max_error = max(max_error, verify_batch(group, actor, pi, wm))
            outcome[name] = entries
        assert start_error <= 1
        tables[split] = outcome
        checks[split] = dict(max_action_error=max_error, max_start_difference_mm=start_error)
        print(split, checks[split], flush=True)

    report = {}
    kinds = {
        "base": ["base"] * len(SEED_PAIRS),
        "earlier": ["previous"] * len(SEED_PAIRS),
        "incumbent": [prior_name(s) for s in SEED_PAIRS],
        **{k: [f"{k}-{s}" for s in SEED_PAIRS] for k in ("control", "margin")},
    }
    for split, streams in [("test", ["test"]), ("noise", ["noise"]), ("combined", list(STREAMS))]:

        def matrix(kind, field):
            return np.array(
                [[[r[field] for r in tables[stream][a]] for a in kinds[kind]] for stream in streams],
                dtype=float,
            )

        success = {k: matrix(k, "success") for k in kinds}
        comparisons = {}
        for target, ref in (
            ("margin", "base"),
            ("margin", "incumbent"),
            ("margin", "control"),
            ("incumbent", "base"),
            ("incumbent", "earlier"),
            ("control", "incumbent"),
        ):
            a, b = success[target], success[ref]
            comparisons[target + "_vs_" + ref] = dict(
                **paired(a.mean(0), b.mean(0)),
                wins_by_pair=(a > b).sum(axis=(0, 2)).tolist(),
                losses_by_pair=(a < b).sum(axis=(0, 2)).tolist(),
            )
        mechanisms = {}
        mode_rates = {}
        for k in kinds:
            failed = 1 - success[k]
            off = matrix(k, "opened_off_center")
            never = matrix(k, "never_open")
            mechanisms[k] = dict(
                off_center_failures_per_100=float((failed * off).mean() * 100),
                never_open_failures_per_100=float((failed * never).mean() * 100),
                off_center_openings_per_100=float(off.mean() * 100),
            )
            mode_rates[k] = {"off_center_failures": failed * off, "never_open_failures": failed * never}
        mechanism_changes = {}
        for metric in mode_rates["margin"]:
            for ref in ("incumbent", "control"):
                summary = paired(mode_rates["margin"][metric].mean(0), mode_rates[ref][metric].mean(0))
                mechanism_changes[metric + "_vs_" + ref] = {
                    key: summary[key] for key in ("difference_pp", "interval_pp")
                }
        report[split] = dict(
            percent={k: 100 * v.mean() for k, v in success.items()},
            counts_by_pair={k: v.sum(axis=(0, 2)).astype(int).tolist() for k, v in success.items()},
            comparisons=comparisons,
            mechanisms=mechanisms,
            mechanism_changes=mechanism_changes,
        )
    write_csv(run / "paired-outcomes.csv", records)
    dump(
        run / "margin-summary.json", dict(episodes=len(records), scenes=len(SCENES), checks=checks, **report)
    )
    print(json.dumps(report["combined"], indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("archive", "audit"))
    p.add_argument("run", type=Path)
    p.add_argument("old", type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    (archive if args.mode == "archive" else audit)(args.run, args.old)
