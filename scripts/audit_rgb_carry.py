"""Check the frozen reward probe's horizontal bearing on real, held-out base rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from rgb_approach_probe import ApproachProbe
from rgb_failure_events import events, first


@torch.inference_mode()
def audit(run):
    folder = run / "test-fresh"
    checkpoint = torch.load(run / "approach.pt", weights_only=True, map_location="cpu")
    probe = ApproachProbe().eval()
    probe.load_state_dict(checkpoint["probe"])
    actual, predicted, outcomes = [], [], []
    trials = 0
    for path in sorted(folder.glob("*-rgb-base.json")):
        with np.load(path.with_suffix(".npz")) as data:
            obs, commands = data["obs"], data["actions"]
            features = data["visual_features"]
            for row in json.loads(path.read_text()):
                trials += 1
                lane = row["lane"]
                state = obs[: row["steps"] + 1, lane]
                held = state[:, 9] > 0.5
                lift = first(held & (state[:, 5] > 0.65))
                if lift is None:
                    continue
                tick = lift + 12
                if tick >= len(state) or not held[lift : tick + 1].all():
                    continue
                if (commands[lift:tick, lane, 3] <= 0).any():
                    continue
                delta = state[tick, 6:8] - state[tick, 3:5]
                if 100 * np.linalg.norm(delta) < 12:
                    continue
                actual.append(delta)
                predicted.append(features[tick, lane])
                outcome = events(state, commands[: row["steps"], lane], row["success"])
                outcomes.append(outcome["mode"])
    assert trials == 320 and actual
    vectors = np.stack(actual)
    actual_unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    response = probe(torch.from_numpy(np.stack(predicted)).float())
    # The reward uses normalized XY directly when scoring commanded direction.
    pred = response[:, :2].numpy()
    pred_unit = pred / np.linalg.norm(pred, axis=1, keepdims=True).clip(1e-8)
    cosine = (pred_unit * actual_unit).sum(1)
    rng = np.random.default_rng(812)
    shuffled = [
        (pred_unit[rng.permutation(len(pred_unit))] * actual_unit).sum(1).mean()
        for _ in range(500)
    ]
    result = {
        "arm": "rgb-base",
        "base_trials": trials,
        "eligible_carries": len(cosine),
        "min_offset_mm": 12,
        "ticks_after_lift": 12,
        "bearing_cosine": float(cosine.mean()),
        "median_bearing_cosine": float(np.median(cosine)),
        "positive_direction_fraction": float((cosine > 0).mean()),
        "shuffled_mean_bearing_cosine": float(np.mean(shuffled)),
        "median_carry_probability": float(response[:, 3].sigmoid().median()),
    }
    outcomes = np.asarray(outcomes)
    result["by_outcome"] = {}
    for name in ("never_centered_no_open", "stacked"):
        sample = cosine[outcomes == name]
        if len(sample):
            result["by_outcome"][name] = {
                "trials": len(sample),
                "bearing_cosine": float(sample.mean()),
                "positive_direction_fraction": float((sample > 0).mean()),
            }
    (run / "carry-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
