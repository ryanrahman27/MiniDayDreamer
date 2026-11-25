"""Evaluate frozen original and residual RSSMs on identical six-action forecast windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from minidaydreamer.data import Dataset
from summarize_stack import cube_forecast


def audit(run, original, first_test, count):
    fitting = Dataset(run / "corpus.npz", 0)
    confirm = Dataset(run / "forecast-check.npz", 0)
    base = torch.load(run / "base.pt", weights_only=True)
    calibration = {r["seed"] for c in base["calibration"] for r in c["outcomes"]}
    test = set(range(first_test, first_test + count))
    confirmation = set(confirm.seeds[confirm.split == "holdout"])
    if confirmation & (set(fitting.seeds) | calibration | test) or test & (calibration | set(fitting.seeds)):
        raise ValueError("Confirmation or policy-test scenes overlap an earlier split")
    result = {}
    for label, data in (("development", fitting), ("confirmation", confirm)):
        ids = np.flatnonzero(data.split == "holdout")
        row = {"episodes": len(ids), "seeds": sorted(map(int, set(data.seeds[ids]))), "models": {}}
        for version, folder in (("original", original), ("residual", run)):
            for arm in ("demos", "varied"):
                c = torch.load(folder / f"{arm}.pt", weights_only=True)
                row["models"][f"{version}_{arm}"] = cube_forecast(c["world"], data, c.get("world_model", "absolute"))
        result[label] = row
    (run / "forecast-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--first-test", type=int, required=True)
    parser.add_argument("--scenes", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.run, args.original, args.first_test, args.scenes), indent=2))
