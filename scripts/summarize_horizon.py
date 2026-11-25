"""Audit the matched-horizon experiment and summarize the two Isaac interventions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_transport import analyze
from probe_transport import run_probes
from summarize_stack import summarize


def paired_motion(rows, baseline, intervention):
    if not rows:
        return {"pairs": 0}
    a = np.array([baseline(r) for r in rows])
    b = np.array([intervention(r) for r in rows])
    return dict(pairs=len(rows), median_base=float(np.median(a)), median_intervention=float(np.median(b)),
                median_paired_difference=float(np.median(b-a)), increased=int(np.sum(b>a)))


def finish(run, prior):
    scores = summarize(run, 50, 100)
    arms = {"base", "original_demos", "original_varied", "demos", "varied", "pulse_base"}
    if set(scores["policies"]) != arms or scores["test_seeds"] != [208, 257]:
        raise ValueError("Unexpected fresh-scene evaluation block")
    old = json.loads((prior / "summary.json").read_text())
    old_first, old_last = old["test_seeds"]
    new = set(range(208, 258))
    forecast = json.loads((run / "horizon-forecast.json").read_text())
    if (old_last >= min(new) or old_first != 158 or new.intersection(forecast["diagnostic_seeds"])
            or scores["corpus_sha256"] != old["corpus_sha256"]):
        raise ValueError("Experiment split or corpus mismatch")
    for a in ("demos", "varied"):
        short = torch.load(prior / f"{a}.pt", weights_only=True)
        long = torch.load(run / f"{a}.pt", weights_only=True)
        if not long["matched_short"] or long["imagination_chunks"] != 8 or long["rl_steps"] != 1500:
            raise ValueError("Unexpected horizon or training budget")
        for module in ("world", "policy"):
            if (short[module].keys() != long[module].keys()
                    or not all(torch.equal(x, long[module][k]) for k, x in short[module].items())):
                raise ValueError(f"Changed {module} weights for {a}")
    if len(list((run / "traces").glob("*.npz"))) != 300:
        raise ValueError("Unexpected number of scored traces")
    motion = analyze(run, run)
    run_probes(run, run)
    pulses = motion["pulses"]
    subset = [r for r in pulses if r["matched"]]
    with (run / "height" / "height-check.json").open() as f:
        heights = json.load(f)
    if len(heights) != len(set(r["seed"] for r in heights)) or any(r["seed"] in new for r in heights):
        raise ValueError("Height check overlaps fresh trials or repeats a scene")
    with (run / "prior" / "transport-events.csv").open() as f:
        intended = sorted(int(r["seed"]) for r in csv.DictReader(f) if r["policy"] == "base"
                          and r["stall_tick"] and int(r["stall_tick"]) <= 164)[:12]
    if [r["seed"] for r in heights] != intended:
        raise ValueError("Height experiment incomplete")
    complete = [r for r in heights if "matched" in r]
    h = [r for r in complete if r["matched"]]
    def height_metrics(rows):
        return {k: paired_motion(rows, lambda r: r["base"][k], lambda r: r["height"][k])
                for k in ("red_rise_mm", "progress_24_mm", "postpulse_request_mm_tick")}
    def xy_progress(rows):
        return paired_motion(rows, lambda r: r["base_progress_mm"], lambda r: r["pulse_progress_mm"])
    result = dict(test_seeds=scores["test_seeds"], policies=scores["policies"], paired=scores["paired"],
                  transport=motion["policies"], xy_pulse=dict(all_progress_mm=xy_progress(pulses),
                                                             matched_progress_mm=xy_progress(subset)),
                  height_pulse=dict(enrolled=len(heights), triggered=sum(r["applied"]>0 for r in heights),
                                    stacks={a: sum(r[a]["success"] for r in heights) for a in ("base", "height")},
                                    all_complete=height_metrics(complete), matched=height_metrics(h)))
    (run / "horizon-summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = finish(args.run, args.prior)
    print(json.dumps({k: v for k, v in out.items() if k != "transport"}, indent=2))
