"""Select a frozen imitation checkpoint by unsteered stacking on development scenes."""
import argparse
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import torch

from minidaydreamer.experiment import policy_episode
from minidaydreamer.models import make_policy
from reward_study import digest, fit_bc, load


STEPS = tuple(range(50, 3001, 50)) + tuple(range(3250, 6001, 250)) + tuple(range(6500, 12001, 500))
DEV = range(5300, 5332)
ACCEPT = range(14, 19)


def next_candidate(scores):
    """Search by training duration, refining brackets and allowing nonmonotonic success rates."""
    if not scores:
        return 1000
    remaining = [s for s in STEPS if s not in scores]
    if not remaining:
        raise RuntimeError("No checkpoint reached the development band")
    tested = sorted(scores)
    for lo, hi in zip(tested, tested[1:]):
        between = [s for s in remaining if lo < s < hi]
        if between and (scores[lo] - 16) * (scores[hi] - 16) < 0:
            return min(between, key=lambda s: abs(s - (lo + hi) / 2))
    if max(scores.values()) < 16 and tested[-1] < STEPS[-1]:
        return min(remaining, key=lambda s: abs(s - min(2 * tested[-1], STEPS[-1])))
    if min(scores.values()) > 16 and tested[0] > STEPS[0]:
        return min(remaining, key=lambda s: abs(s - tested[0] / 2))
    best = min(tested, key=lambda s: (abs(scores[s] - 16), s))
    return min(remaining, key=lambda s: (abs(s - best), s))


def dev_file(run, step, first):
    return run / f"dev-{step}-{first}.json"


def score_shard(run, step, first):
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, enable_cameras=False)
    try:
        from minidaydreamer.isaac import IsaacStack
        env = IsaacStack(cameras=False)
        try:
            checkpoint = load(run / f"candidate-{step}.pt")
            policy = make_policy(checkpoint["policy_kind"]).eval()
            policy.load_state_dict(checkpoint["policy"])
            rows = [dict(seed=seed, **policy_episode(env, policy, None, None, seed)[0])
                    for seed in range(first, first + 8)]
            dest = dev_file(run, step, first)
            partial = dest.with_suffix(".tmp")
            partial.write_text(json.dumps(rows) + "\n")
            partial.replace(dest)
        finally:
            env.close()
    finally:
        launcher.app.close()


def select(run):
    firsts = list(DEV[::8])
    scores = {}
    history = []
    while True:
        step = next_candidate(scores)
        with ExitStack() as logs:
            workers = []
            for first in firsts:
                if dev_file(run, step, first).exists():
                    continue
                log = logs.enter_context((run / f"dev-{step}-{first}.log").open("a"))
                workers.append(subprocess.Popen([sys.executable, __file__, "dev", str(run), "--step", str(step),
                                                 "--first", str(first)], stdout=log, stderr=subprocess.STDOUT))
            codes = [w.wait() for w in workers]
        if any(codes):
            raise RuntimeError(f"Development worker failed at {step} updates: {codes}")
        rows = [r for first in firsts for r in json.loads(dev_file(run, step, first).read_text())]
        if [r["seed"] for r in rows] != list(DEV):
            raise ValueError("Incomplete development scenes")
        scores[step] = sum(r["success"] for r in rows)
        history.append(dict(step=step, stacks=scores[step], outcomes=rows))
        print(f"{step} updates: {scores[step]}/{len(rows)} stable stacks", flush=True)
        if scores[step] in ACCEPT:
            break
    candidate = run / f"candidate-{step}.pt"
    shutil.copyfile(candidate, run / "selected-base.pt")
    result = dict(accepted_counts=list(ACCEPT), selected_step=step, selected_sha256=digest(candidate),
                  development=history)
    (run / "selection.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "select", "dev"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--step", type=int, choices=STEPS)
    parser.add_argument("--first", type=int, choices=DEV[::8])
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "fit":
        fit_bc(args.run, STEPS)
    elif args.command == "select":
        select(args.run.resolve())
    else:
        score_shard(args.run, args.step, args.first)
