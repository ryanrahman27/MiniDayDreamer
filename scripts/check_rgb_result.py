"""Count stable stacks on the paired confirmation trials. Uses the standard library."""

import argparse
import csv
import json
from pathlib import Path

def report(directory):
    protocol_path = directory / "test-wait" / "protocol.json"
    if not protocol_path.is_file():
        protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    start = protocol["first_scene"]
    scenes = range(start, start + protocol["width"] * protocol["batches"])
    arms = protocol["actors"]
    seeds = protocol["actor_seeds"]
    paired = {arm: {} for arm in arms}
    expected = {(scene, scene + offset) for scene in scenes for offset in protocol["noise_offsets"]}

    with (directory / "test-outcomes.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            arm = row["arm"]
            if arm == "rgb-repeat":
                continue  # Only the first noise stream has a repeat.
            scene, noise = int(row["scene"]), int(row["noise_seed"])
            key = (scene, noise)
            if key not in expected or key in paired[arm]:
                raise ValueError(f"Unexpected or duplicate trial for {arm}: {key}")
            if int(row["actor_seed"]) != seeds[(scene - start) % len(seeds)]:
                raise ValueError(f"Wrong actor seed for scene {scene}")
            success = int(row["success"])
            if success not in (0, 1):
                raise ValueError(f"Invalid success value in {arm}: {success}")
            paired[arm][key] = success

    for arm, trials in paired.items():
        if trials.keys() != expected:
            raise ValueError(f"Incomplete paired trials for {arm}")
        stacks = sum(trials.values())
        print(f"{arm:20s} {stacks:3}/{len(trials)}   {stacks / len(trials):5.1%}")

    base, actor = paired["rgb-base"], paired["wait-" + protocol["selected"]]
    wins = sum(actor[k] > base[k] for k in expected)
    losses = sum(actor[k] < base[k] for k in expected)
    gain_points = 100 * (wins - losses) / len(expected)
    print(f"{wins} gains, {losses} losses, {gain_points:+.1f} percentage points")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, help="Scored run containing test-outcomes.csv")
    report(cli.parse_args().directory)
