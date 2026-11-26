"""Plot three end-of-episode stacking outcomes on the matched Isaac cohort."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_placement(run):
    with (run / "paired-outcomes.csv").open() as f:
        rows = list(csv.DictReader(f))
    for arm in ("targeted", "retreat", "base"):
        example = next(
            (r for r in rows if r["arm"] == arm and r["strict"] == "0" and r["survived_withdrawal"] == "1"),
            None,
        )
        if example:
            break
    if not example:
        return
    name = f"{arm}-{example['seed']}"
    with (
        np.load(run / "traces" / (name + ".npz")) as policy,
        np.load(run / "physics" / (name + ".npz")) as test,
    ):
        before, after = policy["obs"][-21:], test["withdrawal_obs"]
    states = np.concatenate((before, after))
    time = (np.arange(len(states)) - len(before) + 1) / 20
    tool, red, blue = (states[:, i : i + 3] * 100 for i in (0, 3, 6))
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)
    axes[0].plot(time, np.linalg.norm(tool - red, axis=1), color="#29627a", label="Tool to red cube")
    axes[0].axhline(50, color="#525e68", ls="--", label="50 mm clearance")
    axes[1].plot(
        time, np.linalg.norm(red[:, :2] - blue[:, :2], axis=1), color="#2b7859", label="Horizontal error"
    )
    axes[1].plot(time, abs(red[:, 2] - blue[:, 2] - 46.8), color="#ae6721", label="Absolute height error")
    for val, color in ((12, "#2b7859"), (8, "#ae6721")):
        axes[1].axhline(val, color=color, ls=":", linewidth=1)
    for ax in axes:
        ax.axvspan(0, 0.6, color="#e6e9eb", alpha=0.75)
        ax.axvline(0, color="#69747b", lw=1)
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False, ncol=2)
        ax.set_ylabel("Distance (mm)")
    axes[1].set_xlabel("Seconds from the end of the policy rollout")
    fig.suptitle(
        f"Released stack during a withdrawal test · {arm} policy · scene {example['seed']}", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(run / "withdrawal-example.png", dpi=160, facecolor="white")
    plt.close(fig)


def plot(run):
    counts = json.loads((run / "retreat-summary.json").read_text())["policies"]
    arms = ["base", "targeted", "retreat"]
    metrics = [
        ("strict", "Policy cleared 50 mm", "#29627a"),
        ("released_terminal", "Released and stable at finish", "#518a65"),
        ("survived_withdrawal", "Survived withdrawal test", "#b67737"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    y = np.arange(3)
    for i, (metric, label, color) in enumerate(metrics):
        values = [counts[arm][metric] for arm in arms]
        bars = ax.barh(y + (i - 1) * 0.24, values, height=0.2, color=color, label=label)
        ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_yticks(y, ["Frozen base", "Current targeted DSRL", "Retreat-reward DSRL"])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Scenes out of 100")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.25, right=0.96, bottom=0.15, top=0.77)
    fig.text(0.035, 0.93, "Cube stability and gripper clearance", fontsize=16, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.025, 0.907), ncol=3, frameon=False, fontsize=9)
    fig.savefig(run / "retreat-outcomes.png", dpi=165, facecolor="white")
    plt.close(fig)
    plot_placement(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot(parser.parse_args().run)
