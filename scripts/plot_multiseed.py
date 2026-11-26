"""Display all five training seeds and the previous checkpoint on each untouched cohort."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiseed_study import TEST_TRAIN


def plot(run):
    data = json.loads((run / "multiseed-summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5), sharey=True)
    colors = ["#227f72", "#aa642a"]
    for ax, split, title in zip(
        axes,
        ("test", "noise", "scalar"),
        ("100 paired scenes", "Same 100, new action noise", "32 new scenes, scalar Isaac"),
    ):
        section = data[split]
        selected = section["winner_vs_base"]
        base = selected["reference_success_pct"]
        prev = section["previous_vs_base"]["mean_difference_pp"]
        y = np.arange(len(TEST_TRAIN))
        ax.axvline(0, color="#404b51", lw=1)
        ax.axvline(prev, color="#6b737b", ls="--", label="Previous checkpoint")
        ax.scatter(selected["actor_differences_pp"], y, s=63, c=colors[0], label="Selected recipe", zorder=3)
        if "from_scratch_vs_base" in section:
            ax.scatter(
                section["from_scratch_vs_base"]["actor_differences_pp"],
                y + 0.17,
                s=35,
                facecolors="none",
                edgecolors=colors[1],
                linewidths=1.5,
                label="From scratch",
                zorder=2,
            )
        ax.set_title(f"{title}\nFrozen base {base:g}%", loc="left", fontsize=11)
        ax.set_yticks(y, [str(seed) for seed in TEST_TRAIN])
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    endpoints = [0]
    for split in ("test", "noise", "scalar"):
        for key in ("winner_vs_base", "previous_vs_base", "from_scratch_vs_base"):
            if key in data[split]:
                endpoints.extend(data[split][key]["actor_differences_pp"])
    lo, hi = np.floor(min(endpoints) / 5) * 5 - 2, np.ceil(max(endpoints) / 5) * 5 + 2
    for ax in axes:
        ax.set_xlim(lo, hi)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Actor-training seed")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.70, wspace=0.13)
    fig.text(0.04, 0.94, "Released-stack success relative to the frozen base", fontsize=16, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.03, 0.91), ncol=3, frameon=False)
    fig.supxlabel("Paired gain over the base (percentage points)", fontsize=11, y=0.035)
    fig.savefig(run / "multiseed-gaps.png", dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    plot(p.parse_args().run)
