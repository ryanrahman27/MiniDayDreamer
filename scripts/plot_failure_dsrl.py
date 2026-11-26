"""Show recurring archived failures beside the four fresh DSRL arms."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def segments(counts):
    modes = counts["modes"]
    stacked = modes.get("stable_stack", 0)
    never = sum(v for k, v in modes.items() if k.startswith("never_open_"))
    off = modes.get("first_open_off_center", 0)
    return [stacked, never, off, sum(modes.values()) - stacked - never - off]


def plot(run):
    archived = json.loads((run / "archived-failure-modes.json").read_text())
    fresh = json.loads((run / "failure-dsrl-summary.json").read_text())
    colors = ["#00796b", "#a74628", "#e3af57", "#cdd2d9"]
    categories = ["Stable stack", "Never opened", "Failed after off-center opening", "Other failure"]
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 4]})
    fig.subplots_adjust(left=0.24, right=0.98, top=0.81, bottom=0.13, hspace=0.52)
    groups = []
    for cohort, arm, label in (
        ("midbase", "mid_base", "Seeds 308–407"),
        ("target", "base", "Seeds 408–507"),
        ("approach", "base", "Seeds 508–607"),
    ):
        entry = archived[cohort][arm]
        groups.append(
            (
                label,
                segments(
                    {"modes": {**entry["failure"]["modes"], "stable_stack": entry["success"]["episodes"]}}
                ),
            )
        )
    current = [
        (label, segments(fresh["policies"][arm]["all"]))
        for arm, label in (
            ("base", "Frozen base"),
            ("original_direct", "Original DSRL"),
            ("matched_direct", "Same starts, direct reward"),
            ("targeted", "Failure-targeted DSRL"),
        )
    ]
    for ax, rows, title in zip(
        axs, (groups, current), ("Archived frozen-base runs", "Fresh paired Isaac runs")
    ):
        for y, (label, parts) in enumerate(rows):
            left = 0
            for i, value in enumerate(parts):
                ax.barh(y, value, left=left, color=colors[i], height=0.75)
                if value >= 5:
                    ax.text(
                        left + value / 2,
                        y,
                        str(value),
                        ha="center",
                        va="center",
                        color="white" if i < 2 else "#212b33",
                        fontsize=9,
                    )
                left += value
        ax.set_yticks(range(len(rows)), [label for label, _ in rows])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[1].set_xlim(0, 100)
    axs[1].set_xlabel("Scenes out of 100")
    fig.text(0.04, 0.95, "Stable stacks and recurring failures", fontsize=16, fontweight="bold")
    fig.legend(
        handles=[Patch(color=c, label=t) for c, t in zip(colors, categories)],
        loc="upper left",
        bbox_to_anchor=(0.03, 0.93),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.savefig(run / "failure-modes.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    plot(p.parse_args().run)
