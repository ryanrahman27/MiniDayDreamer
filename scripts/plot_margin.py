"""Plot the archived paired opening measurements and the locked held-out comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(run):
    with (run / "archived-failure-pairs.csv").open() as f:
        lost = [r for r in csv.DictReader(f) if r["base_won"] == "1"]
    paired = {int(r["scene"]): {} for r in lost}
    for row in lost:
        paired[int(row["scene"])][row["policy"]] = float(row["open_xy_mm"]) if row["open_xy_mm"] else None
    summary = json.loads((run / "margin-summary.json").read_text())["combined"]["comparisons"]
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 5.1), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.085, right=0.965, top=0.79, bottom=0.18, wspace=0.45)
    fig.suptitle(
        "Where the cube was released, and what steering changed", fontsize=17, x=0.055, ha="left", y=0.985
    )
    base, old, margin = "#355876", "#bc6337", "#007a68"
    left, right = ax
    left.set_title("10 archived scenes the base won", loc="left", fontsize=12, fontweight="bold", pad=13)
    left.axvline(12, color="#8a9296", ls="--", lw=1)
    for j, (seed, values) in enumerate(paired.items()):
        b, o = values["base"], values["incumbent-101"]
        left.scatter(b, j, c=base, s=34, zorder=3, label="Frozen base" if j == 0 else None)
        if o is None:
            left.text(16, j, "never opened", va="center", fontsize=8, color=old)
        else:
            left.plot([b, o], [j, j], c="#b7c1c6", lw=1.4)
            left.scatter(o, j, c=old, marker="D", s=27, zorder=3, label="Tuned" if j == 1 else None)
    left.set_yticks(range(len(paired)), list(paired))
    left.invert_yaxis()
    left.set_xlabel("Horizontal offset at first opening (mm)")
    left.set_ylabel("Scene")
    left.set_xlim(0, 25)
    left.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.02, -0.125), fontsize=9)
    right.set_title("128 fresh physical scenes", loc="left", fontsize=12, fontweight="bold", pad=13)
    right.axvline(0, color="#8a9296", ls="--", lw=1)
    info = [
        ("incumbent_vs_base", "Tuned vs. base", base),
        ("margin_vs_base", "Margin vs. base", margin),
        ("margin_vs_incumbent", "Margin vs. tuned", margin),
        ("margin_vs_control", "Margin vs. control", margin),
    ]
    bounds = [0]
    for i, (key, label, col) in enumerate(info):
        row = summary[key]
        lo, hi = row["interval_pp"]
        d = row["difference_pp"]
        y = 3 - i
        right.hlines(y, lo, hi, color=col, lw=2.4)
        right.scatter(d, y, c=col, s=45, zorder=3)
        right.text(d, y + 0.19, f"{d:+.1f}", ha="center", fontsize=9, color=col)
        bounds.extend([lo, hi])
    right.set_yticks(range(3, -1, -1), [s[1] for s in info])
    right.set_xlim(min(bounds) - 3, max(bounds) + 3)
    right.set_ylim(-0.65, 3.68)
    right.set_xlabel("Paired stack-rate change (percentage points)")
    for axis in ax:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#aeb9bc")
        axis.tick_params(length=0, pad=5, labelsize=9)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#e5eaec", lw=0.7)
    fig.text(
        0.57, 0.045, "Dots are paired means. Lines show 95% bootstrap intervals.", fontsize=8, color="#3a4951"
    )
    fig.savefig(run / "release-margin.png", dpi=175, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot(parser.parse_args().run)
