"""Show changes in commanded motion and the two paired Isaac interventions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scatter(ax, rows, xkey, ykey, title):
    if not rows:
        ax.set_title(title)
        ax.text(.5, .5, "No complete paired window", ha="center", transform=ax.transAxes)
        return
    x, y = np.array([(r[xkey], r[ykey]) for r in rows]).T
    matched = np.array([r["matched"] for r in rows])
    low, high = min(0, x.min(), y.min())-3, max(0, x.max(), y.max())+3
    ax.plot([low, high], [low, high], "--", color="#888888", linewidth=1)
    ax.scatter(x[~matched], y[~matched], marker="o", facecolors="none", edgecolors="#858a90", s=34)
    ax.scatter(x[matched], y[matched], color="#21736a", s=34)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Base progress (mm)", ylabel="Intervention progress (mm)")
    ax.set_title(f"{title}\n{matched.sum()} of {len(rows)} pairs closely matched", loc="left", fontsize=10,
                 fontweight="bold")


def plot(run):
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), layout="constrained")
    with (run / "transport-events.csv").open() as f:
        pairs = [(float(r["approach_request_mm_tick"]), float(r["stall_request_mm_tick"]))
                 for r in csv.DictReader(f) if r["policy"] == "base" and r["approach_end"]]
    ax = axes[0]
    if pairs:
        for p in pairs:
            ax.plot([0, 1], p, color="#6b95b2", linewidth=1, alpha=.43)
        ax.plot([0, 1], np.median(pairs, axis=0), "o-", color="#15384e", linewidth=3, label="Median")
        ax.legend(frameon=False, fontsize=8)
    ax.axhline(0, color="#9b9c9d", linewidth=.8)
    ax.set_xticks([0, 1], ["Approach", "Stall"])
    ax.set(xlim=(-.2, 1.2), ylabel="Target-directed command (mm/tick)")
    ax.set_title("Frozen base on fresh scenes", loc="left", fontsize=11, fontweight="bold")
    motion = json.loads((run / "transport-summary.json").read_text())
    scatter(axes[1], motion["pulses"], "base_progress_mm", "pulse_progress_mm", "XY intervention · 12 ticks")
    heights = json.loads((run / "height" / "height-check.json").read_text())
    rows = [dict(x=r["base"]["progress_24_mm"], y=r["height"]["progress_24_mm"], matched=r["matched"])
            for r in heights if "matched" in r]
    scatter(axes[2], rows, "x", "y", "Height intervention · 24 ticks")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("What changes when the cube stalls short of the target?", fontweight="bold", fontsize=14)
    fig.savefig(run / "transport-evidence.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot(parser.parse_args().run)
