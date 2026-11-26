"""Show the learned two-view release cue and paired fresh-scene stacking differences."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rgb_release_geometry import Geometry


def plot(run):
    report = json.loads((run / "summary.json").read_text())
    geom = Geometry().eval()
    geom.load_state_dict(
        torch.load(run / "release-geometry.pt", weights_only=True, map_location="cpu")[
            "geometry"
        ]
    )
    path = sorted((run / "inputs").glob("valid-*.npz"))[0]
    with np.load(path) as f:
        a = f["actions"]
        release = int(np.where(np.diff((a[:, 3] > 0).astype(int)) == -1)[0][0] + 1)
        frames = f["rgb"][[release - 15, release]]
    scores = geom(torch.from_numpy(frames)).tolist()
    fig = plt.figure(figsize=(11.2, 5.0), layout="constrained")
    blocks = fig.add_gridspec(1, 2, width_ratios=[0.88, 1.5])
    side = blocks[0].subgridspec(2, 2, wspace=0.01, hspace=0.01)
    for col, (frame, score) in enumerate(zip(frames, scores, strict=True)):
        for row in range(2):
            ax = fig.add_subplot(side[row, col])
            ax.imshow(frame[3 * row : 3 * row + 3].transpose(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(
                    ("15 ticks earlier" if col == 0 else "Expert release")
                    + f"\nScore {score:.2f}",
                    fontsize=11,
                )
            if col == 0:
                ax.set_ylabel(("Table RGB", "Wrist RGB")[row], fontsize=10)
    ax = fig.add_subplot(blocks[1])
    controls = [
        "rgb-base",
        "rgb-target",
        "rgb-incumbent",
        "fixed-" + report["selected"].removeprefix("release-"),
    ]
    labels = ["Frozen base", "Alignment DSRL", "Progress DSRL", "Fixed noise"]
    pairs = [report["target_minus"][k]["stacks"] for k in controls]
    vals = np.array([p["delta"] for p in pairs]) * 100
    intervals = np.array([p["scene_ci"] for p in pairs]) * 100
    lo = min(-3, float(intervals.min()) - 2)
    hi = max(5, float(intervals.max()) + 4)
    ax.axvline(0, color="#888", lw=1)
    for i, (val, interval) in enumerate(zip(vals, intervals, strict=True)):
        ypos = 3 - i
        ax.plot(
            interval,
            [ypos, ypos],
            color="#267386",
            linewidth=2.1,
            solid_capstyle="round",
        )
        ax.scatter(val, ypos, s=58, color="#12576a", zorder=3)
        ax.annotate(
            f"{val:+.1f}",
            (val, ypos),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.55, 3.66)
    ax.set_yticks([3, 2, 1, 0], labels)
    ax.set_xlabel("Release DSRL minus comparator (percentage points)")
    ax.set_title("Paired change in stable stacks", loc="left", fontsize=14, pad=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.12)
    fig.suptitle(
        "RGB-guided release on fresh Isaac scenes", fontsize=19, fontweight="bold"
    )
    fig.savefig(run / "release-study.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    plot(p.parse_args().run)
