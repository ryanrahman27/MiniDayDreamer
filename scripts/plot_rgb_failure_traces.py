"""Plot measured placement trajectories for matched replays from the original failure cohort."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(run):
    with (run / "original-127.csv").open() as f:
        rows = list(csv.DictReader(f))
    cases = [
        ("never_centered_no_open", "Held outside tolerance"),
        ("seated_no_open", "Reached both tolerances, stayed closed"),
        ("opened_off_center", "Opened off-center"),
    ]
    colors = ["#266c83", "#887128", "#ab5142"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True, sharey="row")
    for col, ((mode, title), color) in enumerate(zip(cases, colors, strict=True)):
        chosen = [r for r in rows if r["mode"] == mode]
        values = np.array([float(r["min_held_xy_mm"]) for r in chosen])
        highlighted = np.argmin(abs(values - np.median(values)))
        for i, row in enumerate(chosen):
            group, off = row["batch_seed"], int(row["noise_seed"]) - int(row["scene"])
            with np.load(
                run / "test-diagnosis" / f"{group}-{off}-rgb-base.npz"
            ) as data:
                obs = data["obs"][:, int(row["lane"])]
            lift = int(row["lift_tick"])
            opened = int(row["opening_tick"]) if row["opening_tick"] else None
            last = opened if opened is not None else int(row["steps"])
            o = obs[lift : last + 1]
            xy = 100 * np.linalg.norm(o[:, 3:5] - o[:, 6:8], axis=1)
            height = 100 * (o[:, 5] - o[:, 8]) - 46.8
            for ax, series in zip(axes[:, col], (xy, height), strict=True):
                strong = i == highlighted
                ax.plot(
                    np.arange(len(series)),
                    series,
                    c=color,
                    alpha=1 if strong else 0.17,
                    lw=2.6 if strong else 0.9,
                    zorder=3 if strong else 1,
                )
                if opened is not None:
                    ax.scatter(
                        len(series) - 1,
                        series[-1],
                        marker="v",
                        s=42 if strong else 14,
                        color=color,
                        alpha=1 if strong else 0.45,
                        zorder=4,
                    )
        axes[0, col].set_title(f"{title}\n{len(chosen)} replays", fontsize=13, pad=10)
        axes[0, col].axhspan(0, 12, color="#8fc19d", alpha=0.26)
        axes[1, col].axhspan(-8, 8, color="#8fc19d", alpha=0.26)
        axes[1, col].axhline(0, c="#729b83", lw=0.7)
        axes[1, col].set_xlabel("Ticks after pickup", fontsize=11)
    axes[0, 0].set_ylabel("Horizontal offset (mm)", fontsize=11)
    axes[1, 0].set_ylabel("Height error (mm)", fontsize=11)
    axes[0, 0].set_ylim(bottom=0)
    for ax in axes.flat:
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.14)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Where the red cube went after pickup",
        ha="left",
        x=0.07,
        y=0.975,
        fontsize=21,
        weight="bold",
    )
    fig.text(
        0.07,
        0.905,
        "Each line is one replay. Green bands mark the placement tolerances. Triangles mark the first open command.",
        fontsize=12,
    )
    fig.text(
        0.07,
        0.016,
        "Darker lines show the run nearest its group’s median closest horizontal approach. Curves stop at opening or episode end.",
        fontsize=10,
        color="#444",
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.105, top=0.77, hspace=0.19, wspace=0.13
    )
    fig.savefig(run / "placement-traces.png", dpi=140, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    draw(p.parse_args().run)
