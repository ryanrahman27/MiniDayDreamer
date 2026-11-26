"""Plot same-input opening, two independent paired tests, and intervention height."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(result):
    prior = json.loads((result / "historical-gap.json").read_text())["same_observation"]
    fig, axes = plt.subplots(
        1, 3, figsize=(13.4, 4.8), gridspec_kw={"width_ratios": [0.84, 1.4, 1.32]}
    )
    ax = axes[0]
    metrics = ("actor_opens_whole_chunk", "fixed_opens_whole_chunk")
    values = [100 * prior[k] / prior["first_interventions"] for k in metrics]
    ax.bar([0, 1], values, width=0.56, color=["#9baab5", "#57748b"])
    for i, value in enumerate(values):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center")
    ax.set_xticks([0, 1], ["Previous\nDSRL", "Fixed"])
    ax.set_ylim(0, 112)
    ax.set_ylabel("All three commands open (%)")
    ax.set_title("Same archived inputs", loc="left")
    labels = ["Frozen base", "Original fixed", "Lower gate only"]
    y = np.array([2, 1, 0])
    axes[1].axvline(0, color="#869198", lw=1)
    axes[2].axvspan(-8, 8, color="#dceae2")
    axes[2].axvline(0, color="#84988b", lw=0.8)
    for folder, legend, off, color in (
        (result, "Initial wait actor", 0.13, "#69859c"),
        (result / "calibrated", "Calibrated wait actor", -0.13, "#10735d"),
    ):
        summary = json.loads((folder / "summary.json").read_text())
        with (folder / "test-outcomes.csv").open() as f:
            rows = list(csv.DictReader(f))
        actor = summary["selected"]
        lower = "lower-" + actor.split("-", 1)[1]
        ax = axes[1]
        for pos, arm in zip(y, ("rgb-base", "fixed-permissive", lower), strict=True):
            comparison = summary["selected_minus"][arm]
            interval = np.array(comparison["scene_ci"]) * 100
            delta = comparison["difference"] * 100
            ax.plot(interval, [pos + off] * 2, lw=2, color=color)
            ax.scatter(
                delta,
                pos + off,
                s=42,
                color=color,
                label=legend if pos == 2 else None,
                zorder=3,
            )
        for pos, arm in zip(y, ("fixed-permissive", lower, actor), strict=True):
            vals = [
                float(r["first_shift_height"])
                for r in rows
                if r["arm"] == arm and r["first_shift_height"]
            ]
            if vals:
                lo, mid, hi = np.percentile(vals, [25, 50, 75])
                axes[2].plot([lo, hi], [pos + off] * 2, lw=2, color=color)
                axes[2].scatter(mid, pos + off, s=42, color=color, zorder=3)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Wait actor minus comparator (pp)")
    axes[1].set_title("Paired change in stacks", loc="left")
    axes[2].set_yticks(y, ["Original fixed", "Lower gate only", "Wait actor"])
    axes[2].set_xlabel("Height above stack position (mm)")
    axes[2].set_title("At first noise shift", loc="left")
    for ax in axes[1:]:
        ax.set_ylim(-0.5, 2.5)
        ax.grid(axis="x", alpha=0.12)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.legend(
        *axes[1].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.61, 0.045),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Choosing when to open the frozen RGB policy",
        fontsize=18,
        weight="bold",
        x=0.055,
        ha="left",
        y=0.96,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.26, top=0.77, wspace=0.62)
    fig.text(
        0.055,
        0.01,
        "Each actor has an independent 160-scene bank. Stack intervals are paired 95% scene bootstraps. Height intervals show interquartile ranges. Shading marks ±8 mm.",
        fontsize=9,
        color="#4d595d",
    )
    fig.savefig(result / "wait-release.png", dpi=155, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", type=Path)
    plot(p.parse_args().result)
