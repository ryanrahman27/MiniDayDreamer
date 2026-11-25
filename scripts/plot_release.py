"""Plot scored Isaac control traces on matched scenes."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from diagnose_stack import release_events


def plot(run, seeds):
    arms = (("base", "Frozen base"), ("demos", "Repaired demos-only RSSM"), ("varied", "Repaired varied RSSM"),
            ("gated_base", "Base with delayed opening"))
    fig, axes = plt.subplots(len(arms), len(seeds), figsize=(5.3 * len(seeds) + 1.2, 8.2),
                             sharex=True, sharey=True, squeeze=False)
    max_xy, max_tick = 12., 0
    for (row, col), ax in np.ndenumerate(axes):
        arm, title = arms[row]
        seed = seeds[col]
        with np.load(run / "traces" / f"{arm}-{seed}.npz") as d:
            obs, actions, success = d["obs"], d["actions"], bool(d["success"])
            event = release_events(obs, actions, d["proposed_grip"], d["aperture_m"])
        ticks = np.arange(len(obs))
        xy = 100 * np.linalg.norm(obs[:, 3:5] - obs[:, 6:8], axis=1)
        max_xy = max(max_xy, float(xy.max()))
        max_tick = max(max_tick, len(actions))
        ax.fill_between(ticks, 0, np.r_[actions[:, 3] <= 0, actions[-1, 3] <= 0].astype(float),
                        step="post", transform=ax.get_xaxis_transform(), color="#c8dcf1", alpha=.8)
        ax.plot(ticks, xy, color="#18466b", linewidth=1.8)
        ax.axhline(12, color="#4e6b53", linestyle="--", linewidth=1)
        lift = event["lift_step"]
        if lift is not None:
            ax.scatter(lift, xy[lift], marker="^", color="#218463", s=40, zorder=3)
        opening = event["first_open_step"]
        if opening is not None:
            ax.scatter(opening, xy[opening], color="#b93e39", s=42, zorder=4)
            ax.annotate(f"{xy[opening]:.0f} mm", (opening, xy[opening]), xytext=(6, 5),
                        textcoords="offset points", color="#9d302d", fontsize=9)
        ax.set_title(f"Scene {seed} · {title}" + (" · stacked" if success else ""),
                     loc="left", fontsize=10, fontweight="bold")
        ax.spines[["right", "top"]].set_visible(False)
    axes[0, 0].set_ylim(0, max_xy * 1.16)
    axes[0, 0].set_xlim(0, max_tick + 1)
    fig.supxlabel("Control tick (20 ticks per second)", fontsize=10)
    fig.supylabel("Horizontal distance between cube centers (mm)", fontsize=10)
    fig.suptitle("Cube transport and gripper timing in Isaac", fontsize=15, fontweight="bold")
    handles = [Patch(facecolor="#c8dcf1", label="Open command"),
               Line2D([], [], marker="^", linestyle="", color="#218463", label="First lift"),
               Line2D([], [], marker="o", linestyle="", color="#b93e39", label="First opening after lift"),
               Line2D([], [], linestyle="--", color="#4e6b53", label="12 mm stacking tolerance")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .945), ncol=4,
               frameon=False, fontsize=9)
    fig.subplots_adjust(left=.09, right=.985, bottom=.085, top=.84, hspace=.37, wspace=.1)
    fig.savefig(run / "release-timeline.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    plot(args.run, args.seeds)
