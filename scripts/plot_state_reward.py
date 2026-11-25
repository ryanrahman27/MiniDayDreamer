"""Plot the paired Isaac outcomes, one column per held-out scene."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot(folder):
    with (folder / "paired-outcomes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    seeds = sorted({int(r["seed"]) for r in rows})
    arms = [
        ("weak_base", "Frozen base"),
        ("weak_demos_learned", "Demos RSSM · learned"),
        ("weak_demos_direct", "Demos RSSM · direct"),
        ("weak_varied_learned", "Varied RSSM · learned"),
        ("weak_varied_direct", "Varied RSSM · direct"),
        ("strong_base", "Frozen base"),
        ("strong_varied_learned", "Varied RSSM · learned"),
        ("strong_varied_direct", "Varied RSSM · direct"),
    ]
    scores = {(r["arm"], int(r["seed"])): int(r["success"]) for r in rows}
    fig, axes = plt.subplots(2, 1, figsize=(10.3, 4.25), height_ratios=[5, 3])
    fig.subplots_adjust(left=.195, right=.895, bottom=.14, top=.86, hspace=.57)
    for ax, group, label in zip(axes, [arms[:5], arms[5:]], ["24-demo imitation policy", "72-demo imitation policy"]):
        matrix = np.array([[scores[name,seed] for seed in seeds] for name, _ in group])
        ax.imshow(matrix, vmin=0, vmax=1, cmap=ListedColormap(["#e7ecee","#00766d"]), aspect="auto",
                  interpolation="nearest")
        ax.set_yticks(range(len(group)), [name for _, name in group],fontsize=9)
        ax.set_xticks([0, 9, 19, 29, 39, 49], [seeds[i] for i in (0,9,19,29,39,49)],fontsize=9)
        ax.set_xticks(np.arange(.5,len(seeds)-.5),minor=True)
        ax.set_yticks(np.arange(.5,len(group)-.5),minor=True)
        ax.grid(which="minor",color="white",linewidth=.6)
        ax.tick_params(which="both", length=0, pad=4)
        ax.set_title(label,loc="left",fontsize=11,fontweight="bold",pad=5)
        for y, count in enumerate(matrix.sum(axis=1)):
            ax.text(len(seeds)-.05,y,f"{count} / {len(seeds)}",va="center",fontsize=9)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("Held-out Isaac scene seed",fontsize=9)
    fig.text(.195,.97,"Stable stacks on matched scenes",va="top",fontsize=16,fontweight="bold")
    fig.text(.895,.965,"Teal = stacked",va="top",ha="right",color="#00645e",fontsize=10)
    fig.savefig(folder / "paired-stacks.png",dpi=170,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    plot(parser.parse_args().run)
