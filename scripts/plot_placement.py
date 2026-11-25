"""Plot measured cube heights and opening locations for the 24 paired stacking losses."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot(folder):
    with (folder / "archived-placement-events.csv").open(newline="") as f:
        pairs = {}
        for r in csv.DictReader(f):
            pairs.setdefault(int(r["seed"]), {})[r["arm"]] = r
    fig, (height, opening) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.81, wspace=.39)
    base_color, direct_color = "#00776d", "#a33a1c"
    height.axhspan(-8, 8, color="#e2f0ed", zorder=0)
    opening.add_patch(Rectangle((0, -8), 12, 16, facecolor="#e2f0ed", edgecolor="none", zorder=0))
    for seed, arms in sorted(pairs.items()):
        b, d = arms["mid_base"], arms["mid_varied_direct"]
        if not d["first_open"]:
            key = "min_centered_held_z_error_mm"
            values = [float(b[key]), float(d[key])]
            height.plot([0, 1], values, color="#bdc5c6", lw=1, zorder=1)
            height.scatter([0, 1], values, s=23, c=[base_color,direct_color], zorder=2)
        else:
            xs = [float(r["open_xy_mm"]) for r in (b,d)]
            ys = [float(r["open_z_error_mm"]) for r in (b,d)]
            opening.plot(xs, ys, color="#bdc5c6", lw=1, zorder=1)
            opening.scatter(xs, ys, s=29, c=[base_color,direct_color], zorder=2)
    height.set(xticks=[0,1], xticklabels=["Base","Direct"], xlim=(-.2,1.2),
               ylabel="Minimum height error while held and centered (mm)")
    opening.set(xlabel="Horizontal error at first open command (mm)", ylabel="Height error at first open command (mm)",
                xlim=(0, max(20,opening.get_xlim()[1])), ylim=(-9, max(34,opening.get_ylim()[1])))
    height.set_title("16 pairs · direct never opened", loc="left", fontsize=11)
    opening.set_title("8 pairs · both policies opened", loc="left", fontsize=11)
    for ax in (height, opening):
        ax.spines[["top","right"]].set_visible(False)
        ax.axhline(0, color="#7da29d", lw=.7, zorder=.5)
    fig.text(.09,.96,"Placement on the 24 lost stacks",va="top",fontsize=16,fontweight="bold")
    fig.text(.72,.95,"● Base",va="top",color=base_color,fontsize=11)
    fig.text(.83,.95,"● Direct",va="top",color=direct_color,fontsize=11)
    fig.savefig(folder/"lost-placements.png",dpi=170,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    plot(parser.parse_args().run)
