"""Plot measured approach trajectories from the original 16 matched never-opened losses."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(run):
    with (run/"aligned-profiles.csv").open(newline="") as f:
        data=[r for r in csv.DictReader(f) if r["source"]=="original"]
    samples={(int(r["seed"]),r["arm"],int(r["relative_tick"])):r for r in data}
    seeds=sorted({int(r["seed"]) for r in data})
    ticks=list(range(-20,13))
    timing=json.loads((run/"climb-summary.json").read_text())
    approach_seconds=timing["original"]["never_open"]["radii"]["80"]["direct"]["relative_tick"]*.05
    fig,axes=plt.subplots(2,1,figsize=(8,5.5),sharex=True)
    fig.subplots_adjust(left=.14,right=.96,top=.81,bottom=.13,hspace=.25)
    for ax,field,ylabel in zip(axes,("height_mm","dz_mm"),("Height above stack (mm)","Rise over next 0.25 s (mm)")):
        for arm,label,color in (("base","Successful base","#00776d"),("direct","Steered loss","#a33a1c")):
            curves=[]
            for t in ticks:
                matched=[s for s in seeds if (s,"base",t) in samples and (s,"direct",t) in samples]
                curves.append(np.percentile([float(samples[s,arm,t][field]) for s in matched], [25,50,75]))
            lo,mid,hi=np.asarray(curves).T
            time=np.asarray(ticks)*.05
            ax.fill_between(time,lo,hi,alpha=.13,color=color,linewidth=0)
            ax.plot(time,mid,color=color,lw=2.4,label=label)
        ax.axvline(0,color="#555555",ls="--",lw=1)
        ax.axvline(approach_seconds,color="#555555",ls=":",lw=1)
        ax.set_ylabel(ylabel)
        ax.spines[["top","right"]].set_visible(False)
    axes[1].axhline(0,color="#777777",lw=.8)
    axes[1].set_xlabel("Seconds from first entry inside 12 mm horizontally")
    axes[0].legend(loc="upper left",ncol=2,frameon=False)
    for time,label in ((approach_seconds,"80 mm (median)"),(0.,"12 mm")):
        axes[0].text(time+.015,.06,label,transform=axes[0].get_xaxis_transform(),size=9,color="#444444")
    fig.text(.14,.95,"The approach to placement",fontsize=16,fontweight="bold")
    fig.text(.14,.885,"Median and middle 50% of paired, still-grasped trajectories",fontsize=10)
    fig.savefig(run/"approach-climb.png",dpi=165,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    plot(p.parse_args().run)
