"""Draw the two overview figures from the published score report."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

root = Path(__file__).resolve().parents[3]
metrics = root / "reports/comparison/metrics"
scores=json.loads((metrics/"scores.json").read_text(encoding="utf-8"))
unseen=scores["unseen_pages"]
keys=("resnet18","swin_tiny","churro","resnet50")
names={"resnet18":"ResNet18","swin_tiny":"Swin-Tiny","churro":"CHURRO","resnet50":"ResNet50"}
navy="#172f3b"
teal="#09796d"
blue="#4c649f"
grey="#68818b"
light="#dbe7e7"
background="#f1f6f5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"text.color":navy,
                     "axes.labelcolor":navy,"svg.fonttype":"none","savefig.facecolor":background})

def canvas(title,subtitle):
    fig, (left,right)=plt.subplots(1,2,figsize=(12.4,5.65),sharey=True,facecolor=background,
                                   gridspec_kw={"wspace":.30})
    fig.subplots_adjust(left=.14,right=.95,bottom=.20,top=.68)
    fig.text(.06,.90,title,ha="left",fontsize=25,color=navy,weight="bold")
    fig.text(.06,.82,subtitle,ha="left",fontsize=11,color=grey)
    for ax in (left,right):
        ax.set_facecolor(background)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="both",length=0,colors=navy,pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="x",color=light)
    return fig,left,right

def export(fig,stem):
    for ext in ("png","svg"):
        fig.savefig(metrics/f"{stem}.{ext}",dpi=300)
    plt.close(fig)

fig,left,right=canvas("Which model best identified the writer?",
    "Nuremberg-9 Letterbooks  ·  One prediction per handwritten line  ·  Macro-F1")
y=np.arange(len(keys))[::-1]
for ax,section,title,start in ((left,scores,"All test lines",84),(right,unseen,"Unseen pages",84)):
    vals=[100*section["models"][m]["macro_f1"] for m in keys]
    ax.set_title(title,loc="left",pad=18,color=navy,weight="bold",fontsize=15)
    ax.set_xlim(start,97.2)
    ax.set_ylim(-.53,3.53)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in keys],weight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}%"))
    for yi,k,value in zip(y,keys,vals,strict=True):
        color=teal if k=="resnet18" else blue if k=="churro" else grey
        ax.hlines(yi,start,value,color=color,alpha=.34,lw=2.5)
        ax.scatter(value,yi,s=87,color=color,zorder=4,edgecolor=background,lw=1)
        ax.annotate(f"{value:.2f}%",(value,yi),xytext=(9,0),textcoords="offset points",
                    va="center",color=color,weight="bold",fontsize=10)
fig.text(.14,.09,"All test lines · 7,636 lines, nine writers",color=grey,fontsize=10)
fig.text(.605,.09,"Unseen pages · 7,583 lines, seven writers",color=grey,fontsize=10)
export(fig,"macro_f1_overview")

fig,left,right=canvas("How much did each model gain over CHURRO?",
    "Macro-F1 difference in percentage points. Values to the right of zero exceed the CHURRO score.")
others=("resnet18","swin_tiny","resnet50")
y=np.arange(len(others))[::-1]
for ax,section,title in ((left,scores,"All test lines"),(right,unseen,"Unseen pages")):
    ax.set_title(title,loc="left",pad=18,color=navy,weight="bold",fontsize=15)
    ax.set_xlim(-3.35,4.9)
    ax.set_ylim(-.6,2.6)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in others],weight="bold")
    ax.set_xticks([-3,-2,-1,0,1,2,3,4])
    ax.axvline(0,color=navy,lw=1)
    for yi,key in zip(y,others,strict=True):
        gap=section["differences_from_churro_pp"][key]["macro_f1"]
        color=teal if gap>=0 else "#a34340"
        ax.barh(yi,gap,height=.43,color=color)
        ax.annotate(f"{gap:+.2f}",(gap,yi),xytext=(7 if gap>=0 else -7,0),textcoords="offset points",
                    va="center",ha="left" if gap>=0 else "right",color=color,weight="bold")
fig.text(.14,.09,"One run per model. Reported deltas are descriptive.",color=grey,fontsize=10)
export(fig,"difference_from_churro")
