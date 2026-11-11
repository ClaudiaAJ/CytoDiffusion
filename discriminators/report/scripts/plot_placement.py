"""Held-out class placement heatmap: excess tree distance of the nearest known class, Poincare vs Euclidean."""
import re, collections, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})
OUT = "/store/CIA/ca662/CytoDiffusion_runs/figures_holdout_placement/"
NAMES = ['artefact','basophil','blast','eosinophil','erythroblast','ig','lymphocyte','monocyte','neutrophil','platelet']
ORDER = ['artefact','blast','lymphocyte','monocyte','neutrophil','eosinophil','basophil','ig','erythroblast','platelet']  # tree order
COLS = [("poincare","base"),("euclid","base"),("poincare","treedepth"),("euclid","treedepth")]
SRC = {64: "/store/CIA/ca662/CytoDiffusion_runs/holdout_tree/results.log",
       4:  "/store/CIA/ca662/CytoDiffusion_runs/holdout_lowdim/results_dim4.log"}

def load(fn):
    cur=None; gap=collections.defaultdict(list); near=collections.defaultdict(list)
    for line in open(fn):
        m=re.match(r"## holdout=(\d+) head=(\w+) cfg=(\w+)",line)
        if m: cur=(NAMES[int(m[1])],m[2],m[3]); continue
        m=re.search(r"nearest known: (\w+)\(T=\d+\).*tree dist (\d+) \(best possible (\d+)\)",line)
        if m and cur: gap[cur].append(int(m[2])-int(m[3])); near[cur].append(m[1])
    return gap,near

fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4), facecolor=SURFACE)
rows=[]
for ax,(dim,fn) in zip(axes,SRC.items()):
    gap,near=load(fn)
    M=np.array([[np.mean(gap[(c,h,g)]) for h,g in COLS] for c in ORDER])
    ax.set_facecolor(SURFACE)
    im=ax.pcolormesh(M, cmap=CMAP, vmin=0, vmax=3, edgecolors=SURFACE, linewidth=2.5)
    ax.set_ylim(len(ORDER),0); ax.set_aspect("auto")
    for i,c in enumerate(ORDER):
        for j,(h,g) in enumerate(COLS):
            v=M[i,j]; modal,cnt=collections.Counter(near[(c,h,g)]).most_common(1)[0]
            fg="white" if v>=1.4 else INK
            ax.text(j+.5,i+.40,f"{v:.1f}",ha="center",va="center",fontsize=10.5,fontweight="semibold",color=fg)
            ax.text(j+.5,i+.72,f"{modal[:12]} {cnt}/3",ha="center",va="center",fontsize=7.4,color=fg)
            rows.append(dict(dim=dim,held_out=c,head=h,config=g,mean_excess=round(v,3),modal_nearest=modal,modal_count=cnt,seeds=len(near[(c,h,g)])))
    ax.set_xticks(np.arange(4)+.5); ax.set_xticklabels(["Poincaré","Euclid.","Poincaré","Euclid."],fontsize=9.5,color=INK2)
    ax.set_yticks(np.arange(len(ORDER))+.5); ax.set_yticklabels(ORDER,fontsize=10,color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.xaxis.tick_top()
    ax.annotate("no tree",xy=(1.0,1.08),xycoords=("data","axes fraction"),ha="center",fontsize=9.5,fontweight="semibold",color=INK)
    ax.annotate("tree + depth loss",xy=(3.0,1.08),xycoords=("data","axes fraction"),ha="center",fontsize=9.5,fontweight="semibold",color=INK)
    ax.annotate(f"{dim}-d embedding head",xy=(0,1.20),xycoords="axes fraction",ha="left",fontsize=12,fontweight="semibold",color=INK)
fig.subplots_adjust(left=0.085,right=0.985,top=0.745,bottom=0.17,wspace=0.30)
cax=fig.add_axes([0.30,0.075,0.40,0.022])
cb=fig.colorbar(im,cax=cax,orientation="horizontal",ticks=[0,1,2,3]); cb.outline.set_visible(False)
cb.set_label("excess tree distance of the nearest known class  (0 = a best-possible lineage neighbour; darker = further away in the tree)",fontsize=9,color=INK2)
cb.ax.tick_params(labelsize=9,colors=INK2,length=0)
fig.text(0.03,0.955,"Where does a held-out class land? Hyperbolic vs Euclidean head",fontsize=15,fontweight="semibold",color=INK)
fig.text(0.03,0.915,"Each row holds one class out of training; cell = how far in the lineage tree its nearest known class is, beyond the best possible (mean of 3 seeds).",fontsize=9.5,color=INK2)
fig.text(0.03,0.885,"Small text = most frequent nearest class (seeds out of 3). Rows ordered by lineage. Frozen ViT features, runnorm head, soft labels + depth_w=5 in the tree columns.",fontsize=9.5,color=INK2)
fig.savefig(OUT+"holdout_placement_heatmap.png",dpi=160,facecolor=SURFACE)
with open(OUT+"holdout_placement_table.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("saved")
