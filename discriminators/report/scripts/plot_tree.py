import sys; sys.path.insert(0,'/mhome/damtp/t/ca662/CytoDiffusion/discriminators')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from class_tree import PARENT, depth
order=["artefact","blast","lymphocyte","ig","neutrophil","eosinophil","basophil","monocyte","erythroblast","platelet"]
x={c:i for i,c in enumerate(order)}
kids={}
for c,p in PARENT.items(): kids.setdefault(p,[]).append(c)
def px(n):
    if n in x: return x[n]
    v=sum(px(k) for k in kids[n])/len(kids[n]); x[n]=v; return v
px("root")
y={n:-depth(n) for n in x}
BG,INK,GR,BL,OR="#fcfcfb","#111","#8a8a85","#2a78d6","#eb6834"
fig,ax=plt.subplots(figsize=(15,6.4),facecolor=BG); ax.set_facecolor(BG)
for c,p in PARENT.items(): ax.plot([x[p],x[c]],[y[p],y[c]],color="#b8b8b2",lw=1.3,zorder=1)
for n in x:
    leaf=n in order; hot=n in("erythroblast","platelet")
    col=OR if hot else (BL if leaf else GR)
    ax.scatter(x[n],y[n],s=170 if leaf else 90,color=col,zorder=3,edgecolor=BG,lw=1.5)
    lab=n.replace("_"," ")+(f"\n(depth {depth(n)})" if leaf else "")
    if leaf: ax.text(x[n],y[n]-0.22,lab,ha="center",va="top",fontsize=9.5,color=INK,linespacing=1.2)
    else: ax.text(x[n]+0.12,y[n]+0.12,lab,ha="left",va="bottom",fontsize=9.5,color="#555")
ax.set_ylim(-5.9,0.5); ax.set_xlim(-0.8,9.8)
ax.set_yticks(range(0,-6,-1)); ax.set_yticklabels([str(i) for i in range(6)],color="#555")
ax.set_ylabel("depth (edges from root)",color="#555")
for s in("top","right","bottom"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#ccc"); ax.set_xticks([])
fig.text(0.04,0.95,"Class hierarchy used for soft labels, tree distance and the depth loss",fontsize=15,fontweight="bold",color=INK,va="top")
fig.text(0.04,0.895,"Blue = leaf classes (the 10 labels); orange = erythroblast and platelet (hold-out focus); gray = internal nodes. Tree distance = edges between two leaves.",fontsize=10,color="#555",va="top")
plt.subplots_adjust(top=0.84,bottom=0.04,left=0.07,right=0.98)
fig.savefig("/store/CIA/ca662/CytoDiffusion_runs/report/figures/00_class_tree.png",dpi=150,facecolor=BG)
