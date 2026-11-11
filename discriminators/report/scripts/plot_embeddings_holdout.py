"""Embedding scatter for held-out classes: where do unseen images land relative to the known classes?
Frozen ViT features, 4-d heads, tree+depth config, seed 0. PCA axes are fitted on KNOWN-class test embeddings only."""
import sys, argparse, numpy as np, torch
sys.path.insert(0,'/mhome/damtp/t/ca662/CytoDiffusion/discriminators')
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import frozen_head_harness as H, class_tree
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"]})
OUT="/store/CIA/ca662/CytoDiffusion_runs/figures_holdout_placement/"
dev=torch.device('cuda:0'); DIM=4; SEED=int(sys.argv[1]) if len(sys.argv)>1 else 0
d=np.load('/store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz')
data=[torch.tensor(d[k]).to(dev) for k in ("train_x","train_y","val_x","val_y","test_x","test_y")]
classes=[str(c) for c in d['classes']]
CASES=[("erythroblast","platelet"),("platelet","erythroblast")]  # (held-out, lineage neighbour)
HEADS=[("euclid","Euclidean"),("poincare","Poincaré")]

def fit(hd, hold):
    data2,held,_=H.remap_holdout(data,classes.index(hold))
    known=[c for c in classes if c!=hold]
    a=argparse.Namespace(model=None,dim=DIM,c=1.0,clip_r=4,norm_mode='runnorm',target_norm=1.0,tau=0.1,lr=1e-3,wd=1e-2,steps=300,batch_size=32,eval_every=50,mixup=0,norm_pen=0,tree_w=0,tree_scale=0.5,soft_t=1,soft_mix=0.8,depth_w=5,
        depth_t=torch.tensor([float(class_tree.depth(n)) for n in known],device=dev),known_names=known,held_name=hold,
        tree_T=torch.tensor(class_tree.tree_distance_matrix(known),dtype=torch.float32,device=dev))
    res=H.run_head(hd,a,data2,dev,SEED,held,True); head=res[-1]; head.eval()
    with torch.no_grad():
        vk=head.log0(head.embed(data2[4])).cpu().numpy(); vh=head.log0(head.embed(held)).cpu().numpy()
    return vk,data2[5].cpu().numpy(),vh,known,a.last_placement

fig,axes=plt.subplots(2,2,figsize=(11.5,10.2),facecolor=SURFACE)
info=[]
for r,(hold,neigh) in enumerate(CASES):
    for c,(hd,hname) in enumerate(HEADS):
        vk,yk,vh,known,pl=fit(hd,hold)
        mu=vk.mean(0); U,S,Vt=np.linalg.svd(vk-mu,full_matrices=False); W=Vt[:2].T
        evr=(S[:2]**2).sum()/(S**2).sum()
        pk,ph=(vk-mu)@W,(vh-mu)@W
        ax=axes[r,c]; ax.set_facecolor(SURFACE)
        ni=known.index(neigh); m=yk==ni
        ax.scatter(pk[~m,0],pk[~m,1],s=6,c=MUTED,alpha=0.30,linewidths=0,zorder=1)
        ax.scatter(pk[m,0],pk[m,1],s=13,c=ORANGE,alpha=0.85,linewidths=0,zorder=2)
        ax.scatter(ph[:,0],ph[:,1],s=15,c=BLUE,alpha=0.85,linewidths=0,zorder=3)
        for k,n in enumerate(known):  # class centroid labels (known classes)
            cx,cy=pk[yk==k].mean(0)
            ax.annotate(n,(cx,cy),fontsize=7.8,color=ORANGE if n==neigh else INK2,ha="center",va="center",fontweight="semibold" if n==neigh else "normal",
                        bbox=dict(boxstyle="round,pad=0.15",fc=SURFACE,ec="none",alpha=0.75),zorder=5)
        hx,hy=ph.mean(0); ax.scatter([hx],[hy],s=70,marker="X",c=BLUE,edgecolors=SURFACE,linewidths=1,zorder=6)
        near=pl["rank"][0][0]
        ax.set_title(f"{hname} · held out: {hold}",loc="left",fontsize=11,fontweight="semibold",color=INK,pad=18)
        ax.annotate(f"nearest known class: {near} (tree dist {pl['rank'][0][2]}) · PCA keeps {100*evr:.0f}% of variance",xy=(0,1.015),xycoords="axes fraction",fontsize=8.5,color=INK2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="datalim")
        for s in ax.spines.values(): s.set_color(GRID)
        info.append((hold,hname,near,pl['rank'][0][2],evr))
fig.subplots_adjust(left=0.03,right=0.985,top=0.835,bottom=0.05,wspace=0.06,hspace=0.20)
fig.text(0.03,0.955,"Where do images of an unseen class land? 4-d heads, Euclidean vs hyperbolic",fontsize=15,fontweight="semibold",color=INK)
fig.text(0.03,0.918,f"Blue = held-out class (never trained on, X = its mean); orange = its lineage neighbour; gray = other known classes. Seed {SEED}, one run per panel.",fontsize=9.5,color=INK2)
fig.text(0.03,0.893,"Tangent-space (log-map) embeddings, tree-aware soft labels + depth loss, frozen ViT features. PCA axes fitted on known classes only; centroid labels = known classes.",fontsize=9.5,color=INK2)
fig.savefig(OUT+f"holdout_embeddings_dim4_seed{SEED}.png",dpi=160,facecolor=SURFACE)
for i in info: print(i)
