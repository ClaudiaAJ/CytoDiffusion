"""All 10 classes trained at dim 4. Are erythroblast and platelet merged or separated, relative to other class pairs?
Per seed: class-centroid distance (tangent-space mean) for every pair, scaled by the pooled within-class spread
(separation ratio, comparable across geometries), the rank of the erythroblast-platelet pair among 45, and the confusion rate."""
import sys, argparse, json, torch, numpy as np
sys.path.insert(0,'/mhome/damtp/t/ca662/CytoDiffusion/discriminators')
import frozen_head_harness as H, class_tree
dev=torch.device('cuda:0')
d=np.load('/store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz')
data=[torch.tensor(d[k]).to(dev) for k in ("train_x","train_y","val_x","val_y","test_x","test_y")]
names=[str(c) for c in d['classes']]; E,Pl=names.index('erythroblast'),names.index('platelet')
xte,yte=data[4],data[5]
T=torch.tensor(class_tree.tree_distance_matrix(names),dtype=torch.float32,device=dev)
depth=torch.tensor([float(class_tree.depth(n)) for n in names],device=dev)
iu=np.triu_indices(10,1)
out={}
for cfg,(st,dw) in {'base':(0,0),'treedepth':(1,5)}.items():
  for hd in ['poincare','euclid']:
    rows=[]
    for seed in range(10):
        a=argparse.Namespace(model=None,dim=4,c=1.0,clip_r=4,norm_mode='runnorm',target_norm=1.0,tau=0.1,lr=1e-3,wd=1e-2,steps=300,batch_size=32,eval_every=50,mixup=0,norm_pen=0,tree_w=0,tree_scale=0.5,soft_t=st,soft_mix=0.8,depth_w=dw,depth_t=depth,known_names=names,held_name=None,tree_T=T)
        res=H.run_head(hd,a,data,dev,seed,None,True); head=res[-1]; head.eval()
        with torch.no_grad():
            z=head.embed(xte); v=head.log0(z); cents=[]; spread=[]
            for k in range(10):
                c=head.exp0(v[yte==k].mean(0)); cents.append(c)
                spread.append(head.dist(z[yte==k],c[None]).pow(2).mean().item())
            C=torch.stack(cents); D=head.dist(C,C).cpu().numpy()
            sep=D/np.sqrt(np.mean(spread))
            pred=head(xte).argmax(1)
            conf_ep=((pred[yte==E]==Pl).float().mean().item()+(pred[yte==Pl]==E).float().mean().item())/2
            rec=[(pred[yte==k]==k).float().mean().item() for k in range(10)]
        s=sep[iu]; ep=sep[E,Pl]
        rows.append(dict(seed=seed,ep_sep=float(ep),ep_rank=int((s<ep).sum())+1,mean_sep=float(s.mean()),ep_over_mean=float(ep/s.mean()),conf_ep=conf_ep,rec_e=rec[E],rec_p=rec[Pl],bal=float(np.mean(rec))))
        print(cfg,hd,rows[-1],flush=True)
    out[f"{cfg}/{hd}"]=rows
json.dump(out,open('/store/CIA/ca662/CytoDiffusion_runs/holdout_lowdim/merge_check.json','w'))
print("ALL DONE")
