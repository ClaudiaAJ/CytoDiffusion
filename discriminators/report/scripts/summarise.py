import re, glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0,'/mhome/damtp/t/ca662/CytoDiffusion/discriminators'); import class_tree
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from scipy.stats import spearmanr
F='/store/CIA/ca662/CytoDiffusion_runs/fullloop'; B='/store/CIA/ca662/CytoDiffusion_runs/balanced_design'
names=['artefact','basophil','blast','eosinophil','erythroblast','ig','lymphocyte','monocyte','neutrophil','platelet']
dep=np.array([class_tree.depth(n) for n in names]); T=class_tree.tree_distance_matrix(names); iu=np.triu_indices(10,1)
def atanh(x): return np.arctanh(np.clip(x,0,1-1e-9))
def geom(head,z,c=1.0):
    if head=='poincare':
        log=lambda p: atanh(np.linalg.norm(p,axis=-1,keepdims=True))*p/np.maximum(np.linalg.norm(p,axis=-1,keepdims=True),1e-12)
        def exp(v): n=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12); return np.tanh(n)*v/n
        def dist(x,y):
            sq=(x**2).sum(-1); sy=(y**2).sum(-1); e=((x-y)**2).sum(-1)
            return np.arccosh(np.maximum(1+2*e/((1-sq)*(1-sy)),1))
        rad=lambda p: 2*atanh(np.linalg.norm(p,axis=-1))
        return log,exp,dist,rad
    return (lambda p:p),(lambda v:v),(lambda x,y:np.linalg.norm(x-y,axis=-1)),(lambda p:np.linalg.norm(p,axis=-1))
def emb_stats(f,head):
    d=np.load(f,allow_pickle=True); z=d['z'].astype(np.float64); y=d['labels']
    log,exp,dist,rad=geom(head,z)
    C=np.stack([exp(log(z[y==k]).mean(0)) for k in range(10)])
    D=dist(C[:,None,:],C[None,:,:]); rho=spearmanr(D[iu],T[iu])[0]
    r=rad(z); rc=np.array([r[y==k].mean() for k in range(10)])
    return rho, spearmanr(rc,dep)[0]
runs={'linear (raw 768-d)':(f'{F}/FullLoop','FullLoop','linear'),'Poincare hard-clip':(f'{F}/FullLoop','FullLoop','poincare'),
      'Poincare runnorm':(f'{F}/FullLoopRunnorm','FullLoopRunnorm','poincare'),'Euclid bottleneck':(f'{F}/FullLoopRunnorm','FullLoopRunnorm','euclid')}
print("PART 1: fine-tuned, all 10 classes (mean +- std over seeds; n = seeds available)")
print(f"{'head':20s} n   acc          bal.acc      macroF1      rho(emb,tree) rad~depth")
for k,(d,exp,h) in runs.items():
    A=[];Bc=[];Fm=[];R=[];RD=[]
    for s in range(3):
        c=f'{d}/{exp}_{h}_vit_base_patch16_384_{s}.csv'
        if not os.path.exists(c): continue
        x=pd.read_csv(c); y,p=x.Label.values,x.Prediction.values
        A.append(100*accuracy_score(y,p)); Bc.append(100*balanced_accuracy_score(y,p)); Fm.append(100*f1_score(y,p,average='macro'))
        if h!='linear':
            r,rd=emb_stats(c.replace('.csv','_embeddings.npz'),h); R.append(r); RD.append(rd)
    ms=lambda v: f"{np.mean(v):5.1f}+-{np.std(v):3.1f}" if v else "   -   "
    ex=lambda v: f"{np.mean(v):5.2f}+-{np.std(v):4.2f}" if v else "     -     "
    print(f"{k:20s} {len(A)}  {ms(A)}   {ms(Bc)}   {ms(Fm)}   {ex(R)}   {ex(RD)}")
print("\nPART 2/3: fine-tuned hold-out, tree-aware (soft labels + depth loss), 3 seeds planned")
rows=[]
for log in sorted(glob.glob(f'{B}/logs/p[48]_*.log')):
    m=re.match(r'p(\d)_(\w+)_(\w+)_s(\d)\.log',os.path.basename(log)); dim,hold,head,seed=int(m[1]),m[2],m[3],int(m[4])
    t=open(log,errors='ignore').read()
    pm=re.search(r"spearman\(emb dist, tree dist\)=([\d.-]+) \| held '\w+' nearest known: (\w+)\(T=(\d+)\).*tree dist (\d+) \(best possible (\d+)\) \| held radius/known mean ([\d.]+)",t)
    am=re.search(r"Test Accuracy: ([\d.]+)  Balanced: ([\d.]+)",t)
    if pm and am: rows.append(dict(dim=dim,hold=hold,head=head,seed=seed,rho=float(pm[1]),near=pm[2],gap=int(pm[4])-int(pm[5]),rad=float(pm[6]),acc=float(am[1]),bal=float(am[2])))
df=pd.DataFrame(rows); df.to_csv(B+'/holdout_results.csv',index=False)
print(f"{'dim':>3s} {'held out':13s}{'head':9s} n  nearest known class (seeds)        correct  mean excess  rho   rad ratio  bal.acc")
for (dim,hold,head),g in df.groupby(['dim','hold','head']):
    near=",".join(g.sort_values('seed').near); print(f"{dim:3d} {hold:13s}{head:9s}{len(g):2d}  {near:36s} {int((g.gap==0).sum())}/{len(g)}     {g.gap.mean():5.2f}      {g.rho.mean():4.2f}  {g.rad.mean():5.2f}     {g.bal.mean():5.1f}")
