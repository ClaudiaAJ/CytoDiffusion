"""Run existing fine-tuned checkpoints over CytoData and save pooled features + logits (val/test, and train for some)."""
import sys, os, numpy as np, torch, timm, importlib.util
from torchvision import datasets, transforms
D = '/mhome/damtp/t/ca662/CytoDiffusion/discriminators'; sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("hyp", f"{D}/train_and_test_hyperbolic.py"); hyp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hyp)
R = '/store/CIA/ca662/CytoDiffusion_runs'; DATA = '/store/CIA/ca662/CytoDiffusion_data/CytoData'; OUT = f'{R}/hyp_accuracy'
dev = torch.device('cuda:0')
tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds = {s: datasets.ImageFolder(f'{DATA}/{s}', transform=tf) for s in ('train', 'val', 'test')}
M = 'vit_base_patch16_384'
specs = [('stock', f'{R}/stock_baseline/saved_models/StockViT_{M}_0.pth', 'stock', {}, True)]
for s in (0, 1, 2):
    specs.append((f'linear_s{s}', f'{R}/fullloop/saved_models/FullLoop_linear_{M}_{s}.pth', 'linear', {}, s == 0))
    specs.append((f'poincare_clip_s{s}', f'{R}/fullloop/saved_models/FullLoop_poincare_{M}_{s}.pth', 'poincare', dict(clip_r=2.0), s == 0))
specs.append(('poincare_runnorm_s0', f'{R}/fullloop/saved_models/FullLoopRunnorm_poincare_{M}_0.pth', 'poincare', dict(norm_mode='runnorm', target_norm=1.0, clip_r=4.0), False))

@torch.no_grad()
def run(model, kind, split):
    loader = torch.utils.data.DataLoader(ds[split], batch_size=32, shuffle=False, num_workers=6)
    F_, L_, Y_ = [], [], []
    for x, y in loader:
        x = x.to(dev)
        with torch.autocast('cuda'):
            if kind == 'stock':
                f = model.forward_head(model.forward_features(x), pre_logits=True); lg = model.head(f)
            else:
                f = model.backbone(x); lg = model.head(f)
        F_.append(f.float().cpu()); L_.append(lg.float().cpu()); Y_.append(y)
    return torch.cat(F_).numpy(), torch.cat(L_).numpy(), torch.cat(Y_).numpy()

for name, ck, kind, kw, with_train in specs:
    o = f'{OUT}/ft_{name}.npz'
    if os.path.exists(o): continue
    if kind == 'stock':
        model = timm.create_model(M, pretrained=False, num_classes=10)
    else:
        model = hyp.Net(M, 10, head=kind, **kw)
    print(name, model.load_state_dict(torch.load(ck, map_location='cpu'), strict=False)); model = model.to(dev).eval()
    out = {'classes': np.array(ds['test'].classes)}
    for split in (('train', 'val', 'test') if with_train else ('val', 'test')):
        f, lg, y = run(model, kind, split)
        out[f'{split}_x'], out[f'{split}_logits'], out[f'{split}_y'] = f, lg, y
    np.savez(o, **out)
    acc = (out['test_logits'].argmax(1) == out['test_y']).mean() * 100
    print(f'{name}: test acc {acc:.1f}  val acc {(out["val_logits"].argmax(1) == out["val_y"]).mean() * 100:.1f}', flush=True)
    del model; torch.cuda.empty_cache()
