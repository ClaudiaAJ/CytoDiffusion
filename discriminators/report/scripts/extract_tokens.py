"""Cache ALL ViT tokens (class token + 576 patch tokens, after the final norm) for train/val/test, fp16."""
import sys, os, numpy as np, torch, timm
from torchvision import datasets, transforms
R = '/store/CIA/ca662/CytoDiffusion_runs'; DATA = '/store/CIA/ca662/CytoDiffusion_data/CytoData'
name = sys.argv[1]; OUT = f'/store/CIA/ca662/CytoDiffusion_data/tokens_{name}'; os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda:0'); M = 'vit_base_patch16_384'
tf = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
if name == 'plain':
    model = timm.create_model(M, pretrained=True, num_classes=0)
else:
    model = timm.create_model(M, pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(f'{R}/stock_baseline/saved_models/StockViT_{M}_0.pth', map_location='cpu'))
model = model.to(dev).eval()
for split in ('train', 'val', 'test'):
    ds = datasets.ImageFolder(f'{DATA}/{split}', transform=tf)
    mm = np.lib.format.open_memmap(f'{OUT}/{split}_tok.npy', mode='w+', dtype=np.float16, shape=(len(ds), 577, 768))
    ys = []; i = 0
    with torch.no_grad():
        for x, y in torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=6):
            with torch.autocast('cuda'):
                t = model.forward_features(x.to(dev))
            mm[i:i + len(y)] = t.float().cpu().numpy().astype(np.float16); i += len(y); ys.append(y)
    mm.flush(); np.save(f'{OUT}/{split}_y.npy', torch.cat(ys).numpy()); print(split, mm.shape, flush=True)
np.save(f'{OUT}/classes.npy', np.array(ds.classes))
