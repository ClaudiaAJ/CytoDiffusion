"""Fast iteration harness for the classifier head.

Step 1 (once, cached): run the frozen pretrained backbone over train/val/test and save the pooled features.
Step 2 (seconds):      train a head (linear or Poincare) on the cached features and report metrics.

The heads are imported from train_and_test_hyperbolic.py, so what works here is the code the full loop uses.
Frozen features do NOT test the fine-tuned setup: confirm any head that works with a full run.

    python frozen_head_harness.py --heads linear poincare --steps 300
"""
import argparse
import importlib.util
import os
import time

import numpy as np
import class_tree
import timm
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hyp", os.path.join(HERE, "train_and_test_hyperbolic.py"))
hyp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hyp)

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def extract(model, ds, device, bs=16):
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def build_cache(a, device):
    model = timm.create_model(a.model, pretrained=True, num_classes=0).to(device).eval()
    eval_tf = transforms.Compose([transforms.Resize((a.img_size, a.img_size)), transforms.ToTensor(), NORM])
    aug_tf = transforms.Compose([
        transforms.Resize((a.img_size, a.img_size)), RandAugment(), transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.125),
        transforms.ToTensor(), NORM])
    out = {}
    for split in ("train", "val", "test"):
        ds = datasets.ImageFolder(os.path.join(a.data_root, split), transform=eval_tf)
        f, y = extract(model, ds, device)
        out[f"{split}_x"], out[f"{split}_y"] = f.numpy(), y.numpy()
        out["classes"] = np.array(ds.classes)
        if split == "train" and a.aug_copies > 0:  # extra augmented views of the training set
            ds_aug = datasets.ImageFolder(os.path.join(a.data_root, split), transform=aug_tf)
            fs, ys = [f], [y]
            for _ in range(a.aug_copies):
                fa, ya = extract(model, ds_aug, device)
                fs.append(fa); ys.append(ya)
            out["train_x"], out["train_y"] = torch.cat(fs).numpy(), torch.cat(ys).numpy()
        print(f"{split}: {out[f'{split}_x'].shape}")
    np.savez(a.cache, **out)
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def bacc(head, x, y):
    """Balanced accuracy (mean per-class recall) in %."""
    head.eval()
    pred = head(x).argmax(1)
    rec = [(pred[y == k] == k).float().mean().item() for k in y.unique()]
    return 100 * float(np.mean(rec))


@torch.no_grad()
def acc(head, x, y):
    head.eval()
    return 100 * (head(x).argmax(1) == y).float().mean().item()


def remap_holdout(data, hold):
    """Drop class `hold` from train/val, relabel the rest 0..K-2; return the held-out test features separately."""
    xtr, ytr, xva, yva, xte, yte = data
    def keep(x, y):
        m = y != hold
        yy = y[m]; yy = yy - (yy > hold).long()
        return x[m], yy
    xtr2, ytr2 = keep(xtr, ytr); xva2, yva2 = keep(xva, yva); xte2, yte2 = keep(xte, yte)
    # the held-out class is never trained on, so every one of its images is a valid out-of-distribution positive
    held = torch.cat([xtr[ytr == hold], xva[yva == hold], xte[yte == hold]])
    return [xtr2, ytr2, xva2, yva2, xte2, yte2], held, xte2


def auroc_scores(head, x_in, x_out):
    """AUROC for held-out (positive) vs known-class test samples. Score = -max logit (and -MSP)."""
    from sklearn.metrics import roc_auc_score
    head.eval()
    with torch.no_grad():
        if hasattr(head, "embed"):  # Poincare head: also score by hyperbolic distance from the origin
            zi, zo = head.embed(x_in), head.embed(x_out)
            ni, no = head.radius(zi), head.radius(zo)
            origin = -torch.cat([ni, no]).cpu().numpy()  # small norm = uncertain = more anomalous
        else:
            origin = None
        li, lo = head(x_in), head(x_out)
    y = np.r_[np.zeros(len(li)), np.ones(len(lo))]
    maxlogit = -torch.cat([li.max(1).values, lo.max(1).values]).cpu().numpy()
    msp = -torch.cat([li.softmax(1).max(1).values, lo.softmax(1).max(1).values]).cpu().numpy()
    return (roc_auc_score(y, maxlogit), roc_auc_score(y, msp),
            roc_auc_score(y, origin) if origin is not None else float("nan"))


@torch.no_grad()
def placement(head, a, xte, yte, holdout_x):
    """Do class centroids follow the tree? Spearman(embedding dist, tree dist) over class pairs, plus where a held-out class lands."""
    from scipy.stats import spearmanr
    head.eval()
    names = list(a.known_names)
    cents = []
    for k in range(len(names)):
        z = head.embed(xte[yte == k])
        cents.append(head.exp0(head.log0(z).mean(0)))  # tangent-space mean (Frechet-ish)
    held = a.held_name
    if holdout_x is not None:
        names = names + [held]
        cents.append(head.exp0(head.log0(head.embed(holdout_x)).mean(0)))
    C = torch.stack(cents)
    Dc = head.dist(C, C).cpu().numpy()
    T = class_tree.tree_distance_matrix(names)
    iu = np.triu_indices(len(names), 1)
    rho = spearmanr(Dc[iu], T[iu])[0]
    out = {"rho": rho, "radius": {n: float(head.radius(c)) for n, c in zip(names, C)}}
    if holdout_x is not None:
        j = len(names) - 1
        order = np.argsort(Dc[j, :j])
        out["rank"] = [(names[i], float(Dc[j, i]), int(T[j, i])) for i in order]
        out["nearest_tree"] = int(T[j, order[0]])
        out["min_possible"] = int(T[j, :j].min())
    return out


def run_head(name, a, data, device, seed, holdout_x=None, return_head=False):
    torch.manual_seed(seed)
    xtr, ytr, xva, yva, xte, yte = data
    n_classes = int(ytr.max().item()) + 1
    if name == "linear":
        head = hyp.LinearHead(xtr.shape[1], n_classes)
    else:
        cls = hyp.EuclideanHead if name == "euclid" else hyp.PoincareHead
        head = cls(xtr.shape[1], n_classes, dim=a.dim, c=a.c, clip_r=a.clip_r, tau=a.tau,
                                  norm_mode=a.norm_mode, target_norm=a.target_norm)
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    n, best_val, best_test, best_bal, rows = xtr.shape[0], -1.0, 0.0, 0.0, []
    for step in range(1, a.steps + 1):
        head.train()
        idx = torch.randperm(n, device=device)[: a.batch_size]
        xb, yb = xtr[idx], ytr[idx]
        if a.mixup > 0:
            lam = float(np.random.beta(a.mixup, a.mixup)); perm = torch.randperm(xb.shape[0], device=device)
            out = head(lam * xb + (1 - lam) * xb[perm])
            loss = lam * F.cross_entropy(out, yb) + (1 - lam) * F.cross_entropy(out, yb[perm])
        else:
            out, z = head(xb, return_embedding=True)
            if a.soft_t > 0:  # tree-aware soft labels: q_k ~ exp(-T[y,k]/soft_t), mixed with one-hot by soft_mix
                q = F.softmax(-a.tree_T[yb] / a.soft_t, dim=1)
                q = (1 - a.soft_mix) * F.one_hot(yb, q.shape[1]).float() + a.soft_mix * q
                loss = -(q * F.log_softmax(out, dim=1)).sum(1).mean()
            else:
                loss = F.cross_entropy(out, yb)
            if a.depth_w > 0 and name in ("poincare", "euclid"):
                # radius should grow with tree depth: relative radius r/mean(r) -> depth/mean(depth)
                r = head.radius(z); dep = a.depth_t[yb]
                loss = loss + a.depth_w * (r / r.mean().detach() - dep / dep.mean()).pow(2).mean()
        if a.tree_w > 0 and name in ("poincare", "euclid"):  # match prototype-to-prototype hyperbolic distance to scaled tree distance
            P = head.prototypes()
            Dp = head.dist(P, P)
            iu = torch.triu_indices(P.shape[0], P.shape[0], 1, device=P.device)
            loss = loss + a.tree_w * (Dp[iu[0], iu[1]] - a.tree_scale * a.tree_T[iu[0], iu[1]]).pow(2).mean()
        if not torch.isfinite(loss):
            print(f"  [{name}] non-finite loss at step {step}")
            break
        opt.zero_grad(); loss.backward(); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            va, te, tr = acc(head, xva, yva), acc(head, xte, yte), acc(head, xtr, ytr)
            with torch.no_grad():
                full = F.cross_entropy(head(xtr), ytr).item()
                pn = head.prototypes().norm(dim=-1).mean().item()
            bal = bacc(head, xte, yte)
            if va >= best_val:
                best_val, best_test, best_bal = va, te, bal
            rows.append((step, full, tr, va, te, head.last_clipped_frac, head.last_preclip_norm, pn, bal))
    auc = auroc_scores(head, xte, holdout_x) if holdout_x is not None else (float("nan"),) * 3
    rad = (float("nan"), float("nan"))
    if hasattr(head, "embed"):  # spread of hyperbolic radius across the (known-class) test samples
        head.eval()
        with torch.no_grad():
            r = head.radius(head.embed(xte))
        rad = (r.mean().item(), (r.std() / r.mean()).item())
    if name in ("poincare", "euclid"):
        a.last_placement = placement(head, a, xte, yte, holdout_x)
    if return_head:
        return rows, best_val, best_test, auc, best_bal, rad, head
    return rows, best_val, best_test, auc, best_bal, rad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_base_patch16_384")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--data_root", default="../data/prepare_data/example_data_train_val_test")
    p.add_argument("--cache", default="features_cache.npz")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--aug_copies", type=int, default=0, help="extra augmented views of the train set in the cache")
    p.add_argument("--heads", nargs="+", default=["linear", "poincare"], choices=["linear", "poincare", "euclid"])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--clip_r", type=float, default=2.0)
    p.add_argument("--norm_mode", choices=["clip", "runnorm", "batchnorm"], default="clip",
                   help="how the tangent vector is scaled before expmap0 (see PoincareHead)")
    p.add_argument("--target_norm", type=float, default=1.0, help="mean |v| the runnorm/batchnorm modes aim for")
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--mixup", type=float, default=0.0, help="mixup alpha applied to features (0 = off)")
    p.add_argument("--holdout", type=int, default=None,
                   help="index into the class list to hold out; trains on the rest and reports AUROC for it")
    p.add_argument("--norm_pen", type=float, default=0.0, help="weight on relu(|v|-clip_r)^2 (Poincare only)")
    p.add_argument("--tree_w", type=float, default=0.0, help="weight on the tree-distance prototype loss (Poincare only)")
    p.add_argument("--tree_scale", type=float, default=1.0, help="hyperbolic distance per tree edge")
    p.add_argument("--soft_t", type=float, default=0.0, help="temperature of tree-aware soft labels (0 = off)")
    p.add_argument("--soft_mix", type=float, default=0.5, help="weight of soft labels vs one-hot")
    p.add_argument("--depth_w", type=float, default=0.0, help="weight on the radius-vs-tree-depth loss")
    p.add_argument("--quiet", action="store_true", help="only print the summary")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = p.parse_args()

    device = torch.device("cuda:0")
    if a.rebuild or not os.path.exists(a.cache):
        t = time.time(); build_cache(a, device); print(f"cache built in {time.time() - t:.0f}s -> {a.cache}")
    d = np.load(a.cache)
    data = [torch.tensor(d[k]).to(device) for k in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")]
    print(f"features: train {tuple(data[0].shape)} val {tuple(data[2].shape)} test {tuple(data[4].shape)}  "
          f"|feat| mean {data[0].norm(dim=1).mean():.1f}")

    holdout_x = None
    if a.holdout is not None:
        classes = [str(c) for c in d["classes"]]
        print(f"holding out class index {a.holdout} ('{classes[a.holdout]}'); n_out_images={int(sum((data[i] == a.holdout).sum() for i in (1, 3, 5)))}")
        data, holdout_x, _ = remap_holdout(data, a.holdout)

    classes = [str(c) for c in d["classes"]]
    a.held_name = classes[a.holdout] if a.holdout is not None else None
    a.known_names = [c for i, c in enumerate(classes) if i != a.holdout]
    a.depth_t = torch.tensor([float(class_tree.depth(n)) for n in a.known_names], device=device)
    a.tree_T = torch.tensor(class_tree.tree_distance_matrix(a.known_names), dtype=torch.float32, device=device)

    summary = {}
    placements = []
    for name in a.heads:
        for seed in a.seeds:
            t = time.time()
            rows, bv, bt, auc, bb, rad = run_head(name, a, data, device, seed, holdout_x)
            if not a.quiet:
                print(f"\n[{name} seed={seed}]  {time.time() - t:.1f}s")
                print(" step  trainCE  train%  val%  test%  clipped  preclip|v|  mean|proto|")
                for r in rows:
                    print(f" {r[0]:4d}  {r[1]:7.3f}  {r[2]:6.1f}  {r[3]:5.1f} {r[4]:6.1f}  {r[5]:7.2f}  {r[6]:10.2f}  {r[7]:11.3f}")
            if name in ("poincare", "euclid"):
                pl = a.last_placement
                msg = f"  placement: spearman(emb dist, tree dist)={pl['rho']:.3f}"
                if "rank" in pl:
                    msg += f" | held '{a.held_name}' nearest known: " + ", ".join(f"{n}(T={t})" for n, _, t in pl["rank"][:3]) + f" -> tree dist {pl['nearest_tree']} (best possible {pl['min_possible']}) | held radius/known mean {pl['radius'][a.held_name] / np.mean([v for k, v in pl['radius'].items() if k != a.held_name]):.2f} (depth {class_tree.depth(a.held_name)})"
                print(msg)
                placements.append(pl)
            summary.setdefault(name, []).append((bv, bt, rows[-1][4] if rows else float("nan"), auc[0], auc[1], auc[2], np.mean([r[5] for r in rows[-1:]]), rows[-1][6], bb, rows[-1][8], rad[0], rad[1]))
    print("\nsummary, mean +- std over seeds (test% on known classes; AUROC for held-out class if --holdout)")
    for name, v in summary.items():
        v = np.array(v); m, sd = v.mean(0), v.std(0)
        extra = f"  AUROC maxlogit {m[3]:.3f}+-{sd[3]:.3f} MSP {m[4]:.3f}+-{sd[4]:.3f} origin-dist {m[5]:.3f}+-{sd[5]:.3f}  [clipped {m[6]:.2f}, preclip|v| {m[7]:.1f}]" if a.holdout is not None else ""
        if name in ("poincare", "euclid"):
            extra += f"  | radius mean {m[10]:.2f} cv {m[11]:.3f}"
        print(f"  {name:9s} test@bestval {m[1]:5.1f}+-{sd[1]:.1f}  test@final {m[2]:5.1f}+-{sd[2]:.1f}  | balanced: @bestval {m[8]:5.1f}+-{sd[8]:.1f} @final {m[9]:5.1f}+-{sd[9]:.1f}{extra}")


if __name__ == "__main__":
    main()
