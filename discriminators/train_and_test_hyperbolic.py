"""Same benchmark as train_and_test.py, but the classifier head is a flat Poincare-ball
prototype head (curvature c, default 1) instead of the stock linear layer.

Everything else (data, augmentation, mixup, AdamW loop, early stopping, test CSV) is unchanged.

    backbone (timm, num_classes=0) -> Linear(feat, dim) -> norm clip -> expmap0
        -> logits_k = -d_c(z, p_k) / tau        p_k = expmap0(v_k), v_k learnable (tangent space)
"""
import argparse
import csv
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from torch.distributions.beta import Beta
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment
from tqdm import tqdm

import class_tree

EPS = 1e-5


# ----------------------------------------------------------------------------- Poincare maths
def _norm(x, dim=-1, keepdim=True):
    # clamp the squared norm so the gradient of ||x|| at x=0 is finite
    return x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(1e-12).sqrt()


def expmap0(v, c):
    """Tangent vector at the origin -> point in the ball of curvature -c."""
    sc = c ** 0.5
    n = _norm(v)
    return torch.tanh(sc * n) * v / (sc * n)


def mobius_add(x, y, c):
    xy = (x * y).sum(-1, keepdim=True)
    x2 = x.pow(2).sum(-1, keepdim=True)
    y2 = y.pow(2).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / den.clamp_min(EPS)


def poincare_dist(x, y, c):
    """Pairwise distance. x: (B,d), y: (K,d) -> (B,K)."""
    sc = c ** 0.5
    diff = mobius_add(-x[:, None, :], y[None, :, :], c)  # (B,K,d)
    n = _norm(diff, keepdim=False)
    return 2 / sc * torch.atanh((sc * n).clamp(max=1 - EPS))


class PoincareHead(nn.Module):
    def __init__(self, in_features, n_classes, dim=64, c=1.0, clip_r=2.0, tau=0.1, norm_mode="clip", target_norm=1.0):
        """norm_mode: 'clip'      -> only the hard norm clip (original behaviour)
                     'runnorm'   -> divide v by a detached running mean of |v|, times target_norm (per-sample radii stay free)
                     'batchnorm' -> BatchNorm1d(affine=False) on v, times target_norm/sqrt(dim) (mean |v| ~ target_norm)
        In the last two the clip is only a safety net, so use a larger clip_r (e.g. 4)."""
        super().__init__()
        self.c, self.clip_r = c, clip_r
        self.norm_mode, self.target_norm, self.dim = norm_mode, target_norm, dim
        if norm_mode == "batchnorm":
            self.bn = nn.BatchNorm1d(dim, affine=False)
        self.register_buffer("run_norm", torch.ones(()))
        self.register_buffer("run_inited", torch.zeros(()))
        # ViT features are large and share a big common component; normalise, then init the projection
        # small so the pre-clip tangent norm starts ~1 (unsaturated) rather than >>clip_r.
        self.feat_norm = nn.LayerNorm(in_features, elementwise_affine=False)
        self.proj = nn.Linear(in_features, dim)
        with torch.no_grad():
            self.proj.weight.mul_(0.3)
            self.proj.bias.zero_()
        # prototypes: random directions at tangent radius ~1 (same scale as the embeddings)
        proto = torch.randn(n_classes, dim)
        self.proto_tangent = nn.Parameter(proto / proto.norm(dim=-1, keepdim=True))
        self.last_clipped_frac = 0.0
        self.log_inv_tau = nn.Parameter(torch.tensor(float(np.log(1.0 / tau))))

    def embed(self, feats):
        v = self.proj(self.feat_norm(feats.float()))
        if self.norm_mode == "batchnorm":
            v = self.bn(v) * (self.target_norm / self.dim ** 0.5)
        elif self.norm_mode == "runnorm":
            if self.training:
                with torch.no_grad():
                    m = _norm(v).mean()
                    if self.run_inited.item() == 0:
                        self.run_norm.copy_(m); self.run_inited.fill_(1)
                    else:
                        self.run_norm.mul_(0.9).add_(0.1 * m)
            v = v * (self.target_norm / self.run_norm.clamp_min(1e-6))
        # feature clipping (Guo et al. 2022): cap the tangent norm so points can't saturate the boundary
        scale = torch.clamp(self.clip_r / _norm(v), max=1.0)
        self._v_norm = _norm(v).squeeze(-1)  # differentiable, per sample (for optional norm penalty)
        self.last_preclip_norm = self._v_norm.mean().item()
        self.last_clipped_frac = (scale < 1).float().mean().item()
        v = v * scale
        return self.exp0(v)

    # geometry hooks (overridden by EuclideanHead)
    def exp0(self, v):
        return expmap0(v, self.c)

    def dist(self, x, y):
        return poincare_dist(x, y, self.c)

    def log0(self, z):
        sc = self.c ** 0.5
        n = z.norm(dim=-1, keepdim=True).clamp(1e-9, 1 - 1e-5)
        return torch.atanh(sc * n) * z / (sc * n)

    def radius(self, z):
        return 2 * torch.atanh(z.norm(dim=-1).clamp(max=1 - 1e-5)) / self.c ** 0.5

    def prototypes(self):
        return self.exp0(self.proto_tangent)

    def forward(self, feats, return_embedding=False):
        with torch.autocast("cuda", enabled=False):  # keep the hyperbolic maths in fp32
            z = self.embed(feats)
            logits = -self.dist(z, self.prototypes()) * self.log_inv_tau.exp()
        return (logits, z) if return_embedding else logits


class EuclideanHead(PoincareHead):
    """Geometry control: identical pipeline to PoincareHead (LayerNorm, projection, norm_mode, distance logits,
    same tau) but the embedding is the tangent vector itself and distances are Euclidean."""
    def exp0(self, v):
        return v

    def dist(self, x, y):
        return torch.cdist(x, y)

    def log0(self, z):
        return z

    def radius(self, z):
        return z.norm(dim=-1)


class LinearHead(nn.Module):
    """Control: the stock timm-style linear classifier, same interface as PoincareHead."""
    def __init__(self, in_features, n_classes, **_):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)
        self.log_inv_tau = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.last_clipped_frac = 0.0
        self.last_preclip_norm = 0.0

    def prototypes(self):
        return self.fc.weight

    def forward(self, feats, return_embedding=False):
        logits = self.fc(feats)
        return (logits, feats) if return_embedding else logits


class Net(nn.Module):
    def __init__(self, model_name, n_classes, head="poincare", **head_kw):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        cls = {"poincare": PoincareHead, "euclid": EuclideanHead, "linear": LinearHead}[head]
        self.head = cls(self.backbone.num_features, n_classes, **head_kw)

    def forward(self, x, return_embedding=False):
        return self.head(self.backbone(x), return_embedding)


# ----------------------------------------------------------------------------- held-out placement
@torch.no_grad()
def placement(head, zk, yk, zh, known, held, device):
    """Where does the held-out class land? Class centroids are tangent-space means; prints a line in the same format
    as frozen_head_harness.py so the same parser works."""
    from scipy.stats import spearmanr
    zk, yk, zh = zk.to(device), yk.to(device), zh.to(device)
    cents = [head.exp0(head.log0(zk[yk == k]).mean(0)) for k in range(len(known))]
    cents.append(head.exp0(head.log0(zh).mean(0)))
    C = torch.stack(cents)
    Dc = head.dist(C, C).cpu().numpy()
    names = known + [held]
    T = class_tree.tree_distance_matrix(names)
    iu = np.triu_indices(len(names), 1)
    rho = spearmanr(Dc[iu], T[iu])[0]
    j = len(names) - 1
    order = np.argsort(Dc[j, :j])
    rad = {n: float(head.radius(c[None])) for n, c in zip(names, C)}
    ratio = rad[held] / np.mean([v for k, v in rad.items() if k != held])
    top = ", ".join(f"{names[i]}(T={int(T[j, i])})" for i in order[:3])
    print(f"  placement: spearman(emb dist, tree dist)={rho:.3f} | held '{held}' nearest known: {top} -> tree dist "
          f"{int(T[j, order[0]])} (best possible {int(T[j, :j].min())}) | held radius/known mean {ratio:.2f} (depth {class_tree.depth(held)})")


# ----------------------------------------------------------------------------- script
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_base_patch16_384")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--head", choices=["poincare", "euclid", "linear"], default="poincare",
                   help="euclid = same bottleneck/prototype pipeline as poincare with Euclidean geometry (control)")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--clip_r", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="backbone lr (as in the baseline)")
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 = as in the baseline script)")
    p.add_argument("--norm_mode", choices=["clip", "runnorm", "batchnorm"], default="clip",
                   help="how the tangent vector is scaled before expmap0 (Poincare only); runnorm/batchnorm keep the radius "
                        "free, use them with --clip_r 4 (the clip is then only a safety net)")
    p.add_argument("--target_norm", type=float, default=1.0, help="mean |v| that runnorm/batchnorm aim for")
    p.add_argument("--norm_pen", type=float, default=0.0,
                   help="weight on relu(|v|-clip_r)^2 so the tangent norm stays inside the clip (Poincare only)")
    p.add_argument("--probe_epochs", type=int, default=0,
                   help="train only the head (backbone frozen) for this many epochs, then unfreeze (LP-FT)")
    p.add_argument("--holdout", default=None, help="class name to hold out of train/val/test (tree-aware analysis on it at the end)")
    p.add_argument("--soft_t", type=float, default=0.0, help="temperature of tree-aware soft labels (0 = off)")
    p.add_argument("--soft_mix", type=float, default=0.8, help="weight of soft labels vs one-hot")
    p.add_argument("--depth_w", type=float, default=0.0, help="weight of the radius-vs-tree-depth loss")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment_name", default="Example-poincare")
    p.add_argument("--data_root", default="../data/prepare_data/example_data_train_val_test")
    p.add_argument("--wandb", action="store_true")
    a = p.parse_args()

    device = torch.device("cuda:0")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    if a.wandb:
        import wandb
        wandb.init(project=f"Discriminator-{a.experiment_name}", config=vars(a))
    log = (lambda d: wandb.log(d)) if a.wandb else (lambda d: None)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((a.img_size, a.img_size)), RandAugment(), transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.125),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((a.img_size, a.img_size)), transforms.ToTensor(), norm])

    train_ds = datasets.ImageFolder(os.path.join(a.data_root, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(a.data_root, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(a.data_root, "test"), transform=eval_tf)
    held_paths = []
    if a.holdout is not None:
        all_classes = list(train_ds.classes)
        hold = all_classes.index(a.holdout)
        for ds in (train_ds, val_ds, test_ds):
            held_paths += [pth for pth, y in ds.samples if y == hold]
            ds.samples = [(pth, y - int(y > hold)) for pth, y in ds.samples if y != hold]
            ds.targets = [y for _, y in ds.samples]
            ds.classes = [c for c in all_classes if c != a.holdout]
            ds.class_to_idx = {c: i for i, c in enumerate(ds.classes)}
        print(f"holding out '{a.holdout}': {len(held_paths)} images across all splits")
    n_classes = len(train_ds.classes)
    print(f"classes ({n_classes}): {train_ds.classes}")
    known = list(train_ds.classes)
    tree_T = depth_t = None
    if a.soft_t > 0 or a.depth_w > 0 or a.holdout is not None:  # needs the class names to be in class_tree
        tree_T = torch.tensor(class_tree.tree_distance_matrix(known), dtype=torch.float32, device=device)
        depth_t = torch.tensor([float(class_tree.depth(n)) for n in known], dtype=torch.float32, device=device)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)

    model = Net(a.model, n_classes, head=a.head, dim=a.dim, c=a.c, clip_r=a.clip_r,
                norm_mode=a.norm_mode, target_norm=a.target_norm).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": a.lr},
        {"params": model.head.parameters(), "lr": a.head_lr},
    ])
    ckpt = f"saved_models/{a.experiment_name}_{a.head}_{a.model}_{a.seed}.pth"
    os.makedirs("saved_models", exist_ok=True)

    def mixup(x, y, alpha=0.3):
        lam = Beta(alpha, alpha).sample().item()
        idx = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1 - lam) * x[idx], y, y[idx], lam

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
        return 100 * correct / total

    best, bad = 0.0, 0
    for epoch in range(a.epochs):
        model.train()
        probing = epoch < a.probe_epochs
        for prm in model.backbone.parameters():
            prm.requires_grad_(not probing)
        if probing:
            model.backbone.eval()
        if epoch == a.probe_epochs and a.probe_epochs > 0:
            print(f"-- unfreezing backbone at epoch {epoch + 1}")
        running, clipped, prenorm = 0.0, [], []
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            images, ya, yb, lam = mixup(images, labels)
            optimizer.zero_grad()
            if a.depth_w > 0:
                logits, z = model(images, return_embedding=True)
            else:
                logits = model(images)
            if a.soft_t > 0:  # tree-aware soft labels, mixed with one-hot (applied to both mixup targets)
                def ce(y):
                    q = F.softmax(-tree_T[y] / a.soft_t, dim=1)
                    q = (1 - a.soft_mix) * F.one_hot(y, n_classes).float() + a.soft_mix * q
                    return -(q * F.log_softmax(logits, dim=1)).sum(1).mean()
                loss = lam * ce(ya) + (1 - lam) * ce(yb)
            else:
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            if a.depth_w > 0:  # relative radius -> relative depth (mixup: depth of the mixed label)
                r = model.head.radius(z)
                dep = lam * depth_t[ya] + (1 - lam) * depth_t[yb]
                loss = loss + a.depth_w * (r / r.mean().detach() - dep / dep.mean()).pow(2).mean()
            if a.norm_pen > 0 and a.head in ("poincare", "euclid"):
                loss = loss + a.norm_pen * torch.relu(model.head._v_norm - a.clip_r).pow(2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}: {loss.item()}")
            loss.backward()
            optimizer.step()
            running += loss.item()
            clipped.append(model.head.last_clipped_frac)
            prenorm.append(model.head.last_preclip_norm)
        val_acc = evaluate(val_loader)
        with torch.no_grad():
            protos = model.head.prototypes()
            proto_norm = protos.norm(dim=-1).mean().item()
        print(f"Epoch {epoch + 1}, Loss: {running / len(train_loader):.4f}, Validation Accuracy: {val_acc}, "
              f"mean|proto|: {proto_norm:.3f}, 1/tau: {model.head.log_inv_tau.exp().item():.2f}, clipped: {np.mean(clipped):.2f}, preclip|v|: {np.mean(prenorm):.2f}")
        log({"epoch": epoch, "loss": running / len(train_loader), "accuracy": val_acc, "proto_norm": proto_norm})
        if val_acc >= best:
            best, bad = val_acc, 0
            torch.save(model.state_dict(), ckpt)
            print(f"Saved model with accuracy: {best}")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ---- test: also dump embeddings so hierarchy / held-out analysis can be done later
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    labels_all, preds_all, diffs, embs = [], [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            logits, z = model(images.to(device), return_embedding=True)
            top2 = torch.topk(logits, 2, dim=1).values
            labels_all += labels.tolist()
            preds_all += logits.argmax(1).cpu().tolist()
            diffs += (top2[:, 0] - top2[:, 1]).cpu().tolist()
            embs.append(z.cpu().numpy())
    z_held = None
    if a.holdout is not None and hasattr(model.head, "exp0"):
        from PIL import Image
        class HeldDS(torch.utils.data.Dataset):
            def __len__(self): return len(held_paths)
            def __getitem__(self, i): return eval_tf(Image.open(held_paths[i]).convert("RGB"))
        held_loader = torch.utils.data.DataLoader(HeldDS(), batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)
        zh = []
        with torch.no_grad():
            for images in held_loader:
                zh.append(model(images.to(device), return_embedding=True)[1].cpu())
        z_held = torch.cat(zh)
        placement(model.head, torch.tensor(np.concatenate(embs)), torch.tensor(labels_all), z_held, known, a.holdout, device)
    names = [s[0].split("/")[-1] for s in test_ds.samples]
    acc = 100 * np.mean(np.array(labels_all) == np.array(preds_all))
    bal = balanced_accuracy_score(labels_all, preds_all) * 100
    print(f"Test Accuracy: {acc}  Balanced: {bal}")
    log({"test_accuracy": acc, "test_balanced_accuracy": bal})

    os.makedirs(a.experiment_name, exist_ok=True)
    base = os.path.join(a.experiment_name, f"{a.experiment_name}_{a.head}_{a.model}_{a.seed}")
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Image_Name", "Label", "Prediction", "Logit_Difference"])
        w.writerows(zip(names, labels_all, preds_all, diffs))
    np.savez(base + "_embeddings.npz", z=np.concatenate(embs), labels=labels_all, names=names,
             prototypes=model.head.prototypes().detach().cpu().numpy(), classes=train_ds.classes, c=a.c, head=a.head,
             z_held=z_held.numpy() if z_held is not None else np.zeros((0,)), held_name=a.holdout or "")
    print(f"Saved {base}.csv and {base}_embeddings.npz")


if __name__ == "__main__":
    main()
