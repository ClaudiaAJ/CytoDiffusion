"""Control for the radius-as-novelty-signal: is a class's radius lower when the class is held out than when it is trained on?

For each seed and class k: train (a) a head on all classes, (b) a head with class k held out. Both use the same hyper-parameters.
Compare the hyperbolic radius of class k's *test* images under (a) and (b). Delta = R_held - R_trained.
Delta < 0 consistently => novelty pulls samples towards the origin. Delta ~ 0 => the radius is a property of the class, not of novelty.
"""
import argparse
import importlib.util
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hh", os.path.join(HERE, "frozen_head_harness.py"))
hh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hh)


@torch.no_grad()
def radius(head, x):
    head.eval()
    z = head.embed(x)
    return (2 * torch.atanh(z.norm(dim=-1).clamp(max=1 - 1e-5)) / head.c ** 0.5).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--norm_mode", default="runnorm")
    p.add_argument("--target_norm", type=float, default=1.0)
    p.add_argument("--clip_r", type=float, default=4.0)
    a = p.parse_args()
    a.dim, a.c, a.tau, a.lr, a.wd, a.batch_size = 64, 1.0, 0.1, 1e-3, 1e-2, 32
    a.mixup, a.norm_pen, a.eval_every = 0.0, 0.0, a.steps
    os.makedirs(a.out_dir, exist_ok=True)

    device = torch.device("cuda:0")
    d = np.load(a.cache)
    classes = [str(c) for c in d["classes"]]
    full = [torch.tensor(d[k]).to(device) for k in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")]
    xte, yte = full[4], full[5]
    K = len(classes)
    R_tr = np.zeros((len(a.seeds), K)); R_ho = np.zeros_like(R_tr); R_oth = np.zeros_like(R_tr)
    for si, seed in enumerate(a.seeds):
        head = hh.run_head("poincare", a, full, device, seed, None, return_head=True)[-1]
        for k in range(K):
            R_tr[si, k] = radius(head, xte[yte == k]).mean()
        for k in range(K):
            data_k, _, _ = hh.remap_holdout(full, k)
            head_k = hh.run_head("poincare", a, data_k, device, seed, None, return_head=True)[-1]
            R_ho[si, k] = radius(head_k, xte[yte == k]).mean()
            R_oth[si, k] = radius(head_k, xte[yte != k]).mean()
        print(f"seed {seed} done", flush=True)

    np.savez(os.path.join(a.out_dir, "radius_control.npz"), R_trained=R_tr, R_held=R_ho, R_other_known=R_oth, classes=classes, seeds=a.seeds)
    print(f"\nmean hyperbolic radius of each class's TEST images ({a.norm_mode}, target {a.target_norm}); mean +- std over {len(a.seeds)} seeds")
    print(f"{'class':13s} {'trained on':>12s} {'held out':>12s} {'delta':>14s} {'seeds delta<0':>14s} {'other known (held-out head)':>28s}")
    for k, c in enumerate(classes):
        dl = R_ho[:, k] - R_tr[:, k]
        print(f"{c:13s} {R_tr[:, k].mean():6.2f}+-{R_tr[:, k].std():4.2f} {R_ho[:, k].mean():6.2f}+-{R_ho[:, k].std():4.2f} "
              f"{dl.mean():+7.3f}+-{dl.std():5.3f} {int((dl < 0).sum()):>9d}/{len(dl):<4d} {R_oth[:, k].mean():18.2f}")
    dl = R_ho - R_tr
    print(f"{'all classes':13s} {R_tr.mean():6.2f}      {R_ho.mean():6.2f}      {dl.mean():+7.3f}  (class-averaged)")


if __name__ == "__main__":
    main()
