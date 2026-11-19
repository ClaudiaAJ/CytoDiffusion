"""Tree consistency of the PBC fine-tuned test embeddings (flat labels, no tree loss).

rho = Spearman(class-centroid distance, tree distance) over the 28 class pairs.
Centroids: Euclidean mean (linear, euclid); mean in the tangent space at the origin mapped back with exp0 (poincare).
Distance: the head's own (Euclidean; hyperbolic for poincare).
radius-depth = Spearman(per-class mean radius, tree depth); radius = |z| (Euclidean) or d(0, z) (Poincare).

    python ft_tree_consistency.py
"""
import importlib.util
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, D)
import class_tree  # noqa: E402

spec = importlib.util.spec_from_file_location("hyp", os.path.join(D, "train_and_test_hyperbolic.py"))
hyp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hyp)
B = "/store/CIA/ca662/CytoDiffusion_runs/pbc/finetune/PBC_FT/PBC_FT_%s_vit_base_patch16_384_%d_embeddings.npz"


def pdist(x, y, c):
    return hyp.poincare_dist(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), c).numpy()


for h in ("poincare", "euclid", "linear"):
    R, RD = [], []
    for seed in range(3):
        d = np.load(B % (h, seed)); z = d["z"].astype(np.float64); y = d["labels"]; cl = [str(c) for c in d["classes"]]
        T = class_tree.tree_distance_matrix(cl); dep = np.array([class_tree.depth(n) for n in cl], float)
        iu = np.triu_indices(len(cl), 1); K = range(len(cl))
        if h == "poincare":
            c = float(d["c"]); n = np.linalg.norm(z, axis=1, keepdims=True)
            v = np.arctanh(np.clip(np.sqrt(c) * n, 0, 1 - 1e-7)) * z / (n * np.sqrt(c))
            cen = hyp.expmap0(torch.tensor(np.stack([v[y == k].mean(0) for k in K])), c).numpy()
            Dc = pdist(cen, cen, c); rad = pdist(np.zeros((1, z.shape[1])), z, c)[0]
        else:
            cen = np.stack([z[y == k].mean(0) for k in K])
            Dc = np.linalg.norm(cen[:, None] - cen[None], axis=-1); rad = np.linalg.norm(z, axis=1)
        R.append(spearmanr(Dc[iu], T[iu])[0])
        RD.append(spearmanr([rad[y == k].mean() for k in K], dep)[0])
    print(f"{h:9s} rho {np.mean(R):.3f} +- {np.std(R, ddof=1):.3f}   radius-depth {np.mean(RD):.2f} +- {np.std(RD, ddof=1):.2f}"
          f"   per-seed rho {np.round(R, 3)}")
