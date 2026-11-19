"""Figures for the PBC report (report/PBC_REPORT.md).

    python report/scripts/pbc_figures.py     # writes report/figures/PBC_*.png

PBC_00_class_tree.png        the approved tree restricted to the 8 PBC classes
PBC_A1_embeddings.png        fine-tuned test embeddings (seed 0), 2D PCA, one panel per class
PBC_B_holdout_nearest.png    nearest known class of each held-out class (frozen features), counts over 2 label settings x 3 seeds
"""
import collections
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import class_tree  # noqa: E402

RUNS = "/store/CIA/ca662/CytoDiffusion_runs/pbc"
OUT = os.path.join(HERE, "..", "figures")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
ACCENT = "#2a78d6"
CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                       "#2a78d6", "#1c5cab", "#104281", "#0d366b"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
CLASSES = ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]


def tree_figure():
    order = ["lymphocyte", "ig", "neutrophil", "eosinophil", "basophil", "monocyte", "erythroblast", "platelet"]
    keep = set()
    for c in order:
        keep.update(class_tree.path_to_root(c))
    parent = {c: p for c, p in class_tree.PARENT.items() if c in keep}
    kids = collections.defaultdict(list)
    for c, p in parent.items():
        kids[p].append(c)
    x = {c: i for i, c in enumerate(order)}

    def px(n):
        if n not in x:
            x[n] = sum(px(k) for k in kids[n]) / len(kids[n])
        return x[n]
    px("root")
    y = {n: -class_tree.depth(n) for n in x}
    fig, ax = plt.subplots(figsize=(12, 6))
    for c, p in parent.items():
        ax.plot([x[p], x[c]], [y[p], y[c]], color="#b8b8b2", lw=1.3, zorder=1)
    for n in x:
        leaf = n in order
        ax.scatter(x[n], y[n], s=170 if leaf else 90, color=ACCENT if leaf else MUTED, zorder=3, edgecolor=SURFACE, lw=1.5)
        if leaf:
            ax.text(x[n], y[n] - 0.22, f"{n}\n(depth {class_tree.depth(n)})", ha="center", va="top", fontsize=9.5, linespacing=1.2)
        else:
            ax.text(x[n] + 0.12, y[n] + 0.12, n.replace("_", " "), ha="left", va="bottom", fontsize=9.5, color=INK2)
    ax.set_ylim(-5.9, 0.5); ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_yticks(range(0, -6, -1)); ax.set_yticklabels([str(i) for i in range(6)])
    ax.set_ylabel("depth (edges from root)")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#ccc"); ax.set_xticks([])
    fig.text(0.04, 0.95, "Class tree for PBC (approved CytoData tree without artefact and blast)", fontsize=14, fontweight="bold", va="top")
    fig.text(0.04, 0.895, "Blue = the 8 PBC labels; gray = internal nodes. Tree distance = edges between two leaves.", fontsize=10, color=INK2, va="top")
    plt.subplots_adjust(top=0.84, bottom=0.04, left=0.07, right=0.98)
    fig.savefig(os.path.join(OUT, "PBC_00_class_tree.png"), dpi=150)
    plt.close(fig)


def pca2(z):
    zc = z - z.mean(0)
    u, s, _ = np.linalg.svd(zc, full_matrices=False)
    return u[:, :2] * s[:2], (s[:2] ** 2).sum() / (s ** 2).sum()


def embeddings_figure():
    heads = [("linear", "Linear head (768-d)"), ("euclid", "Euclidean head (64-d)"), ("poincare", "Poincaré head (64-d)")]
    fig, axes = plt.subplots(3, 8, figsize=(20, 8.4))
    for r, (h, title) in enumerate(heads):
        d = np.load(f"{RUNS}/finetune/PBC_FT/PBC_FT_{h}_vit_base_patch16_384_0_embeddings.npz")
        z, y, cl = d["z"].astype(np.float64), d["labels"], [str(c) for c in d["classes"]]
        if h == "poincare":  # PCA in the tangent space at the origin
            c = float(d["c"]); n = np.linalg.norm(z, axis=1, keepdims=True)
            z = np.arctanh(np.clip(np.sqrt(c) * n, 0, 1 - 1e-7)) * z / (n * np.sqrt(c))
        xy, ev = pca2(z)
        for k, name in enumerate(cl):
            ax = axes[r, k]; m = y == k
            ax.scatter(xy[~m, 0], xy[~m, 1], s=4, c=MUTED, alpha=0.25, linewidths=0, zorder=1)
            ax.scatter(xy[m, 0], xy[m, 1], s=8, c=ACCENT, alpha=0.9, linewidths=0, zorder=2)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if r == 0:
                ax.set_title(name, fontsize=11)
        axes[r, 0].set_ylabel(f"{title}\nPCA var {ev:.0%}", fontsize=10, labelpad=8)
    fig.text(0.01, 0.985, "PBC fine-tuned test embeddings (seed 0, flat labels), 2D PCA", fontsize=14, fontweight="bold", va="top")
    fig.text(0.01, 0.95, "Blue = the class named above the column; gray = all other test images. Poincaré head: running-norm, c=1; its embeddings are mapped to the tangent space at the origin (log map) before PCA.",
             fontsize=10, color=INK2, va="top")
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.06, right=0.995, wspace=0.05, hspace=0.08)
    fig.savefig(os.path.join(OUT, "PBC_A1_embeddings.png"), dpi=130)
    plt.close(fig)


def holdout_figure():
    pat = re.compile(r"held '(\w+)' nearest known: (\w+)\(T=")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
    panels = [(64, "poincare"), (64, "euclid"), (4, "poincare"), (4, "euclid")]
    for ax, (dim, head) in zip(axes, panels):
        M = np.zeros((8, 8), int); hd = None
        for line in open(f"{RUNS}/holdout_dim{dim}.log"):
            m = re.match(r"## holdout=\d+ head=(\w+)", line)
            if m:
                hd = m.group(1); continue
            m = pat.search(line)
            if m and hd == head:
                M[CLASSES.index(m.group(1)), CLASSES.index(m.group(2))] += 1
        T = class_tree.tree_distance_matrix(CLASSES)
        ax.imshow(np.ma.masked_where(np.eye(8, dtype=bool), M), cmap=CMAP, vmin=0, vmax=6)
        for i in range(8):
            best = T[i][np.arange(8) != i].min()
            for j in range(8):
                if i == j:
                    ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=9); continue
                if T[i, j] == best:  # best-possible neighbour(s) in the tree
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
                if M[i, j]:
                    ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=9, color="white" if M[i, j] >= 4 else INK)
        ax.set_xticks(range(8)); ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=9)
        ax.set_yticks(range(8)); ax.set_yticklabels(CLASSES if ax is axes[0] else [], fontsize=9)
        ax.set_title(f"{'Poincaré' if head == 'poincare' else 'Euclidean'}, dim {dim}", fontsize=11)
        ax.set_xlabel("nearest known class"); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("held-out class")
    fig.text(0.01, 0.98, "PBC hold-out placement (frozen features): nearest known class of the held-out class", fontsize=14, fontweight="bold", va="top")
    fig.text(0.01, 0.925, "Counts over 6 runs per row (flat + hierarchy labels, 3 seeds each). Outlined cells = best possible neighbour(s) in the tree.", fontsize=10, color=INK2, va="top")
    plt.subplots_adjust(top=0.84, bottom=0.2, left=0.07, right=0.995, wspace=0.08)
    fig.savefig(os.path.join(OUT, "PBC_B_holdout_nearest.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    tree_figure(); embeddings_figure(); holdout_figure()
    print("written to", os.path.abspath(OUT))
