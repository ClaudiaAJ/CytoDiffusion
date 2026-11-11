"""Euclidean (linear head) vs hyperbolic (Poincare head) embeddings from train_and_test_hyperbolic.py.

Reads the *_embeddings.npz files written at test time and writes
  embeddings_pca_small_multiples.png   one panel per class (that class highlighted, the rest gray), 2D PCA
  class_distance_heatmaps.png          mean pairwise distance between classes (diagonal = within-class spread)
  radius_by_class.png                  (only with --poincare_free) hyperbolic radius of each class's test embeddings
  class_distance_{euclid_cosine,poincare[,poincare_free]}.csv, radius_by_class.csv   the numbers behind the figures

Rerun it on the hierarchical-label runs to get the same figures for a like-for-like comparison.

    python plot_embeddings.py --linear A.npz --poincare B.npz [--poincare_free C.npz] --out_dir figs --label "flat labels, seed 0"
  --poincare       head with a hard norm clip (radius fixed)      --poincare_free   head with a free radius (e.g. runnorm)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- design tokens (dataviz skill reference palette, light mode) ---
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ACCENT = "#2a78d6"  # categorical slot 1 (blue): the highlighted class
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # sequential 100 -> 700
CMAP = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["z"].astype(np.float64), d["labels"], [str(c) for c in d["classes"]], float(d["c"])


def pca2(x):
    xc = x - x.mean(0)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    return u[:, :2] * s[:2], (s[:2] ** 2).sum() / (s ** 2).sum()


def poincare_dist_matrix(z, c):
    """d(x,y) = arcosh(1 + 2c|x-y|^2 / ((1-c|x|^2)(1-c|y|^2))) / sqrt(c)"""
    sq = (z ** 2).sum(1)
    e2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    arg = 1 + 2 * c * e2 / ((1 - c * sq)[:, None] * (1 - c * sq)[None, :])
    return np.arccosh(np.maximum(arg, 1.0)) / np.sqrt(c)


def cosine_dist_matrix(x):
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    return 1 - xn @ xn.T


def class_mean_matrix(D, y, k):
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            block = D[np.ix_(y == i, y == j)]
            if i == j:  # exclude the zero self-distances
                n = block.shape[0]
                M[i, j] = block.sum() / max(n * (n - 1), 1)
            else:
                M[i, j] = block.mean()
    return M


def style_axes(ax, circle=False):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID); sp.set_linewidth(1)
    if circle:
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), color=AXIS, lw=1, zorder=0)  # ball boundary
    ax.set_aspect("equal")


def small_multiples(fig, outer, xy, y, classes, title, note, circle):
    sub = outer.subgridspec(2, 5, wspace=0.06, hspace=0.22)
    lim = 1.08 if circle else np.abs(xy).max() * 1.08
    for k, name in enumerate(classes):
        ax = fig.add_subplot(sub[k // 5, k % 5])
        other = y != k
        ax.scatter(xy[other, 0], xy[other, 1], s=6, c=MUTED, alpha=0.30, linewidths=0, zorder=1)
        ax.scatter(xy[~other, 0], xy[~other, 1], s=16, c=ACCENT, alpha=0.95, edgecolors=SURFACE, linewidths=0.3, zorder=2)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        style_axes(ax, circle)
        ax.set_title(f"{name}  (n={int((~other).sum())})", fontsize=9.5, color=INK, loc="left", pad=4)
        if k == 0:
            ax.annotate(title, xy=(0, 1.38), xycoords="axes fraction", fontsize=11.5, fontweight="semibold", color=INK)
            ax.annotate(note, xy=(0, 1.22), xycoords="axes fraction", fontsize=8.5, color=INK2)


def hyp_radius(z, c):
    return 2 * np.arctanh(np.minimum(np.linalg.norm(z, axis=1), 1 - 1e-9)) / np.sqrt(c)


def radius_figure(r, y, classes, path, label):
    """Per-class distribution of hyperbolic radius: bar = middle 50%, whisker = 5th-95th percentile, dot = median."""
    k = len(classes)
    stats = []
    for i in range(k):
        q = np.percentile(r[y == i], [5, 25, 50, 75, 95])
        stats.append((classes[i], int((y == i).sum()), *q))
    order = np.argsort([st[4] for st in stats])  # smallest median radius at the top
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(left=0.17, right=0.95, top=0.78, bottom=0.12)
    fig.text(0.03, 0.955, "How far from the origin does each class sit?", fontsize=15, fontweight="semibold", color=INK)
    fig.text(0.03, 0.915, f"Hyperbolic radius of test embeddings, free-radius head{(' · ' + label) if label else ''}.", fontsize=9.5, color=INK2)
    fig.text(0.03, 0.885, "Dot = median, bar = middle 50%, whisker = 5th–95th percentile. Smaller radius = nearer the centre of the ball.", fontsize=9.5, color=INK2)
    for row, i in enumerate(order):
        name, n, p5, p25, p50, p75, p95 = stats[i]
        ax.plot([p5, p95], [row, row], color=AXIS, lw=1.5, solid_capstyle="round", zorder=1)
        ax.plot([p25, p75], [row, row], color=ACCENT, lw=7, solid_capstyle="round", zorder=2)
        ax.plot([p50], [row], marker="o", ms=8, mfc=ACCENT, mec=SURFACE, mew=2, zorder=3)
    ax.set_yticks(range(k)); ax.set_yticklabels([f"{stats[i][0]}  (n={stats[i][1]})" for i in order], fontsize=9.5, color=INK2)
    ax.set_ylim(k - 0.5, -0.5)
    lo, hi = min(st[2] for st in stats), max(st[6] for st in stats)
    mid, span = (lo + hi) / 2, max(hi - lo, 0.5)  # minimum span so a near-constant radius doesn't blow up the axis
    ax.set_xlim(mid - span * 0.55, mid + span * 0.55)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    ax.set_xlabel("hyperbolic radius  (2·artanh|z|)", fontsize=9.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, colors=MUTED); ax.tick_params(axis="y", labelcolor=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(path, dpi=140); plt.close(fig)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--linear", required=True); p.add_argument("--poincare", required=True)
    p.add_argument("--poincare_free", default=None, help="optional: embeddings of a free-radius Poincare head")
    p.add_argument("--out_dir", required=True); p.add_argument("--label", default="")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    zl, yl, classes, _ = load(a.linear)
    zp, yp, classes_p, c = load(a.poincare)
    assert classes == classes_p and (yl == yp).all(), "the files must hold the same test images in the same order"
    sets = [("euclid_cosine", zl, "Euclidean — linear head, 768-d pooled ViT features", None),
            ("poincare", zp, "Hyperbolic — Poincaré head, hard norm clip (radius fixed), 64-d ball embeddings (c = 1)", "fixed")]
    if a.poincare_free:
        zf, yf, classes_f, cf = load(a.poincare_free)
        assert classes == classes_f and (yl == yf).all(), "the free-radius file must hold the same test images in the same order"
        sets.append(("poincare_free", zf, "Hyperbolic — Poincaré head, free radius (running-norm scaling), 64-d ball embeddings (c = 1)", "free"))
    k, nb = len(classes), len(sets)

    # ---------------- figure 1: small multiples, one block per embedding ----------------
    H = 5.6 * nb
    fig = plt.figure(figsize=(15, H))
    gs = fig.add_gridspec(nb, 1, hspace=0.34, left=0.03, right=0.985, top=1 - 1.85 / H, bottom=0.03 / (nb / 2))
    fig.text(0.03, 1 - 0.31 / H, "Test-set embeddings after fine-tuning: Euclidean vs hyperbolic heads", fontsize=15, fontweight="semibold", color=INK)
    fig.text(0.03, 1 - 0.58 / H, f"ViT-B/16, 1,000 test images, 10 classes {('· ' + a.label) if a.label else ''}. One panel per class: that class in blue, all other test images in gray. 2D PCA of the embedding.",
             fontsize=9.5, color=INK2)
    for b, (key, z, ttl, kind) in enumerate(sets):
        xy, var = pca2(z)
        if kind is None:
            note, circle = f"PC1 + PC2 explain {100 * var:.0f}% of the variance. Axes in feature units (equal aspect).", False
        else:
            r = hyp_radius(z, c)
            circle = True
            if kind == "fixed":
                note = (f"PC1 + PC2 of the ball coordinates explain {100 * var:.0f}% of the variance. Circle = ball boundary. All embeddings sit at hyperbolic radius "
                        f"{r.min():.2f} to {r.max():.2f}; the 2D radius shown is a projection artefact.")
            else:
                note = (f"PC1 + PC2 of the ball coordinates explain {100 * var:.0f}% of the variance. Circle = ball boundary. Hyperbolic radius varies across samples: "
                        f"mean {r.mean():.2f}, std {r.std():.2f}, range {r.min():.2f} to {r.max():.2f}. The 2D radius is only an approximate view of it.")
        small_multiples(fig, gs[b], xy, yl, classes, ttl, note, circle)
    f1 = os.path.join(a.out_dir, "embeddings_pca_small_multiples.png")
    fig.savefig(f1, dpi=140); plt.close(fig)

    # ---------------- figure 2: class distance heatmaps ----------------
    mats = []
    for key, z, ttl, kind in sets:
        D = cosine_dist_matrix(z) if kind is None else poincare_dist_matrix(z, c)
        M = class_mean_matrix(D, yl, k); mats.append(M)
        np.savetxt(os.path.join(a.out_dir, f"class_distance_{key}.csv"), M, delimiter=",", header=",".join(classes), comments="", fmt="%.5f")
    short = {"euclid_cosine": ("Euclidean — cosine distance, 768-d", "cosine distance"),
             "poincare": ("Hyperbolic, hard clip — Poincaré distance", "hyperbolic distance"),
             "poincare_free": ("Hyperbolic, free radius — Poincaré distance", "hyperbolic distance")}
    fig, axes = plt.subplots(1, nb, figsize=(6.8 * nb, 6.6))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.07 if nb > 2 else 0.10, right=0.97, top=0.80, bottom=0.19, wspace=0.55 if nb > 2 else 0.42)
    fig.text(0.03, 0.955, "How far apart are the classes?  Mean pairwise distance between test embeddings", fontsize=15, fontweight="semibold", color=INK)
    fig.text(0.03, 0.915, "Darker = farther apart. Diagonal = spread within a class. Classes in alphabetical order (no hierarchy assumed). Each panel has its own scale, starting at its smallest value (not zero).",
             fontsize=9.5, color=INK2)
    for ax, M, (key, z, ttl, kind) in zip(axes, mats, sets):
        t, unit = short[key]
        im = ax.pcolormesh(M, cmap=CMAP, vmin=M.min(), vmax=M.max(), edgecolors=SURFACE, linewidth=2)
        ax.set_xlim(0, k); ax.set_ylim(k, 0); ax.set_aspect("equal")
        ax.set_xticks(np.arange(k) + 0.5); ax.set_yticks(np.arange(k) + 0.5)
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9, color=INK2)
        ax.set_yticklabels(classes, fontsize=9, color=INK2)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(t, fontsize=10.5, fontweight="semibold", color=INK, loc="left", pad=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(unit, fontsize=9, color=INK2); cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    rp = hyp_radius(zp, c)
    fig.text(0.03, 0.035, f"Hard-clip head: all 1,000 embeddings lie on one shell (hyperbolic radius {rp.min():.2f} to {rp.max():.2f}), so its distances depend on direction only."
             + ("  Free-radius head: radius also contributes." if a.poincare_free else ""), fontsize=9.5, color=INK2)
    f2 = os.path.join(a.out_dir, "class_distance_heatmaps.png")
    fig.savefig(f2, dpi=140); plt.close(fig)

    written = [f1, f2]
    # ---------------- figure 3: radius per class (free-radius head only) ----------------
    if a.poincare_free:
        rf = hyp_radius(zf, c)
        f3 = os.path.join(a.out_dir, "radius_by_class.png")
        stats = radius_figure(rf, yl, classes, f3, a.label)
        with open(os.path.join(a.out_dir, "radius_by_class.csv"), "w") as fh:
            fh.write("class,n,p5,p25,median,p75,p95\n")
            for st in stats:
                fh.write(f"{st[0]},{st[1]}," + ",".join(f"{v:.4f}" for v in st[2:]) + "\n")
        written.append(f3)
        print(f"free-radius head: hyperbolic radius mean {rf.mean():.3f} std {rf.std():.3f} range {rf.min():.3f}..{rf.max():.3f}")
    print("wrote", *written)


if __name__ == "__main__":
    main()
