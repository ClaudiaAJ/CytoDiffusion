"""Summarise holdout_dim*.log: per (dim, head, cfg) and per held-out class, mean rho, excess tree distance, #best, radius ratio.

    python summarise_holdout.py
"""
import collections, re, sys
import numpy as np

O = "/store/CIA/ca662/CytoDiffusion_runs/pbc"
pat = re.compile(r"spearman\(emb dist, tree dist\)=([-\d.]+) \| held '(\w+)'.*?-> tree dist (\d+) \(best possible (\d+)\) \| held radius/known mean ([\d.]+)")
for dim in (64, 4):
    rows = collections.defaultdict(list)
    hd = cfg = None
    for line in open(f"{O}/holdout_dim{dim}.log"):
        m = re.match(r"## holdout=\d+ head=(\w+) cfg=(\w+)", line)
        if m:
            hd, cfg = m.groups(); continue
        m = pat.search(line)
        if m:
            rho, held, nt, mp, rr = m.groups()
            rows[(hd, cfg, held)].append((float(rho), int(nt) - int(mp), int(nt) == int(mp), float(rr)))
    classes = sorted({k[2] for k in rows})
    print(f"\n=== dim {dim} (3 seeds per cell) ===")
    print(f"{'head':9s} {'cfg':10s} {'rho':>6s} {'excess':>7s} {'#best':>7s} {'radratio':>8s}")
    for hd in ("poincare", "euclid"):
        for cfg in ("base", "treedepth"):
            a = np.array([r for c in classes for r in rows[(hd, cfg, c)]])
            print(f"{hd:9s} {cfg:10s} {a[:,0].mean():6.3f} {a[:,1].mean():7.2f} {int(a[:,2].sum()):3d}/{len(a):<3d} {a[:,3].mean():8.2f}")
    print("\nper held-out class: excess (#best/3)")
    print(f"{'class':13s}" + "".join(f"{h[:4]+'-'+c[:4]:>14s}" for h in ("poincare", "euclid") for c in ("base", "treedepth")))
    for cl in classes:
        s = f"{cl:13s}"
        for hd in ("poincare", "euclid"):
            for cfg in ("base", "treedepth"):
                a = np.array(rows[(hd, cfg, cl)])
                s += f"{a[:,1].mean():9.2f} ({int(a[:,2].sum())}/3)"
        print(s)
