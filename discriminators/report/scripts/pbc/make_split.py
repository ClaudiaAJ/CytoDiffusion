"""Deduplicated, stratified train/val/test split of PBC (Acevedo et al. 2020) as ImageFolder symlinks.

- Exact duplicates (same md5) are reduced to one copy; neutrophil/BNE_191112.jpg (identical to
  eosinophil/EO_225902.jpg) is dropped, keeping the eosinophil label.
- Split 70/10/20 per class (CytoData proportions: 3500/494/1000), seed 0.

    python make_split.py
"""
import collections, hashlib, os, random

SRC = "/store/CIA/ca662/CytoDiffusion_data/PBC/PBC_dataset_normal_DIB"
DST = "/store/CIA/ca662/CytoDiffusion_data/PBC/split"
FRAC = {"train": 0.7, "val": 0.1, "test": 0.2}

by_hash = collections.defaultdict(list)
for cls in sorted(os.listdir(SRC)):
    for f in sorted(os.listdir(os.path.join(SRC, cls))):
        if f.startswith("."):  # neutrophil/.DS_169665.jpg is a macOS .DS_Store file, not an image
            continue
        p = os.path.join(SRC, cls, f)
        by_hash[hashlib.md5(open(p, "rb").read()).hexdigest()].append((cls, f))

keep = collections.defaultdict(list)
dropped = []
for h, items in by_hash.items():
    if len({c for c, _ in items}) > 1:  # cross-class duplicate: keep the non-neutrophil copy
        items = sorted(items, key=lambda t: t[0] == "neutrophil")
    keep[items[0][0]].append(items[0][1])
    dropped += items[1:]
print("dropped duplicates:", len(dropped), dropped)

rng = random.Random(0)
for cls, files in sorted(keep.items()):
    files = sorted(files); rng.shuffle(files)
    n = len(files); ntr = round(FRAC["train"] * n); nva = round(FRAC["val"] * n)
    parts = {"train": files[:ntr], "val": files[ntr:ntr + nva], "test": files[ntr + nva:]}
    for split, fs in parts.items():
        d = os.path.join(DST, split, cls); os.makedirs(d, exist_ok=True)
        for f in fs:
            os.symlink(os.path.join(SRC, cls, f), os.path.join(d, f))
    print(f"{cls:13s} total {n:5d}  " + "  ".join(f"{s} {len(v)}" for s, v in parts.items()))
