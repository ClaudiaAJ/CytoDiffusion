"""Approved lineage tree for the 10 CytoData classes (leaves-only; blast and ig attach as leaves).

Usage:
    python class_tree.py            # prints tree, parent table, tree-distance matrix
    from class_tree import PARENT, LEAVES, tree_distance_matrix
"""
import numpy as np

# child -> parent
PARENT = {
    "haematopoietic": "root",
    "artefact": "root",
    "blast": "haematopoietic",
    "lymphoid": "haematopoietic",
    "myeloid": "haematopoietic",
    "lymphocyte": "lymphoid",
    "gm_lineage": "myeloid",            # granulocyte-monocyte lineage
    "me_lineage": "myeloid",            # megakaryocyte-erythroid lineage
    "granulocytes": "gm_lineage",
    "monocyte": "gm_lineage",
    "ig": "granulocytes",
    "neutrophil": "granulocytes",
    "eosinophil": "granulocytes",
    "basophil": "granulocytes",
    "erythroblast": "me_lineage",
    "platelet": "me_lineage",
}

LEAVES = ["artefact", "basophil", "blast", "eosinophil", "erythroblast",
          "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]  # folder (alphabetical) order


def path_to_root(n):
    p = [n]
    while n != "root":
        n = PARENT[n]
        p.append(n)
    return p


def tree_distance(a, b):
    """Number of edges between nodes a and b."""
    pa, pb = path_to_root(a), path_to_root(b)
    common = next(x for x in pa if x in pb)
    return pa.index(common) + pb.index(common)


def tree_distance_matrix(classes=LEAVES):
    return np.array([[tree_distance(a, b) for b in classes] for a in classes], dtype=float)


def depth(n):
    return len(path_to_root(n)) - 1


if __name__ == "__main__":
    print("parent table (child -> parent):")
    for c, p in PARENT.items():
        print(f"  {c:15s} -> {p}")
    print("\nleaf depths:", {c: depth(c) for c in LEAVES})
    D = tree_distance_matrix()
    w = 6
    print("\ntree distance (edges):")
    print(" " * 13 + "".join(f"{c[:5]:>{w}s}" for c in LEAVES))
    for c, row in zip(LEAVES, D):
        print(f"{c:13s}" + "".join(f"{int(v):>{w}d}" for v in row))
