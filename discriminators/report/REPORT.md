# Euclidean vs Poincaré-ball embedding heads on CytoData: setup, methods, results

Data: public CytoData blood-cell images (BioStudies S-BSST2156), 10 classes, 1,000 test images
(artefact 175, basophil 27, blast 20, eosinophil 47, erythroblast 38, ig 44, lymphocyte 213, monocyte 98, neutrophil 319, platelet 19).
Backbone: ViT-B/16-384 (timm `augreg_in21k_ft_in1k`), identical across all arms. Only the embedding head changes.

## 0. Common setup
**Heads** (all: non-affine LayerNorm -> trained linear projection to dim d -> norm control -> map -> prototype logits)
- Linear: stock classifier on 768-d pooled features.
- Euclidean bottleneck: same pipeline, identity map, Euclidean distance, logits = -dist(z, prototype)/tau.
- Poincaré: same pipeline, exp map at origin, curvature c=1 fixed, logits = -d_hyp(z, prototype)/tau, prototypes = tangent vectors mapped by exp0.
- Norm control: `runnorm` (divide by detached running mean norm x target; target 1 -> mean hyperbolic radius ~2 because d(0, exp0 v) = 2|v|), `batchnorm`, or hard clip (all points on one shell, radius 4).

**Two training regimes**
- Frozen: ViT features cached once (80 s), head trained on 768-d vectors, no mixup. Fast; used for sweeps, hold-outs, 10-seed replications.
- Fine-tuned: LP-FT (5 probe epochs, then unfreeze), 20 epochs, mixup 0.3, RandAugment. Same loop as `discriminators/train_and_test.py`.

**Stock baseline** (original script, ViT only, 384 px, 1 seed): accuracy 88.6%, balanced accuracy 85.9%, macro-F1 86.2; lowest recall ig 52.3%.
Paper table, ViT-B/16 row: accuracy 84.4%, sensitivity 81.1%. The 4-point gap is not explained (test split, weights, seeds not checked).

**Class tree** (`discriminators/class_tree.py`): root -> {artefact, haematopoietic}; haematopoietic -> {blast, lymphoid -> lymphocyte, myeloid -> {gm_lineage -> {granulocytes -> {ig, neutrophil, eosinophil, basophil}, monocyte}, me_lineage -> {erythroblast, platelet}}}.
Leaf depths: artefact 1, blast 2, lymphocyte 3, monocyte/erythroblast/platelet 4, neutrophil/eosinophil/basophil/ig 5. Tree distance = number of edges.

**Class tree figure**

![tree](figures/00_class_tree.png)
*Fig 0: the 10 labels as leaves of the lineage tree; vertical position = depth (used by the depth loss), tree distance = edges between leaves.*

**Metrics**
- Accuracy, balanced accuracy, AUROC for out-of-distribution scores (max-logit, MSP, distance to origin).
- rho: Spearman correlation between class-centroid distances (tangent-space mean) in the head and tree distances over all class pairs.
- Radius-depth correlation: correlation of class mean radius with leaf depth.
- Held-out placement: train on 9 classes, embed the held-out class, find the nearest known class by head distance; "excess" = its tree distance minus the best possible tree distance (0 = nearest known class is a tree neighbour); radius ratio = held-out centroid radius / mean known centroid radius.

---
# A. Flat-hierarchy experiments (one-hot labels, cross-entropy, no tree information)

## A1. Fine-tuned, 10 classes, 3 seeds (64-d)
| head | accuracy % | balanced acc % |
|---|---|---|
| Linear (768-d) | 87.5 | 84.7 +- 1.7 |
| Euclidean bottleneck | 87.5 +- 0.8 | 85.7 +- 2.4 |
| Poincaré, running-norm | 87.0 +- 0.6 | 84.7 +- 1.7 |
| Poincaré, hard clip | 87.0 | 83.3 +- 2.6 |

Differences between heads are within seed spread. Tree consistency without a tree loss: rho <= 0.10 for both geometries; radius-depth correlation about 0.
(An earlier "Poincaré 1-3 points lower" statement came from seed 0 only and was retracted.)

![embeddings](figures/A1_embeddings_finetuned_flat.png)
*Fig A1: test-set embeddings (2-D PCA, one panel per class in blue) for the linear, hard-clip Poincaré and running-norm Poincaré heads. Classes form distinct clusters in all three. In the hard-clip panel all points lie on one hyperbolic radius (4.00); in the running-norm panel the radius ranges from 1.34 to 2.43 (std 0.21).*

![distances](figures/A2_class_distance_heatmaps.png)
*Fig A2: class-centroid distance matrices, Euclidean/cosine vs Poincaré.*

![radius](figures/A3_radius_by_class.png)
*Fig A3: radius per class (Poincaré, running-norm). Radius varies by class but is unrelated to tree depth when no depth loss is used.*

## A2. Frozen features, hold-one-class-out, flat labels (3 seeds, 64-d)
Train on 9 classes, embed the 10th.
- Accuracy on the remaining classes: Poincaré 78.4, Euclidean 78.6; balanced accuracy 70.3 vs 70.6.
- Mean excess tree distance of the nearest known class (over 10 hold-outs): 1.30 for both heads; nearest known class is a best-possible tree neighbour in 4/10 hold-outs for both.
- rho: Poincaré 0.14, Euclidean 0.18.
- Held-out radius / known radius: Poincaré 0.92, Euclidean 0.86.
- Pattern by class: held-out neutrophil, eosinophil, basophil, monocyte land next to a granulocyte (excess 0). Held-out artefact, lymphocyte, blast, ig, platelet land next to a class 2-5 edges farther than the best option (platelet's nearest known class is artefact, lymphocyte or basophil at tree distance 5, while erythroblast is at 2).

## A3. Radius of a class when it is held out (frozen, running-norm, 10 seeds)
Mean hyperbolic radius of a class's test images: trained-on 1.96 vs held-out 1.71 (class-averaged), difference -0.24. Held-out radius was lower in 10/10 seeds for artefact, basophil, eosinophil, lymphocyte, monocyte, neutrophil, platelet; 9/10 erythroblast; 7/10 blast; 6/10 ig. Known classes sit at about 1.96-2.00.

## A4. Low-dimensional heads, frozen, flat labels (hold-outs 10 classes, 3 seeds)
| dim | Poincaré acc / bal | Euclid acc / bal | rho P / E |
|---|---|---|---|
| 2 | 49.1 / 29.0 | 53.4 / 34.9 | 0.11 / 0.06 |
| 4 | 66.7 / 39.7 | 70.3 / 52.2 | 0.16 / 0.17 |
| 8 | 75.0 / 59.8 | 76.1 / 63.9 | 0.20 / 0.21 |
| 64 | 78.4 / 70.3 | 78.6 / 70.6 | 0.14 / 0.18 |

Accuracy falls steeply below 8 dimensions in both geometries; Euclidean is equal or higher at every dimension. Mean excess at dim 4: Poincaré 1.10, Euclidean 1.30.

---
# B. Hierarchy experiments

## B0. Methods
1. Tree-aware soft labels: target q_k proportional to exp(-T[y,k]/t), t = 1, mixed with the one-hot label (weight 0.8 on q).
2. Depth-radius loss: 5 x mean((r / mean(r) - depth / mean(depth))^2), with the mean radius detached. With mixup, depth is interpolated by the mixing weight.
3. Tried and dropped: a prototype tree loss. It fitted the prototypes (rho 0.96) but not the embeddings (rho 0.10-0.18) and lowered accuracy.
"treedepth" below = soft labels + depth loss.

## B1. Frozen features, all 10 classes trained (64-d, 3 seeds)
- Soft labels raise embedding-tree consistency rho from about 0.17 to about 0.47.
- The depth loss makes class radius follow depth (correlation 0.84-0.90) in both geometries.
- Cost: balanced accuracy 70.3 -> 65.3 (Poincaré), 70.6 -> 63.3 (Euclidean) in the hold-out runs.
- Euclidean rho is higher than Poincaré in 10/10 classes' hold-out runs (0.50 vs 0.37 on average at 64-d).

## B2. Frozen features, hold-one-class-out with hierarchy (3 seeds per cell)
| dim | head | config | rho | held-out radius ratio | mean excess | hold-outs with excess 0 (of 10) |
|---|---|---|---|---|---|---|
| 64 | Poincaré | flat | 0.14 | 0.92 | 1.30 | 4 |
| 64 | Poincaré | treedepth | 0.37 | 0.98 | 1.27 | 4 |
| 64 | Euclid | flat | 0.18 | 0.86 | 1.30 | 4 |
| 64 | Euclid | treedepth | 0.50 | 0.97 | 1.20 | 4 |
| 4 | Poincaré | flat | 0.16 | 0.96 | 1.10 | 2 |
| 4 | Poincaré | treedepth | 0.32 | 0.99 | 0.73 | 5 |
| 4 | Euclid | flat | 0.17 | 0.75 | 1.30 | 3 |
| 4 | Euclid | treedepth | 0.49 | 0.95 | 1.40 | 2 |

![placement](figures/B1_holdout_placement_heatmap.png)
*Fig B1: held-out placement table (excess tree distance per held-out class and head/config).*

At 64-d, held-out placement is the same for both geometries (excess 1.2-1.3). At dim 4 with hierarchy, Poincaré has the lowest mean excess (0.73 vs 1.40).

## B3. Dim-4 replication and merge check (frozen, 10 seeds, erythroblast/platelet)
- Poincaré treedepth, dim 4: the nearest known class was the best-possible neighbour in 8/10 seeds (erythroblast) and 7/10 (platelet); Euclidean: 3/10 and 1/10.
- Merge check (all 10 classes trained, dim 4): erythroblast and platelet recall is 17% and 2% for Poincaré treedepth, 39% and 11% for Euclid treedepth (flat: 35%/10% Poincaré, 71%/36% Euclid). The erythroblast-platelet pair ranks within the 5 closest of 45 class pairs in 7/10 seeds for Poincaré treedepth (Euclid treedepth 2/10, flat Euclid 1/10, flat Poincaré 2/10).
- Reading: the Poincaré dim-4 placement coincides with these two classes being merged in the embedding and rarely recognised.

![embeddings holdout](figures/B2_holdout_embeddings_dim4.png)
*Fig B2: frozen dim-4 heads, treedepth, seed 0; held-out class in blue (X = mean), its lineage neighbour in orange, other known classes gray. Poincaré: held-out erythroblast overlaps platelet (nearest known: platelet, tree distance 2); held-out platelet sits near erythroblast. Euclidean: held-out erythroblast nearest to lymphocyte (tree distance 5), held-out platelet nearest to artefact (5). Seed 0 is one run per panel; across 10 seeds Euclid matched the neighbour in 3/10 and 1/10.*

## B4. Fine-tuned, hierarchy, hold-out erythroblast / platelet (balanced design)
| dim | held out | head | seeds | rho | radius ratio | acc % | balanced % |
|---|---|---|---|---|---|---|---|
| 4 | erythroblast | Euclid | 3 | 0.52 | 0.73 | 81.0 | 61.3 |
| 4 | erythroblast | Poincaré | 3 | 0.33 | 0.66 | 81.5 | 58.2 |
| 4 | platelet | Euclid | 3 | 0.39 | 0.79 | 79.9 | 58.6 |
| 4 | platelet | Poincaré | 3 | 0.27 | 0.76 | 81.2 | 62.3 |
| 8 | erythroblast | Euclid | 2 | 0.57 | 0.69 | 85.2 | 76.8 |
| 8 | erythroblast | Poincaré | 2 | 0.30 | 0.76 | 86.6 | 80.7 |
| 8 | platelet | Euclid | 1 | 0.42 | 0.67 | 83.7 | 77.1 |
| 8 | platelet | Poincaré | 2 | 0.23 | 0.70 | 85.4 | 78.1 |

- The dim-4 frozen result (Poincaré neighbour match) did not appear when fine-tuned: 0/3 seeds with the best-possible neighbour for either head on either class. Held-out erythroblast's nearest known class is artefact in all 6 runs (excess 3). Held-out platelet's nearest known class is lymphocyte, artefact, ig or blast (excess 3, or 2 when blast).
- Held-out radius ratio is 0.66-0.79 in both geometries (held-out classes sit closer to the origin than known classes).
- rho is higher for Euclidean in all four paired cells.
- Dim 8 is incomplete (cancelled): seed 1 platelet Euclid and all of seed 2 are missing.

## B5. Mixup control (frozen features with feature-space mixup 0.3)
Setup: the frozen hold-out grid was re-run with mixup alpha 0.3 on the cached feature vectors, mixing the one-hot or soft labels and the depth target by the same weight (same as the fine-tuned loop). 10 hold-outs x 2 heads x {flat, hierarchy} x 3 seeds, at 64-d and 4-d, otherwise identical to B2 (300 steps). Dim 8 and dim 2 were not re-run.

| dim | head | config | rho | acc % | balanced % | mean excess | held-out radius ratio |
|---|---|---|---|---|---|---|---|
| 64 | Poincaré | flat | 0.14 -> 0.14 | 78.4 -> 77.1 | 70.3 -> 67.0 | 1.30 -> 1.60 | 0.92 -> 0.95 |
| 64 | Poincaré | hierarchy | 0.37 -> 0.35 | 77.9 -> 76.8 | 65.3 -> 61.9 | 1.27 -> 0.90 | 0.98 -> 0.98 |
| 64 | Euclid | flat | 0.18 -> 0.15 | 78.6 -> 77.8 | 70.5 -> 68.1 | 1.30 -> 1.70 | 0.86 -> 0.93 |
| 64 | Euclid | hierarchy | 0.50 -> 0.49 | 77.7 -> 76.5 | 63.3 -> 60.4 | 1.20 -> 1.00 | 0.97 -> 0.97 |
| 4 | Poincaré | flat | 0.16 -> 0.16 | 66.7 -> 65.1 | 39.7 -> 35.6 | 1.10 -> 0.87 | 0.96 -> 0.97 |
| 4 | Poincaré | hierarchy | 0.32 -> 0.31 | 63.5 -> 63.1 | 31.7 -> 30.1 | 0.73 -> 0.87 | 0.99 -> 0.99 |
| 4 | Euclid | flat | 0.17 -> 0.13 | 70.3 -> 69.9 | 52.2 -> 49.5 | 1.30 -> 1.33 | 0.75 -> 0.81 |
| 4 | Euclid | hierarchy | 0.49 -> 0.45 | 66.3 -> 65.6 | 38.4 -> 37.4 | 1.40 -> 1.07 | 0.95 -> 0.97 |

(each cell: without mixup -> with mixup; excess is averaged over seeds here, not taken per class as in B2)

- Mixup lowers accuracy by 0.4-1.6 points and balanced accuracy by 1-3.5 points. Rho changes by at most 0.04.
- Euclidean rho stays above Poincaré rho under hierarchy training (0.49 vs 0.35 at 64-d, 0.45 vs 0.31 at 4-d).
- At dim 4 with hierarchy, the Poincaré-vs-Euclid gap in mean excess falls from 0.67 (0.73 vs 1.40) to 0.20 (0.87 vs 1.07).
- At 64-d, mean excess changes by 0.2-0.4 in either direction depending on the cell. No significance test was run.
- Number of hold-out classes whose nearest known class is a best-possible tree neighbour (in at least 2 of 3 seeds), without -> with mixup: dim 4 Poincaré hierarchy 7 -> 6, Euclid hierarchy 4 -> 6; dim 64 Poincaré hierarchy 4 -> 6, Euclid hierarchy 4 -> 5.
- Feature-space mixup is not image mixup, and RandAugment is not reproduced, so the arms are still not identical.

Files: `holdout_mixup/results_dim{64,4}.log`, `holdout_mixup/run.sh`; harness flag `--mixup 0.3`.

## B6. Curvature sweep (frozen features, no mixup)
Setup: Poincaré head with c in {0.1, 0.25, 0.5, 1, 2, 4}, everything else identical to B2 (runnorm, target norm 1, tau 0.1, 300 steps, soft labels + depth loss for the hierarchy config), 10 hold-outs x {flat, hierarchy} x 3 seeds, at 64-d and 4-d. The Euclidean head is the c -> 0 limit. The c=1 rerun reproduces the original c=1 numbers exactly.
Before this sweep, `radius()` and `log0()` in `train_and_test_hyperbolic.py` were corrected to use the ball radius 1/sqrt(c) (they assumed 1). At c=1 results are unchanged; earlier results are unaffected.
Columns: rho, accuracy %, balanced accuracy %, mean excess, number of hold-out classes (of 10) with a best-possible nearest neighbour in at least 2 of 3 seeds, held-out radius ratio.

**64-d**

| c | config | rho | acc | bal | excess | #best | radius ratio |
|---|---|---|---|---|---|---|---|
| 0 (Euclid) | flat | 0.18 | 78.6 | 70.5 | 1.30 | 5 | 0.86 |
| 0 (Euclid) | hier. | 0.50 | 77.7 | 63.3 | 1.20 | 4 | 0.97 |
| 0.1 | flat | 0.20 | 77.9 | 69.1 | 1.30 | 5 | 0.95 |
| 0.1 | hier. | 0.48 | 77.2 | 62.8 | 1.13 | 5 | 0.99 |
| 0.25 | flat | 0.19 | 78.0 | 69.2 | 1.30 | 5 | 0.95 |
| 0.25 | hier. | 0.47 | 77.3 | 63.1 | 1.13 | 5 | 0.98 |
| 0.5 | flat | 0.17 | 78.1 | 69.7 | 1.30 | 5 | 0.94 |
| 0.5 | hier. | 0.46 | 77.5 | 63.9 | 1.33 | 4 | 0.98 |
| 1 | flat | 0.14 | 78.4 | 70.3 | 1.30 | 5 | 0.92 |
| 1 | hier. | 0.37 | 77.9 | 65.3 | 1.27 | 4 | 0.98 |
| 2 | flat | 0.10 | 78.5 | 70.9 | 1.23 | 5 | 0.89 |
| 2 | hier. | 0.20 | 78.1 | 65.7 | 1.00 | 5 | 0.97 |
| 4 | flat | 0.05 | 78.5 | 70.1 | 1.37 | 5 | 0.86 |
| 4 | hier. | 0.04 | 77.2 | 63.0 | 1.63 | 3 | 0.96 |

**4-d**

| c | config | rho | acc | bal | excess | #best | radius ratio |
|---|---|---|---|---|---|---|---|
| 0 (Euclid) | flat | 0.17 | 70.3 | 52.2 | 1.30 | 4 | 0.75 |
| 0 (Euclid) | hier. | 0.49 | 66.3 | 38.4 | 1.40 | 4 | 0.95 |
| 0.1 | flat | 0.17 | 68.1 | 48.9 | 1.13 | 4 | 0.83 |
| 0.1 | hier. | 0.47 | 65.0 | 39.1 | 1.23 | 4 | 0.98 |
| 0.25 | flat | 0.18 | 67.8 | 47.3 | 1.20 | 4 | 0.90 |
| 0.25 | hier. | 0.43 | 65.5 | 35.9 | 1.07 | 6 | 0.99 |
| 0.5 | flat | 0.18 | 67.5 | 44.0 | 1.30 | 4 | 0.91 |
| 0.5 | hier. | 0.37 | 64.9 | 34.0 | 0.87 | 6 | 0.99 |
| 1 | flat | 0.16 | 66.7 | 39.7 | 1.10 | 6 | 0.96 |
| 1 | hier. | 0.32 | 63.5 | 31.7 | 0.73 | 7 | 0.99 |
| 2 | flat | 0.16 | 64.5 | 33.9 | 1.13 | 4 | 0.97 |
| 2 | hier. | 0.26 | 62.0 | 28.2 | 0.73 | 6 | 0.99 |
| 4 | flat | 0.15 | 61.9 | 30.2 | 1.17 | 5 | 0.97 |
| 4 | hier. | 0.20 | 59.9 | 25.9 | 1.13 | 5 | 0.99 |

- Under hierarchy training, rho decreases as curvature increases: 0.48 -> 0.04 (64-d) and 0.47 -> 0.20 (4-d) from c=0.1 to c=4. The Euclidean values (0.50, 0.49) continue the same curve. Flat rho is 0.15-0.18 at 4-d for all c and falls from 0.20 to 0.05 at 64-d.
- Accuracy at 64-d is 77.2-78.5% for all c. At 4-d, accuracy falls from 68.1% (c=0.1) to 61.9% (c=4) for flat labels and from 65.0% to 59.9% for hierarchy; balanced accuracy falls from 48.9 to 30.2 (flat) and from 39.1 to 25.9 (hierarchy).
- Mean excess at 64-d has no clear trend with c (hierarchy: 1.13, 1.13, 1.33, 1.27, 1.00, 1.63; Euclid 1.20).
- At 4-d with hierarchy, mean excess is lowest at c=1 and c=2 (0.73), 0.87 at c=0.5, against 1.40 for Euclid and 1.13 at c=4. The number of best-neighbour hold-outs is 6, 7 and 6 at c=0.5, 1, 2 (Euclid 4). This is an average over 10 classes with 3 seeds each; B5 showed the c=1 gap to Euclid shrinks with mixup (0.87 vs 1.07).
- The held-out radius ratio is 0.96-0.99 for all hierarchy runs, including Euclid (0.95-0.97).
- No significance test was run.

Files: `curvature_sweep/results_dim{64,4}_c{c}.log`, `curvature_sweep/run.sh`.

---
# C. Limitations
- Frozen and fine-tuned arms still differ: B5 adds feature-space mixup to the frozen arm, but images are not mixed and RandAugment is not reproduced. Fine-tuned runs without mixup were not done (about 10 h for hold-outs at dim 4).
- No fine-tuned flat hold-out control at dim 4/8; no contrastive loss; curvature is swept only on frozen features (B6), fixed at c=1 in the fine-tuned runs.
- The paper's CytoDiffusion backbone is not used. The stock ViT baseline is 4 points above the paper's ViT row, cause not checked.
- Seeds: 3 for most cells, 10 for the frozen dim-4 replication and the radius check; dim-8 fine-tuned rows have 1-2 seeds.
- Tree is my draft (approved by you); distances depend on it.

# D. Files
Everything is under `/store/CIA/ca662/CytoDiffusion_runs/`: `fullloop/`, `holdout_tree/`, `holdout_lowdim/`, `radius_control/`, `balanced_design/` (`holdout_results.csv`), `figures_holdout_placement/` (table CSV), `stock_baseline/`; code in `discriminators/`.
