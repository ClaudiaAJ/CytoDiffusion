#!/bin/bash
# New head types (hyperplane/MLR heads, product-manifold heads) vs matched Euclidean controls. 10 classes, flat labels, 900 steps, 5 seeds.
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/hyp_accuracy2
for cache in /store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz /store/CIA/ca662/CytoDiffusion_runs/hyp_accuracy/ft_stock.npz; do
for hd in linear euclid poincare euclid_mlr poincare_mlr prod_ee prod_eh prod_hh; do
echo "## cache=$(basename $cache) head=$hd"
$PY frozen_head_harness.py --cache $cache --heads $hd --norm_mode runnorm --target_norm 1 --clip_r 4 --steps 900 --quiet --seeds 0 1 2 3 4 2>&1 | grep -E "test@bestval|rror|Trace|non-finite"
done; done > $OUT/heads2_results.log 2>&1
echo DONE >> $OUT/heads2_results.log
