#!/bin/bash
# control: soft labels + depth loss trained on a SCRAMBLED tree (classes re-assigned to leaves, 3 different derangements),
# same settings as the approved-tree "treedepth" runs in holdout_tree (64-d) / holdout_lowdim (4-d). No mixup, c=1.
# Placement metrics are measured against the TRUE tree. One process at a time.
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/scrambled_tree
for dim in 64 4; do
for sc in 1 2 3; do
for h in 0 1 2 3 4 5 6 7 8 9; do for hd in poincare euclid; do
echo "## holdout=$h head=$hd cfg=scramble$sc"
$PY frozen_head_harness.py --cache /store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz --dim $dim --norm_mode runnorm --target_norm 1 --clip_r 4 --heads $hd --soft_t 1 --soft_mix 0.8 --depth_w 5 --scramble $sc --holdout $h --quiet --seeds 0 1 2 2>&1 | grep -E "scrambled|placement|test@bestval|rror|Trace|non-finite"
done; done >> $OUT/results_dim${dim}_scramble${sc}.log 2>&1
echo "$(date +%F_%T) dim $dim scramble $sc DONE" >> $OUT/progress.log
done; done
echo "$(date +%F_%T) ALL DONE" >> $OUT/progress.log
