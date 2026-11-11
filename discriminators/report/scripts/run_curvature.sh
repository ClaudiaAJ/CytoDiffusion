#!/bin/bash
# curvature sweep, frozen features, NO mixup (identical settings to holdout_tree / holdout_lowdim, which are the c=1 reference).
# Poincare head only; c in {0.1,0.25,0.5,1,2,4} (c=1 is a reproducibility check); dims 64 and 4; 10 hold-outs x {flat, treedepth} x 3 seeds.
# one process at a time.
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/curvature_sweep
for dim in 64 4; do
for c in 0.1 0.25 0.5 1 2 4; do
for h in 0 1 2 3 4 5 6 7 8 9; do for cfg in "base 0 0" "treedepth 1 5"; do set -- $cfg
echo "## holdout=$h head=poincare cfg=$1"
$PY frozen_head_harness.py --cache /store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz --dim $dim --c $c --norm_mode runnorm --target_norm 1 --clip_r 4 --heads poincare --soft_t $2 --soft_mix 0.8 --depth_w $3 --holdout $h --quiet --seeds 0 1 2 2>&1 | grep -E "placement|test@bestval|rror|Trace|non-finite"
done; done >> $OUT/results_dim${dim}_c${c}.log 2>&1
echo "$(date +%F_%T) dim $dim c $c DONE" >> $OUT/progress.log
done; done
echo "$(date +%F_%T) ALL DONE" >> $OUT/progress.log
