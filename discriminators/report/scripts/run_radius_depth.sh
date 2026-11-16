#!/bin/bash
# Re-run with radius-vs-depth correlation logged: (a) scrambled tree x3 shuffles, (b) flat and approved-tree reference.
# Same settings as scrambled_tree/, holdout_tree/ (64-d) and holdout_lowdim/ (4-d): runnorm, target norm 1, tau 0.1, c=1, no mixup,
# 10 hold-outs x {poincare,euclid} x 3 seeds. One process at a time.
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/radius_depth
CACHE=/store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz
for dim in 64 4; do
for cfg in "flat 0 0 0" "approved 1 5 0" "scramble1 1 5 1" "scramble2 1 5 2" "scramble3 1 5 3"; do set -- $cfg
for h in 0 1 2 3 4 5 6 7 8 9; do for hd in poincare euclid; do
echo "## holdout=$h head=$hd cfg=$1"
$PY frozen_head_harness.py --cache $CACHE --dim $dim --norm_mode runnorm --target_norm 1 --clip_r 4 --heads $hd --soft_t $2 --soft_mix 0.8 --depth_w $3 --scramble $4 --holdout $h --quiet --seeds 0 1 2 2>&1 | grep -E "placement|test@bestval|rror|Trace|non-finite"
done; done >> $OUT/results_dim${dim}_$1.log 2>&1
echo "$(date +%F_%T) dim $dim $1 DONE" >> $OUT/progress.log
done; done
echo "$(date +%F_%T) ALL DONE" >> $OUT/progress.log
