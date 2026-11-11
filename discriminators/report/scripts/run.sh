#!/bin/bash
# low-dim heads: dims 2, 8, 4 x all 10 hold-outs x {poincare,euclid} x {base, treedepth}, 3 seeds; one process at a time
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_VISIBLE_DEVICES=0
for dim in 2 8 4; do
for h in 0 1 2 3 4 5 6 7 8 9; do for hd in poincare euclid; do for cfg in "base 0 0" "treedepth 1 5"; do set -- $cfg
echo "## holdout=$h head=$hd cfg=$1"
$PY frozen_head_harness.py --cache /store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz --dim $dim --norm_mode runnorm --target_norm 1 --clip_r 4 --heads $hd --soft_t $2 --soft_mix 0.8 --depth_w $3 --holdout $h --quiet --seeds 0 1 2 2>&1 | grep -E "placement|test@bestval|rror|Trace"
done; done; done >> results_dim$dim.log 2>&1
echo "DIM $dim DONE" >> progress.log
done
echo "ALL DONE" >> progress.log
