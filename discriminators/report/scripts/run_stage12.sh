#!/bin/bash
# Stage 1 (Riemannian prototypes, learnable curvature) and stage 2 (MLP heads, Euclidean control vs hyperbolic).
# Frozen features, no mixup, runnorm target norm 1, tau 0.1, c init 1, soft_t 1 / soft_mix 0.8 / depth_w 5 for "treedepth".
# 10 hold-outs x 3 seeds, dims 64 and 4. One process at a time. Resumable: a (dim, head, cfg) log is only kept once it ends with "CELL DONE".
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/stage12
CACHE=/store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz
for dim in 64 4; do
for hd in poincare euclid poincare_riem poincare_riem_lc euclid_mlp poincare_mlp poincare_mlp_riem_lc; do
for cfg in "base 0 0" "treedepth 1 5"; do set -- $cfg
f=$OUT/results_dim${dim}_${hd}_$1.log
if [ -f $f ] && tail -1 $f | grep -q "CELL DONE"; then continue; fi
rm -f $f.part
for h in 0 1 2 3 4 5 6 7 8 9; do
echo "## holdout=$h head=$hd cfg=$1" >> $f.part
$PY frozen_head_harness.py --cache $CACHE --dim $dim --norm_mode runnorm --target_norm 1 --clip_r 4 --heads $hd --soft_t $2 --soft_mix 0.8 --depth_w $3 --holdout $h --quiet --seeds 0 1 2 2>&1 | grep -E "placement|test@bestval|rror|Trace|non-finite" >> $f.part
done
echo "CELL DONE" >> $f.part; mv $f.part $f
echo "$(date +%F_%T) dim $dim $hd $1 DONE" >> $OUT/progress.log
done; done; done
echo "$(date +%F_%T) ALL DONE" >> $OUT/progress.log
