#!/bin/bash
# Prototype-movement check: does the stage-1 null result come from the prototypes barely moving?
# heads {poincare (tangent prototypes, AdamW), poincare_riem (Riemannian prototypes), euclid} x prototype lr multiplier {1,10,100}
# hierarchy config (soft labels + depth loss), 10 hold-outs x 3 seeds, dims 64 and 4, frozen features, no mixup, c=1.
# One process at a time; resumable (a log counts only if it ends with "CELL DONE").
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/proto_lr
CACHE=/store/CIA/ca662/CytoDiffusion_data/features/vitb384_plain.npz
for dim in 64 4; do
for hd in poincare poincare_riem euclid; do
for m in 1 10 100; do
f=$OUT/results_dim${dim}_${hd}_x${m}.log
if [ -f $f ] && tail -1 $f | grep -q "CELL DONE"; then continue; fi
rm -f $f.part
for h in 0 1 2 3 4 5 6 7 8 9; do
echo "## holdout=$h head=$hd cfg=treedepth" >> $f.part
$PY frozen_head_harness.py --cache $CACHE --dim $dim --norm_mode runnorm --target_norm 1 --clip_r 4 --heads $hd --lr_proto_mult $m --soft_t 1 --soft_mix 0.8 --depth_w 5 --holdout $h --quiet --seeds 0 1 2 2>&1 | grep -E "placement|test@bestval|rror|Trace|non-finite" >> $f.part
done
echo "CELL DONE" >> $f.part; mv $f.part $f
echo "$(date +%F_%T) dim $dim $hd x$m DONE" >> $OUT/progress.log
done; done; done
echo "$(date +%F_%T) ALL DONE" >> $OUT/progress.log
