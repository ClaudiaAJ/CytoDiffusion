#!/bin/bash
# Post-hoc heads on features of FINE-TUNED ViT backbones (all 10 classes, no hold-out). 5 seeds, test accuracy at the best-validation step.
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/hyp_accuracy
for bb in stock linear_s0 poincare_clip_s0; do
for cfg in "flat 0 0" "hier 1 5"; do set -- $cfg
for hd in linear euclid poincare euclid_mlp poincare_mlp poincare_riem_lc; do
echo "## backbone=$bb head=$hd cfg=$1"
$PY frozen_head_harness.py --cache $OUT/ft_$bb.npz --dim 64 --norm_mode runnorm --target_norm 1 --clip_r 4 --heads $hd --soft_t $2 --soft_mix 0.8 --depth_w $3 --quiet --seeds 0 1 2 3 4 2>&1 | grep -E "test@bestval|rror|Trace|non-finite"
done; done; done > $OUT/heads_results.log 2>&1
echo DONE >> $OUT/heads_results.log
