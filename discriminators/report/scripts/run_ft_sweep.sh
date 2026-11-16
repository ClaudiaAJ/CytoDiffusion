#!/bin/bash
# Validation-selected sweep on the fine-tuned stock ViT features (10 classes): each geometry gets its own hyperparameter search.
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/hyp_accuracy
until grep -q "^DONE" $OUT/heads_results.log 2>/dev/null; do sleep 5; done
run() { echo "## $*"; $PY frozen_head_harness.py --cache $OUT/ft_stock.npz --norm_mode runnorm --target_norm 1 --clip_r 4 --soft_mix 0.8 --quiet --seeds 0 1 2 "$@" 2>&1 | grep -E "test@bestval|rror|Trace|non-finite"; }
for cfg in "0 0" "1 5"; do set -- $cfg; st=$1; dw=$2
for dim in 16 64; do for tau in 0.05 0.1 0.3; do for m in 1 10; do
run --heads euclid --dim $dim --tau $tau --lr_proto_mult $m --soft_t $st --depth_w $dw
for c in 0.1 1 2; do
run --heads poincare --c $c --dim $dim --tau $tau --lr_proto_mult $m --soft_t $st --depth_w $dw
done; done; done; done; done > $OUT/sweep_results.log 2>&1
echo DONE >> $OUT/sweep_results.log
