#!/bin/bash
# PBC frozen-feature experiments, same settings as CytoData holdout_tree (dim 64) and holdout_lowdim (dim 4).
# Sequential, one process at a time.
#  1) no hold-out: linear / poincare / euclid x {base (flat), treedepth (hierarchy)}, 5 seeds
#  2) all 8 hold-outs x {poincare, euclid} x {base, treedepth}, 3 seeds
O=/store/CIA/ca662/CytoDiffusion_runs/pbc
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
CACHE=/store/CIA/ca662/CytoDiffusion_data/features/pbc_vitb384_plain.npz
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
cd /mhome/damtp/t/ca662/CytoDiffusion/discriminators
COMMON="--cache $CACHE --norm_mode runnorm --target_norm 1 --clip_r 4 --soft_mix 0.8 --quiet"
for dim in 64 4; do
  for cfg in "base 0 0" "treedepth 1 5"; do set -- $cfg
    echo "## noholdout dim=$dim cfg=$1"
    $PY frozen_head_harness.py $COMMON --dim $dim --heads linear poincare euclid --soft_t $2 --depth_w $3 --seeds 0 1 2 3 4 2>&1 | grep -E "test@bestval|rror|Trace"
  done >> $O/accuracy_dim$dim.log 2>&1
  for h in 0 1 2 3 4 5 6 7; do for hd in poincare euclid; do for cfg in "base 0 0" "treedepth 1 5"; do set -- $cfg
    echo "## holdout=$h head=$hd cfg=$1"
    $PY frozen_head_harness.py $COMMON --dim $dim --heads $hd --soft_t $2 --depth_w $3 --holdout $h --seeds 0 1 2 2>&1 | grep -E "placement|test@bestval|rror|Trace"
  done; done; done >> $O/holdout_dim$dim.log 2>&1
  echo "$(date +%F_%T) DIM $dim DONE" >> $O/progress.log
done
echo "$(date +%F_%T) ALL DONE" >> $O/progress.log
