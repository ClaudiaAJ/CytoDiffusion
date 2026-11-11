#!/bin/bash
# Balanced design, fine-tuned ViT (20 epochs, 5 probe), ONE training process at a time, resumable (skips runs whose CSV exists).
#  Part 1: matched 64-d pair, all 10 classes: poincare runnorm seeds 1,2 (seed 0 already exists) + euclid runnorm seeds 0,1,2
#  Part 2: dim 4, hold out erythroblast / platelet, poincare vs euclid, tree-aware (soft labels + depth loss), seeds 0-2
#  Part 3: same as part 2 at dim 8
R=/store/CIA/ca662/CytoDiffusion_runs/balanced_design
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
SC=/mhome/damtp/t/ca662/CytoDiffusion/discriminators/train_and_test_hyperbolic.py
DATA=/store/CIA/ca662/CytoDiffusion_data/CytoData
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
COMMON="--data_root $DATA --epochs 20 --probe_epochs 5 --norm_mode runnorm --target_norm 1 --clip_r 4 --num_workers 6"

run() {  # run <outdir> <experiment> <head> <seed> <log> [extra args...]
  local outdir=$1 name=$2 head=$3 seed=$4 log=$5; shift 5
  local csv=$outdir/$name/${name}_${head}_vit_base_patch16_384_${seed}.csv
  [ -f "$csv" ] && { echo "$(date +%F_%T) skip (done): $name $head seed$seed" >> $R/progress.log; return; }
  local gpu free
  for try in $(seq 1 288); do
    read gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr | head -1 | tr ',' ' ')
    [ "$free" -ge 14000 ] && break
    sleep 300
  done
  echo "$(date +%F_%T) start $name $head seed$seed on gpu$gpu (${free}MiB free)" >> $R/progress.log
  mkdir -p $outdir && cd $outdir
  CUDA_VISIBLE_DEVICES=$gpu $PY $SC $COMMON --experiment_name $name --head $head --seed $seed "$@" > $R/logs/$log 2>&1
  local rc=$?
  echo "$(date +%F_%T) end   $name $head seed$seed exit=$rc" >> $R/progress.log
  rm -f $outdir/saved_models/${name}_${head}_vit_base_patch16_384_${seed}.pth   # checkpoint is 345 MB; csv + embeddings are kept
}

F=/store/CIA/ca662/CytoDiffusion_runs/fullloop
# ---- Part 1
run $F FullLoopRunnorm euclid   0 p1_euclid_s0.log
run $F FullLoopRunnorm poincare 1 p1_poincare_s1.log
run $F FullLoopRunnorm euclid   1 p1_euclid_s1.log
run $F FullLoopRunnorm poincare 2 p1_poincare_s2.log
run $F FullLoopRunnorm euclid   2 p1_euclid_s2.log
echo "$(date +%F_%T) PART 1 DONE" >> $R/progress.log
# ---- Parts 2 and 3
for dim in 4 8; do
  for seed in 0 1 2; do for hold in erythroblast platelet; do for head in poincare euclid; do
    run $R/holdout_d$dim FTHold_d${dim}_$hold $head $seed p${dim}_${hold}_${head}_s${seed}.log \
        --dim $dim --holdout $hold --soft_t 1 --soft_mix 0.8 --depth_w 5
  done; done; done
  echo "$(date +%F_%T) PART (dim $dim) DONE" >> $R/progress.log
done
echo "$(date +%F_%T) ALL DONE" >> $R/progress.log
