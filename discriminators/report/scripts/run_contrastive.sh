#!/bin/bash
# Fine-tuned ViT with a supervised contrastive term on the embedding (weight 0.5, cross-batch memory 1024), on top of the usual
# cross-entropy on prototype-distance logits. Arms: poincare (hyperbolic distance) vs euclid (Euclidean distance); same pipeline otherwise.
# Settings identical to fullloop/FullLoopRunnorm (20 epochs, 5 probe, runnorm, mixup 0.3), seeds 0-2. ONE process at a time, resumable.
R=/store/CIA/ca662/CytoDiffusion_runs/contrastive
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
SC=/mhome/damtp/t/ca662/CytoDiffusion/discriminators/train_and_test_hyperbolic.py
DATA=/store/CIA/ca662/CytoDiffusion_data/CytoData
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
COMMON="--data_root $DATA --epochs 20 --probe_epochs 5 --norm_mode runnorm --target_norm 1 --clip_r 4 --num_workers 6 --con_w 0.5"
name=FTCon
cd $R
for seed in 0 1 2; do for head in poincare euclid; do
  csv=$R/$name/${name}_${head}_vit_base_patch16_384_${seed}.csv
  [ -f "$csv" ] && { echo "$(date +%F_%T) skip (done): $head seed$seed" >> $R/progress.log; continue; }
  for try in $(seq 1 288); do
    read gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr | head -1 | tr ',' ' ')
    [ "$free" -ge 14000 ] && break
    sleep 300
  done
  echo "$(date +%F_%T) start $head seed$seed on gpu$gpu (${free}MiB free)" >> $R/progress.log
  CUDA_VISIBLE_DEVICES=$gpu $PY $SC $COMMON --experiment_name $name --head $head --seed $seed > $R/logs/${head}_s${seed}.log 2>&1
  echo "$(date +%F_%T) end   $head seed$seed exit=$?" >> $R/progress.log
  rm -f $R/saved_models/${name}_${head}_vit_base_patch16_384_${seed}.pth
done; done
echo "$(date +%F_%T) ALL DONE" >> $R/progress.log
