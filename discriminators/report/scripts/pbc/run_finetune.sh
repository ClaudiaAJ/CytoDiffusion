#!/bin/bash
# PBC fine-tuned comparison, same loop as CytoData A1 (fullloop/): 20 epochs, 5 probe epochs, mixup 0.3, 64-d, c=1.
# linear / euclid (runnorm) / poincare (runnorm), seeds 0-2. Sequential; waits for run_frozen.sh to finish first.
# Resumable: a run is skipped if its test CSV already exists.
O=/store/CIA/ca662/CytoDiffusion_runs/pbc
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
SC=/mhome/damtp/t/ca662/CytoDiffusion/discriminators/train_and_test_hyperbolic.py
DATA=/store/CIA/ca662/CytoDiffusion_data/PBC/split
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
until grep -q "ALL DONE" $O/progress.log 2>/dev/null; do sleep 60; done
mkdir -p $O/finetune/logs; cd $O/finetune
for seed in 0 1 2; do
  for head in poincare euclid linear; do
    name=PBC_FT
    csv=$O/finetune/$name/${name}_${head}_vit_base_patch16_384_${seed}.csv
    [ -f "$csv" ] && { echo "$(date +%F_%T) skip $head seed=$seed (done)"; continue; }
    for try in $(seq 1 144); do
      read gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | sort -t, -k2 -nr | head -1 | tr ',' ' ')
      [ "$free" -ge 14000 ] && break
      echo "$(date +%F_%T) no GPU with 14GB free (best: gpu$gpu ${free}MiB), waiting"; sleep 300
    done
    extra=""; [ $head != linear ] && extra="--norm_mode runnorm --target_norm 1 --clip_r 4"
    echo "$(date +%F_%T) start head=$head seed=$seed on gpu$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY $SC --data_root $DATA --experiment_name $name --head $head --seed $seed \
        --epochs 20 --probe_epochs 5 --num_workers 6 $extra > $O/finetune/logs/${head}_seed${seed}.log 2>&1
    echo "$(date +%F_%T) end   head=$head seed=$seed exit=$?"
  done
done
echo "$(date +%F_%T) FT ALL DONE"
