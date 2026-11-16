#!/bin/bash
# After the head grid: cache all ViT tokens (frozen pretrained backbone, then the fine-tuned stock backbone) and compare token pooling
# in Euclidean vs hyperbolic space. One process at a time.
PY=/store/CIA/ca662/anaconda/envs/CytoDiffusion/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
OUT=/store/CIA/ca662/CytoDiffusion_runs/hyp_accuracy2
cd $OUT
until grep -q "^DONE" $OUT/heads2_results.log 2>/dev/null; do sleep 5; done
V="cls_euclid cls_poincare mean_euclid normw_euclid midpoint_poincare attn_euclid attn_poincare"
for bb in plain stock; do
  T=/store/CIA/ca662/CytoDiffusion_data/tokens_$bb
  [ -f $T/classes.npy ] || $PY extract_tokens.py $bb > extract_$bb.log 2>&1
  echo "## backbone=$bb" >> tokens_results.log
  $PY token_pool.py --tokens $T --variants $V --out $OUT/token_preds_$bb.npz 2>&1 | grep -E "test@bestval|rror|Trace|non-finite" >> tokens_results.log
done
echo DONE >> tokens_results.log
