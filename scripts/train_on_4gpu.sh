set -euo pipefail

# ========= GPU / torchrun =========
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_PORT=29501
NPROC=4

# ========= Paths =========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
CI_ROOT=/path/to/coco-inpainted/train
COCO_IMG=/path/to/datasets/images/train2017
COCO_META=/path/to/datasets/images/meta_data
DTD=/path/to/dtd/images

# ========= Save names =========
BEST_CP=best_model_change_pairs_model.pth

# ========= Training hyperparams =========
EPOCHS=50
BATCH=96
VAL_BATCH=96
WORKERS=8
LR=1e-4
SCHED=constant
WD=0.005
TARGET_SIZE=256
EARLY_STOP=0          # >0 monitors val loss (or train if no val); 0=off
GAMMA=0.95            # only for scheduler=exponential
ETA_MIN=1e-6          # only for scheduler=cosine

# ========= coco-inpainted mixing =========
LOCAL_P=0.5
EX_P=0.5
EX_MAX_AREA=0.5

# ========= Augmentation =========
AUG_AFFINE_TRANSLATE=0.05
AUG_AFFINE_SCALE_MIN=0.95
AUG_AFFINE_SCALE_MAX=1.05
AUG_AFFINE_DEGREES=30
AUG_AFFINE_P=1.0
AUG_JIT_B=0.05
AUG_JIT_C=0.05
AUG_JIT_S=0.05
AUG_JIT_H=0.03
AUG_JITTER_P=1.0

cd "$REPO"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  train/train_change_pairs.py \
  --coco-inpainted-root "$CI_ROOT" \
  --coco-inpainted-local-region-p "$LOCAL_P" \
  --coco-inpainted-exchange-p "$EX_P" \
  --coco-inpainted-coco-images "$COCO_IMG" \
  --coco-inpainted-coco-metadata "$COCO_META" \
  --coco-inpainted-exchange-max-area "$EX_MAX_AREA" \
  --coco-inpainted-dtd-root "$DTD" \
  --target-size "$TARGET_SIZE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --val-batch-size "$VAL_BATCH" \
  --num-workers "$WORKERS" \
  --lr "$LR" \
  --scheduler "$SCHED" \
  --gamma "$GAMMA" \
  --eta-min "$ETA_MIN" \
  --weight-decay "$WD" \
  --early-stopping-patience "$EARLY_STOP" \
  --val-each-epoch \
  --best-path "$BEST_CP" \
  --no-epoch-checkpoints \
  --aug-affine-translate "$AUG_AFFINE_TRANSLATE" \
  --aug-affine-scale-min "$AUG_AFFINE_SCALE_MIN" \
  --aug-affine-scale-max "$AUG_AFFINE_SCALE_MAX" \
  --aug-affine-degrees "$AUG_AFFINE_DEGREES" \
  --aug-affine-p "$AUG_AFFINE_P" \
  --aug-jitter-b "$AUG_JIT_B" \
  --aug-jitter-c "$AUG_JIT_C" \
  --aug-jitter-s "$AUG_JIT_S" \
  --aug-jitter-h "$AUG_JIT_H" \
  --aug-jitter-p "$AUG_JITTER_P"

