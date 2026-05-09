#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
CKPT=metauas-256.pth
MVTEC_ROOT=~/datasets/mvtec_ad
VISA_ROOT=~/datasets/visa
BATCH=32
DEVICE=cuda:0
SEEDS="1,2,3,5,7,9"

# ========= Execution =========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

echo "=== MVTec AD (oneprompt, multi-seed avg) ==="
python "${REPO}/eval/eval_mvtec.py" \
  --dataset mvtec --mode oneprompt \
  --checkpoint "$CKPT" \
  --data-root "$MVTEC_ROOT" \
  --batch-size "$BATCH" \
  --device "$DEVICE" \
  --oneprompt-dir "${REPO}/eval/MVTec-AD" \
  --oneprompt-seeds "$SEEDS"

echo ""
echo "=== MVTec AD (topk) ==="
python "${REPO}/eval/eval_mvtec.py" \
  --dataset mvtec --mode topk \
  --checkpoint "$CKPT" \
  --data-root "$MVTEC_ROOT" \
  --batch-size "$BATCH" \
  --device "$DEVICE" \
  --topk-json "${REPO}/eval/MVTec-AD/test-train-top10pair-eb4.json" \
  --top-k 1

echo ""
echo "=== VisA (oneprompt, multi-seed avg) ==="
python "${REPO}/eval/eval_mvtec.py" \
  --dataset visa --mode oneprompt \
  --checkpoint "$CKPT" \
  --data-root "$VISA_ROOT" \
  --batch-size "$BATCH" \
  --device "$DEVICE" \
  --oneprompt-dir "${REPO}/eval/VisA-AD" \
  --oneprompt-seeds "$SEEDS"

echo ""
echo "=== VisA (topk) ==="
python "${REPO}/eval/eval_mvtec.py" \
  --dataset visa --mode topk \
  --checkpoint "$CKPT" \
  --data-root "$VISA_ROOT" \
  --batch-size "$BATCH" \
  --device "$DEVICE" \
  --topk-json "${REPO}/eval/VisA-AD/test-train-top10pair-eb4.json" \
  --top-k 1
