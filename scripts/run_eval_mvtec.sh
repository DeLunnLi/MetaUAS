#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
CKPT=best_model.pth
MVTEC_ROOT=~/datasets/mvtec_ad
BATCH=32
DEVICE=cuda:0
SEEDS="1,2,3,5,7,9"

# ========= Execution =========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

python "${REPO}/eval/eval_mvtec.py" \
  --checkpoint "$CKPT" \
  --mvtec-root "$MVTEC_ROOT" \
  --batch-size "$BATCH" \
  --device "$DEVICE" \
  --oneprompt-dir "${REPO}/eval/MVTec-AD" \
  --oneprompt-seeds "$SEEDS"
