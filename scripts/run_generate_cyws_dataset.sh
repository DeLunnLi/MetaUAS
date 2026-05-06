#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
DATA_DIR="/path/to/datasets/images"
SPLIT="train2017"

OUT_CYWS="/path/to/repo/data/coco-inpainted-generated/train"
OUTPUT_DIR="${OUT_CYWS}/_lama_jpg"

# 4 GPUs × 4 workers each = 16 parallel processes
GPU_IDS="0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

VARIANTS_PER_IMAGE=4
CYWS_TRAIN_RATE=0.95
MAX_MASK_AREA_RATIO=0.7
UNIQUE_IOU_THRESH=1.0
UNIQUE_MAX_TRIES=100
CHUNK_SIZE=1000
MIN_COCO_ANN_AREA=0

QUIET_PNG_WARNINGS=1
QUIET_UNIQUE_WARNINGS=1

CLEAN_BEFORE_RUN=1
RESET_CYWS_OUTPUTS=1
RESET_LAMA_JPGS=1

# ========= Execution =========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
cd "$REPO"

if [[ ! -d "${DATA_DIR}/${SPLIT}" ]]; then
  echo "[error] COCO image dir not found: ${DATA_DIR}/${SPLIT}" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/annotations/instances_${SPLIT}.json" ]]; then
  echo "[error] COCO annotation not found: ${DATA_DIR}/annotations/instances_${SPLIT}.json" >&2
  exit 1
fi

if [[ "$CLEAN_BEFORE_RUN" == "1" ]]; then
  if [[ -z "${OUT_CYWS}" || "${OUT_CYWS}" == "/" ]]; then
    echo "[error] Invalid OUT_CYWS: '${OUT_CYWS}'" >&2
    exit 1
  fi
  if [[ -z "${OUTPUT_DIR}" || "${OUTPUT_DIR}" == "/" ]]; then
    echo "[error] Invalid OUTPUT_DIR: '${OUTPUT_DIR}'" >&2
    exit 1
  fi

  echo "[clean] remove LaMa intermediates under: ${OUT_CYWS}"
  rm -rf "${OUT_CYWS}"/.lama_inputs_* "${OUT_CYWS}"/.lama_outputs_* "${OUT_CYWS}"/.lama_best_* \
         "${OUT_CYWS}"/.lama_retry_in_* "${OUT_CYWS}"/.lama_retry_out_* 2>/dev/null || true

  if [[ "$RESET_CYWS_OUTPUTS" == "1" ]]; then
    echo "[clean] reset CYWS outputs under: ${OUT_CYWS}"
    rm -rf "${OUT_CYWS}/images_and_masks" "${OUT_CYWS}/inpainted" 2>/dev/null || true
    rm -f "${OUT_CYWS}/data_split.pkl" 2>/dev/null || true
  fi

  if [[ "$RESET_LAMA_JPGS" == "1" ]]; then
    echo "[clean] reset LaMa jpg outputs under: ${OUTPUT_DIR}"
    rm -f "${OUTPUT_DIR}"/*.jpg "${OUTPUT_DIR}"/*.jpeg "${OUTPUT_DIR}"/*.png 2>/dev/null || true
  fi
fi

CMD=(
  python dataset/pre_process_coco.py
  --data-dir "$DATA_DIR"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
  --gpu-ids "$GPU_IDS"
  --cyws-root "$OUT_CYWS"
  --cyws-variants-per-image "$VARIANTS_PER_IMAGE"
  --cyws-train-rate "$CYWS_TRAIN_RATE"
  --max-mask-area-ratio "$MAX_MASK_AREA_RATIO"
  --cyws-unique-iou-thresh "$UNIQUE_IOU_THRESH"
  --cyws-unique-max-tries "$UNIQUE_MAX_TRIES"
  --chunk-size "$CHUNK_SIZE"
  --min-coco-ann-area "$MIN_COCO_ANN_AREA"
)

if [[ "$QUIET_PNG_WARNINGS" == "1" ]]; then
  CMD+=(--quiet-png-warnings)
  export PIPELINE_QUIET_PNG_WARNINGS=1
fi
if [[ "$QUIET_UNIQUE_WARNINGS" == "1" ]]; then
  CMD+=(--quiet-unique-warnings)
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

