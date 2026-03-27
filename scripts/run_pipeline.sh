#!/usr/bin/env bash
set -euo pipefail

# End-to-end FF++ pipeline — TemporalTriStreamDetector:
# 1) Create split manifest from CSV       (split 720/140/140)
# 2) Extract faces — 32 frames/video      (uniform, real & fake)
# 3) Train TemporalTriStreamDetector      (2-phase: spatial → temporal)
#
# Extract 32 frames/video để VideoSequenceDataset có đủ frame để sample T=16.
# Tỉ lệ split 72/14/14 (~720/140/140 video).
#
# Usage:
#   bash scripts/run_pipeline.sh \
#     --dataset-root /path/to/ffpp_root \
#     --phase1-epochs 8 --phase2-epochs 25 \
#     --num-workers 8 --device cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT=""
SEED="42"
DEVICE="cuda"
PHASE1_EPOCHS="8"
PHASE2_EPOCHS="25"
NUM_WORKERS="8"
BATCH_SIZE="2"
LR="3e-4"
LR_BACKBONE="5e-5"
FRAMES_PER_VIDEO="32"          # extract 32 frames/video → sample T=16 khi train
N_FRAMES="16"
SPLIT_NAME="ffpp_c23_split_72_14_14"
USE_AMP="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)      DATASET_ROOT="$2";      shift 2 ;;
    --seed)              SEED="$2";               shift 2 ;;
    --device)            DEVICE="$2";             shift 2 ;;
    --phase1-epochs)     PHASE1_EPOCHS="$2";      shift 2 ;;
    --phase2-epochs)     PHASE2_EPOCHS="$2";      shift 2 ;;
    --num-workers)       NUM_WORKERS="$2";        shift 2 ;;
    --batch-size)        BATCH_SIZE="$2";         shift 2 ;;
    --lr)                LR="$2";                 shift 2 ;;
    --lr-backbone)       LR_BACKBONE="$2";        shift 2 ;;
    --frames-per-video)  FRAMES_PER_VIDEO="$2";   shift 2 ;;
    --n-frames)          N_FRAMES="$2";           shift 2 ;;
    --split-name)        SPLIT_NAME="$2";         shift 2 ;;
    --amp)               USE_AMP="1";             shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "ERROR: --dataset-root is required"
  exit 1
fi

DATASET_ROOT="$(cd "${DATASET_ROOT}" && pwd)"
CSV_DIR="${DATASET_ROOT}/csv"
SPLIT_DIR="${DATASET_ROOT}/splits"
EXTRACTED_ROOT="${DATASET_ROOT}/extracted_faces"
OUTPUT_DIR="${PROJECT_ROOT}/outputs"
MANIFEST_PATH="${SPLIT_DIR}/${SPLIT_NAME}.json"

mkdir -p "${SPLIT_DIR}" "${EXTRACTED_ROOT}" "${OUTPUT_DIR}"

TOTAL_EPOCHS=$((PHASE1_EPOCHS + PHASE2_EPOCHS))
echo "============================================================"
echo "  FF++ Pipeline — TemporalTriStreamDetector"
echo "  Dataset   : ${DATASET_ROOT}"
echo "  Split     : 72/14/14  (train/val/test)"
echo "  Extract   : ${FRAMES_PER_VIDEO} frames/video  →  T=${N_FRAMES} khi train"
echo "  Phase 1   : ${PHASE1_EPOCHS} epochs (spatial-only)"
echo "  Phase 2   : ${PHASE2_EPOCHS} epochs (full temporal)"
echo "  Total     : ${TOTAL_EPOCHS} epochs  |  LR: ${LR}  |  Batch: ${BATCH_SIZE}"
echo "============================================================"

# ── Bước 1: Tạo split manifest ─────────────────────────────────────────────
echo ""
echo "=== [1/4] Tạo split manifest từ CSV (72/14/14) ==="
python3 "${SCRIPT_DIR}/create_split_manifest_from_csv.py" \
  --dataset-root "${DATASET_ROOT}" \
  --csv-dir "${CSV_DIR}" \
  --train-ratio 0.72 \
  --val-ratio   0.14 \
  --test-ratio  0.14 \
  --seed "${SEED}" \
  --output "${MANIFEST_PATH}"

# ── Bước 2: Extract faces TRAIN ────────────────────────────────────────────
echo ""
echo "=== [2/4] Extract TRAIN faces (${FRAMES_PER_VIDEO} frames/video) ==="
python3 "${SCRIPT_DIR}/extract_from_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --split train \
  --output-root "${EXTRACTED_ROOT}" \
  --real-class original \
  --real-frames-per-video "${FRAMES_PER_VIDEO}" \
  --fake-frames-per-video "${FRAMES_PER_VIDEO}" \
  --sampling-strategy uniform \
  --device "${DEVICE}"

# ── Bước 3: Extract faces VAL + TEST ───────────────────────────────────────
echo ""
echo "=== [3/4] Extract VAL/TEST faces (${FRAMES_PER_VIDEO} frames/video) ==="
for SPLIT in val test; do
  python3 "${SCRIPT_DIR}/extract_from_manifest.py" \
    --manifest "${MANIFEST_PATH}" \
    --split "${SPLIT}" \
    --output-root "${EXTRACTED_ROOT}" \
    --real-class original \
    --real-frames-per-video "${FRAMES_PER_VIDEO}" \
    --fake-frames-per-video "${FRAMES_PER_VIDEO}" \
    --sampling-strategy uniform \
    --device "${DEVICE}"
done

# ── Bước 4: Train ──────────────────────────────────────────────────────────
echo ""
echo "=== [4/4] Train TemporalTriStreamDetector (2-phase) ==="
TRAIN_CMD=(
  python3 "${SCRIPT_DIR}/train.py"
  --train-real "${EXTRACTED_ROOT}/train/original"
  --train-fake
    "${EXTRACTED_ROOT}/train/DeepFakeDetection"
    "${EXTRACTED_ROOT}/train/Deepfakes"
    "${EXTRACTED_ROOT}/train/Face2Face"
    "${EXTRACTED_ROOT}/train/FaceShifter"
    "${EXTRACTED_ROOT}/train/FaceSwap"
    "${EXTRACTED_ROOT}/train/NeuralTextures"
  --val-real "${EXTRACTED_ROOT}/val/original"
  --val-fake
    "${EXTRACTED_ROOT}/val/DeepFakeDetection"
    "${EXTRACTED_ROOT}/val/Deepfakes"
    "${EXTRACTED_ROOT}/val/Face2Face"
    "${EXTRACTED_ROOT}/val/FaceShifter"
    "${EXTRACTED_ROOT}/val/FaceSwap"
    "${EXTRACTED_ROOT}/val/NeuralTextures"
  --model efficientnet-b4
  --n-frames "${N_FRAMES}"
  --srm-filters 30
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --phase1-epochs "${PHASE1_EPOCHS}"
  --phase2-epochs "${PHASE2_EPOCHS}"
  --lr "${LR}"
  --lr-backbone "${LR_BACKBONE}"
  --augmentation medium
  --balanced-sampler
  --focal-loss
  --focal-gamma 2.0
  --best-metric auc
  --bce
  --grad-checkpoint
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${USE_AMP}" == "1" ]]; then
  TRAIN_CMD+=(--amp)
fi

PYTORCH_ALLOC_CONF=expandable_segments:True "${TRAIN_CMD[@]}"

echo ""
echo "============================================================"
echo "  Pipeline hoàn tất!"
echo "  Manifest : ${MANIFEST_PATH}"
echo "  Faces    : ${EXTRACTED_ROOT}"
echo "  Outputs  : ${OUTPUT_DIR}"
echo "============================================================"
