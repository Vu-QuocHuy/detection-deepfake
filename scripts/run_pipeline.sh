#!/usr/bin/env bash
set -euo pipefail

# End-to-end FF++ pipeline:
# 1) Create split manifest from CSV
# 2) Extract faces for train/val/test using per-class quotas
# 3) Train tri-stream model (RGB + Frequency + SRM-like)
#
# Usage:
#   bash scripts/run_pipeline.sh \
#     --dataset-root /path/to/ffpp_root \
#     --epochs 20 --num-workers 6 --device cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT=""
SEED="42"
DEVICE="cuda"
EPOCHS="20"
NUM_WORKERS="6"
BATCH_SIZE="1"
LR="1e-4"
SPLIT_NAME="ffpp_c23_split_80_10_10"
USE_AMP="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --split-name) SPLIT_NAME="$2"; shift 2 ;;
    --no-amp) USE_AMP="0"; shift 1 ;;
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

echo "=== [1/4] Create split manifest from CSV ==="
python3 "${SCRIPT_DIR}/create_split_manifest_from_csv.py" \
  --dataset-root "${DATASET_ROOT}" \
  --csv-dir "${CSV_DIR}" \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --seed "${SEED}" \
  --output "${MANIFEST_PATH}"

echo "=== [2/4] Extract TRAIN faces (real=24, fake=4) ==="
python3 "${SCRIPT_DIR}/extract_from_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --split train \
  --output-root "${EXTRACTED_ROOT}" \
  --real-class original \
  --real-frames-per-video 24 \
  --fake-frames-per-video 4 \
  --sampling-strategy uniform \
  --device "${DEVICE}"

echo "=== [3/4] Extract VAL/TEST faces (real=16, fake=3) ==="
python3 "${SCRIPT_DIR}/extract_from_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --split val \
  --output-root "${EXTRACTED_ROOT}" \
  --real-class original \
  --real-frames-per-video 16 \
  --fake-frames-per-video 3 \
  --sampling-strategy uniform \
  --device "${DEVICE}"

python3 "${SCRIPT_DIR}/extract_from_manifest.py" \
  --manifest "${MANIFEST_PATH}" \
  --split test \
  --output-root "${EXTRACTED_ROOT}" \
  --real-class original \
  --real-frames-per-video 16 \
  --fake-frames-per-video 3 \
  --sampling-strategy uniform \
  --device "${DEVICE}"

echo "=== [4/4] Train tri-stream model ==="
TRAIN_CMD=(
  python3 "${SCRIPT_DIR}/train.py"
  --multistream
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
  --freq-model efficientnet-b2
  --srm-model efficientnet-b0
  --srm-filters 30
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ "${USE_AMP}" == "1" ]]; then
  TRAIN_CMD+=(--amp)
fi

"${TRAIN_CMD[@]}"

echo
echo "Pipeline completed."
echo "Manifest: ${MANIFEST_PATH}"
echo "Faces:    ${EXTRACTED_ROOT}"
echo "Outputs:  ${OUTPUT_DIR}"
