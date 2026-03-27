#!/usr/bin/env bash
# ============================================================
# run_local.sh  —  Temporal training trên máy local
# ============================================================
# Hardware: RTX 5060 Ti 16 GB  +  i7-14700F (20 cores)
#
# Kiến trúc: TemporalTriStreamDetector
#   Backbone  : EfficientNet-B4 (shared weights, feat=1792-d)
#   Frames    : T=16 (cố định)
#   Transformer: 2 layers × 8 heads, CLS token aggregation
#   Output    : FC(1) + BCEWithLogitsLoss + pos_weight
#
# 2-Phase training (Hướng 2: Pretrain spatial → finetune temporal):
# ─────────────────────────────────────────────────────────────
# Phase 1 (epoch 1~8) — Spatial-only:
#   • Transformer FROZEN, forward = mean-pool frame features
#   • Chỉ train: spatial encoder (EffNet) + channel attention + FC(1)
#   • Spatial encoder học frame-level artifact detection trước
#   • LR = 3e-4 (đầy đủ cho spatial)
#   • ~9 min/epoch → 8 epochs ≈ 1.2h
#
# Phase 2 (epoch 9~33) — Full temporal:
#   • Unfreeze ALL: spatial + Transformer + classifier
#   • Transformer học temporal aggregation trên features đã ổn định
#   • LR backbone = 5e-5 (thấp), LR temporal head = 3e-4
#   • ~15 min/epoch → 25 epochs ≈ 6.2h
#
# Tổng: ~7.4h — trong khả năng máy local overnight
#
# VRAM ước tính với grad_checkpoint ON:
#   B4 + T=16 + batch=2 + grad_ckpt → ~13-15 GB
#   (grad_checkpoint làm VRAM độc lập với T — chỉ 1 frame live lúc backprop)
#
# Kết quả dự đoán:
#   FF++ AUC    : ~97-98%
#   Celeb-DF AUC: ~79-84%  (nhờ medium augmentation + temporal context)
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ── CONFIG ───────────────────────────────────────────────────
DATA_ROOT="data/extracted"
OUTPUT_DIR="outputs/temporal_b4_T16"
MODEL="efficientnet-b4"
N_FRAMES=16                  # T=16 cố định — đủ bắt temporal artifact
BATCH=2                      # B4 + grad_checkpoint + T=16: batch=2 fit 16GB
WORKERS=8                    # i7-14700F 20 cores, để 12 dự phòng
PHASE1_EPOCHS=8              # Spatial pre-training (Transformer frozen)
PHASE2_EPOCHS=25             # Full temporal fine-tuning
LR=3e-4                      # LR cho spatial (Phase 1) và temporal head (Phase 2)
LR_BACKBONE=5e-5             # LR cho backbone trong Phase 2 (10x thấp hơn)
AUGMENTATION="medium"        # JPEG compression + blur + color jitter
FOCAL_GAMMA=2.0
GRAD_CHECKPOINT=true         # Bắt buộc với B4 — giảm activation memory ~6x

# Fake classes
FAKE_CLASSES=(
    "DeepFakeDetection"
    "Deepfakes"
    "Face2Face"
    "FaceShifter"
    "FaceSwap"
    "NeuralTextures"
)

# ── Build fake paths ──────────────────────────────────────────
TRAIN_FAKE_ARGS=""
VAL_FAKE_ARGS=""
for cls in "${FAKE_CLASSES[@]}"; do
    if [ -d "$DATA_ROOT/train/$cls" ]; then
        TRAIN_FAKE_ARGS+=" $DATA_ROOT/train/$cls"
    fi
    if [ -d "$DATA_ROOT/val/$cls" ]; then
        VAL_FAKE_ARGS+=" $DATA_ROOT/val/$cls"
    fi
done

# ── Kiểm tra dữ liệu ─────────────────────────────────────────
if [ ! -d "$DATA_ROOT/train/original" ]; then
    echo "ERROR: Không tìm thấy $DATA_ROOT/train/original"
    echo "       Chạy scripts/run_pipeline.sh để extract dữ liệu trước."
    exit 1
fi

if [ -z "$TRAIN_FAKE_ARGS" ]; then
    echo "ERROR: Không tìm thấy fake classes trong $DATA_ROOT/train/"
    exit 1
fi

# ── Summary ──────────────────────────────────────────────────
TOTAL_EPOCHS=$((PHASE1_EPOCHS + PHASE2_EPOCHS))
echo "======================================================"
echo " Temporal TriStream B4 — Local Training (T=16)"
echo "======================================================"
echo " Model       : $MODEL  (EfficientNet-B4, feat=1792-d)"
echo " Frames/vid  : $N_FRAMES  (fixed T=16)"
echo " Batch       : $BATCH video sequences"
echo " VRAM est.   : ~13-15 GB  (grad_checkpoint ON)"
echo " Workers     : $WORKERS"
echo " Phase 1     : $PHASE1_EPOCHS epochs (spatial-only, Transformer frozen)"
echo " Phase 2     : $PHASE2_EPOCHS epochs (full temporal, all unfrozen)"
echo " Total       : $TOTAL_EPOCHS epochs"
echo " Augment     : $AUGMENTATION (JPEG+blur+color jitter)"
echo " Loss        : FocalBCELoss + pos_weight"
echo " Output      : $OUTPUT_DIR"
echo "======================================================"
echo ""

GRAD_CKPT_FLAG=""
if [ "$GRAD_CHECKPOINT" = "true" ]; then
    GRAD_CKPT_FLAG="--grad-checkpoint"
fi

PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 scripts/train.py \
    --train-real   "$DATA_ROOT/train/original" \
    --train-fake   $TRAIN_FAKE_ARGS \
    --val-real     "$DATA_ROOT/val/original" \
    --val-fake     $VAL_FAKE_ARGS \
    --output-dir   "$OUTPUT_DIR" \
    --model        "$MODEL" \
    --n-frames     "$N_FRAMES" \
    --sampling     "random" \
    --num-heads    8 \
    --num-layers   2 \
    --batch-size   "$BATCH" \
    --num-workers  "$WORKERS" \
    --phase1-epochs "$PHASE1_EPOCHS" \
    --phase2-epochs "$PHASE2_EPOCHS" \
    --lr           "$LR" \
    --lr-backbone  "$LR_BACKBONE" \
    --augmentation "$AUGMENTATION" \
    --amp \
    --bce \
    --balanced-sampler \
    --focal-loss \
    --focal-gamma  "$FOCAL_GAMMA" \
    --best-metric  auc \
    $GRAD_CKPT_FLAG

echo ""
echo "Training complete!"
echo "Best model (Phase 2) : $OUTPUT_DIR/checkpoints/best_model.pth"
echo "Spatial pretrain     : $OUTPUT_DIR/checkpoints/spatial_pretrain.pth"
echo "Logs                 : $OUTPUT_DIR/logs/training.log"
