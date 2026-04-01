#!/usr/bin/env python3
"""
Temporal DeepFake Detection Training
=====================================
Trains TemporalTriStreamDetector on video sequences (T=16 frames/video).

2-Phase training strategy (recommended):
  Phase 1 (--phase1-epochs, default=8):
    → Freeze Temporal Transformer.
    → Forward uses MEAN-POOL of frame features (no Transformer interference).
    → Only spatial encoder + channel attention + FC(1) update.
    → Spatial features become meaningful per-frame representations first.

  Phase 2 (--phase2-epochs, default=25):
    → Unfreeze ALL parameters (spatial + Transformer + classifier).
    → Full temporal forward via Transformer.
    → Transformer learns to aggregate already-good spatial features.
    → Use lower LR for spatial backbone (--lr-backbone).

Usage example (local RTX 5060 Ti 16GB):
  python scripts/train.py \\
    --train-real  data/extracted/train/original \\
    --train-fake  data/extracted/train/DeepFakeDetection \\
                  data/extracted/train/Deepfakes \\
                  data/extracted/train/Face2Face \\
                  data/extracted/train/FaceShifter \\
                  data/extracted/train/FaceSwap \\
                  data/extracted/train/NeuralTextures \\
    --val-real    data/extracted/val/original \\
    --val-fake    data/extracted/val/DeepFakeDetection \\
    --output-dir  outputs/temporal \\
    --model       efficientnet-b4 \\
    --n-frames    16 \\
    --phase1-epochs 8 \\
    --phase2-epochs 25 \\
    --batch-size  2 \\
    --amp \\
    --focal-loss \\
    --balanced-sampler \\
    --grad-checkpoint
"""

import argparse
from contextlib import nullcontext
import os
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from deepfake_detector.data import get_train_transforms, get_val_transforms
from deepfake_detector.data.dataset import (
    CombinedVideoDataset,
    VideoSequenceDataset,
)
from deepfake_detector.models.multistream import _effnet_feat_dim, _effnet_input_size
from deepfake_detector.models.temporal import TemporalTriStreamDetector
from deepfake_detector.utils import (
    calculate_comprehensive_metrics,
    plot_training_history,
    setup_logger,
)


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for FC(2) + CrossEntropy path."""

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class FocalBCELoss(nn.Module):
    """
    Focal Loss for FC(1) + BCE path (binary classification).
    FL(p) = -alpha * (1-p)^gamma * log(p)

    pos_weight balances class imbalance (= neg_count / pos_count).
    gamma focuses training on hard examples.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, 1] or [B],  targets: [B] float (0.0 or 1.0)
        logits = logits.squeeze(-1)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)   # prob of correct class
        return ((1.0 - pt) ** self.gamma * bce).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_preds_probs(logits: torch.Tensor, bce_output: bool):
    """
    Convert raw logits to (preds [B], probs_real [B]) based on output mode.
    bce_output=True : logits [B,1] → sigmoid → prob_real; pred = (prob > 0.5)
    bce_output=False: logits [B,2] → softmax → prob_real = col 1
    """
    if bce_output:
        logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        preds = (probs >= 0.5).astype(np.int64)
    else:
        logits_np = logits.cpu().numpy().astype(np.float64)
        probs = softmax(logits_np, axis=1)[:, 1]
        preds = np.argmax(logits_np, axis=1)
    return preds, probs


def _make_grad_scaler(enabled: bool):
    """Create a GradScaler compatible across torch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_cuda(enabled: bool):
    """Return an autocast context manager compatible across torch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def _infer_checkpoint_phase(ckpt: dict, ckpt_path: str, phase1_epochs: int) -> int:
    """Infer saved training phase (1 or 2) from checkpoint metadata/path."""
    metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    phase = metrics.get("phase") if isinstance(metrics, dict) else None
    if phase in (1, 2):
        return int(phase)

    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    if isinstance(epoch, int):
        return 2 if epoch > phase1_epochs else 1

    m = re.search(r"_P([12])(?:\.|$)", Path(ckpt_path).name)
    if m:
        return int(m.group(1))

    return 1


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, epoch, bce_output: bool):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for seqs, labels in pbar:
        seqs   = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_cuda(scaler.is_enabled()):
            logits = model(seqs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * seqs.size(0)
        preds, _ = _get_preds_probs(logits.detach(), bce_output)
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    epoch_prec = precision_score(all_labels, all_preds, zero_division=0)
    epoch_rec  = recall_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, epoch_acc, epoch_prec, epoch_rec


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, epoch, bce_output: bool):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for seqs, labels in pbar:
        seqs   = seqs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(seqs)
        loss   = criterion(logits, labels)

        running_loss += loss.item() * seqs.size(0)
        preds, probs = _get_preds_probs(logits, bce_output)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(loader.dataset)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    epoch_acc  = accuracy_score(labels_arr, all_preds)
    conf_mat   = confusion_matrix(labels_arr, all_preds)
    probs_arr  = np.asarray(all_probs, dtype=np.float64)
    metrics    = calculate_comprehensive_metrics(probs=probs_arr, labels=labels_arr)

    return epoch_loss, epoch_acc, conf_mat, metrics, probs_arr, labels_arr


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Temporal TriStream DeepFake Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument("--train-real", nargs="+", required=True)
    parser.add_argument("--train-fake", nargs="+", required=True)
    parser.add_argument("--val-real",   nargs="+", required=True)
    parser.add_argument("--val-fake",   nargs="+", required=True)

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/temporal")

    # Model
    parser.add_argument("--model",       type=str, default="efficientnet-b4",
                        help="EfficientNet backbone. B4 needs --grad-checkpoint + batch=2.")
    parser.add_argument("--augmentation", type=str, default="medium",
                        choices=["light", "medium", "heavy"],
                        help="Augmentation level. 'medium' adds JPEG compression+blur "
                             "for cross-dataset generalization.")
    parser.add_argument("--bce",          action="store_true", default=True,
                        help="Use FC(1)+BCEWithLogitsLoss (recommended for binary).")
    parser.add_argument("--no-bce",       dest="bce", action="store_false",
                        help="Use FC(2)+CrossEntropyLoss (legacy).")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing on backbone. "
                             "Required for B4 fine-tuning on ≤16GB VRAM. "
                             "Reduces activation memory ~6x, ~30%% slower.")
    parser.add_argument("--n-frames",    type=int, default=16,
                        help="Fixed T: number of frames per video sequence")
    parser.add_argument("--sampling",    type=str, default="uniform",
                        choices=["uniform", "random"],
                        help="Frame selection strategy (uniform or random for training aug)")
    parser.add_argument("--num-heads",   type=int, default=8,
                        help="Temporal Transformer attention heads")
    parser.add_argument("--num-layers",  type=int, default=2,
                        help="Temporal Transformer encoder layers")
    parser.add_argument("--srm-filters", type=int, default=30)

    # Training
    parser.add_argument("--batch-size",  type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--amp",         action="store_true",
                        help="Mixed precision (recommended on GPU)")

    # 2-Phase training (new recommended strategy)
    # Phase 1: freeze Transformer, train spatial encoder (mean-pool forward)
    # Phase 2: unfreeze all, full temporal Transformer forward
    parser.add_argument("--phase1-epochs", type=int, default=8,
                        help="Phase 1: spatial-only training (Transformer frozen). "
                             "Forward uses mean-pool — clean gradients to spatial encoder.")
    parser.add_argument("--phase2-epochs", type=int, default=25,
                        help="Phase 2: full temporal training (all params active). "
                             "Transformer learns on stable spatial features.")
    parser.add_argument("--lr-backbone",   type=float, default=5e-5,
                        help="LR for spatial backbone in Phase 2 (lower than temporal head).")

    # Class imbalance
    parser.add_argument("--balanced-sampler",  action="store_true")
    parser.add_argument("--focal-loss",        action="store_true")
    parser.add_argument("--focal-gamma",       type=float, default=2.0)

    # Evaluation
    parser.add_argument("--best-metric",  type=str, default="auc",
                        choices=["acc", "auc", "f1"])
    parser.add_argument("--f1-threshold", type=float, default=0.5)

    # Resuming / transfer
    parser.add_argument("--resume",              type=str, default=None,
                        help="Resume full temporal checkpoint")
    parser.add_argument("--pretrained-spatial",  type=str, default=None,
                        help="Bootstrap from a frame-level TriStream checkpoint")

    args = parser.parse_args()

    # Windows: DataLoader multiprocessing uses torch shared memory; without this,
    # collate can fail with "Couldn't open shared file mapping" / WinError 1455
    # when RAM or page file is tight (often right after a phase change).
    if sys.platform == "win32":
        torch.multiprocessing.set_sharing_strategy("file_system")

    # ── Directories ───────────────────────────────────────────────────────────
    ckpt_dir    = Path(args.output_dir) / "checkpoints"
    log_dir     = Path(args.output_dir) / "logs"
    results_dir = Path(args.output_dir) / "results"
    for d in [ckpt_dir, log_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="temporal_training",
        log_file=str(log_dir / "training.log"),
        level="INFO",
    )

    if sys.platform == "win32":
        logger.info(
            "Windows: torch multiprocessing sharing strategy is file_system (DataLoader). "
            "If collate still fails with shared file mapping / 1455, use --num-workers 0."
        )

    logger.info("=" * 60)
    logger.info("Temporal DeepFake Detection Training")
    logger.info("=" * 60)
    logger.info(
        f"2-Phase strategy:\n"
        f"  Phase 1 ({args.phase1_epochs} epochs): spatial-only, Transformer FROZEN, mean-pool forward\n"
        f"  Phase 2 ({args.phase2_epochs} epochs): full temporal, all params active"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Transforms ────────────────────────────────────────────────────────────
    image_size = _effnet_input_size(args.model)
    train_tfm  = get_train_transforms(image_size, augmentation_level=args.augmentation)
    val_tfm    = get_val_transforms(image_size)
    logger.info(f"Augmentation level : {args.augmentation}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info(f"Loading video sequences (T={args.n_frames} frames)…")

    train_real_cfg = [(p, -1) for p in args.train_real]
    train_fake_cfg = [(p, -1) for p in args.train_fake]
    val_real_cfg   = [(p, -1) for p in args.val_real]
    val_fake_cfg   = [(p, -1) for p in args.val_fake]

    train_ds = CombinedVideoDataset(
        VideoSequenceDataset(train_real_cfg, is_real=True,  transform=train_tfm,
                             n_frames=args.n_frames, sampling=args.sampling),
        VideoSequenceDataset(train_fake_cfg, is_real=False, transform=train_tfm,
                             n_frames=args.n_frames, sampling=args.sampling),
    )
    val_ds = CombinedVideoDataset(
        VideoSequenceDataset(val_real_cfg, is_real=True,  transform=val_tfm,
                             n_frames=args.n_frames, sampling="uniform"),
        VideoSequenceDataset(val_fake_cfg, is_real=False, transform=val_tfm,
                             n_frames=args.n_frames, sampling="uniform"),
    )

    logger.info(f"Train: {len(train_ds)} videos  |  Val: {len(val_ds)} videos")

    # ── Samplers ──────────────────────────────────────────────────────────────
    train_sampler = None
    if args.balanced_sampler:
        labels_np = np.asarray(train_ds.labels, dtype=np.int64)
        counts    = np.bincount(labels_np, minlength=2).astype(np.float64)
        counts[counts == 0] = 1.0
        weights   = (1.0 / counts)[labels_np]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )
        logger.info(
            f"WeightedRandomSampler: fake={int(counts[0])}, real={int(counts[1])} videos"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    bce_output = args.bce
    logger.info(
        f"Creating TemporalTriStreamDetector\n"
        f"  Backbone       : {args.model}\n"
        f"  Frames/vid     : {args.n_frames}  (fixed T)\n"
        f"  Transformer    : {args.num_layers} layers × {args.num_heads} heads\n"
        f"  Feat dim       : {_effnet_feat_dim(args.model)}-d\n"
        f"  Output         : {'FC(1) + BCEWithLogitsLoss' if bce_output else 'FC(2) + CrossEntropyLoss'}\n"
        f"  Grad checkpoint: {args.grad_checkpoint}"
    )

    model = TemporalTriStreamDetector(
        backbone=args.model,
        n_frames=args.n_frames,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        srm_filters=args.srm_filters,
        pretrained=True,
        bce_output=bce_output,
        use_grad_checkpoint=args.grad_checkpoint,
    ).to(device)

    if args.pretrained_spatial:
        logger.info(f"Loading spatial weights from: {args.pretrained_spatial}")
        model.load_spatial_weights(args.pretrained_spatial, device=str(device))

    # Start in Phase 1: Transformer frozen, spatial trains first
    if args.phase1_epochs > 0:
        model.set_phase(1)
        current_phase = 1
        logger.info("[Phase 1 START] Transformer FROZEN. Forward: mean-pool of frame features.")
    else:
        model.set_phase(2)
        current_phase = 2
        logger.info("[Phase 2 START] Full temporal training from epoch 1.")

    total, trainable = model.count_parameters()
    logger.info(f"Total parameters     : {total:,}")
    logger.info(f"Trainable parameters : {trainable:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Compute pos_weight = neg_count / pos_count for imbalanced dataset
    labels_np = np.asarray(train_ds.labels, dtype=np.int64)
    counts    = np.bincount(labels_np, minlength=2).astype(np.float64)
    pos_weight_val = counts[0] / max(counts[1], 1.0)   # fake / real ratio
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    logger.info(f"pos_weight (neg/pos ratio): {pos_weight_val:.2f}")

    if bce_output:
        if args.focal_loss:
            criterion = FocalBCELoss(
                gamma=args.focal_gamma,
                pos_weight=pos_weight_tensor,
            )
            logger.info(f"Loss: FocalBCELoss(gamma={args.focal_gamma}, pos_weight={pos_weight_val:.2f})")
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            logger.info(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight_val:.2f})")
    else:
        # Legacy FC(2) path
        if args.focal_loss:
            criterion = FocalLoss(gamma=args.focal_gamma)
            logger.info(f"Loss: FocalLoss(gamma={args.focal_gamma})")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Loss: CrossEntropyLoss")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Phase 1: only spatial params are trainable
    # Phase 2: all params with differential LR (backbone lower than temporal head)
    spatial_params  = list(model.spatial.parameters())
    temporal_params = list(model.temporal.parameters()) + list(model.classifier.parameters())

    total_epochs = args.phase1_epochs + args.phase2_epochs

    def _build_optimizer_scheduler(phase: int):
        if phase == 2:
            opt = AdamW([
                {"params": spatial_params,  "lr": args.lr_backbone},
                {"params": temporal_params, "lr": args.lr},
            ], weight_decay=1e-3)
            num_steps = len(train_loader) * max(args.phase2_epochs, 1)
            warmup = len(train_loader) * min(3, max(args.phase2_epochs // 6, 1))
        else:
            opt = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, weight_decay=1e-3,
            )
            num_steps = len(train_loader) * max(args.phase1_epochs, 1)
            warmup = len(train_loader) * min(3, max(args.phase1_epochs // 3, 1))

        sch = get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=warmup, num_training_steps=num_steps
        )
        return opt, sch

    optimizer, scheduler = _build_optimizer_scheduler(current_phase)

    # ── AMP scaler ────────────────────────────────────────────────────────────
    use_amp = bool(args.amp and device.type == "cuda")
    scaler  = _make_grad_scaler(use_amp)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_score  = 0.0

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        ckpt = model.load_checkpoint(args.resume, device=str(device))
        if isinstance(ckpt, dict):
            resume_phase = _infer_checkpoint_phase(ckpt, args.resume, args.phase1_epochs)
            if resume_phase != current_phase:
                model.set_phase(resume_phase)
                current_phase = resume_phase
                optimizer, scheduler = _build_optimizer_scheduler(current_phase)
                logger.info(
                    f"Resume checkpoint indicates Phase {resume_phase}; "
                    f"optimizer/scheduler rebuilt to match checkpoint phase."
                )

            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except ValueError as e:
                    logger.warning(
                        "Could not load optimizer state (likely parameter-group mismatch). "
                        "Continuing with fresh optimizer state. Details: %s", e
                    )
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as e:  # scheduler format can differ across runs
                    logger.warning(
                        "Could not load scheduler state. Continuing with fresh scheduler. "
                        "Details: %s", e
                    )
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"])

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
    }

    for epoch in range(start_epoch + 1, total_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{total_epochs}")

        # ── Phase transition: Phase 1 → Phase 2 ───────────────────────────────
        if current_phase == 1 and epoch > args.phase1_epochs:
            model.set_phase(2)
            current_phase = 2

            # Rebuild optimizer with differential LR:
            #   backbone: small LR (features already good from Phase 1)
            #   temporal head: full LR (learning from scratch)
            optimizer = AdamW([
                {"params": spatial_params,  "lr": args.lr_backbone},
                {"params": temporal_params, "lr": args.lr},
            ], weight_decay=1e-3)
            num_steps_p2 = len(train_loader) * args.phase2_epochs
            warmup_p2    = len(train_loader) * min(3, max(args.phase2_epochs // 6, 1))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_p2, num_training_steps=num_steps_p2
            )
            # Reset AMP scaler for new phase
            scaler = _make_grad_scaler(use_amp)

            trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"[Phase 2 START] Epoch {epoch}: ALL params unfrozen.\n"
                f"  Spatial LR   : {args.lr_backbone}\n"
                f"  Temporal LR  : {args.lr}\n"
                f"  Trainable    : {trainable_now:,} params\n"
                f"  Forward mode : full Temporal Transformer"
            )

        # Train
        train_loss, train_acc, train_prec, train_rec = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, bce_output
        )

        # Validate
        val_loss, val_acc, conf_mat, val_metrics, probs, labels_arr = validate_one_epoch(
            model, val_loader, criterion, device, epoch, bce_output
        )

        val_auc     = float(val_metrics.get("auc_roc", 0.0))
        opt_thr     = float(val_metrics.get("optimal_threshold", args.f1_threshold))
        preds_thr = (probs >= args.f1_threshold).astype(np.int64)
        preds_opt = (probs >= opt_thr).astype(np.int64)

        val_f1      = float(f1_score(labels_arr, preds_thr, zero_division=0))
        val_prec    = float(precision_score(labels_arr, preds_thr, zero_division=0))
        val_rec     = float(recall_score(labels_arr, preds_thr, zero_division=0))
        val_f1_opt  = float(f1_score(labels_arr, preds_opt, zero_division=0))
        val_prec_opt = float(precision_score(labels_arr, preds_opt, zero_division=0))
        val_rec_opt  = float(recall_score(labels_arr, preds_opt, zero_division=0))
        val_acc_opt = float(accuracy_score(labels_arr, preds_opt))

        logger.info(
            f"Train  loss={train_loss:.4f}  acc={train_acc:.4f}  "
            f"P={train_prec:.4f}  R={train_rec:.4f}"
        )
        logger.info(
            f"Val    loss={val_loss:.4f}  acc={val_acc:.4f}  "
            f"AUC={val_auc:.4f}  F1={val_f1:.4f}  "
            f"P={val_prec:.4f}  R={val_rec:.4f}  "
            f"F1@opt({opt_thr:.3f})={val_f1_opt:.4f}  "
            f"P@opt={val_prec_opt:.4f}  R@opt={val_rec_opt:.4f}  acc@opt={val_acc_opt:.4f}"
        )
        logger.info(f"Confusion matrix:\n{conf_mat}")

        phase_tag = f"P{current_phase}"
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_precision"].append(train_prec)
        history["train_recall"].append(train_rec)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)

        # Save per-epoch checkpoint (tagged with phase)
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}_{phase_tag}.pth"
        model.save_checkpoint(
            str(ckpt_path),
            epoch=epoch,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            metrics={
                "phase": current_phase,
                "val_acc": val_acc, "val_loss": val_loss,
                "val_auc": val_auc, "val_f1": val_f1,
                "val_precision": val_prec, "val_recall": val_rec,
                "val_f1_opt": val_f1_opt,
                "val_precision_opt": val_prec_opt, "val_recall_opt": val_rec_opt,
                "val_acc_opt": val_acc_opt,
                "val_optimal_threshold": opt_thr,
            },
        )

        # Save best model (only from Phase 2 — Phase 1 is spatial-only pre-training)
        if current_phase == 2:
            score = {"acc": val_acc, "auc": val_auc, "f1": val_f1}[args.best_metric]
            if score > best_score:
                best_score = score
                best_path  = ckpt_dir / "best_model.pth"
                model.save_checkpoint(str(best_path), epoch=epoch)
                logger.info(f"Best model saved [{args.best_metric}={best_score:.4f}]")
        else:
            # In Phase 1, always save as spatial_pretrain.pth for possible bootstrap
            spatial_path = ckpt_dir / "spatial_pretrain.pth"
            model.save_checkpoint(str(spatial_path), epoch=epoch)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_training_history(
        history,
        metrics=["loss", "accuracy", "precision", "recall"],
        save_path=str(results_dir / "training_history.png"),
        show=False,
    )

    logger.info("Training complete!")
    logger.info(f"Best {args.best_metric} (Phase 2): {best_score:.4f}")


if __name__ == "__main__":
    main()
