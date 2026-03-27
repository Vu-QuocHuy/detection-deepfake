#!/usr/bin/env python3
"""
Testing and Evaluation Script — TemporalTriStreamDetector
==========================================================
Evaluates a trained TemporalTriStreamDetector on a test split.
Loads video sequences (T frames/video) and reports comprehensive metrics.

Usage:
  python scripts/test.py \\
    --test-real  data/extracted/test/original \\
    --test-fake  data/extracted/test/DeepFakeDetection \\
                 data/extracted/test/Deepfakes \\
                 data/extracted/test/Face2Face \\
                 data/extracted/test/FaceShifter \\
                 data/extracted/test/FaceSwap \\
                 data/extracted/test/NeuralTextures \\
    --checkpoint outputs/temporal_b4_T16/checkpoints/best_model.pth \\
    --output-dir test_results \\
    --n-frames 16 \\
    --batch-size 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detector.data import get_val_transforms
from deepfake_detector.data.dataset import CombinedVideoDataset, VideoSequenceDataset
from deepfake_detector.models.multistream import _effnet_input_size
from deepfake_detector.models.temporal import TemporalTriStreamDetector
from deepfake_detector.utils import (
    calculate_comprehensive_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics,
    setup_logger,
)
from deepfake_detector.utils.metrics import get_EER_states


@torch.no_grad()
def evaluate(model: TemporalTriStreamDetector, loader: DataLoader,
             device: torch.device, bce_output: bool):
    """Run inference and return (probs, labels) for the full test set."""
    model.eval()
    all_probs, all_labels = [], []

    for seqs, labels in tqdm(loader, desc="Testing"):
        seqs = seqs.to(device, non_blocking=True)
        logits = model(seqs)

        if bce_output:
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        else:
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

        all_probs.extend(probs.astype(np.float64))
        all_labels.extend(labels.numpy())

    return np.asarray(all_probs), np.asarray(all_labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TemporalTriStreamDetector on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--test-real",   nargs="+", required=True)
    parser.add_argument("--test-fake",   nargs="+", required=True)
    parser.add_argument("--checkpoint",  type=str,  required=True)
    parser.add_argument("--output-dir",  type=str,  default="test_results")
    parser.add_argument("--model",       type=str,  default="efficientnet-b4")
    parser.add_argument("--n-frames",    type=int,  default=16)
    parser.add_argument("--batch-size",  type=int,  default=4)
    parser.add_argument("--num-workers", type=int,  default=4)
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Decision threshold for FAKE/REAL (applied to fake-prob)")
    parser.add_argument("--save-predictions", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="testing",
        log_file=str(out_dir / "test.log"),
        level="INFO",
    )
    logger.info("=" * 60)
    logger.info("TemporalTriStreamDetector — Test Evaluation")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load checkpoint to determine model config ─────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone  = ckpt.get("backbone",  args.model)
    n_frames  = ckpt.get("n_frames",  args.n_frames)
    bce_output = ckpt.get("bce_output", True)
    logger.info(f"  backbone={backbone}  n_frames={n_frames}  bce_output={bce_output}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    image_size = _effnet_input_size(backbone)
    tfm = get_val_transforms(image_size)

    test_real_cfg = [(p, -1) for p in args.test_real]
    test_fake_cfg = [(p, -1) for p in args.test_fake]

    test_ds = CombinedVideoDataset(
        VideoSequenceDataset(test_real_cfg, is_real=True,  transform=tfm,
                             n_frames=n_frames, sampling="uniform"),
        VideoSequenceDataset(test_fake_cfg, is_real=False, transform=tfm,
                             n_frames=n_frames, sampling="uniform"),
    )
    logger.info(f"Test set: {len(test_ds)} videos")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TemporalTriStreamDetector(
        backbone=backbone,
        n_frames=n_frames,
        bce_output=bce_output,
        pretrained=False,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    probs, labels = evaluate(model, loader, device, bce_output)

    # probs = P(fake) for BCE, P(real) for CE.
    # Convention: labels 0=fake 1=real → we keep probs as P(fake) for BCE.
    # For metrics that expect P(real): use 1 - probs when bce_output=True.
    if bce_output:
        # BCE sigmoid → probs = P(fake=1 | BCE label convention: 1=fake)
        # But our label convention is 0=fake,1=real → need to adjust.
        # Actually: in train.py, labels are 0=fake,1=real passed to BCE
        # which uses the label directly, so sigmoid(logit) ≈ P(label=1) = P(real)
        probs_real = probs
    else:
        probs_real = probs

    pred_labels = (probs_real >= args.threshold).astype(np.int64)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = calculate_comprehensive_metrics(probs=probs_real, labels=labels)
    print_metrics(metrics, title="Test Results")

    metrics_file = out_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("TemporalTriStreamDetector — Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"checkpoint  : {args.checkpoint}\n")
        f.write(f"backbone    : {backbone}\n")
        f.write(f"n_frames    : {n_frames}\n")
        f.write(f"test videos : {len(test_ds)}\n")
        f.write(f"threshold   : {args.threshold}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Metrics saved to: {metrics_file}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    conf_mat = confusion_matrix(labels, pred_labels)
    logger.info(f"Confusion matrix:\n{conf_mat}")
    plot_confusion_matrix(
        conf_mat,
        class_names=["Fake", "Real"],
        title="Test Confusion Matrix",
        save_path=str(out_dir / "confusion_matrix.png"),
        show=False,
    )

    # ── ROC curve ────────────────────────────────────────────────────────────
    EER, optimal_thr, FRR_list, FAR_list = get_EER_states(probs_real, labels)
    plot_roc_curve(
        FRR_list, FAR_list, EER,
        title=f"ROC Curve (EER={EER:.4f})",
        save_path=str(out_dir / "roc_curve.png"),
        show=False,
    )
    logger.info(f"EER={EER:.4f}  Optimal threshold={optimal_thr:.4f}")

    # ── Save per-video predictions ────────────────────────────────────────────
    if args.save_predictions:
        df = pd.DataFrame({
            "true_label":  labels,
            "pred_label":  pred_labels,
            "prob_real":   probs_real,
            "prob_fake":   1.0 - probs_real,
        })
        df.to_csv(out_dir / "predictions.csv", index=False)
        logger.info(f"Predictions saved to: {out_dir / 'predictions.csv'}")

    logger.info(f"\nTest complete! Results → {out_dir}")


if __name__ == "__main__":
    main()
