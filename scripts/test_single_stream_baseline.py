#!/usr/bin/env python3
"""Test checkpoint baseline một luồng (không dùng test.py / fusion)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detector.baselines.temporal_single_stream import TemporalSingleStreamAblation
from deepfake_detector.data import get_val_transforms
from deepfake_detector.data.dataset import CombinedVideoDataset, VideoSequenceDataset
from deepfake_detector.models.multistream import _effnet_input_size
from deepfake_detector.utils import (
    calculate_comprehensive_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_metrics,
    setup_logger,
)
from deepfake_detector.utils.metrics import get_EER_states


@torch.no_grad()
def evaluate(model: TemporalSingleStreamAblation, loader: DataLoader,
             device: torch.device, bce_output: bool):
    model.eval()
    all_probs, all_labels = [], []
    for seqs, labels in tqdm(loader, desc="Testing"):
        seqs = seqs.to(device, non_blocking=True)
        logits = model(seqs)
        if bce_output:
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        else:
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.astype(np.float64))
        all_labels.extend(labels.numpy())
    return np.asarray(all_probs), np.asarray(all_labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate single-stream baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-real", nargs="+", required=True)
    parser.add_argument("--test-fake", nargs="+", required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="test_results_single_stream")
    parser.add_argument("--model", type=str, default="efficientnet-b4")
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--use-val-threshold",
        action="store_true",
        help="Dung metrics.val_optimal_threshold trong checkpoint.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--single-stream",
        type=str,
        default=None,
        choices=["rgb", "freq", "srm"],
        help="Neu checkpoint khong co single_stream (ban cu).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="test_single_stream",
        log_file=str(out_dir / "test.log"),
        level="INFO",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = ckpt.get("backbone", args.model)
    n_frames = ckpt.get("n_frames", args.n_frames)
    bce_output = ckpt.get("bce_output", True)
    ckpt_stream = ckpt.get("single_stream")
    spatial_forward = ckpt.get("spatial_forward")

    if spatial_forward == "single_stream":
        if ckpt_stream not in ("rgb", "freq", "srm"):
            logger.error("Checkpoint single_stream invalid.")
            sys.exit(1)
        if args.single_stream is not None and args.single_stream != ckpt_stream:
            logger.error("--single-stream khac voi checkpoint.")
            sys.exit(1)
        stream = ckpt_stream
    elif args.single_stream is not None:
        stream = args.single_stream
        logger.warning("Checkpoint thieu metadata; dung --single-stream ban cung cap.")
    else:
        logger.error("Thieu single_stream trong checkpoint; hay them --single-stream.")
        sys.exit(1)

    logger.info(f"backbone={backbone} n_frames={n_frames} single_stream={stream}")

    thr_eff = float(args.threshold)
    if args.use_val_threshold:
        vm = ckpt.get("metrics") or {}
        vo = vm.get("val_optimal_threshold")
        if vo is not None:
            thr_eff = float(vo)
            logger.info(f"decision_threshold={thr_eff:.4f} (checkpoint val_optimal_threshold)")
        else:
            logger.warning(
                "Checkpoint khong co val_optimal_threshold; dung --threshold=%s",
                args.threshold,
            )

    image_size = _effnet_input_size(backbone)
    tfm = get_val_transforms(image_size)
    test_ds = CombinedVideoDataset(
        VideoSequenceDataset(
            [(p, -1) for p in args.test_real], is_real=True, transform=tfm,
            n_frames=n_frames, sampling="uniform",
        ),
        VideoSequenceDataset(
            [(p, -1) for p in args.test_fake], is_real=False, transform=tfm,
            n_frames=n_frames, sampling="uniform",
        ),
    )
    logger.info(f"Test set: {len(test_ds)} videos")
    if len(test_ds) == 0:
        logger.error("No test videos. Kiem tra --test-real / --test-fake.")
        sys.exit(1)

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TemporalSingleStreamAblation(
        backbone=backbone,
        n_frames=n_frames,
        bce_output=bce_output,
        pretrained=False,
        single_stream=stream,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model = model.to(device)
    model.set_phase(2)

    probs, labels = evaluate(model, loader, device, bce_output)
    probs_real = probs
    pred_labels = (probs_real >= thr_eff).astype(np.int64)
    metrics = calculate_comprehensive_metrics(
        probs=probs_real,
        labels=labels,
        fixed_decision_threshold=thr_eff,
    )
    print_metrics(metrics, title=f"Single-stream ({stream})")

    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"single_stream: {stream}\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"threshold_used: {thr_eff}\n")
        f.write(f"use_val_threshold: {args.use_val_threshold}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    plot_confusion_matrix(
        confusion_matrix(labels, pred_labels),
        class_names=["Fake", "Real"],
        title=f"Confusion ({stream})",
        save_path=str(out_dir / "confusion_matrix.png"),
        show=False,
    )
    EER, optimal_thr, FRR_list, FAR_list = get_EER_states(probs_real, labels)
    plot_roc_curve(
        FRR_list, FAR_list, EER,
        title=f"ROC EER={EER:.4f}",
        save_path=str(out_dir / "roc_curve.png"),
        show=False,
    )
    if args.save_predictions:
        pd.DataFrame({
            "true_label": labels,
            "pred_label": pred_labels,
            "prob_real": probs_real,
            "prob_fake": 1.0 - probs_real,
        }).to_csv(out_dir / "predictions.csv", index=False)

    logger.info(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
