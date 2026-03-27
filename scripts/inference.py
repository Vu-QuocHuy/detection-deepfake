#!/usr/bin/env python3
"""
Inference Script — TemporalTriStreamDetector
=============================================
Run video-level deepfake detection on:
  • A directory of face-crop images (extracted from one video)
  • A parent directory containing multiple per-video subdirectories

The model requires a sequence of T frames per video.
If a directory has more than T images, T are sampled uniformly.
If fewer than T, the last frame is repeated to pad.

Usage examples:
  # Single video (directory of face crops)
  python scripts/inference.py \\
    --input data/extracted/test/Deepfakes/video_001 \\
    --checkpoint outputs/temporal_b4_T16/checkpoints/best_model.pth

  # Multiple videos (parent directory with per-video subdirs)
  python scripts/inference.py \\
    --input data/extracted/test/Deepfakes \\
    --checkpoint outputs/temporal_b4_T16/checkpoints/best_model.pth \\
    --output results.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from deepfake_detector.data import get_val_transforms
from deepfake_detector.models.multistream import _effnet_input_size
from deepfake_detector.models.temporal import TemporalTriStreamDetector
from deepfake_detector.utils import setup_logger

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _load_frames(frame_paths: list[str], transform, n_frames: int) -> torch.Tensor:
    """
    Load exactly n_frames from frame_paths into [1, T, 3, H, W].
    Uniform sampling if len > n_frames, pad with last frame if len < n_frames.
    """
    paths = sorted(frame_paths)
    total = len(paths)

    if total == 0:
        raise ValueError("No frames found")

    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = list(range(total)) + [total - 1] * (n_frames - total)

    frames = []
    for i in indices:
        img = cv2.imread(paths[i])
        if img is None:
            img = cv2.imread(paths[min(i + 1, total - 1)])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = transform(image=img_rgb)["image"]   # [3, H, W]
        frames.append(t)

    seq = torch.stack(frames, dim=0).unsqueeze(0)   # [1, T, 3, H, W]
    return seq


@torch.no_grad()
def predict_video(model: TemporalTriStreamDetector, seq: torch.Tensor,
                  device: torch.device, bce_output: bool, threshold: float):
    """
    Returns (label, prob_fake, prob_real).
    prob_real = P(real), prob_fake = P(fake).
    """
    model.eval()
    seq = seq.to(device)
    logit = model(seq)

    if bce_output:
        prob_real = torch.sigmoid(logit.squeeze()).item()
    else:
        import torch.nn.functional as F
        prob_real = F.softmax(logit, dim=1)[0, 1].item()

    prob_fake = 1.0 - prob_real
    label = "REAL" if prob_real >= threshold else "FAKE"
    return label, prob_fake, prob_real


def get_video_dirs(input_path: Path) -> list[tuple[str, Path]]:
    """
    Return list of (video_name, dir_path).
    If input_path itself contains images → treat as single video.
    If input_path contains subdirs → treat each subdir as a video.
    """
    direct_images = [p for p in input_path.iterdir()
                     if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if direct_images:
        return [(input_path.name, input_path)]

    subdirs = sorted([p for p in input_path.iterdir() if p.is_dir()])
    return [(d.name, d) for d in subdirs]


def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Inference — TemporalTriStreamDetector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      type=str, required=True,
                        help="Directory of face crops (single video) or parent dir (multiple videos)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained TemporalTriStreamDetector checkpoint")
    parser.add_argument("--model",      type=str, default="efficientnet-b4",
                        help="EfficientNet backbone (overridden by checkpoint if available)")
    parser.add_argument("--n-frames",   type=int, default=16,
                        help="T frames per video (overridden by checkpoint if available)")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="P(real) threshold: >= threshold → REAL")
    parser.add_argument("--output",     type=str, default=None,
                        help="Save results to CSV (optional)")

    args = parser.parse_args()

    logger = setup_logger(name="inference", level="INFO")
    logger.info("=" * 60)
    logger.info("DeepFake Detection Inference")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    logger.info(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone   = ckpt.get("backbone",   args.model)
    n_frames   = ckpt.get("n_frames",   args.n_frames)
    bce_output = ckpt.get("bce_output", True)
    logger.info(f"  backbone={backbone}  n_frames={n_frames}  bce={bce_output}")

    model = TemporalTriStreamDetector(
        backbone=backbone,
        n_frames=n_frames,
        bce_output=bce_output,
        pretrained=False,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)

    image_size = _effnet_input_size(backbone)
    transform  = get_val_transforms(image_size)

    # ── Collect videos ────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.is_dir():
        logger.error(f"Input must be a directory: {input_path}")
        return

    videos = get_video_dirs(input_path)
    logger.info(f"Found {len(videos)} video(s) in {input_path}")

    results = []
    for vid_name, vid_dir in tqdm(videos, desc="Processing"):
        frame_paths = [str(p) for p in vid_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS]
        if not frame_paths:
            logger.warning(f"No images in {vid_dir}, skipping.")
            continue

        try:
            seq = _load_frames(frame_paths, transform, n_frames)
            label, prob_fake, prob_real = predict_video(
                model, seq, device, bce_output, args.threshold
            )
            results.append({
                "video":      vid_name,
                "prediction": label,
                "prob_real":  round(prob_real, 4),
                "prob_fake":  round(prob_fake, 4),
                "confidence": round(max(prob_real, prob_fake), 4),
            })
        except Exception as e:
            logger.error(f"Error on {vid_dir}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    total = len(results)
    n_real = sum(1 for r in results if r["prediction"] == "REAL")
    n_fake = total - n_real
    print(f"Total videos  : {total}")
    print(f"Predicted REAL: {n_real}")
    print(f"Predicted FAKE: {n_fake}")
    print("=" * 60)

    if total <= 20:
        for r in results:
            print(f"  {r['video']:40s}  {r['prediction']}  "
                  f"(real={r['prob_real']:.3f}, fake={r['prob_fake']:.3f})")

    if args.output:
        pd.DataFrame(results).to_csv(args.output, index=False)
        logger.info(f"Results saved to: {args.output}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
