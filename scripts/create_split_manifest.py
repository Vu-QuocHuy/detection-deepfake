#!/usr/bin/env python3
"""
Create reproducible train/val/test split manifest for FF++ style folders.

Expected input structure:
  <input_root>/<class_name>/*.mp4

Output manifest JSON format:
{
  "seed": 42,
  "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
  "classes": ["original", "deepfakes", ...],
  "data": {
    "train": {"original": ["..."], "deepfakes": ["..."], ...},
    "val": {...},
    "test": {...}
  }
}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos(class_dir: Path) -> List[str]:
    files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    files = sorted(files, key=lambda p: p.name)
    return [str(p.resolve()) for p in files]


def split_items(items: List[str], train_ratio: float, val_ratio: float, test_ratio: float, rng: random.Random):
    if not items:
        return [], [], []
    temp = items[:]
    rng.shuffle(temp)
    n = len(temp)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Keep all remainder in test.
    n_test = n - n_train - n_val
    train = temp[:n_train]
    val = temp[n_train:n_train + n_val]
    test = temp[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Create split manifest for deepfake datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=str, required=True, help="Root directory containing class subfolders")
    parser.add_argument("--classes", type=str, nargs="+", required=True, help="Class folder names to include")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, required=True, help="Output manifest JSON path")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    rng = random.Random(args.seed)

    manifest: Dict[str, object] = {
        "seed": args.seed,
        "splits": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "classes": args.classes,
        "data": {
            "train": {},
            "val": {},
            "test": {},
        },
    }

    for class_name in args.classes:
        class_dir = input_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")

        videos = list_videos(class_dir)
        train, val, test = split_items(
            videos, args.train_ratio, args.val_ratio, args.test_ratio, rng
        )
        manifest["data"]["train"][class_name] = train
        manifest["data"]["val"][class_name] = val
        manifest["data"]["test"][class_name] = test
        print(
            f"{class_name}: total={len(videos)} train={len(train)} val={len(val)} test={len(test)}"
        )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved split manifest to: {output}")


if __name__ == "__main__":
    main()

