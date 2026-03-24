#!/usr/bin/env python3
"""
Extract face crops from videos according to a split manifest JSON.

This script helps enforce per-video frame quotas for balanced REAL/FAKE-any:
 - e.g. original -> 24 frames/video
 - fake classes -> 4 frames/video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from extract_faces import FastMTCNN, process_video


def _load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_quota(class_name: str, real_class: str, real_quota: int, fake_quota: int) -> int:
    return real_quota if class_name == real_class else fake_quota


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops using split manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to split manifest JSON")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True, help="Split to extract")
    parser.add_argument("--output-root", type=str, required=True, help="Root output folder for extracted faces")
    parser.add_argument("--real-class", type=str, default="original", help="Name of the real class")
    parser.add_argument("--real-frames-per-video", type=int, default=24, help="Frames per real video")
    parser.add_argument("--fake-frames-per-video", type=int, default=4, help="Frames per fake video")
    parser.add_argument("--batch-size", type=int, default=60, help="Batch size passed to face extractor")
    parser.add_argument("--sampling-strategy", type=str, default="uniform", choices=["uniform", "stride"], help="Sampling strategy")
    parser.add_argument("--frame-skip", type=int, default=30, help="Stride mode: process every Nth frame")
    parser.add_argument("--stride", type=int, default=1, help="MTCNN detect stride")
    parser.add_argument("--margin", type=int, default=50, help="MTCNN margin")
    parser.add_argument("--min-face-size", type=int, default=100, help="MTCNN minimum face size")
    parser.add_argument("--square-margin", type=float, default=0.30, help="Square crop margin fraction")
    parser.add_argument("--pad-mode", type=str, default="reflect", choices=["reflect", "constant"], help="Padding mode for square crop")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    manifest = _load_manifest(manifest_path)
    split_data: Dict[str, List[str]] = manifest["data"][args.split]

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mtcnn = FastMTCNN(
        stride=args.stride,
        margin=args.margin,
        min_face_size=args.min_face_size,
        device=args.device,
        square_crop=True,
        square_margin=args.square_margin,
        pad_mode=args.pad_mode,
    )

    grand_total_frames = 0
    grand_total_faces = 0

    classes = sorted(split_data.keys())
    for class_name in classes:
        video_paths = split_data[class_name]
        class_out = output_root / args.split / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        frames_per_video = _resolve_quota(
            class_name,
            real_class=args.real_class,
            real_quota=args.real_frames_per_video,
            fake_quota=args.fake_frames_per_video,
        )
        print(
            f"\n[{args.split}] class={class_name} videos={len(video_paths)} "
            f"frames_per_video={frames_per_video}"
        )

        for video_path in tqdm(video_paths, desc=f"{args.split}:{class_name}"):
            try:
                frames, faces = process_video(
                    video_path=video_path,
                    output_dir=str(class_out),
                    mtcnn=mtcnn,
                    batch_size=args.batch_size,
                    frame_skip=args.frame_skip,
                    frames_per_video=frames_per_video,
                    sampling_strategy=args.sampling_strategy,
                )
                grand_total_frames += frames
                grand_total_faces += faces
            except Exception as exc:
                print(f"Error processing {video_path}: {exc}")

    print("\nExtraction complete.")
    print(f"Total sampled frames: {grand_total_frames}")
    print(f"Total extracted faces: {grand_total_faces}")
    print(f"Saved to: {output_root}")


if __name__ == "__main__":
    main()

