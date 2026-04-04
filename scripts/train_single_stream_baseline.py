#!/usr/bin/env python3
"""Train baseline một luồng: patch temporal rồi gọi train.main() (checkpoint có single_stream)."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _apply_single_stream_patch(stream: str) -> None:
    import deepfake_detector.models.temporal as temporal_mod
    from deepfake_detector.baselines.temporal_single_stream import TemporalSingleStreamAblation

    class _Patched(TemporalSingleStreamAblation):
        def __init__(self, *a, **k):
            super().__init__(*a, single_stream=stream, **k)

    temporal_mod.TemporalTriStreamDetector = _Patched  # noqa: SLF001


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--single-stream",
        type=str,
        required=True,
        choices=["rgb", "freq", "srm"],
    )
    pre_args, rest = pre.parse_known_args()
    _apply_single_stream_patch(pre_args.single_stream)

    train_path = Path(__file__).resolve().parent / "train.py"
    spec = importlib.util.spec_from_file_location("_ff_train_single_stream", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {train_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]] + rest
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
