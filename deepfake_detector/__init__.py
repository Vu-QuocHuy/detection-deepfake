"""
DeepFake Detection using EfficientNet
A robust, production-ready deepfake detection framework.
"""

__version__ = "2.0.0"
__author__ = "Umit Kacar"
__license__ = "MIT"

from deepfake_detector.models import TriStreamDeepFakeDetector, TemporalTriStreamDetector
from deepfake_detector.data import (
    CombinedVideoDataset,
    VideoSequenceDataset,
    get_train_transforms,
    get_val_transforms,
)
from deepfake_detector.utils import setup_logger, calculate_metrics

__all__ = [
    "TriStreamDeepFakeDetector",
    "TemporalTriStreamDetector",
    "VideoSequenceDataset",
    "CombinedVideoDataset",
    "get_train_transforms",
    "get_val_transforms",
    "setup_logger",
    "calculate_metrics",
]